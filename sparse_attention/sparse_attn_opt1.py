"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)
Credits: OpenAI kernel team

Extra Credits:
- Original flash attention paper (https://arxiv.org/abs/2205.14135)
- Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)

"""

import pytest
import torch

import triton
import triton.language as tl
import torch.nn.functional as F


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


@triton.jit
def _attn_fwd_inner_casual_false(acc, l_i, m_i, q, nnz_id,  #
                    K_block_ptr, V_block_ptr,  #
                    start_m, qk_scale,  #
                    TOPK: tl.constexpr,  #
                    l_offset: tl.constexpr, stride_lm: tl.constexpr, stride_ln: tl.constexpr,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr, fp8_v: tl.constexpr):
    
    l_offset =  nnz_id + l_offset + start_m * stride_lm
    for nnz_id in range(TOPK):
        present_nnz_id = tl.load(l_offset + nnz_id * stride_ln)
        start_n = present_nnz_id * BLOCK_N
        start_n = tl.multiple_of(start_n, BLOCK_N)
        start_n = start_n.to(tl.int32)

        # -- compute qk ----
        k = tl.load(tl.advance(K_block_ptr, (0, start_n)), boundary_check=(0, 1), padding_option="zero") # make sure that the last block is padded with zero
        qk = tl.dot(q, k)
        max = tl.max(qk, 1)
        m_ij = tl.maximum(m_i, max)
        qk -= m_ij[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = tl.load(tl.advance(V_block_ptr, (start_n, 0)), boundary_check=(0, 1), padding_option="zero") # make sure that the last block is padded with zero
        p = p.to(tl.float16)
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        m_i = m_ij
    acc = acc / l_i[:, None]
    return acc

@triton.jit
def _attn_fwd_inner_casual_true(acc, l_i, m_i, q, nnz_id,  #
                    K_block_ptr, V_block_ptr,  #
                    start_m, qk_scale,  #
                    TOPK: tl.constexpr,  #
                    l_offset: tl.constexpr, stride_lm: tl.constexpr, stride_ln: tl.constexpr,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr, fp8_v: tl.constexpr):
    
    l_offset =  nnz_id + l_offset + start_m * stride_lm
    for nnz_id in range(TOPK):
        present_nnz_id = tl.load(l_offset + nnz_id * stride_ln)
        if start_m >= present_nnz_id:
            start_n = present_nnz_id * BLOCK_N
            start_n = tl.multiple_of(start_n, BLOCK_N)
            start_n = start_n.to(tl.int32)
            # -- compute qk ----
            k = tl.load(tl.advance(K_block_ptr, (0, start_n)), boundary_check=(0, 1), padding_option="zero") # make sure that the last block is padded with zero
            qk = tl.dot(q, k)
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk + tl.where(mask, 0, -1.0e6)
            max = tl.max(qk, 1)
            m_ij = tl.maximum(m_i, max)
            qk -= m_ij[:, None]

            p = tl.math.exp2(qk)
            l_ij = tl.sum(p, 1)
            # -- update m_i and l_i
            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            # -- update output accumulator --
            acc = acc * alpha[:, None]
            # update acc
            v = tl.load(tl.advance(V_block_ptr, (start_n, 0)), boundary_check=(0, 1), padding_option="zero") # make sure that the last block is padded with zero
            p = p.to(tl.float16)
            acc = tl.dot(p, v, acc)
            # update m_i and l_i
            m_i = m_ij
    acc = acc / l_i[:, None]
    return acc

# We don't run auto-tuning every time to keep the tutorial fast. Keeping
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
    for BM, BN in [(64, 64)]\
    for s in ([1] if is_hip() else [3, 4, 7])\
    for w in [4, 8]\
]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True


@triton.autotune(list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM"])
@triton.jit
def _attn_fwd(Q, K, V, sm_scale, M, Out,  #
              nnz_id,  #
              stride_qz, stride_qh, stride_qm, stride_qk,  #
              stride_kz, stride_kh, stride_kn, stride_kk,  #
              stride_vz, stride_vh, stride_vk, stride_vn,  #
              stride_oz, stride_oh, stride_om, stride_on,  #
              stride_lz, stride_lh, stride_lm, stride_ln,  #
              Z, H, 
              N_CTX,  #
              n_rep,  #
              TOPK: tl.constexpr,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              STAGE: tl.constexpr  #
              ):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    off_kvh = off_h // n_rep
    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_offset = off_z.to(tl.int64) * stride_kz + off_kvh.to(tl.int64) * stride_kh
    v_offset = off_z.to(tl.int64) * stride_vz + off_kvh.to(tl.int64) * stride_vh
    l_offset = off_z.to(tl.int64) * stride_lz + off_h.to(tl.int64) * stride_lh

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float8e5 else (1, 0)
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=v_order,
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + q_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero") # make sure that the last block is padded with zero
    q = (q * qk_scale).to(tl.float16)
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE == 1:
        acc = _attn_fwd_inner_casual_false(acc, l_i, m_i, q, nnz_id,  #
                                        K_block_ptr, V_block_ptr,  #
                                        start_m, qk_scale,  #
                                        TOPK,
                                        l_offset, stride_lm, stride_ln,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5  #
                                        )
    elif STAGE == 3:
        acc = _attn_fwd_inner_casual_true(acc, l_i, m_i, q, nnz_id,  #
                                        K_block_ptr, V_block_ptr,  #
                                        start_m, qk_scale,  #
                                        TOPK,
                                        l_offset, stride_lm, stride_ln,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5  #
                                        )

    # epilogue
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), boundary_check=(0, 1))


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, nnz_id, causal, sm_scale):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        NUM_HEADS_Q, NUM_HEADS_K, NUM_HEADS_V = q.shape[1], k.shape[1], v.shape[1]
        assert NUM_HEADS_K == NUM_HEADS_V
        n_rep = NUM_HEADS_Q // NUM_HEADS_K
        o = torch.empty_like(q)
        autotuned_config = _attn_fwd.configs[0]
        BLOCK_N = autotuned_config.kwargs["BLOCK_N"]
        topk = min(nnz_id.shape[-1], q.shape[2]//BLOCK_N)
        stage = 3 if causal else 1
        extra_kern_args = {}
        # Tuning for AMD target
        if is_hip():
            waves_per_eu = 3 if HEAD_DIM_K <= 64 else 2
            extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}

        grid = lambda args: (triton.cdiv(q.shape[2], args["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        _attn_fwd[grid](
            q, k, v, sm_scale, M, o,  #
            nnz_id,  #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
            nnz_id.stride(0), nnz_id.stride(1), nnz_id.stride(2), nnz_id.stride(3),  #
            q.shape[0], q.shape[1],  #
            N_CTX=q.shape[2],  #
            n_rep=n_rep,  #
            TOPK = topk,  #
            HEAD_DIM=HEAD_DIM_K,  #
            STAGE=stage,  #
            **extra_kern_args)
        return o


attention = _attention.apply

from flash_attn.flash_attn_interface import flash_attn_kvpacked_func
from utils import block_topk
import time


@pytest.mark.parametrize("Z, H, N_CTX, HEAD_DIM", [(2, 32, 4000, 128)])
@pytest.mark.parametrize("causal", [True, False])
def test_op(Z, H, N_CTX, HEAD_DIM, causal, dtype=torch.float16):
    torch.manual_seed(20)
    n_rep = 2
    q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, H//n_rep, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((Z, H//n_rep, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    sm_scale = 0.5
    # reference implementation
    k_ref = k[:, :, None, :, :].expand(Z, H//n_rep, n_rep, N_CTX, HEAD_DIM).reshape(Z, H, N_CTX, HEAD_DIM)
    v_ref = v[:, :, None, :, :].expand(Z, H//n_rep, n_rep, N_CTX, HEAD_DIM).reshape(Z, H, N_CTX, HEAD_DIM)

    q_ref = torch.nn.functional.pad(q, (0, 0, 0, (64 - N_CTX % 64) % 64), value=0)
    k_ref = torch.nn.functional.pad(k_ref, (0, 0, 0, (64 - N_CTX % 64) % 64), value=0)
    v_ref = torch.nn.functional.pad(v_ref, (0, 0, 0, (64 - N_CTX % 64) % 64), value=0)

    nearest_ctx = ((N_CTX + 63) // 64) * 64
    M = torch.tril(torch.ones((nearest_ctx, nearest_ctx), device="cuda"))
    p = torch.matmul(q_ref, k_ref.transpose(2, 3)) * sm_scale

    if causal:
        p[:, :, M == 0] = float("-inf")
    mxp = torch.max_pool2d(p, (64, 64), stride=(64, 64))
    p = block_topk(p, 3, 64)

    p = torch.softmax(p, dim=-1)
    ref_out = torch.matmul(p, v_ref)[:,:,:N_CTX,:]

    # triton implementation
    nnz_id = torch.topk(mxp, 3, dim=-1).indices
    tri_out= attention(q, k, v, nnz_id, causal, sm_scale)
    
    # compare
    assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)

try:
    from flash_attn.flash_attn_interface import \
        flash_attn_qkvpacked_func as flash_attn_func
    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False

TORCH_HAS_FP8 = None
BATCH, N_HEADS, HEAD_DIM, N_CTX = 4, 32, 128, 32768
BLOCK_SIZE = 64
# vary seq length for fixed head and batch=4
configs = []
for mode in ["fwd"]:
    for causal in [True]:
        if mode == "bwd" and not causal:
            continue
        configs.append(
            triton.testing.Benchmark(
                x_names=["topk"],
                x_vals=[i for i in range(0, 141, 5)],
                line_arg="provider",
                line_vals=["triton-fp16"] + (["flash"] if HAS_FLASH else []),
                line_names=["Triton [FP16]"] + (["Flash-2"] if HAS_FLASH else []),
                styles=[("red", "-"), ("blue", "-"), ("green", "-")],
                ylabel="ms",
                plot_name=f"sparse-attention-batch{BATCH}-head{N_HEADS}-seq{N_CTX}-causal={causal}",
                args={
                    "H": N_HEADS,
                    "BATCH": BATCH,
                    "HEAD_DIM": HEAD_DIM,
                    "N_CTX": N_CTX,
                    "BLOCK_SIZE": BLOCK_SIZE,
                    "mode": mode,
                    "causal": causal,
                },
            ))


@triton.testing.perf_report(configs)
def bench_flash_attention_gqa(BATCH, H, N_CTX, HEAD_DIM, topk, BLOCK_SIZE, causal, mode, provider, device="cuda"):
    assert mode in ["fwd"]
    warmup = 25
    rep = 100
    dtype = torch.float16
    if "triton" in provider:
        q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, H//4, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, H//4, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        sm_scale = 1.3
        M = torch.tril(torch.ones((N_CTX // BLOCK_SIZE, N_CTX // BLOCK_SIZE), device="cuda"))
        p = torch.randn((BATCH, H, N_CTX // BLOCK_SIZE, N_CTX // BLOCK_SIZE), dtype=dtype, device=device)
        if causal:
            p[:, :, M == 0] = float("-inf")
        nnz_id = torch.topk(p, topk, dim=-1).indices
        fn = lambda: attention(q, k, v, nnz_id, causal, sm_scale)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    if provider == "flash":
        q = torch.randn((BATCH, N_CTX, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        kv = torch.randn((BATCH, N_CTX, 2, H//4, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        fn = lambda: flash_attn_kvpacked_func(q, kv, causal=causal)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    return ms

if __name__ == "__main__":
    # only works on post-Ampere GPUs right now
    pytest.main([__file__])
    # bench_flash_attention.run(save_path="./fused-pooling-flashattn/", print_data=True)
    # bench_flash_attention_gqa.run(save_path="./sparse_attn_32k/", print_data=True)
    