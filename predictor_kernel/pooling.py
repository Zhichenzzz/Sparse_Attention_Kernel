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


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"



# We don't run auto-tuning every time to keep the tutorial fast. Keeping
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
    for BM in [64]\
    for BN in [128]\
    for s in ([1] if is_hip() else [3, 4, 7])\
    for w in [4, 8]\
]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True



class pooling_mm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, w):
        BATCH, N_HEADS, N_CTX, HEAD_DIM = q.shape
        # w n_head, hidden_dim, head_dim
        # shape constraints
        assert HEAD_DIM == w.shape[1]
        # HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # # when v is in float8_e5m2 it is transposed.
        # HEAD_DIM_V = v.shape[-1]
        # assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        # assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        # NUM_HEADS_Q, NUM_HEADS_K, NUM_HEADS_V = q.shape[1], k.shape[1], v.shape[1]
        # assert NUM_HEADS_K == NUM_HEADS_V
        # n_rep = NUM_HEADS_Q // NUM_HEADS_K
        
        autotuned_config = max_pooling.configs[0]
        BLOCK_M = autotuned_config.kwargs["BLOCK_M"]
        p_out = torch.empty((BATCH, N_HEADS, N_CTX // BLOCK_M, HEAD_DIM), device=q.device, dtype=q.dtype)
        n_d = triton.cdiv(q.shape[2], BLOCK_M)
        o = torch.empty((BATCH, N_HEADS, n_d, w.shape[-1]), device=q.device, dtype=q.dtype)
        # R = torch.full((q.shape[0], q.shape[1], q.shape[2], n_d), -65504.0, device=q.device, dtype=torch.float16)
        # Po = torch.zeros((q.shape[0], q.shape[1], n_d, n_d), device=q.device, dtype=torch.float16)
        # stage = 3 if causal else 1
        extra_kern_args = {}
        # Tuning for AMD target
        if is_hip():
            waves_per_eu = 3 if HEAD_DIM <= 64 else 2
            extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}

        grid = lambda args: (triton.cdiv(N_CTX, args["BLOCK_M"]), BATCH * N_HEADS, 1)
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        max_pooling[grid](
            q, p_out, #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            p_out.stride(0), p_out.stride(1), p_out.stride(2), p_out.stride(3),
            N_HEADS, N_CTX, HEAD_DIM, 
            **extra_kern_args,)
        o = torch.matmul(p_out, w)
        # matmul_kernel[grid](
        #     q, w, o, #
        #     q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
        #     w.stride(0), w.stride(1), w.stride(2),  #
        #     o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
        #     N_HEADS, N_CTX,  #
        #     w.shape[-1],  #
        #     HEAD_DIM,  #
        #     N_DOWNSAMPLE=n_d,
        #     **extra_kern_args, #
        # )

        # fused_pooling_mm[grid](
        #     q, w, o, #
        #     q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
        #     w.stride(0), w.stride(1), w.stride(2),  #
        #     o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
        #     N_HEADS, N_CTX,  #
        #     w.shape[-1],  #
        #     HEAD_DIM,  #
        #     N_DOWNSAMPLE=n_d,
        #     **extra_kern_args, #
        # )

        return o
@triton.autotune(list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM"])
@triton.jit
def max_pooling(Q, Out,
                stride_qz, stride_qh, stride_qm, stride_qk,  #
                stride_oz, stride_oh, stride_om, stride_ok,  #
                H, N_CTX,  #
                HEAD_DIM: tl.constexpr,  #
                BLOCK_M: tl.constexpr,  #
                BLOCK_N: tl.constexpr,  #
                ):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh
    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    Out_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(N_CTX // BLOCK_M, HEAD_DIM),
        strides=(stride_om, stride_ok),
        offsets=(start_m, 0),
        block_shape=(1, HEAD_DIM),
        order=(0, 1),
    )
    q = tl.load(Q_block_ptr)
    # print(q.shape)
    q = tl.max(q, 0).to(tl.float16)[None, :]
    # print(q.shape)
    tl.store(Out_block_ptr, q.to(Out.type.element_ty))

# @triton.autotune(list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM"])
# @triton.jit
# def matmul_kernel(
#         a_ptr, b_ptr, c_ptr,
# @triton.autotune(list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM"])
# @triton.jit
# def fused_pooling_mm(Q, W, Out, #
#                 stride_qz, stride_qh, stride_qm, stride_qk,  #
#                 stride_wh, stride_wk, stride_wn,  #
#                 stride_oz, stride_oh, stride_om, stride_on,  #
#                 H, N_CTX,  #
#                 HIDDEN_DIM: tl.constexpr,  #
#                 HEAD_DIM: tl.constexpr,  #
#                 N_DOWNSAMPLE: tl.constexpr,  #
#                 BLOCK_M: tl.constexpr,  #
#                 BLOCK_N: tl.constexpr,  #
#               ):
#     tl.static_assert(BLOCK_N <= HEAD_DIM)
#     start_m = tl.program_id(0)
#     off_hz = tl.program_id(1)
#     off_z = off_hz // H
#     off_h = off_hz % H
#     q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
#     w_offset = off_h.to(tl.int64) * stride_wh

#     # block pointers
#     Q_block_ptr = tl.make_block_ptr(
#         base=Q + q_offset,
#         shape=(N_CTX, HEAD_DIM),
#         strides=(stride_qm, stride_qk),
#         offsets=(start_m * BLOCK_M, 0),
#         block_shape=(BLOCK_M, BLOCK_N),
#         order=(1, 0),
#     )

#     W_block_ptr = tl.make_block_ptr(
#         base=W + w_offset,
#         shape=(HEAD_DIM, HIDDEN_DIM), 
#         strides=(stride_wk, stride_wn),
#         offsets=(0, 0),
#         block_shape=(BLOCK_N, HIDDEN_DIM),
#         order=(0, 1),
#     )
#     O_block_ptr = tl.make_block_ptr(
#         base=Out + q_offset,
#         shape=(N_DOWNSAMPLE, HIDDEN_DIM),
#         strides=(stride_om, stride_on),
#         offsets=(start_m, 0),
#         block_shape=(1, HIDDEN_DIM),
#         order=(1, 0),
#     )  
#     # initialize offsets
#     offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
#     offs_n = tl.arange(0, BLOCK_N)
#     # initialize pointer to m and l
#     # m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
#     # l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
#     acc = tl.zeros([1, HIDDEN_DIM], dtype=tl.float16)

#     # load q: it will stay in SRAM throughout
#     # q = tl.load(Q_block_ptr)
#     # stage 1: off-band
#     # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
#     # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
#     acc = kernel(acc, Q_block_ptr, W_block_ptr,   #
#                 BLOCK_M, HEAD_DIM, BLOCK_N,  #
#                 N_CTX, HEAD_DIM,  #
#                                     )
#     tl.store(O_block_ptr, acc.to(Out.type.element_ty))

# @triton.jit
# def kernel(acc, Q_block_ptr,  #
#             W_block_ptr,  #
#             BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
#             N_CTX: tl.constexpr, HIDDEN_DIM: tl.constexpr):
    
#     for start_n in range(0, HEAD_DIM, BLOCK_N):
#         q = tl.load(Q_block_ptr)
#         # print(q.shape)
#         w = tl.load(W_block_ptr)
#         start_n = tl.multiple_of(start_n, BLOCK_N)
#         # -- compute qk ----
#         q_max = tl.max(q, 0).to(tl.float16)[None, :]
#         # print(q_max.shape)
#         w = tl.load(W_block_ptr)
#         print(w.shape)
#         acc = tl.dot(q_max, w, acc)

#         Q_block_ptr = tl.advance(Q_block_ptr, (0, BLOCK_N))
#         W_block_ptr = tl.advance(W_block_ptr, (BLOCK_N, 0))
    
#     return acc


pooling = pooling_mm.apply


@pytest.mark.parametrize("Z, H, N_CTX, HEAD_DIM, HIDDEN_DIM", [(4, 32, 16384, 128, 256)])
def test_op(Z, H, N_CTX, HEAD_DIM, HIDDEN_DIM, dtype=torch.float16):
    torch.manual_seed(20)
    q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    w = (torch.empty((H, HEAD_DIM, HIDDEN_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    tri_out = pooling(q, w)
    q = q.transpose(-1, -2).reshape(-1, HEAD_DIM, N_CTX)
    q = torch.max_pool1d(q, kernel_size=64, stride=64, ceil_mode=True)
    q = q.transpose(-1, -2).reshape(Z, H, -1, HEAD_DIM)
    ref_out = torch.matmul(q, w)
    assert torch.allclose(tri_out, ref_out, atol=1e-3, rtol=1e-3)



try:
    from flash_attn.flash_attn_interface import \
        flash_attn_qkvpacked_func as flash_attn_func
    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False

TORCH_HAS_FP8 = False
BATCH, N_HEADS, HEAD_DIM, HIDDEN_DIM= 2, 32, 128, 128
# vary seq length for fixed head and batch=4
configs = []
for mode in ["fwd"]:
    for causal in [True, False]:
        if mode == "bwd" and not causal:
            continue
        configs.append(
            triton.testing.Benchmark(
                x_names=["N_CTX"],
                x_vals=[2**i for i in range(10, 16)],
                line_arg="provider",
                line_vals=["triton-fp16"] + (["triton-fp8"] if TORCH_HAS_FP8 else []) +
                (["flash"] if HAS_FLASH else []),
                line_names=["Triton [FP16] + Pooling"] + (["Triton [FP8] + Pooling"] if TORCH_HAS_FP8 else []) +
                (["Flash-2"] if HAS_FLASH else []),
                styles=[("red", "-"), ("blue", "-"), ("green", "-")],
                ylabel="ms",
                plot_name=f"fused-pooling-attention-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}-{mode}-causal={causal}",
                args={
                    "H": N_HEADS,
                    "BATCH": BATCH,
                    "HEAD_DIM": HEAD_DIM,
                    "HIDDEN_DIM": HIDDEN_DIM,
                    "mode": mode,
                    "causal": causal,
                },
            ))
        
@pytest.mark.skip
def test_pooling(q,w):
    BATCH, N_HEADS, N_CTX, HEAD_DIM = q.shape
    q = q.transpose(-1, -2).reshape(-1, HEAD_DIM, N_CTX)
    q = torch.max_pool1d(q, kernel_size=64, stride=64, ceil_mode=True)
    q = q.transpose(-1, -2).reshape(BATCH, N_HEADS, -1, HEAD_DIM)
    o = torch.matmul(q, w)
    return o

@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, HEAD_DIM, HIDDEN_DIM, causal, mode, provider, device="cuda"):
    assert mode in ["fwd"]
    warmup = 25
    rep = 100
    dtype = torch.float16
    if "triton" in provider:
        q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        w = (torch.empty((H, HEAD_DIM, HIDDEN_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
        fn = lambda: pooling(q, w)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    if provider == "flash":
        q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        w = (torch.empty((H, HEAD_DIM, HIDDEN_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
        fn = lambda: test_pooling(q, w)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    return ms


if __name__ == "__main__":
    # only works on post-Ampere GPUs right now
    bench_flash_attention.run(save_path="./test/", print_data=True)
    pytest.main([__file__])