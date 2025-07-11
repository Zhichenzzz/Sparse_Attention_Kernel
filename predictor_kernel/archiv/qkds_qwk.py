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
    for BN in [64]\
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
    def forward(ctx, q, k, w):
        BATCH, N_HEADS, N_CTX, HEAD_DIM = q.shape
        K_HEADS = k.shape[1]
        HIDDEN_DIM = w.shape[-1]
        n_rep = N_HEADS // K_HEADS
        # shape constraints
        assert HEAD_DIM == w.shape[1]
        autotuned_config = max_pooling.configs[0]
        BLOCK_M = autotuned_config.kwargs["BLOCK_M"]
        q_down = torch.empty((BATCH, N_HEADS, N_CTX // BLOCK_M, HEAD_DIM), device=q.device, dtype=q.dtype)
        k_down = torch.empty((BATCH, K_HEADS, N_CTX // BLOCK_M, HIDDEN_DIM), device=q.device, dtype=q.dtype)
        n_d = triton.cdiv(q.shape[2], BLOCK_M)
        o = torch.empty((BATCH, N_HEADS, n_d, n_d), device=q.device, dtype=q.dtype) - float("inf")
        extra_kern_args = {}
        # Tuning for AMD target
        if is_hip():
            waves_per_eu = 3 if HEAD_DIM <= 64 else 2
            extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}

        grid = lambda args: (triton.cdiv(N_CTX, args["BLOCK_M"]), BATCH * N_HEADS, 1)
        max_pooling[grid](
            q, k, q_down, k_down, n_rep,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            q_down.stride(0), q_down.stride(1), q_down.stride(2), q_down.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            k_down.stride(0), k_down.stride(1), k_down.stride(2), k_down.stride(3),
            N_HEADS, N_CTX, HEAD_DIM, HIDDEN_DIM,
            **extra_kern_args,)
        
        grid = lambda args: (triton.cdiv(n_d, args["BLOCK_M"]), BATCH * N_HEADS, 1)
        
        # qw = torch.matmul(q_down, w)
        # matmul_kernel_causal[grid](
        #     qw, k_down, o,
        #     qw.stride(0), qw.stride(1), qw.stride(2), qw.stride(3),
        #     k_down.stride(0), k_down.stride(1), k_down.stride(2), k_down.stride(3),
        #     o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        #     N_HEADS, n_rep, n_d, HIDDEN_DIM,
        #     **extra_kern_args,)
        
        fused_matmul[grid](
            q_down, k_down, w, o, n_rep,
            q_down.stride(0), q_down.stride(1), q_down.stride(2), q_down.stride(3),
            k_down.stride(0), k_down.stride(1), k_down.stride(2), k_down.stride(3),
            w.stride(0), w.stride(1), w.stride(2),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            N_HEADS, n_d, HEAD_DIM, HIDDEN_DIM, 
            **extra_kern_args,)
        
        nnz_id = torch.topk(o, 5, dim=-1).indices

        return nnz_id

@triton.autotune(list(filter(keep, configs)), key=["n_d", "HIDDEN_DIM"])
@triton.jit
def matmul_kernel_causal(
        a_ptr, b_ptr, c_ptr,
        stride_az, stride_ah, stride_am, stride_ak,  #
        stride_bz, stride_bh, stride_bk, stride_bn,  #
        stride_cz, stride_ch, stride_cm, stride_cn,  #
        H, n_rep,
        n_d: tl.constexpr,
        HIDDEN_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr, 
        BLOCK_N: tl.constexpr, 
):
    start_m = tl.program_id(axis=0)
    off_hz = tl.program_id(axis=1)
    off_z = off_hz // H
    off_h = off_hz % H
    off_bh = off_h // n_rep
    a_offset = off_z.to(tl.int64) * stride_az + off_h.to(tl.int64) * stride_ah
    b_offset = off_z.to(tl.int64) * stride_bz + off_bh.to(tl.int64) * stride_bh
    c_offset = off_z.to(tl.int64) * stride_cz + off_h.to(tl.int64) * stride_ch

    a_block_ptr = tl.make_block_ptr(
        base = a_ptr + a_offset,
        shape=(BLOCK_M, HIDDEN_DIM),
        strides=(stride_am, stride_ak),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HIDDEN_DIM),
        order=(0, 1)
    )
    b_block_ptr = tl.make_block_ptr(
        base = b_ptr + b_offset,
        shape=(HIDDEN_DIM, n_d),
        strides=(stride_bn, stride_bk),
        offsets=(0, 0),
        block_shape=(HIDDEN_DIM, BLOCK_N),
        order=(0, 1)
    )
    c_block_ptr = tl.make_block_ptr(
        base = c_ptr + c_offset,
        shape=(BLOCK_M, n_d),
        strides=(stride_cm, stride_cn),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(0, 1)
    )

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    lo, hi = 0, (start_m + 1) * BLOCK_M

    for start_n in range(lo, hi, BLOCK_N):
        q = tl.load(a_block_ptr)
        k = tl.load(b_block_ptr)
        qk = tl.dot(q, k)
        mask = offs_m[:, None] >= (start_n + offs_n[None, :])
        qk = qk + tl.where(mask, 0, -1.0e6)
        qk = qk.to(tl.float16)
        tl.store(c_block_ptr, qk)
        b_block_ptr = tl.advance(b_block_ptr, (0, BLOCK_N))
        c_block_ptr = tl.advance(c_block_ptr, (0, BLOCK_N))  


@triton.autotune(list(filter(keep, configs)), key=["n_d", "HEAD_DIM"])
@triton.jit
def fused_matmul(Q, K, W, O, n_rep,  #
                stride_qz, stride_qh, stride_qm, stride_qk,  #
                stride_kz, stride_kh, stride_km, stride_kn,  #
                stride_wh, stride_wk, stride_wn,  #
                stride_oz, stride_oh, stride_om, stride_on,  #
                H, 
                n_d: tl.constexpr,  #
                HEAD_DIM: tl.constexpr,  #
                HIDDEN_DIM: tl.constexpr,  #
                BLOCK_M: tl.constexpr,  #
                BLOCK_N: tl.constexpr,  #
                ):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    off_kh = off_h // n_rep
    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_offset = off_z.to(tl.int64) * stride_kz + off_kh.to(tl.int64) * stride_kh
    w_offset = off_h.to(tl.int64) * stride_wh
    o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh
    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(n_d, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(0, 1),
    )
    W_block_ptr = tl.make_block_ptr(
        base=W + w_offset,
        shape=(HEAD_DIM, HIDDEN_DIM),
        strides=(stride_wk, stride_wn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(HIDDEN_DIM, n_d),
        strides=(stride_kn, stride_km),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_M),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        base=O + o_offset,
        shape=(BLOCK_M, n_d),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_M),
        order=(0, 1),
    )
    q = tl.load(Q_block_ptr)
    lo, hi = 0, (start_m + 1) * BLOCK_M
    for start_k in range(lo, hi, BLOCK_M):
        acc = tl.zeros((BLOCK_M, BLOCK_M), dtype=tl.float16)
        for start_n in range(0, HIDDEN_DIM, BLOCK_N):
            
            w = tl.load(tl.advance(W_block_ptr, (0, start_n)))
            qw = tl.dot(q, w).to(tl.float16)
            k = tl.load(tl.advance(K_block_ptr, (start_n, 0)))
            o = tl.dot(qw, k).to(tl.float16)
            acc = acc + o
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = start_k + tl.arange(0, BLOCK_M)
        mask = offs_m[:, None] >= offs_n[None, :]
        acc = acc + tl.where(mask, 0, -1.0e6)
        acc = acc.to(tl.float16)
        # print(acc.shape)
        tl.store(O_block_ptr, acc)
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_M))
        O_block_ptr = tl.advance(O_block_ptr, (0, BLOCK_M))


@triton.autotune(list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM"])
@triton.jit
def max_pooling(Q, K, Q_down, K_down, n_rep,  #
                stride_qz, stride_qh, stride_qm, stride_qk,  #
                stride_qdz, stride_qdh, stride_qdm, stride_qdk,  #
                stride_kz, stride_kh, stride_km, stride_kk,  #
                stride_kdz, stride_kdh, stride_kdm, stride_kdk,  #
                H, N_CTX,  #
                HEAD_DIM: tl.constexpr,  #
                HIDDEN_DIM: tl.constexpr,  #
                BLOCK_M: tl.constexpr,  #
                BLOCK_N: tl.constexpr,  #
                ):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    off_kh = off_h // n_rep
    off_num = off_h % n_rep
    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    qd_offset = off_z.to(tl.int64) * stride_qdz + off_h.to(tl.int64) * stride_qdh
    k_offset = off_z.to(tl.int64) * stride_kz + off_kh.to(tl.int64) * stride_kh
    kd_offset = off_z.to(tl.int64) * stride_kdz + off_kh.to(tl.int64) * stride_kdh
    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    Q_down_block_ptr = tl.make_block_ptr(
        base=Q_down + qd_offset,
        shape=(N_CTX // BLOCK_M, HEAD_DIM),
        strides=(stride_qdm, stride_qdk),
        offsets=(start_m, 0),
        block_shape=(1, HEAD_DIM),
        order=(0, 1),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_km, stride_kk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    K_down_block_ptr = tl.make_block_ptr(
        base=K_down + kd_offset,
        shape=(N_CTX // BLOCK_M, HIDDEN_DIM),
        strides=(stride_kdm, stride_kdk),
        offsets=(start_m, 0),
        block_shape=(1, HEAD_DIM),
        order=(0, 1),
    )
    q = tl.load(Q_block_ptr)
    q = tl.sum(q, 0)[None, :] / BLOCK_M
    tl.store(Q_down_block_ptr, q.to(Q_down.type.element_ty))

    # K Dowmsample
    if off_num == 0:
        k = tl.load(K_block_ptr)
        k_max = tl.max(k, 0).to(tl.float16)[None, :]
        tl.store(K_down_block_ptr, k_max.to(K_down.type.element_ty))
        K_down_block_ptr = tl.advance(K_down_block_ptr, (0, HEAD_DIM))
        k_min = tl.min(k, 0).to(tl.float16)[None, :]
        tl.store(K_down_block_ptr, k_min.to(K_down.type.element_ty))
        


pooling = pooling_mm.apply

@pytest.mark.skip
def torch_predictor(q,k,w):
    BATCH, N_HEADS, N_CTX, HEAD_DIM = q.shape
    K_HEADS = k.shape[1]
    HIDDEN_DIM = w.shape[-1]

    q = q.transpose(-1, -2).reshape(-1, HEAD_DIM, N_CTX)
    q = torch.avg_pool1d(q, kernel_size=64, stride=64, ceil_mode=True)
    q = q.transpose(-1, -2).reshape(BATCH, N_HEADS, -1, HEAD_DIM)

    k = k.transpose(-1, -2).reshape(-1, HEAD_DIM, N_CTX)
    k_max = torch.max_pool1d(k, kernel_size=64, stride=64, ceil_mode=True)
    k_min = -torch.max_pool1d(-k, kernel_size=64, stride=64, ceil_mode=True)
    k = torch.cat((k_max, k_min), dim=1)
    k = k.transpose(-1, -2).reshape(BATCH, K_HEADS, -1, HIDDEN_DIM)

    p = torch.matmul(q, w)
    k = k[:, :, None, :, :].expand(BATCH, K_HEADS, N_HEADS // K_HEADS, N_CTX // 64, HIDDEN_DIM).reshape(BATCH, N_HEADS, N_CTX // 64, HIDDEN_DIM)
    o = torch.matmul(p, k.transpose(-1, -2))

    M = torch.tril(torch.ones((N_CTX // 64, N_CTX // 64), device="cuda"))
    o[:, :, M == 0] = float("-inf")
    nnz_id = torch.topk(o, 5, dim=-1).indices

    return nnz_id

@pytest.mark.parametrize("Z, H, N_CTX, HEAD_DIM, HIDDEN_DIM", [(4, 32, 16384, 128, 256)])
def test_op(Z, H, N_CTX, HEAD_DIM, HIDDEN_DIM, dtype=torch.float16):
    torch.manual_seed(20)
    q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, H // 4, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    w = (torch.empty((H, HEAD_DIM, HIDDEN_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    triton_out = pooling(q, k, w)
    q = q.transpose(-1, -2).reshape(-1, HEAD_DIM, N_CTX)
    q = torch.avg_pool1d(q, kernel_size=64, stride=64, ceil_mode=True)
    q = q.transpose(-1, -2).reshape(Z, H, -1, HEAD_DIM)

    # error is from avg_pool1d
    # assert torch.allclose(q, q_, atol=1e-5, rtol=0)

    k = k.transpose(-1, -2).reshape(-1, HEAD_DIM, N_CTX)
    k_max = torch.max_pool1d(k, kernel_size=64, stride=64, ceil_mode=True)
    k_min = -torch.max_pool1d(-k, kernel_size=64, stride=64, ceil_mode=True)
    k = torch.cat((k_max, k_min), dim=1)
    k = k.view(Z, H // 4, HIDDEN_DIM, -1)
    k = k.transpose(-1, -2).reshape(Z, H // 4, -1, HIDDEN_DIM)

    p = torch.matmul(q, w)
    k = k[:, :, None, :, :].expand(Z, H // 4, 4, N_CTX // 64, HIDDEN_DIM).reshape(Z, H, N_CTX // 64, HIDDEN_DIM)
    o = torch.matmul(p, k.transpose(-1, -2))

    M = torch.tril(torch.ones((N_CTX // 64, N_CTX // 64), device="cuda"))
    o[:, :, M == 0] = float("-inf")
    nnz_id = torch.topk(o, 5, dim=-1).indices
    assert torch.allclose(nnz_id, triton_out, atol=1e-3, rtol=0)





BATCH, N_HEADS, HEAD_DIM, HIDDEN_DIM= 4, 32, 128, 256
# vary seq length for fixed head and batch=4
configs = []
for mode in ["fwd"]:
    configs.append(
        triton.testing.Benchmark(
            x_names=["N_CTX"],
            x_vals=[2**i for i in range(10, 17)],
            line_arg="provider",
            line_vals=["triton-fp16"] + ["torch-fp16"],
            line_names=["Triton-fp16"] + ["Torch-fp16"],
            styles=[("red", "-"), ("blue", "-")],
            ylabel="ms",
            plot_name=f"pooling-matmul-batch{BATCH}-hidden{HIDDEN_DIM}-d{HEAD_DIM}-{mode}",
            args={
                "H": N_HEADS,
                "BATCH": BATCH,
                "HEAD_DIM": HEAD_DIM,
                "HIDDEN_DIM": HIDDEN_DIM,
                "mode": mode,
            },
        ))
    

@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, HEAD_DIM, HIDDEN_DIM, mode, provider, device="cuda"):
    assert mode in ["fwd"]
    warmup = 25
    rep = 100
    dtype = torch.float16
    if "triton" in provider:
        q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, H // 4, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        w = torch.randn((H, HEAD_DIM, HIDDEN_DIM), dtype=dtype, device="cuda")
        fn = lambda: pooling(q, k, w)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    if "torch" in provider:
        q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, H // 4, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        w = torch.randn((H, HEAD_DIM, HIDDEN_DIM), dtype=dtype, device="cuda")
        fn = lambda: torch_predictor(q, k, w)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    return ms


if __name__ == "__main__":
    # only works on post-Ampere GPUs right now
    bench_flash_attention.run(save_path="./test_total/", print_data=True)
    pytest.main([__file__])