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
        autotuned_config = max_pooling.configs[0]
        BLOCK_M = autotuned_config.kwargs["BLOCK_M"]
        p_out = torch.empty((BATCH, N_HEADS, N_CTX // BLOCK_M, HEAD_DIM), device=q.device, dtype=q.dtype)
        n_d = triton.cdiv(q.shape[2], BLOCK_M)
        o = torch.empty((BATCH, N_HEADS, n_d, w.shape[-1]), device=q.device, dtype=q.dtype)
        extra_kern_args = {}
        # Tuning for AMD target
        if is_hip():
            waves_per_eu = 3 if HEAD_DIM <= 64 else 2
            extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}

        grid = lambda args: (triton.cdiv(N_CTX, args["BLOCK_M"]), BATCH * N_HEADS, 1)
        max_pooling[grid](
            q, p_out, #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            p_out.stride(0), p_out.stride(1), p_out.stride(2), p_out.stride(3),
            N_HEADS, N_CTX, HEAD_DIM, 
            **extra_kern_args,)
        o = torch.matmul(p_out, w)
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
    q = tl.max(q, 0).to(tl.float16)[None, :]
    tl.store(Out_block_ptr, q.to(Out.type.element_ty))


pooling = pooling_mm.apply

@pytest.mark.skip
def test_pooling(q,k,w):
    BATCH, N_HEADS, N_CTX, HEAD_DIM = q.shape
    K_HEADS = k.shape[1]

    q = q.transpose(-1, -2).reshape(-1, HEAD_DIM, N_CTX)
    q = torch.max_pool1d(q, kernel_size=64, stride=64, ceil_mode=True)
    q = q.transpose(-1, -2).reshape(BATCH, N_HEADS, -1, HEAD_DIM)

    k = k.transpose(-1, -2).reshape(-1, HEAD_DIM, N_CTX)
    k = torch.max_pool1d(k, kernel_size=64, stride=64, ceil_mode=True)
    k = k.transpose(-1, -2).reshape(BATCH, K_HEADS, -1, HEAD_DIM)

    p = torch.matmul(q, w)
    k = k[:, :, None, :, :].expand(BATCH, K_HEADS, N_HEADS // K_HEADS, N_CTX // 64, HEAD_DIM).reshape(BATCH, N_HEADS, N_CTX // 64, HEAD_DIM)
    o = torch.matmul(p, k.transpose(-1, -2))
    return o

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



BATCH, N_HEADS, HEAD_DIM, HIDDEN_DIM= 4, 32, 128, 128
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
        w = torch.randn((H, HEAD_DIM, HIDDEN_DIM), dtype=dtype, device="cuda")
        fn = lambda: pooling(q, w)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    if "torch" in provider:
        q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, H // 4, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        w = torch.randn((H, HEAD_DIM, HIDDEN_DIM), dtype=dtype, device="cuda")
        fn = lambda: test_pooling(q, k, w)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    return ms


if __name__ == "__main__":
    # only works on post-Ampere GPUs right now
    bench_flash_attention.run(save_path="./test/", print_data=True)
    # pytest.main([__file__])