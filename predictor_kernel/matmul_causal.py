import torch

import triton
import triton.language as tl

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

def get_autotune_config():
    cuda_configs = [
        triton.Config(
            {'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN},
            num_stages=ns, num_warps=nw
        )
        for BM, BN in [(64, 64), (32, 32)]\
        for ns in ([3, 4, 7])\
        for nw in ([4, 8])\
        
    ]
    return cuda_configs

@triton.autotune(
    configs=get_autotune_config(),
    key=['Z', 'H', 'N', 'K'],
)
@triton.jit
def matmul_kernel_causal(
        a_ptr, b_ptr, c_ptr,
        stride_az, stride_ah, stride_am, stride_ak,  #
        stride_bz, stride_bh, stride_bk, stride_bn,  #
        stride_cz, stride_ch, stride_cm, stride_cn,  #
        Z, H, n_rep,
        N: tl.constexpr,
        K: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr, 
        BLOCK_SIZE_N: tl.constexpr, 
):
    off_hz = tl.program_id(axis=0)
    start_m = tl.program_id(axis=1)
    off_z = off_hz // H
    off_h = off_hz % H
    off_bh = off_h // n_rep
    a_offset = off_z.to(tl.int64) * stride_az + off_h.to(tl.int64) * stride_ah
    b_offset = off_z.to(tl.int64) * stride_bz + off_bh.to(tl.int64) * stride_bh
    c_offset = off_z.to(tl.int64) * stride_cz + off_h.to(tl.int64) * stride_ch

    a_block_ptr = tl.make_block_ptr(
        base = a_ptr + a_offset,
        shape=(BLOCK_SIZE_M, K),
        strides=(stride_am, stride_ak),
        offsets=(start_m * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, K),
        order=(0, 1)
    )
    b_block_ptr = tl.make_block_ptr(
        base = b_ptr + b_offset,
        shape=(K, N),
        strides=(stride_bn, stride_bk),
        offsets=(0, 0),
        block_shape=(K, BLOCK_SIZE_N),
        order=(0, 1)
    )
    c_block_ptr = tl.make_block_ptr(
        base = c_ptr + c_offset,
        shape=(BLOCK_SIZE_M, N),
        strides=(stride_cm, stride_cn),
        offsets=(start_m * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(0, 1)
    )

    offs_m = start_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    lo, hi = 0, (start_m + 1) * BLOCK_SIZE_M

    for start_n in range(lo, hi, BLOCK_SIZE_N):
        q = tl.load(a_block_ptr)
        k = tl.load(b_block_ptr)
        qk = tl.dot(q, k)
        mask = offs_m[:, None] >= (start_n + offs_n[None, :])
        qk = qk + tl.where(mask, 0, -1.0e6)
        qk = qk.to(tl.float16)
        tl.store(c_block_ptr, qk)
        b_block_ptr = tl.advance(b_block_ptr, (0, BLOCK_SIZE_N))
        c_block_ptr = tl.advance(c_block_ptr, (0, BLOCK_SIZE_N))  


def matmul(a, b, causal=False):
    # Check constraints.
    assert a.is_contiguous(), "Matrix A must be contiguous"
    Z, H_a, M, K = a.shape
    Z, H_b, M, K = b.shape
    n_rep = H_a // H_b
    # Allocates output.
    c = torch.empty((Z, H_a, M, M), device=a.device, dtype=torch.float16) - 1.0e6
    grid = lambda META: (Z * H_a, triton.cdiv(M, META['BLOCK_SIZE_M']) )
    matmul_kernel_causal[grid](
        a, b, c,
        a.stride(0), a.stride(1), a.stride(2), a.stride(3),
        b.stride(0), b.stride(1), b.stride(2), b.stride(3),
        c.stride(0), c.stride(1), c.stride(2), c.stride(3),
        Z, H_a, n_rep, M, K, 
    )
    return c

def torch_matmul(a, b, causal=False):
    BATCH, N_HEADS, N_DOWNSAMPLE, HIDDEN_DIM = a.shape
    b = b[:, :, None, :, :].expand(BATCH, N_HEADS // 4, 4, N_DOWNSAMPLE, HIDDEN_DIM).reshape(BATCH, N_HEADS, N_DOWNSAMPLE, HIDDEN_DIM)
    torch_output = torch.matmul(a, b.transpose(-1, -2))
    if causal:
        mask = torch.tril(torch.ones((N_DOWNSAMPLE, N_DOWNSAMPLE), device=a.device, dtype=torch.bool))
        torch_output[:, :, mask == 0] = float("-inf")
    return torch_output

def test():
    BATCH, N_HEADS, N_DOWNSAMPLE, HIDDEN_DIM = 1, 32, 1024, 256
    a = torch.randn(BATCH, N_HEADS, N_DOWNSAMPLE, HIDDEN_DIM, device='cuda', dtype=torch.float16)
    b = torch.randn(BATCH, N_HEADS // 4, N_DOWNSAMPLE, HIDDEN_DIM, device='cuda', dtype=torch.float16)
    for causal in [True]:
        triton_output = matmul(a, b, causal=causal)
        torch_output = torch_matmul(a, b, causal=causal)
        rtol = 0
        if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
            print("✅ Triton and Torch match")
        else:
            print("❌ Triton and Torch differ")


BATCH, N_HEADS, HIDDEN_DIM = 4, 32, 256
configs = []
for causal in [True]:
    configs.append(
        triton.testing.Benchmark(
        x_names=['N_DOWNSAMPLE'],  # Argument names to use as an x-axis for the plot.
        x_vals=[ 2**i for i in range(7, 11)],  # Different possible values for `x_name`.
        x_log=False,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['torch', 'triton'],  # Possible values for `line_arg`.
        line_names=['Torch', 'Triton'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='ms',  # Label name for the y-axis.
        plot_name=f'matmul_causal_{causal}_B{BATCH}_H{N_HEADS}_D{HIDDEN_DIM}',  # Display name for the plot.
        args={'BATCH': BATCH, 'N_HEADS': N_HEADS, 'HIDDEN_DIM': HIDDEN_DIM, 'causal': causal},  # Constant arguments to pass to `benchmark`.
    ))
    

@triton.testing.perf_report(configs)
# BATCH, N_HEADS, HEAD_DIM, HIDDEN_DIM
def benchmark(BATCH, N_HEADS, N_DOWNSAMPLE, HIDDEN_DIM, causal, provider):
    a = torch.randn(BATCH, N_HEADS, N_DOWNSAMPLE, HIDDEN_DIM, device='cuda', dtype=torch.float16)
    b = torch.randn(BATCH, N_HEADS // 4, N_DOWNSAMPLE, HIDDEN_DIM, device='cuda', dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        fn = lambda: torch_matmul(a, b, causal)
        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b, causal), quantiles=quantiles)
    return ms, min_ms, max_ms



test()
benchmark.run(save_path="./matmul_causal_gqa",print_data=True)