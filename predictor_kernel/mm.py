import torch

import triton
import triton.language as tl
import math

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

def get_autotune_config():
    cuda_configs = [
        triton.Config(
            {'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, 'BLOCK_SIZE_K': BK, 'GROUP_SIZE_M': GM},
            num_stages=ns, num_warps=nw
        )
        for BM in [64, 32]\
        for BN in [64]\
        for BK in [64]\
        for GM in [8]\
        for ns in ([3, 4, 7])\
        for nw in ([4, 8])\
        
    ]
    return cuda_configs

@triton.autotune(
    configs=get_autotune_config(),
    key=['Z', 'H', 'M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_az, stride_ah, stride_am, stride_ak,  #
        stride_bz, stride_bh, stride_bk, stride_bn,  #
        stride_cz, stride_ch, stride_cm, stride_cn,  #
        Z, H,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, 
        BLOCK_SIZE_N: tl.constexpr, 
        BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
):
    off_hz = tl.program_id(axis=0)
    pid = tl.program_id(axis=1)
    off_z = off_hz // H
    off_h = off_hz % H
    a_offset = off_z.to(tl.int64) * stride_az + off_h.to(tl.int64) * stride_ah
    b_offset = off_z.to(tl.int64) * stride_bz + off_h.to(tl.int64) * stride_bh
    c_offset = off_z.to(tl.int64) * stride_cz + off_h.to(tl.int64) * stride_ch

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + a_offset + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + b_offset + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + c_offset + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (off_z < Z) & (off_h < H) & (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b):
    # Check constraints.
    assert a.is_contiguous(), "Matrix A must be contiguous"
    Z, H, M, K = a.shape
    Z, H, K, N = b.shape
    # Allocates output.
    c = torch.empty((Z, H, M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (Z * H, triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1), a.stride(2), a.stride(3),
        b.stride(0), b.stride(1), b.stride(2), b.stride(3),
        c.stride(0), c.stride(1), c.stride(2), c.stride(3),
        Z, H,
    )
    return c

def test():
    Z, H, M, N, K = 1, 32, 2048, 2048, 128
    a = torch.randn(Z, H, M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(Z, H, K, M, device='cuda', dtype=torch.float16)
    triton_output = matmul(a, b)
    torch_output = torch.matmul(a, b)
    print(f"triton_output_with_fp16_inputs={triton_output}")
    print(f"torch_output_with_fp16_inputs={torch_output}")
    rtol = 0
    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")


configs = []

configs.append(
    triton.testing.Benchmark(
    x_names=['Z'],  # Argument names to use as an x-axis for the plot.
    x_vals=[ i for i in range(0, 6)],  # Different possible values for `x_name`.
    x_log=False,  # x axis is logarithmic.
    line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
    line_vals=['torch', 'triton'],  # Possible values for `line_arg`.
    line_names=['Torch', 'Triton'],  # Label name for the lines.
    styles=[('blue', '-'), ('green', '-')],  # Line styles.
    ylabel='ms',  # Label name for the y-axis.
    plot_name=f'batch matmul',  # Name for the plot. Used also as a file name for saving the plot.
    args={"M":512, "K":128, "H":32},  # Values for function arguments not in `x_names` and `y_name`.
))
    

@triton.testing.perf_report(configs)
def benchmark(Z, H, M, K, provider):
    a = torch.randn(Z, H, M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(Z, H, K, M, device='cuda', dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
    return ms, min_ms, max_ms



test()
benchmark.run(save_path="./mm",print_data=True)