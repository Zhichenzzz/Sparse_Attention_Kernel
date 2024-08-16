import torch

import triton
import triton.language as tl

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

def get_autotune_config():
    cuda_configs = [
        triton.Config(
            {'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, 'BLOCK_SIZE_K': BK, 'GROUP_SIZE_M': GM},
            num_stages=ns, num_warps=nw
        )
        for BM, BN in [(64, 64), (32, 32)]\
        for BK in [16, 32, 64]\
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
    
    ## if we don't want to group k
    # pid_m = pid // num_pid_n
    # pid_n = pid % num_pid_n

    # ----------------------------------------------------------
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + a_offset + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + b_offset + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # if pid_m >= pid_n:
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
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

    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # mask = offs_cm[:, None] >= offs_cn[None, :]
    # c = c + tl.where(mask, 0, -1.0e6)
    c_ptrs = c_ptr + c_offset + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (off_z < Z) & (off_h < H) & (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.autotune(
    configs=get_autotune_config(),
    key=['Z', 'H', 'N', 'K'],
)
@triton.jit
def matmul_kernel_causal(
        a_ptr, b_ptr, c_ptr,
        N: tl.constexpr,
        K: tl.constexpr,
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
    start_m = tl.program_id(axis=1)
    off_z = off_hz // H
    off_h = off_hz % H
    a_offset = off_z.to(tl.int64) * stride_az + off_h.to(tl.int64) * stride_ah
    b_offset = off_z.to(tl.int64) * stride_bz + off_h.to(tl.int64) * stride_bh
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

    ##  This is the original code for group k  

    # for start_n in range(lo, hi, BLOCK_SIZE_N):
    #     acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)

    #     for start_k in range(0, K, BLOCK_SIZE_K):
    #         q = tl.load(tl.advance(a_block_ptr, (0, start_k)))
    #         k = tl.load(tl.advance(b_block_ptr, (start_k, 0)))
    #         acc = tl.dot(q, k, acc)
    #         # a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
    #         # b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))

    #     mask = offs_m[:, None] >= (start_n + offs_n[None, :])
    #     acc = acc + tl.where(mask, 0, -1.0e6)
    #     acc = acc.to(tl.float16)
    #     # print(acc.shape)
    #     tl.store(c_block_ptr, acc)
    #     b_block_ptr = tl.advance(b_block_ptr, (0, BLOCK_SIZE_N))
    #     c_block_ptr = tl.advance(c_block_ptr, (0, BLOCK_SIZE_N))


def matmul(a, b, causal=False):
    # Check constraints.
    assert a.is_contiguous(), "Matrix A must be contiguous"
    Z, H, M, K = a.shape
    Z, H, N, K = b.shape
    # Allocates output.
    c = torch.empty((Z, H, M, N), device=a.device, dtype=torch.float16) - 1.0e6
    # 1D launch kernel where each block gets its own program.
    
    # if causal == False:
    #     grid = lambda META: (Z * H, triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']))
    #     matmul_kernel[grid](
    #         a, b, c,
    #         M, N, K,
    #         a.stride(0), a.stride(1), a.stride(2), a.stride(3),
    #         b.stride(0), b.stride(1), b.stride(2), b.stride(3),
    #         c.stride(0), c.stride(1), c.stride(2), c.stride(3),
    #         Z, H,
    #     )
    # else:
    grid = lambda META: (Z * H, triton.cdiv(M, META['BLOCK_SIZE_M']) )
    matmul_kernel_causal[grid](
        a, b, c,
        N, K,
        a.stride(0), a.stride(1), a.stride(2), a.stride(3),
        b.stride(0), b.stride(1), b.stride(2), b.stride(3),
        c.stride(0), c.stride(1), c.stride(2), c.stride(3),
        Z, H,
    )
    return c

def torch_matmul(a, b, causal=False):
    M = a.shape[-2]
    torch_output = torch.matmul(a, b.transpose(-1, -2))
    if causal:
        mask = torch.tril(torch.ones((M, M), device="cuda"))
        torch_output[:, :, mask == 0] = float("-inf")
    return torch_output

def test():
    BATCH, N_HEADS, N_DOWNSAMPLE, HIDDEN_DIM = 1, 32, 1024, 256
    a = torch.randn(BATCH, N_HEADS, N_DOWNSAMPLE, HIDDEN_DIM, device='cuda', dtype=torch.float16)
    b = torch.randn(BATCH, N_HEADS, N_DOWNSAMPLE, HIDDEN_DIM, device='cuda', dtype=torch.float16)
    for causal in [True]:
        triton_output = matmul(a, b, causal=causal)
        torch_output = torch_matmul(a, b, causal=causal)
        rtol = 0
        print(triton_output)
        print(torch_output)
        if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
            print("✅ Triton and Torch match")
        else:
            print("❌ Triton and Torch differ")



configs = []
for causal in [True]:
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
        plot_name=f'matmul-causal-{causal}',  # Name for the plot. Used also as a file name for saving the plot.
        args={"M":1024, "K":256, "H":32, "causal":causal},  # Constant arguments for the function.
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
# benchmark.run(save_path="./matmul",print_data=True)