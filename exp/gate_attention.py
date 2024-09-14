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
    for s in ([1] if is_hip() else [3])\
    for w in [4]\
]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True


@triton.autotune(list(filter(keep, configs)), key=["N_DOWNSAMPLE", "HIDDEN_DIM"])
@triton.jit
def matmul_kernel_causal(
        a_ptr, b_ptr, c_ptr,
        stride_az, stride_ah, stride_am, stride_ak,  #
        stride_bz, stride_bh, stride_bk, stride_bn,  #
        stride_cz, stride_ch, stride_cm, stride_cn,  #
        H, n_rep,
        N_DOWNSAMPLE: tl.constexpr,
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
        shape=(HIDDEN_DIM, N_DOWNSAMPLE),
        strides=(stride_bn, stride_bk),
        offsets=(0, 0),
        block_shape=(HIDDEN_DIM, BLOCK_N),
        order=(0, 1)
    )
    c_block_ptr = tl.make_block_ptr(
        base = c_ptr + c_offset,
        shape=(BLOCK_M, N_DOWNSAMPLE),
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



@triton.autotune(list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM"])
@triton.jit
def qk_downsampling(Q, K, Q_down, K_down, n_rep, #
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
    q = q.to(tl.float32)
    q = (tl.sum(q, 0)[None, :] / BLOCK_M).to(tl.float16)
    tl.store(Q_down_block_ptr, q)

    # K Dowmsample
    if off_num == 0:
        k = tl.load(K_block_ptr)
        k_max = tl.max(k, 0).to(tl.float16)[None, :]
        tl.store(K_down_block_ptr, k_max.to(K_down.type.element_ty))
        K_down_block_ptr = tl.advance(K_down_block_ptr, (0, HEAD_DIM))
        k_min = tl.min(k, 0).to(tl.float16)[None, :]
        tl.store(K_down_block_ptr, k_min.to(K_down.type.element_ty))
        

class mask_predictor_kernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, w, topk):
        BATCH, N_HEADS, N_CTX, HEAD_DIM = q.shape
        K_HEADS = k.shape[1]
        HIDDEN_DIM = w.shape[-1]
        n_rep = N_HEADS // K_HEADS
        # shape constraints
        assert HEAD_DIM == w.shape[1]
        autotuned_config = qk_downsampling.configs[0]
        BLOCK_M = autotuned_config.kwargs["BLOCK_M"]
        q_down = torch.empty((BATCH, N_HEADS, N_CTX // BLOCK_M, HEAD_DIM), device=q.device, dtype=q.dtype)
        k_down = torch.empty((BATCH, K_HEADS, N_CTX // BLOCK_M, HIDDEN_DIM), device=q.device, dtype=q.dtype)
        N_DOWNSAMPLE = triton.cdiv(q.shape[2], BLOCK_M)
        o = torch.empty((BATCH, N_HEADS, N_DOWNSAMPLE, N_DOWNSAMPLE), device=q.device, dtype=q.dtype) - float("inf")
        extra_kern_args = {}
        # Tuning for AMD target
        if is_hip():
            waves_per_eu = 3 if HEAD_DIM <= 64 else 2
            extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}

        # Numerical error in the kernel if q dtype is float16
        grid = lambda args: (triton.cdiv(N_CTX, args["BLOCK_M"]), BATCH * N_HEADS, 1)
        qk_downsampling[grid](
            q, k, q_down, k_down, n_rep,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            q_down.stride(0), q_down.stride(1), q_down.stride(2), q_down.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            k_down.stride(0), k_down.stride(1), k_down.stride(2), k_down.stride(3),
            N_HEADS, N_CTX, HEAD_DIM, HIDDEN_DIM,
            **extra_kern_args,)
        
        grid = lambda args: (triton.cdiv(N_DOWNSAMPLE, args["BLOCK_M"]), BATCH * N_HEADS, 1)
        
        qw = torch.matmul(q_down, w)
        matmul_kernel_causal[grid](
            qw, k_down, o,
            qw.stride(0), qw.stride(1), qw.stride(2), qw.stride(3),
            k_down.stride(0), k_down.stride(1), k_down.stride(2), k_down.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            N_HEADS, n_rep, N_DOWNSAMPLE, HIDDEN_DIM, 
            **extra_kern_args,)
        
        nnz_id = torch.topk(o, topk, dim=-1).indices

        return nnz_id

triton_mask_predictor = mask_predictor_kernel.apply


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
        k = tl.load(tl.advance(K_block_ptr, (0, start_n)))
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
        v = tl.load(tl.advance(V_block_ptr, (start_n, 0)))
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
            k = tl.load(tl.advance(K_block_ptr, (0, start_n)))
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
            v = tl.load(tl.advance(V_block_ptr, (start_n, 0)))
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
    q = tl.load(Q_block_ptr)
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
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


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