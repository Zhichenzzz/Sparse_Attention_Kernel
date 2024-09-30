import numpy as np
import random
import triton
import triton.language as tl
import torch

next_power_of_2 = triton.next_power_of_2
MAX_FUSED_SIZE = 65536
ROPE_GROUP_SIZE = 4


def calculate_settings(n):
    """
    Calculate the blocksize and number of warps required for triton kernels

    Args:
        n: head dimension

    Returns:
        BLOCK_SIZE
        num_warps
    """
    BLOCK_SIZE = next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(f"Cannot launch Triton kernel since n = {n} exceeds "\
                           f"the maximum CUDA blocksize = {MAX_FUSED_SIZE}.")
    num_warps = 4
    if   BLOCK_SIZE >= 32768: num_warps = 32
    elif BLOCK_SIZE >=  8192: num_warps = 16
    elif BLOCK_SIZE >=  2048: num_warps = 8
    return BLOCK_SIZE, num_warps


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def rope_ref(q, cos, sin, position_ids, unsqueeze_dim=1):
    """
    Apply RoPE to the {Q, K} matrix using the efficient method mentioned in the 
    RoFormer paper

    Args:
        q: a torch tensor representing {Q, K} matrix 
        cos: the cos matrix
        sin: the sin matrix 
        position_ids
        unsqueeze_dim

    Returns:
        The {Q, K} matrix, with RoPE applied
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    return q_embed

