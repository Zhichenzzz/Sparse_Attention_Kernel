import torch

def block_topk(p, top_k, block_size):
    Z, H, N_CTX, N_CTX = p.shape
    ref = torch.max_pool2d(p, (block_size, block_size), stride=(block_size, block_size))

    _, topk_indices = torch.topk(ref, top_k, dim=-1)

    mask = torch.full_like(p, -float('inf')).to(torch.float16)

    for z in range(Z):
        for h in range(H):
            for i in range(ref.shape[2]):
                for k in range(top_k): 
                    idx = topk_indices[z, h, i, k]
                    start_row = i * block_size
                    start_col = idx * block_size
                    end_row = start_row + block_size
                    end_col = start_col + block_size
                    if end_col > N_CTX: 
                        end_col = N_CTX
                    mask[z, h, start_row:end_row, start_col:end_col] = p[z, h, start_row:end_row, start_col:end_col]
    mask = mask.to(torch.float16)
    return mask