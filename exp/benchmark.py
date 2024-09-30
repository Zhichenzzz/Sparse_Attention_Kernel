import torch
# from MoA.models.llama.density_calculation import lut_attention_density
# from MoA.kernels.block_sparse_attention_lut import _sparse_attention
# sparse_attention = _sparse_attention.apply
from tqdm import tqdm
from flash_attn import flash_attn_func, flash_attn_kvpacked_func


device = "cuda"
dtype = torch.float16
seq_len = 8192
batch_size = 1
nheads = 32
nheads_k = nheads // 4
d = 128

q = torch.randn(batch_size, nheads, seq_len, d, device=device, dtype=dtype)
k = torch.randn(batch_size, nheads_k, seq_len, d, device=device, dtype=dtype)
v = torch.randn(batch_size, nheads_k, seq_len, d, device=device, dtype=dtype)
sm_scale = 1.3


lut = torch.load("7b_50/lut_result/lut_16384_plan_0.pt")
density_list = lut_attention_density(lut)[0]
sum = 0
for d in density_list:
    sum += d
avg = sum / len(density_list)

sum_latency = 0

for i in tqdm(range(len(lut))):
    lut_cuda = lut[i].to(torch.int32).to(device)
    
    # Warm-up run to ensure kernels are loaded and any one-time setups are done
    for j in range(5):
        sparse_attention(q, k, v, sm_scale, lut_cuda)
    torch.cuda.synchronize()  # Ensure warm-up is complete

    # Start timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    # Call the sparse_attention kernel
    for j in range(10):
        output = sparse_attention(q, k, v, sm_scale, lut_cuda)
    end_event.record()
    
    # Wait for the events to be recorded
    torch.cuda.synchronize()
    
    # Calculate elapsed time
    elapsed_time_ms = start_event.elapsed_time(end_event) / 10

    print("sparsity", 1 - density_list[i], "latency", elapsed_time_ms)
    sum_latency += elapsed_time_ms

avg_latency = sum_latency / len(lut)
print("avg density", avg)
print("avg latency", avg_latency)
    
    
###########################################################################################################################
# print()
# print("######## flash-attn-2 ##############")
# print()
# q = torch.randn(batch_size, seq_len, nheads, d, device=device, dtype=dtype)
# kv = torch.randn(batch_size, seq_len, 2, nheads_k, d, device=device, dtype=dtype)

# for j in range(25):
#     flash_attn_kvpacked_func(q, kv, causal=True)
# torch.cuda.synchronize()  # Ensure warm-up is complete

# # Start timing
# start_event = torch.cuda.Event(enable_timing=True)
# end_event = torch.cuda.Event(enable_timing=True)

# start_event.record()
# # Call the sparse_attention kernel
# for j in range(100):
#     flash_attn_kvpacked_func(q, kv, causal=True)
# end_event.record()

# # Wait for the events to be recorded
# torch.cuda.synchronize()

# # Calculate elapsed time
# elapsed_time_ms = start_event.elapsed_time(end_event) / 100
# print(f"Elapsed Time for flash-attn-2: {elapsed_time_ms:.3f} ms")