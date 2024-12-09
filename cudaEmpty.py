import torch
torch.cuda.empty_cache()
torch.cuda.memory.empty_cache()
print(torch.cuda.memory_summary(device=None, abbreviated=False))
torch.cuda.mem_get_info