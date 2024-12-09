import torch

# Example tensor operation
x = torch.randn((1000, 1000), device='cuda')

# After some GPU operations
print(torch.cuda.memory_summary(device=None, abbreviated=False))
