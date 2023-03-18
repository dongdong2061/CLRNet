import torch

a = torch.rand(2,2)
print(a)
b = a.repeat(2,1,1)
print(b)