import einops
import torch
ten = torch.Tensor([0,0,1,1,2,2,3,3,4,4])
print(einops.repeat(ten, '(a b)-> (a 3) b',b=2))
