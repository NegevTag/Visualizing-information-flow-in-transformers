import torch
import einops as ein
a = torch.zeros([4,4])
print(ein.repeat(a,"d1 d2-> (k d1) d2",k=0))