import torch.nn as nn
import torch


# class testmoudule(nn.Module):
#     def __init__(self):
#         super(testmoudule,self).__init__()
#         self.sample_point = 36
#         self.n_strips = 71

#         self.register_buffer(name='sample_x_indexs', tensor=(torch.linspace(
#             0, 1, steps=self.sample_points, dtype=torch.float32) *
#                                 self.n_strips).long())
        
    
#     def forward(self):
#         print(self.sample_x_indexs)
#         print(self.sample_x_indexs.shape)

# a = torch.linspace(
#             1, 0, steps=36, dtype=torch.float32)      
# b = (a*71).long()
# c = torch.flip((1 - b.float() / 71), dims=[-1])
# print(a,b,c)
# d = torch.flip(c,dims=[-1])
# print(d)
# print(b)
# print(b.shape)
# print(a)
# print(a.repeat(192,1))
# print(1-a.repeat(192,1))
# print((a+(1-a.repeat(192,1))).shape)
# t = torch.tensor([ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 
#         36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 71])
# print(6+t)
# priors_on_featmap = torch.randn(192,78)
# print(priors_on_featmap.shape)
# priors_on_featmap = priors_on_featmap.clone()[..., 6+t]
# print(priors_on_featmap.shape)
# prior_xs = torch.flip(priors_on_featmap, dims=[2]) 
# print(prior_xs)
# prior_xs = prior_xs.view(1, 192, -1, 1)
# print(prior_xs)

t = torch.tensor([[-2,1],[2,3]])
print(t.sigmoid())


# if __name__ == "__mian__":
#     t = testmoudule()
#     t.forward()