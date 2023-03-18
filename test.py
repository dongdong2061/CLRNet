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

a = torch.linspace(
            0, 1, steps=36, dtype=torch.float32)      
b = (a*71).long()
print(b)
print(b.shape)




# if __name__ == "__mian__":
#     t = testmoudule()
#     t.forward()