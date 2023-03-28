import torch
from torch import nn
from sanet import sa_layer


class Cross_Branch(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 groups):
        super(Cross_Branch,self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.groups = groups
        self.layer = sa_layer(self.in_channel,groups=self.groups)

    def forward(self,x):
        return self.layer(x)
    
class Main_Branch(nn.Module):
    def __Init__(self,
                 in_channel,
                 out_channel):
        super(Main_Branch,self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
