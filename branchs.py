import torch
from torch import nn


class Cross_Branch(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,):
        super(Cross_Branch,self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        