from torch import nn


#用于冻结BN
def freeze(m):
    for i, k in m.named_children():
        if isinstance(k, nn.BatchNorm2d):
            k.eval()
        else:
            freeze(k)

