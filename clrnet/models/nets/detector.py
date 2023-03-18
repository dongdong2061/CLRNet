import torch.nn as nn
import torch

from clrnet.models.registry import NETS
from ..registry import build_backbones, build_aggregator, build_heads, build_necks


@NETS.register_module
class Detector(nn.Module):
    def __init__(self, cfg):
        super(Detector, self).__init__()
        self.cfg = cfg
        self.backbone = build_backbones(cfg)
        self.aggregator = build_aggregator(cfg) if cfg.haskey('aggregator') else None
        self.neck = build_necks(cfg) if cfg.haskey('neck') else None
        self.heads = build_heads(cfg)
    
    def get_lanes(self):
        return self.heads.get_lanes(output)

    def forward(self, batch):
        output = {}
        #[torch.Size([24, 64, 80, 200]), torch.Size([24, 128, 40, 100]), torch.Size([24, 256, 20, 50]), torch.Size([24, 512, 10, 25])]
        fea = self.backbone(batch['img'] if isinstance(batch, dict) else batch)
        # print(type(fea))
        # backbone_size = [x.shape for x in fea]
        # print("backbone_size",backbone_size)
        #no use
        if self.aggregator:
            fea[-1] = self.aggregator(fea[-1])
            print(type(fea))
            aggregator_size   = [x.shape for x in fea]
            print(aggregator_size)
        if self.neck:
            #[torch.Size([24, 64, 40, 100]), torch.Size([24, 64, 20, 50]), torch.Size([24, 64, 10, 25])]
            fea = self.neck(fea)
            neck_size = [x.shape for x in fea]
            print('neck_size',neck_size)
            print(type(fea))

        if self.training:
            output = self.heads(fea, batch=batch)
        else:
            output = self.heads(fea)
        # output_size = [x.shape for x in output]
        # print('output_size',output_size)
        print(output)
        print(type(output))
        #{'loss': tensor(10.5187, device='cuda:0', grad_fn=<AddBackward0>), 'loss_stats': {'loss': tensor(10.5187, device='cuda:0', 
        # grad_fn=<AddBackward0>), 'cls_loss': tensor(4.5798, device='cuda:0', grad_fn=<MulBackward0>), 
        # 'reg_xytl_loss': tensor(3.2644, device='cuda:0', grad_fn=<MulBackward0>), 
        # 'seg_loss': tensor(0.9903, device='cuda:0', grad_fn=<MulBackward0>), 
        # 'iou_loss': tensor(1.6842, device='cuda:0', grad_fn=<MulBackward0>), 
        # 'stage_0_acc': tensor([98.2111], device='cuda:0'), 'stage_1_acc': tensor([98.2111], device='cuda:0'), 
        # 'stage_2_acc': tensor([98.2110], device='cuda:0')}}
        return output
