import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import build_backbone


class AET_MultiEncoder(nn.Module):
    def __init__(self, cfg):
        super(AET_MultiEncoder, self).__init__()
        self.backbone = self._init_backbone(cfg.backbone)
        self.backbone_ = self._init_backbone(cfg.backbone)
        self.predictor = self._init_predictor(cfg.predictor)
        self.aet_head = self._init_aet_head(cfg.aet_head)

    def _init_backbone(self, cfg):
        return build_backbone(cfg)

    def _init_predictor(self, cfg):
        dim_in = cfg.dim_in
        dim_mid = cfg.dim_mid
        dim_out = cfg.dim_out
        return nn.Sequential(
            nn.Linear(dim_in, dim_mid),
            nn.BatchNorm1d(dim_mid),
            nn.ReLU(inplace=True),
            nn.Linear(dim_mid, dim_out),
        )

    def _init_aet_head(self, cfg):
        dim_in = cfg.dim_in
        dim_mid = cfg.dim_mid
        dim_out = cfg.dim_out
        return nn.Sequential(
            nn.Linear(dim_in, dim_mid),
            nn.BatchNorm1d(dim_mid),
            nn.ReLU(inplace=True),
            nn.Linear(dim_mid, dim_out),
        )

    def _global_pool(self, feat):
        num_channels = feat.size(1)
        return F.avg_pool2d(feat, (feat.size(2), feat.size(3))).view(
            -1, num_channels)

    def forward(self, x1, x2):
        r11 = self.backbone(x1)[-1]
        r12 = self.backbone_(x1)[-1]
        r2 = self.backbone(x2)[-1]

        r11 = self._global_pool(r11)
        r12 = self._global_pool(r12)
        r2 = self._global_pool(r2)

        p1 = self.predictor(r11)
        p2 = self.predictor(r12)

        pred_aet = self.aet_head(torch.cat((r11, r2), dim=1))

        return p1, p2, r11.detach(), r12.detach(), pred_aet


if __name__ == '__main__':
    from addict import Dict

    cfg = Dict(
        backbone=dict(
            type='MyResNetV1d',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch',
        ),
        predictor=dict(
            dim_in=2048,
            dim_mid=1024,
            dim_out=2048,
        ),
        aet_head=dict(
            dim_in=2 * 2048,
            dim_mid=1024,
            dim_out=9,
        ),
    )
    model = AET_MultiEncoder(cfg)
    data = torch.ones(size=[7, 3, 64, 64])
    output = model.backbone(data)
    for i in output:
        print(i.shape)
    output = model(data, data)
    for i in output:
        print(i.shape)
