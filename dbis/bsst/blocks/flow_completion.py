"""The recurrent flow completion network of ProPainter (Zhou et al., ICCV 2023, https://github.com/sczhou/ProPainter), reused by BSSTNet. ProPainter's own code is under
the S-Lab License 1.0 (non-commercial use only)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .align import FlowAlignment
from .common import lrelu


class BidirectionalPropagation(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.channel = c
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        for i, m in enumerate(("backward_", "forward_")):
            self.deform_align[m] = FlowAlignment(2 * c, c)
            self.backbone[m] = nn.Sequential(nn.Conv2d((2 + i) * c, c, 3, 1, 1), lrelu(), nn.Conv2d(c, c, 3, 1, 1))
        self.fusion = nn.Conv2d(2 * c, c, 1, 1, 0)

    def forward(self, x):
        b, t, c, h, w = x.shape
        feats = {"spatial": [x[:, i, :, :, :] for i in range(t)]}
        for name in ("backward_", "forward_"):
            feats[name] = []
            frame_idx = range(t - 1, -1, -1) if "backward" in name else range(t)
            feat_prop = x.new_zeros(b, self.channel, h, w)
            for i, idx in enumerate(frame_idx):
                feat_current = feats["spatial"][idx]
                if i > 0:
                    cond_n1 = feat_prop
                    feat_n2 = torch.zeros_like(feat_prop)
                    cond_n2 = torch.zeros_like(cond_n1)
                    if i > 1:
                        feat_n2 = feats[name][-2]
                        cond_n2 = feat_n2
                    cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                    feat_prop = self.deform_align[name](torch.cat([feat_prop, feat_n2], dim=1), cond)
                feat = [feat_current] + [feats[k][idx] for k in feats if k not in ("spatial", name)] + [feat_prop]
                feat_prop = feat_prop + self.backbone[name](torch.cat(feat, dim=1))
                feats[name].append(feat_prop)
            if "backward" in name:
                feats[name] = feats[name][::-1]
        outputs = [self.fusion(torch.cat([feats[k][i] for k in feats if k != "spatial"], dim=1)) for i in range(t)]
        return torch.stack(outputs, dim=1) + x


class Deconv(nn.Module):
    def __init__(self, cin, cout, k=3, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size=k, stride=1, padding=padding)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True))


class P3DBlock(nn.Module):
    def __init__(self, cin, cout, k, stride, padding):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(cin, cout, (1, k, k), (1, stride, stride), (0, padding, padding)),
                                   lrelu(0.2))
        self.conv2 = nn.Sequential(nn.Conv3d(cout, cout, (3, 1, 1), (1, 1, 1), (2, 0, 0), dilation=(2, 1, 1)))

    def forward(self, x):
        return self.conv2(self.conv1(x))


class FlowCompletion(nn.Module):
    """RecurrentFlowCompleteNet: refines the RAFT flows of a patch from (blur map, flow)."""

    def __init__(self):
        super().__init__()
        self.downsample = nn.Sequential(nn.Conv3d(3, 32, (1, 5, 5), (1, 2, 2), (0, 2, 2), padding_mode="replicate"),
                                        lrelu(0.2))
        self.encoder1 = nn.Sequential(P3DBlock(32, 32, 3, 1, 1), lrelu(0.2), P3DBlock(32, 64, 3, 2, 1), lrelu(0.2))
        self.mid_dilation = nn.Sequential(
            nn.Conv3d(64, 64, (1, 3, 3), (1, 1, 1), padding=(0, 3, 3), dilation=(1, 3, 3)), lrelu(0.2),
            nn.Conv3d(64, 64, (1, 3, 3), (1, 1, 1), padding=(0, 2, 2), dilation=(1, 2, 2)), lrelu(0.2),
            nn.Conv3d(64, 64, (1, 3, 3), (1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1)), lrelu(0.2))
        self.feat_prop_module = BidirectionalPropagation(64)
        self.decoder1 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), lrelu(0.2), Deconv(64, 32, 3, 1), lrelu(0.2))
        self.upsample = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), lrelu(0.2), Deconv(32, 2, 3, 1))

    def forward(self, blur_motion_map, flows):
        b, t, _, h, w = blur_motion_map.size()
        inputs = torch.cat((blur_motion_map.permute(0, 2, 1, 3, 4), flows.permute(0, 2, 1, 3, 4)), dim=1)
        x = self.downsample(inputs)
        feat_e1 = self.encoder1(x)
        feat_prop = self.feat_prop_module(self.mid_dilation(feat_e1).permute(0, 2, 1, 3, 4))
        feat_prop = feat_prop.view(-1, 64, h // 4, w // 4)
        _, c, _, h_f, w_f = feat_e1.shape
        feat_d2 = feat_prop + feat_e1.permute(0, 2, 1, 3, 4).contiguous().view(-1, c, h_f, w_f)
        flow = self.upsample(self.decoder1(feat_d2))
        return flow.view(b, t, 2, h, w) + flows

    def forward_bidirect_flow(self, blur_motion_map, flows_forward, flows_backward):
        bm_forward = blur_motion_map[:, :-1, ...].contiguous()
        bm_backward = blur_motion_map[:, 1:, ...].contiguous()
        pred_backward = self.forward(bm_backward, flows_backward)
        pred_forward = self.forward(torch.flip(bm_forward, dims=[1]), torch.flip(flows_forward, dims=[1]))
        return torch.flip(pred_forward, dims=[1]), pred_backward
