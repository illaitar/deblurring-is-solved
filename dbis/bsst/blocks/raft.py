"""RAFT (Teed and Deng, ECCV 2020, https://github.com/princeton-vl/RAFT), BSD 3-Clause License, Copyright (c) 2020,
princeton-vl; as vendored by BSSTNet (basicsr/archs/RAFT)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import _const


def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device), indexing="ij")
    return torch.stack(coords[::-1], dim=0).float()[None].repeat(batch, 1, 1, 1)


def bilinear_sampler(img, coords):
    h, w = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (w - 1) - 1
    ygrid = 2 * ygrid / (h - 1) - 1
    return F.grid_sample(img, torch.cat([xgrid, ygrid], dim=-1), align_corners=True)


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels, self.radius = num_levels, radius
        batch, dim, ht, wd = fmap1.shape
        corr = torch.matmul(fmap1.view(batch, dim, ht * wd).transpose(1, 2), fmap2.view(batch, dim, ht * wd))
        corr = corr.view(batch, ht, wd, 1, ht, wd) / torch.sqrt(torch.tensor(dim).float())
        corr = corr.reshape(batch * ht * wd, 1, ht, wd)
        self.corr_pyramid = [corr]
        for _ in range(num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        def make():   # the authors build it on the CPU and copy it at every call
            d = torch.linspace(-r, r, 2 * r + 1)
            return torch.stack(torch.meshgrid(d, d, indexing="ij"), dim=-1).to(coords.device)
        delta = _const(("corr_delta", r, coords.device), make).view(1, 2 * r + 1, 2 * r + 1, 2)
        out = []
        for i in range(self.num_levels):
            centroid = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2 ** i
            corr = bilinear_sampler(self.corr_pyramid[i], centroid + delta)
            out.append(corr.view(batch, h1, w1, -1))
        return torch.cat(out, dim=-1).permute(0, 3, 1, 2).contiguous().float()


def raft_norm(kind, c):
    return nn.BatchNorm2d(c) if kind == "batch" else nn.InstanceNorm2d(c)


class ResidualBlock(nn.Module):
    """RAFT's ResidualBlock; its norm3 is the second module of downsample (the authors also register it as norm3)."""

    def __init__(self, cin, c, norm_fn, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, c, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(c, c, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.norm1, self.norm2 = raft_norm(norm_fn, c), raft_norm(norm_fn, c)
        self.downsample = (None if stride == 1 else
                           nn.Sequential(nn.Conv2d(cin, c, kernel_size=1, stride=stride), raft_norm(norm_fn, c)))

    def forward(self, x):
        y = self.relu(self.norm1(self.conv1(x)))
        y = self.relu(self.norm2(self.conv2(y)))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x + y)


class BasicEncoder(nn.Module):
    def __init__(self, output_dim, norm_fn):
        super().__init__()
        self.norm1 = raft_norm(norm_fn, 64)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.layer1 = nn.Sequential(ResidualBlock(64, 64, norm_fn, 1), ResidualBlock(64, 64, norm_fn, 1))
        self.layer2 = nn.Sequential(ResidualBlock(64, 96, norm_fn, 2), ResidualBlock(96, 96, norm_fn, 1))
        self.layer3 = nn.Sequential(ResidualBlock(96, 128, norm_fn, 2), ResidualBlock(128, 128, norm_fn, 1))
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

    def forward(self, x):
        is_list = isinstance(x, (tuple, list))
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.conv2(self.layer3(self.layer2(self.layer1(x))))
        return torch.split(x, [batch_dim, batch_dim], dim=0) if is_list else x


class BasicMotionEncoder(nn.Module):
    def __init__(self, cor_planes=4 * 81):
        super().__init__()
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 192, 128 - 2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc2(F.relu(self.convc1(corr))))
        flo = F.relu(self.convf2(F.relu(self.convf1(flow))))
        out = F.relu(self.conv(torch.cat([cor, flo], dim=1)))
        return torch.cat([out, flow], dim=1)


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=256):
        super().__init__()
        c = hidden_dim + input_dim
        self.convz1 = nn.Conv2d(c, hidden_dim, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(c, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(c, hidden_dim, (1, 5), padding=(0, 2))
        self.convz2 = nn.Conv2d(c, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(c, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(c, hidden_dim, (5, 1), padding=(2, 0))

    def forward(self, h, x):
        for cz, cr, cq in ((self.convz1, self.convr1, self.convq1), (self.convz2, self.convr2, self.convq2)):
            hx = torch.cat([h, x], dim=1)
            z = torch.sigmoid(cz(hx))
            r = torch.sigmoid(cr(hx))
            q = torch.tanh(cq(torch.cat([r * h, x], dim=1)))
            h = (1 - z) * h + z * q
        return h


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class BasicUpdateBlock(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.encoder = BasicMotionEncoder()
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        self.mask = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(256, 64 * 9, 1, padding=0))

    def forward(self, net, inp, corr, flow, with_mask):
        inp = torch.cat([inp, self.encoder(flow, corr)], dim=1)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)
        return net, (.25 * self.mask(net) if with_mask else None), delta_flow


class RAFT(nn.Module):
    """RAFT (large, raft-things weights) as vendored by BSSTNet: forward(image1, image2) in [0, 1] -> the flow of the
    last of 20 iterations, upsampled to the input size (test_mode=True)."""

    def __init__(self):
        super().__init__()
        self.hidden_dim = self.context_dim = 128
        self.fnet = BasicEncoder(256, "instance")
        self.cnet = BasicEncoder(256, "batch")
        self.update_block = BasicUpdateBlock(hidden_dim=128)

    @staticmethod
    def upsample_flow(flow, mask):
        n, _, h, w = flow.shape
        mask = torch.softmax(mask.view(n, 1, 9, 8, 8, h, w), dim=2)
        up_flow = F.unfold(8 * flow, [3, 3], padding=1).view(n, 2, 9, 1, 1, h, w)
        up_flow = torch.sum(mask * up_flow, dim=2).permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(n, 2, 8 * h, 8 * w)

    def forward(self, image1, image2, iters=20):
        image1 = (2 * (image1 / 1.) - 1.0).contiguous()
        image2 = (2 * (image2 / 1.) - 1.0).contiguous()
        fmap1, fmap2 = self.fnet([image1, image2])
        corr_fn = CorrBlock(fmap1, fmap2, radius=4)
        net, inp = torch.split(self.cnet(image1), [self.hidden_dim, self.context_dim], dim=1)
        net, inp = torch.tanh(net), torch.relu(inp)
        n, _, h, w = image1.shape
        coords0 = coords_grid(n, h // 8, w // 8, image1.device)
        coords1 = coords_grid(n, h // 8, w // 8, image1.device)
        for itr in range(iters):   # (the authors upsample the flow at every iteration and return the last one)
            corr = corr_fn(coords1)
            net, up_mask, delta_flow = self.update_block(net, inp, corr, coords1 - coords0, itr == iters - 1)
            coords1 = coords1 + delta_flow
        return self.upsample_flow(coords1 - coords0, up_mask)
