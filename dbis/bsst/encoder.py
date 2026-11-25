"""The per-frame encoder of BSST: the first part of the network, before the temporal propagation. The full network
(net.py, BSST) subclasses it, so its modules keep their names (concat, down01, encoder_level1, ...) in the weights."""
import torch.nn as nn

from .blocks.cnn import CAB
from .blocks.common import lrelu


class BSSTEncoder(nn.Module):
    """forward(x (n, 3, h, w) in [0, 1]) -> [full size (n, n_feat, h, w), 1/2 (n, n_feat, h/2, w/2), 1/4 (n, mid, h/4,
    w/4)]: the features BSST takes from every frame (the first and second as skip connections of its decoder, the third
    into the propagation). h and w multiples of 4. feature_info lists channels and reduction of each (as timm's
    features_only models do)."""

    def __init__(self, mid=192, n_feat=64, **_):
        super().__init__()
        self.concat = nn.Sequential(nn.Conv2d(3, n_feat, 3, 1, 1, bias=False), lrelu(), CAB(n_feat))
        self.down01 = nn.Sequential(nn.Conv2d(n_feat, n_feat, 3, 2, 1, bias=False), lrelu())
        self.encoder_level1 = CAB(n_feat)
        self.encoder_level1_1 = CAB(n_feat)
        self.down12 = nn.Sequential(nn.Conv2d(n_feat, mid, 3, 2, 1, bias=False), lrelu())
        self.encoder_level2 = CAB(mid)
        self.encoder_level2_1 = CAB(mid)
        self.feature_info = [{"num_chs": n_feat, "reduction": 1, "module": "concat"},
                             {"num_chs": n_feat, "reduction": 2, "module": "encoder_level1_1"},
                             {"num_chs": mid, "reduction": 4, "module": "encoder_level2_1"}]

    def encode(self, x):
        x = self.concat(x)
        enc11 = self.encoder_level1_1(self.encoder_level1(self.down01(x)))
        enc22 = self.encoder_level2_1(self.encoder_level2(self.down12(enc11)))
        return [x, enc11, enc22]

    def forward(self, x):
        return self.encode(x)
