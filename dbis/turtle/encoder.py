"""The encoder of Turtle: the input conv and the three encoder levels of its U-Net, which see only the current frame
(their blocks carry no history). The full network (net.py, Turtle) subclasses it, so its modules keep their names
(input_projection, encoder_level1 to 3, down1_2, down2_3, down3_4) in the weights."""
import torch.nn as nn
import torch.nn.functional as F

from .blocks.common import Downsample
from .blocks.level import LevelBlock


class TurtleEncoder(nn.Module):
    """forward(x (n, 3, h, w) in [0, 1]) -> [64 ch at 1, 128 at 1/2, 256 at 1/4, 512 at 1/8 (the input of the bottleneck,
    after down3_4)]. h and w are zero-padded to a multiple of 32, as the network does, and the features keep that size.
    The defaults are the GoPro model of the authors (Turtle_Deblur_Gopro.yml)."""

    def __init__(self, dim=64, enc_blocks=(2, 6, 10), num_heads=(1, 2, 4, 8), ffn_expansion_factor=2.5, bias=False,
                 LayerNorm_type="WithBias", enc_attn=(("ReducedAttn", "ReducedAttn"), ("ReducedAttn", "ReducedAttn"),
                                                       ("Channel", "Channel")), enc_ffw=("FFW", "FFW", "GFFW"),
                 num_frames_tocache=3, **_):
        super().__init__()
        self.input_projection = nn.Conv2d(3, dim, 3, 1, 1, bias=bias)
        kw = dict(ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type,
                  num_frames_tocache=num_frames_tocache)
        self.encoder_level1 = LevelBlock(dim, num_blocks=enc_blocks[0], attn_type1=enc_attn[0][0],
                                         attn_type2=enc_attn[0][1], FFW_type=enc_ffw[0], num_heads=num_heads[0], **kw)
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = LevelBlock(dim * 2, num_blocks=enc_blocks[1], attn_type1=enc_attn[1][0],
                                         attn_type2=enc_attn[1][1], FFW_type=enc_ffw[1], num_heads=num_heads[1], **kw)
        self.down2_3 = Downsample(dim * 2)
        self.encoder_level3 = LevelBlock(dim * 4, num_blocks=enc_blocks[2], attn_type1=enc_attn[2][0],
                                         attn_type2=enc_attn[2][1], FFW_type=enc_ffw[2], num_heads=num_heads[2], **kw)
        self.down3_4 = Downsample(dim * 4)
        self.padder_size = 32
        self.feature_info = [{"num_chs": dim, "reduction": 1, "module": "encoder_level1"},
                             {"num_chs": dim * 2, "reduction": 2, "module": "encoder_level2"},
                             {"num_chs": dim * 4, "reduction": 4, "module": "encoder_level3"},
                             {"num_chs": dim * 8, "reduction": 8, "module": "down3_4"}]

    def pad(self, x):
        """Zero padding of h and w to a multiple of 32 (the authors' check_image_size)."""
        h, w = x.shape[-2:]
        return F.pad(x, (0, -w % self.padder_size, 0, -h % self.padder_size))

    def encode(self, x):
        e1, _, _ = self.encoder_level1(self.input_projection(x.float()))
        e2, _, _ = self.encoder_level2(self.down1_2(e1))
        e3, _, _ = self.encoder_level3(self.down2_3(e2))
        return [e1, e2, e3, self.down3_4(e3)]

    def forward(self, x):
        return self.encode(self.pad(x))
