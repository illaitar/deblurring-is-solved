"""Turtle's transformer block and the stacks of them that make a level of the U-Net.

Source: basicsr/models/archs/turtle_t1_arch.py of https://github.com/Ascend-Research/Turtle (commit 7094f42), MIT
License, Copyright (c) 2024 Huawei Technologies Co., Ltd. The same modules and weight names.
"""
import torch.nn as nn

from .attention import ChannelAttention, ReducedAttn
from .common import FeedForward, GatedFeedForward, LayerNorm
from .history import CausalHistoryModel, FrameHistoryRouter


class TurtleAttnBlock(nn.Module):
    """x + attention(norm1(x)), then + feed-forward(norm2(x)). Attention: "Channel", "ReducedAttn", "FHR" (history
    without alignment), "CHM" (history with alignment) or "NoAttn"; feed-forward: "GFFW" (gated) or "FFW"."""

    def __init__(self, dim, ffn_expansion_factor, bias, LayerNorm_type, num_heads=1, Scale_patchsize=1,
                 attention_type="Channel", FFW_type="GFFW", num_frames_tocache=1, fast=False):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        if attention_type == "Channel":
            self.attn = ChannelAttention(dim, num_heads, bias)
        elif attention_type == "ReducedAttn":
            self.attn = ReducedAttn(dim)
        elif attention_type == "FHR":
            self.attn = FrameHistoryRouter(dim, num_heads, bias, num_frames_tocache)
        elif attention_type == "CHM":
            self.attn = CausalHistoryModel(dim, num_heads, bias, Scale_patchsize, num_frames_tocache, fast=fast)
        elif attention_type == "NoAttn":
            self.attn = None
        else:
            raise ValueError(f"attention type {attention_type!r}")
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        if FFW_type == "GFFW":
            self.ffn = GatedFeedForward(dim, ffn_expansion_factor, bias)
        elif FFW_type == "FFW":
            self.ffn = FeedForward(dim)
        else:
            raise ValueError(f"feed-forward type {FFW_type!r}")

    def forward(self, x, k_cached=None, v_cached=None):
        if self.attn is None:
            return x + self.ffn(self.norm2(x)), None, None
        a, k, v = self.attn(self.norm1(x), k_cached, v_cached)
        x = x + a
        return x + self.ffn(self.norm2(x)), k, v


class LevelBlock(nn.Module):
    """num_blocks Turtle blocks: attn_type1 in all but the last, attn_type2 (the one that may carry history) in the
    last; the cache goes to and comes from the last block only."""

    def __init__(self, dim, ffn_expansion_factor, bias, LayerNorm_type, num_blocks, attn_type1="Channel",
                 attn_type2="CHM", FFW_type="GFFW", num_frames_tocache=1, num_heads=1, Scale_patchsize=1, fast=False):
        super().__init__()
        self.num_blocks = num_blocks
        kw = dict(dim=dim, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                  LayerNorm_type=LayerNorm_type, FFW_type=FFW_type, num_frames_tocache=num_frames_tocache,
                  Scale_patchsize=Scale_patchsize, fast=fast)
        self.transformer_blocks = nn.ModuleList([TurtleAttnBlock(attention_type=attn_type1, **kw)
                                                 for _ in range(num_blocks - 1)]
                                                + [TurtleAttnBlock(attention_type=attn_type2, **kw)])

    def forward(self, x, k_cached=None, v_cached=None):
        for blk in self.transformer_blocks[:-1]:
            x, _, _ = blk(x)
        return self.transformer_blocks[-1](x, k_cached, v_cached)


class LatentCacheBlock(nn.Module):
    """The bottleneck: attn_type1 in the first block, attn_type2 in the middle ones, attn_type3 in the last; the first
    and the last carry their own caches."""

    def __init__(self, dim, ffn_expansion_factor, bias, LayerNorm_type, num_blocks, attn_type1="FHR",
                 attn_type2="Channel", attn_type3="FHR", FFW_type="GFFW", num_frames_tocache=1, num_heads=1):
        super().__init__()
        if num_blocks < 2:
            raise ValueError("LatentCacheBlock needs at least 2 blocks")
        self.num_blocks = num_blocks
        kw = dict(dim=dim, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                  LayerNorm_type=LayerNorm_type, FFW_type=FFW_type, num_frames_tocache=num_frames_tocache)
        self.transformer_blocks = nn.ModuleList([TurtleAttnBlock(attention_type=attn_type1, **kw)]
                                                + [TurtleAttnBlock(attention_type=attn_type2, **kw)
                                                   for _ in range(num_blocks - 2)]
                                                + [TurtleAttnBlock(attention_type=attn_type3, **kw)])

    def forward(self, x, k1_cached=None, v1_cached=None, k2_cached=None, v2_cached=None):
        out, k1, v1 = self.transformer_blocks[0](x, k1_cached, v1_cached)
        for blk in self.transformer_blocks[1:-1]:
            out, _, _ = blk(out)
        out, k2, v2 = self.transformer_blocks[-1](out, k2_cached, v2_cached)
        return out, k1, v1, k2, v2
