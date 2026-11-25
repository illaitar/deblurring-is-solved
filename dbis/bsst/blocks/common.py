"""Shared helpers of the BSSTNet blocks: per-shape constants cached on the device, the LeakyReLU of the authors."""
import torch.nn as nn


_CONST = {}


def _const(key, make):
    """Constants (grids, masks, index tables) made once per shape and device."""
    v = _CONST.get(key)
    if v is None:
        v = _CONST[key] = make()
    return v


def lrelu(slope=0.1):
    return nn.LeakyReLU(negative_slope=slope, inplace=True)
