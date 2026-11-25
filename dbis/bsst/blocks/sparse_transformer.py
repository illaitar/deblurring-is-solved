"""The temporal sparse transformer of ProPainter (Zhou et al., ICCV 2023, https://github.com/sczhou/ProPainter), reused by BSSTNet. ProPainter's own code is under
the S-Lab License 1.0 (non-commercial use only)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import _const


class SoftSplit(nn.Module):
    def __init__(self, c, hidden, kernel_size, stride, padding):
        super().__init__()
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding
        self.embedding = nn.Linear(kernel_size[0] * kernel_size[1] * c, hidden)

    def forward(self, x, b, output_size):
        f_h = int((output_size[0] + 2 * self.padding[0] - (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        f_w = int((output_size[1] + 2 * self.padding[1] - (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)
        feat = F.unfold(x, self.kernel_size, 1, self.padding, self.stride).permute(0, 2, 1)
        feat = self.embedding(feat)
        return feat.view(b, -1, f_h, f_w, feat.size(2))


class SoftComp(nn.Module):
    def __init__(self, c, hidden, kernel_size, stride, padding):
        super().__init__()
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding
        self.embedding = nn.Linear(hidden, kernel_size[0] * kernel_size[1] * c)
        self.bias_conv = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t, output_size):
        b_, _, _, _, c_ = x.shape
        feat = self.embedding(x.view(b_, -1, c_))
        b, _, c = feat.size()
        feat = feat.view(b * t, -1, c).permute(0, 2, 1)
        feat = F.fold(feat, output_size=output_size, kernel_size=self.kernel_size, stride=self.stride,
                      padding=self.padding)
        return self.bias_conv(feat)


class FusionFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim=1000, t2t=None):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(dim, hidden_dim))
        self.fc2 = nn.Sequential(nn.GELU(), nn.Linear(hidden_dim, dim))
        self.t2t = t2t
        self.kernel_shape = t2t["kernel_size"][0] * t2t["kernel_size"][1]

    def forward(self, x, output_size):
        p = self.t2t
        n_vecs = 1
        for i, d in enumerate(p["kernel_size"]):
            n_vecs *= int((output_size[i] + 2 * p["padding"][i] - (d - 1) - 1) / p["stride"][i] + 1)
        x = self.fc1(x)
        b, n, c = x.size()
        normalizer = x.new_ones(b, n, self.kernel_shape).view(-1, n_vecs, self.kernel_shape).permute(0, 2, 1)
        normalizer = F.fold(normalizer, output_size=output_size, kernel_size=p["kernel_size"], padding=p["padding"],
                            stride=p["stride"])
        x = F.fold(x.view(-1, n_vecs, c).permute(0, 2, 1), output_size=output_size, kernel_size=p["kernel_size"],
                   padding=p["padding"], stride=p["stride"])
        x = F.unfold(x / normalizer, kernel_size=p["kernel_size"], padding=p["padding"],
                     stride=p["stride"]).permute(0, 2, 1).contiguous().view(b, n, c)
        return self.fc2(x)


def sparse_partition(x, window_size, n_head):
    b, t, h, w, c = x.shape
    x = x.view(b, t, h // window_size[0], window_size[0], w // window_size[1], window_size[1], n_head, c // n_head)
    return x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()


def roll_with_zero_padding(x, shifts, dims):
    for dim, shift in zip(dims, shifts):
        size = x.size(dim)
        x = torch.roll(x, shift, dim)
        if shift > 0:
            x.index_fill_(dim, torch.arange(0, shift), 0)
        elif shift < 0:
            x.index_fill_(dim, torch.arange(size + shift, size), 0)
    return x


def sparse_attention_constants(window_size, device="cpu"):
    """(mask_padding, valid_ind_rolled): the buffers of the authors' SparseWindowAttention, built for its hard-coded
    32 x 32 token map (dropped from the weights, rebuilt here)."""
    def make():
        e = tuple((i + 1) // 2 for i in window_size)
        nh = math.ceil(32 / window_size[0]) * window_size[0]
        nw = math.ceil(32 / window_size[1]) * window_size[1]
        masks, pads = [], []
        for sy, sx in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            m = torch.ones(window_size[0], window_size[1])
            m[(slice(None, -e[0]) if sy < 0 else slice(e[0], None)),
              (slice(None, -e[1]) if sx < 0 else slice(e[1], None))] = 0
            masks.append(m)
            pads.append(roll_with_zero_padding(torch.ones(nh, nw), (sy * e[0], sx * e[1]), (0, 1)))
        valid = torch.stack(masks, 0).flatten(0).nonzero(as_tuple=False).view(-1)
        return torch.stack(pads, 0).to(device), valid.to(device)
    return _const(("sparse", tuple(window_size), torch.device(device)), make)


class SparseWindowAttention(nn.Module):
    sdpa = False      # True: F.scaled_dot_product_attention (flash) for both attention products

    def __init__(self, dim, n_head, window_size, pool_size=(4, 4)):
        super().__init__()
        self.key = nn.Linear(dim, dim, True)
        self.query = nn.Linear(dim, dim, True)
        self.value = nn.Linear(dim, dim, True)
        self.proj = nn.Linear(dim, dim)
        self.n_head = n_head
        self.window_size = tuple(window_size)
        self.pool_layer = nn.Conv2d(dim, dim, kernel_size=pool_size, stride=pool_size, padding=(0, 0), groups=dim)
        self.expand_size = tuple((i + 1) // 2 for i in window_size)

    def forward(self, x, indices):
        """x (b, t, h, w, c); indices[i] = (masked queries, masked keys/values, unmasked queries) of patch i, the
        indices the authors take with qmask[i].nonzero(), kvmask[i].nonzero(), (1 - qmask[i]).nonzero()."""
        b, t, h, w, c = x.shape
        w_h, w_w = self.window_size
        c_head = c // self.n_head
        n_wh, n_ww = math.ceil(h / w_h), math.ceil(w / w_w)
        new_h, new_w = n_wh * w_h, n_ww * w_w
        pad_r, pad_b = new_w - w, new_h - h
        t_token = t // 4
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b, 0, 0), mode="constant", value=0)
        mask_padding, valid_ind_rolled = sparse_attention_constants(self.window_size, x.device)
        q, k, v = self.query(x), self.key(x), self.value(x)

        def part(a):
            return sparse_partition(a, self.window_size, self.n_head).view(b, n_wh * n_ww, self.n_head, t,
                                                                         w_h * w_w, c_head)
        win_q, win_k, win_v = part(q.contiguous()), part(k.contiguous()), part(v.contiguous())
        es = self.expand_size
        rolled_k, rolled_v = [], []
        for j, (sy, sx) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
            m = mask_padding[j].view(1, 1, new_h, new_w, 1)
            rolled_k.append(part(torch.roll(k, shifts=(sy * es[0], sx * es[1]), dims=(2, 3)) * m))
            rolled_v.append(part(torch.roll(v, shifts=(sy * es[0], sx * es[1]), dims=(2, 3)) * m))
        rool_k = torch.cat(rolled_k, 4).contiguous()[:, :, :, :, valid_ind_rolled]
        rool_v = torch.cat(rolled_v, 4).contiguous()[:, :, :, :, valid_ind_rolled]
        roll_n = rool_k.shape[4]
        win_k = torch.cat((win_k, rool_k.view(b, n_wh * n_ww, self.n_head, t, roll_n, c_head)), dim=4)
        win_v = torch.cat((win_v, rool_v.view(b, n_wh * n_ww, self.n_head, t, roll_n, c_head)), dim=4)
        pool_x = self.pool_layer(x.view(b * t, new_h, new_w, c).permute(0, 3, 1, 2))
        _, _, p_h, p_w = pool_x.shape
        pool_x = pool_x.permute(0, 2, 3, 1).view(b, t, p_h, p_w, c)
        for proj, name in ((self.key, "k"), (self.value, "v")):
            pool = proj(pool_x).unsqueeze(1).repeat(1, n_wh * n_ww, 1, 1, 1, 1)
            pool = pool.view(b, n_wh * n_ww, t, p_h, p_w, self.n_head, c_head).permute(0, 1, 5, 2, 3, 4, 6)
            pool = pool.contiguous().view(b, n_wh * n_ww, self.n_head, t, p_h * p_w, c_head)
            if name == "k":
                win_k = torch.cat((win_k, pool), dim=4)
            else:
                win_v = torch.cat((win_v, pool), dim=4)

        twin_q = win_q.permute(0, 1, 3, 2, 4, 5).reshape(b, n_wh * n_ww * t, self.n_head, w_h * w_w, c_head)
        twin_k = win_k.reshape(b, n_wh * n_ww, self.n_head, t, -1, c_head)
        twin_v = win_v.reshape(b, n_wh * n_ww, self.n_head, t, -1, c_head)
        twin_k = twin_k.permute(0, 1, 3, 2, 4, 5).reshape(b, n_wh * n_ww * t, self.n_head, -1, c_head)
        twin_v = twin_v.permute(0, 1, 3, 2, 4, 5).reshape(b, n_wh * n_ww * t, self.n_head, -1, c_head)
        out = torch.zeros_like(twin_q)
        nt = w_h * w_w
        for i in range(b):
            mask_ind_i, kvmask_ind_i, unmask_ind_i = indices[i]
            mask_n = len(mask_ind_i)
            if mask_n > 0:   # masked windows: attention over the 12 selected frames, all tokens
                win_q_t = twin_q[i, mask_ind_i]
                win_k_t = twin_k[i, kvmask_ind_i].view(mask_n, self.n_head, -1, c_head)
                win_v_t = twin_v[i, kvmask_ind_i].view(mask_n, self.n_head, -1, c_head)
                kv_win_token = win_v_t.shape[-2]
                win_k_t = win_k_t.view(-1, t_token, self.n_head, kv_win_token, c_head)
                win_v_t = win_v_t.view(-1, t_token, self.n_head, kv_win_token, c_head)
                n_win = win_v_t.shape[0]
                win_k_t = win_k_t.permute(0, 2, 1, 3, 4).reshape(n_win, self.n_head, t_token * kv_win_token, c_head)
                win_v_t = win_v_t.permute(0, 2, 1, 3, 4).reshape(n_win, self.n_head, t_token * kv_win_token, c_head)
                win_q_t = win_q_t.view(n_win, t_token, self.n_head, nt, c_head)
                win_q_t = win_q_t.permute(0, 2, 1, 3, 4).reshape(n_win, self.n_head, t_token * nt, c_head)
                if self.sdpa:
                    y_t = F.scaled_dot_product_attention(win_q_t, win_k_t, win_v_t)
                else:
                    att_t = (win_q_t @ win_k_t.transpose(-2, -1)) * (1.0 / math.sqrt(win_q_t.size(-1)))
                    y_t = F.softmax(att_t, dim=-1) @ win_v_t
                y_t = y_t.view(n_win, self.n_head, t_token, nt, c_head)
                y_t = y_t.permute(0, 2, 1, 3, 4).reshape(n_win * t_token, self.n_head, nt, c_head)
                out[i, mask_ind_i] = y_t.view(-1, self.n_head, nt, c_head)
            n_un = len(unmask_ind_i)   # unmasked windows: groups of 12 consecutive (window, frame) tokens
            win_q_s = twin_q[i, unmask_ind_i]
            win_k_s = twin_k[i, unmask_ind_i, :, :nt].view(n_un, self.n_head, nt, c_head)
            win_v_s = twin_v[i, unmask_ind_i, :, :nt].view(n_un, self.n_head, nt, c_head)
            win_q_s = win_q_s.view(-1, t_token, self.n_head, nt, c_head)
            n_un_token = win_q_s.shape[0]
            win_q_s = win_q_s.permute(0, 2, 1, 3, 4).reshape(n_un_token, self.n_head, t_token * nt, c_head)
            win_k_s = win_k_s.view(-1, t_token, self.n_head, nt, c_head)
            win_k_s = win_k_s.permute(0, 2, 1, 3, 4).reshape(n_un_token, self.n_head, t_token * nt, c_head)
            win_v_s = win_v_s.view(-1, t_token, self.n_head, nt, c_head)
            win_v_s = win_v_s.permute(0, 2, 1, 3, 4).reshape(n_un_token, self.n_head, t_token * nt, c_head)
            if self.sdpa:
                y_s = F.scaled_dot_product_attention(win_q_s, win_k_s, win_v_s)
            else:
                att_s = (win_q_s @ win_k_s.transpose(-2, -1)) * (1.0 / math.sqrt(win_q_s.size(-1)))
                y_s = F.softmax(att_s, dim=-1) @ win_v_s
            y_s = y_s.view(n_un_token, self.n_head, t_token, nt, c_head)
            out[i, unmask_ind_i] = y_s.permute(0, 2, 1, 3, 4).reshape(n_un_token * t_token, self.n_head, nt, c_head)
        out = out.view(b, n_wh, n_ww, t, self.n_head, w_h, w_w, c_head)
        out = out.permute(0, 3, 1, 5, 2, 6, 4, 7).contiguous().view(b, t, new_h, new_w, c)
        if pad_r > 0 or pad_b > 0:
            out = out[:, :, :h, :w, :]
        return self.proj(out)


class TemporalSparseTransformer(nn.Module):
    def __init__(self, dim, n_head, window_size, pool_size, t2t):
        super().__init__()
        self.attention = SparseWindowAttention(dim, n_head, window_size, pool_size)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FusionFeedForward(dim, t2t=t2t)

    def forward(self, x, fold_x_size, indices):
        b, t, h, w, c = x.shape
        x = x + self.attention(self.norm1(x), indices)
        y = self.norm2(x)
        return x + self.mlp(y.view(b, t * h * w, c), fold_x_size).view(b, t, h, w, c)


class TemporalSparseTransformerBlock(nn.Module):
    def __init__(self, dim, n_head, window_size, pool_size, depths, t2t):
        super().__init__()
        self.transformer = nn.Sequential(*[TemporalSparseTransformer(dim, n_head, window_size, pool_size, t2t)
                                           for _ in range(depths)])
        self.depths = depths

    def forward(self, x, fold_x_size, qmasks, kvmasks):
        b, t = x.shape[:2]
        n_win = math.ceil(x.shape[2] / 8) * math.ceil(x.shape[3] / 8)
        # the authors' qmask.view(b, t, windows).permute(0, 2, 1).reshape(b, windows * t), per depth; the indices of
        # every depth and patch at once (one sync instead of two per depth and patch)
        flat = [torch.stack([m.view(b, t, n_win).permute(0, 2, 1).reshape(b, n_win * t) for m in ms])
                for ms in (qmasks, kvmasks)]
        sel = torch.stack((flat[0] != 0, flat[1] != 0, (1 - flat[0]) != 0))          # (3, depth, b, windows * t)
        counts = sel.sum(-1).flatten().tolist()
        parts = torch.split(sel.nonzero(as_tuple=False)[:, -1], counts)
        d = len(qmasks)
        for i in range(self.depths):
            idx = [tuple(parts[(j * d + i) * b + p] for j in range(3)) for p in range(b)]
            x = self.transformer[i](x, fold_x_size, idx)
        return x
