"""
U-Net backbone for DDPM (Ho, Jain, Abbeel 2020), following the architecture
description in Appendix B of the paper:

  - Backbone: PixelCNN++ / Wide ResNet-style U-Net (Ronneberger et al. 2015).
  - GroupNorm used in place of weight normalization ("to make the
    implementation simpler", as stated in the paper).
  - Two convolutional residual blocks per resolution level.
  - Self-attention blocks at the 16x16 resolution, inserted between the
    convolutional residual blocks.
  - Diffusion time t specified via a Transformer sinusoidal position
    embedding, added into every residual block.

The paper's Appendix B is a high-level description and does not spell out
every bookkeeping detail (e.g. exact skip-connection wiring, up/downsampling
op). Where the paper is silent we follow the implementation choices of the
official released code (Ho et al., github.com/hojonathanho/diffusion), which
is the de facto operational definition of "the DDPM architecture" used by
essentially all faithful re-implementations. These choices are called out
in METHODOLOGY.md.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Transformer sinusoidal position embedding (Vaswani et al. 2017),
    used by the paper to embed the diffusion timestep t."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def group_norm(channels: int, num_groups: int = 32) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=min(num_groups, channels), num_channels=channels)


class ResidualBlock(nn.Module):
    """Two conv residual block with GroupNorm + swish, and the timestep
    embedding injected as a per-channel bias after the first conv (matches
    the official DDPM implementation)."""

    def __init__(self, in_ch: int, out_ch: int, temb_dim: int, dropout: float):
        super().__init__()
        self.norm1 = group_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.temb_proj = nn.Linear(temb_dim, out_ch)
        self.norm2 = group_norm(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        # zero-init the last conv of each residual block, as in the
        # official implementation, so each block starts as identity.
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(swish(self.norm1(x)))
        h = h + self.temb_proj(swish(temb))[:, :, None, None]
        h = self.conv2(self.dropout(swish(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Self-attention over spatial positions, applied at a fixed set of
    resolutions per the paper (16x16 for the CIFAR-10 configuration)."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = group_norm(channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        hn = self.norm(x)
        q = self.q(hn).reshape(b, c, h * w).permute(0, 2, 1)  # B,HW,C
        k = self.k(hn).reshape(b, c, h * w)                    # B,C,HW
        v = self.v(hn).reshape(b, c, h * w).permute(0, 2, 1)   # B,HW,C

        attn = torch.bmm(q, k) * (c ** -0.5)                   # B,HW,HW
        attn = torch.softmax(attn, dim=-1)
        out = torch.bmm(attn, v)                                # B,HW,C
        out = out.permute(0, 2, 1).reshape(b, c, h, w)
        return x + self.proj_out(out)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNet(nn.Module):
    """DDPM U-Net: predicts epsilon (the noise) given a noisy image x_t and
    timestep t.

    Args:
        image_channels: number of image channels (3 for RGB).
        base_channels: channel count at the first resolution level. The
            paper uses 128 for CIFAR-10; we expose it as a parameter so the
            quick CPU validation run can use a narrower network (documented
            deviation, see METHODOLOGY.md).
        channel_mult: per-resolution-level channel multipliers. Paper uses
            (1, 2, 2, 2) for the 32x32 CIFAR-10 model.
        num_res_blocks: residual blocks per resolution level ("two
            convolutional residual blocks per resolution level" per paper).
        attn_resolutions: spatial resolutions (in pixels) at which
            self-attention is inserted (paper: 16x16).
        dropout: dropout probability inside residual blocks (paper uses 0.1
            for CIFAR-10).
    """

    def __init__(
        self,
        image_size: int = 32,
        image_channels: int = 3,
        base_channels: int = 128,
        channel_mult=(1, 2, 2, 2),
        num_res_blocks: int = 2,
        attn_resolutions=(16,),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.image_size = image_size
        temb_dim = base_channels * 4

        self.temb_mlp = nn.Sequential(
            nn.Linear(base_channels, temb_dim),
            nn.SiLU(),
            nn.Linear(temb_dim, temb_dim),
        )
        self.base_channels = base_channels

        self.conv_in = nn.Conv2d(image_channels, base_channels, 3, padding=1)

        # ---- Downsampling path ----
        self.down_blocks = nn.ModuleList()
        self.down_attns = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        chs = [base_channels]  # tracks channel count of every stored skip
        cur_ch = base_channels
        cur_res = image_size
        num_levels = len(channel_mult)
        for level, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            level_blocks = nn.ModuleList()
            level_attns = nn.ModuleList()
            for _ in range(num_res_blocks):
                level_blocks.append(ResidualBlock(cur_ch, out_ch, temb_dim, dropout))
                cur_ch = out_ch
                level_attns.append(AttentionBlock(cur_ch) if cur_res in attn_resolutions else nn.Identity())
                chs.append(cur_ch)
            self.down_blocks.append(level_blocks)
            self.down_attns.append(level_attns)
            if level != num_levels - 1:
                self.downsamples.append(Downsample(cur_ch))
                chs.append(cur_ch)
                cur_res //= 2
            else:
                self.downsamples.append(None)

        # ---- Middle ----
        self.mid_block1 = ResidualBlock(cur_ch, cur_ch, temb_dim, dropout)
        self.mid_attn = AttentionBlock(cur_ch)
        self.mid_block2 = ResidualBlock(cur_ch, cur_ch, temb_dim, dropout)

        # ---- Upsampling path (mirrors down path, num_res_blocks+1 blocks
        # per level to consume the extra skip stored at each downsample) ----
        self.up_blocks = nn.ModuleList()
        self.up_attns = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = base_channels * mult
            level_blocks = nn.ModuleList()
            level_attns = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                skip_ch = chs.pop()
                level_blocks.append(ResidualBlock(cur_ch + skip_ch, out_ch, temb_dim, dropout))
                cur_ch = out_ch
                level_attns.append(AttentionBlock(cur_ch) if cur_res in attn_resolutions else nn.Identity())
            self.up_blocks.append(level_blocks)
            self.up_attns.append(level_attns)
            if level != 0:
                self.upsamples.append(Upsample(cur_ch))
                cur_res *= 2
            else:
                self.upsamples.append(None)

        assert not chs, "skip connection bookkeeping mismatch"

        self.norm_out = group_norm(cur_ch)
        self.conv_out = nn.Conv2d(cur_ch, image_channels, 3, padding=1)
        # zero-init final conv, as in the official implementation.
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.temb_mlp(timestep_embedding(t, self.base_channels))

        h = self.conv_in(x)
        skips = [h]
        for level_blocks, level_attns, downsample in zip(self.down_blocks, self.down_attns, self.downsamples):
            for block, attn in zip(level_blocks, level_attns):
                h = block(h, temb)
                h = attn(h)
                skips.append(h)
            if downsample is not None:
                h = downsample(h)
                skips.append(h)

        h = self.mid_block1(h, temb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, temb)

        for level_blocks, level_attns, upsample in zip(self.up_blocks, self.up_attns, self.upsamples):
            for block, attn in zip(level_blocks, level_attns):
                h = block(torch.cat([h, skips.pop()], dim=1), temb)
                h = attn(h)
            if upsample is not None:
                h = upsample(h)

        h = self.conv_out(swish(self.norm_out(h)))
        return h
