"""HiFi-GAN style vocoder/Generator for S2.  Mirrors module.models.Generator.

Differences from MeloTTS Generator: upsample_rates/kernels are different here
(10,8,2,2,2 / 16,16,8,2,2 → 32kHz instead of MeloTTS' 44.1kHz) but the topology
is identical: a stack of ConvTranspose1d upsamplers, each followed by N residual
blocks whose outputs are averaged.
"""

from typing import List

import mlx.core as mx
import mlx.nn as nn

from modules import Conv1dPT, ConvTranspose1dPT, LRELU_SLOPE, get_padding


class ResBlock1(nn.Module):
    """3 dilations × (LReLU + conv1 + LReLU + conv2 + residual)."""

    def __init__(self, channels: int, kernel_size: int = 3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1: List[Conv1dPT] = [
            Conv1dPT(channels, channels, kernel_size,
                     dilation=d, padding=get_padding(kernel_size, d))
            for d in dilation
        ]
        self.convs2: List[Conv1dPT] = [
            Conv1dPT(channels, channels, kernel_size,
                     dilation=1, padding=get_padding(kernel_size, 1))
            for _ in dilation
        ]

    def __call__(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = nn.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = nn.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = x + xt
        return x


class Generator(nn.Module):
    def __init__(
        self,
        initial_channel: int,
        resblock: str,
        resblock_kernel_sizes: List[int],
        resblock_dilation_sizes: List[List[int]],
        upsample_rates: List[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: List[int],
        gin_channels: int = 0,
    ):
        super().__init__()
        assert resblock == "1", "Only ResBlock1 used by S2 v2ProTw"
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        self.conv_pre = Conv1dPT(initial_channel, upsample_initial_channel, 7, padding=3)

        self.ups: List[ConvTranspose1dPT] = []
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            in_ch = upsample_initial_channel // (2 ** i)
            out_ch = upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(
                ConvTranspose1dPT(in_ch, out_ch, k, stride=u, padding=(k - u) // 2)
            )

        self.resblocks: List[ResBlock1] = []
        for i in range(self.num_upsamples):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock1(ch, k, d))

        last_ch = upsample_initial_channel // (2 ** self.num_upsamples)
        self.conv_post = Conv1dPT(last_ch, 1, 7, padding=3, bias=False)

        if gin_channels != 0:
            self.cond = Conv1dPT(gin_channels, upsample_initial_channel, 1)
        else:
            self.cond = None

    def __call__(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None and self.cond is not None:
            x = x + self.cond(g)
        for i in range(self.num_upsamples):
            x = nn.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                blk_out = self.resblocks[i * self.num_kernels + j](x)
                xs = blk_out if xs is None else xs + blk_out
            x = xs / self.num_kernels
        x = nn.leaky_relu(x, LRELU_SLOPE)
        x = self.conv_post(x)
        return mx.tanh(x)
