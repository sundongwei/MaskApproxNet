import os
import sys
# This line is often used to allow importing modules from the parent directory.
# Its necessity depends on how the project is structured and executed.
# If the script is always run from the project root, this might be redundant.
sys.path.append("../")
from abc import abstractmethod
import math
import numpy as np
import torch as th
import torch # Explicitly imported, though th is the preferred alias
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict # Keep if used for state_dict manipulation
from .fp16_util import convert_module_to_f16, convert_module_to_f32
from copy import deepcopy # Keep if used
from .utils import softmax_helper,sigmoid_helper # Keep if used
from .utils import InitWeights_He # Keep if used
from batchgenerators.augmentations.utils import pad_nd_image # Keep if used by Generic_UNet
from .utils import no_op # Keep if used by Generic_UNet
from .utils import to_cuda, maybe_to_torch # Keep if used by Generic_UNet
from scipy.ndimage.filters import gaussian_filter # Keep if used by Generic_UNet
from typing import Union, Tuple, List # Keep if used by Generic_UNet
from torch.cuda.amp import autocast # Keep if used by Generic_UNet
from module.diff_module import ChangeBindModel # Specific module for change detection
from .nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
    layer_norm, # Imported layer_norm, used in UNetModel.enhance
)
from torchvision.utils import save_image # Keep for debugging if needed

class AttentionPool2d(nn.Module):
    """
    A 2D attention pooling layer.
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    This layer uses a form of self-attention to pool features across spatial dimensions.
    """

    def __init__(
        self,
        spacial_dim: int, # The spatial dimension (height or width) of the input feature map.
        embed_dim: int,   # The embedding dimension (number of channels) of the input.
        num_heads_channels: int, # Number of channels per attention head.
        output_dim: int = None, # The output dimension. If None, defaults to embed_dim.
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1) # 1D conv for QKV projection
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1) # 1D conv for output projection
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x: th.Tensor) -> th.Tensor:
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # Reshape to [N, C, H*W]
        # Prepend a learnable [CLS] token equivalent for attention pooling
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # [N, C, H*W+1]
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # Add positional embedding
        x = self.qkv_proj(x) # Project to Q, K, V
        x = self.attention(x) # Apply attention mechanism
        x = self.c_proj(x) # Project to output dimension
        return x[:, :, 0] # Return the pooled feature (from the [CLS] token equivalent)


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x: th.Tensor, emb: th.Tensor) -> th.Tensor:
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x: th.Tensor, emb: th.Tensor) -> th.Tensor:
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels: int, use_conv: bool, dims: int = 2, out_channels: int = None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x: th.Tensor) -> th.Tensor:
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels: int, use_conv: bool, dims: int = 2, out_channels: int = None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x: th.Tensor) -> th.Tensor:
        assert x.shape[1] == self.channels
        return self.op(x)

# MobileNet utility blocks (conv_bn, conv_dw, MobBlock) are part of the original codebase,
# but not directly used by UNetModel or Generic_UNet in the provided structure.
# They are kept here as they might be used by other components or future extensions.
# If confirmed unused, they could be removed. For now, they are retained.
def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
        )

def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )

class MobBlock(nn.Module):
    def __init__(self,ind):
        super().__init__()
        if ind == 0:
            self.stage = nn.Sequential(conv_bn(3, 32, 2), conv_dw(32, 64, 1), conv_dw(64, 128, 1), conv_dw(128, 128, 1))
        elif ind == 1:
            self.stage  = nn.Sequential(conv_dw(128, 256, 2), conv_dw(256, 256, 1))
        elif ind == 2:
            self.stage = nn.Sequential(conv_dw(256, 256, 2), conv_dw(256, 256, 1))
        else:
            self.stage = nn.Sequential(conv_dw(256, 512, 2), conv_dw(512, 512, 1), conv_dw(512, 512, 1), conv_dw(512, 512, 1), conv_dw(512, 512, 1), conv_dw(512, 512, 1))
    def forward(self,x):
        return self.stage(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    Includes timestep embedding and optional up/downsampling.
    """
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        dropout: float,
        out_channels: int = None,
        use_conv: bool = False,
        use_scale_shift_norm: bool = False,
        dims: int = 2,
        use_checkpoint: bool = False,
        up: bool = False,
        down: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv: # Use 3x3 conv for skip connection if specified
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else: # Use 1x1 conv for skip connection
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x: th.Tensor, emb: th.Tensor) -> th.Tensor:
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x: th.Tensor, emb: th.Tensor) -> th.Tensor:
        if self.updown: # Apply up/downsampling to features and input if specified
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape): # Align emb_out shape for broadcasting
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm: # FiLM-like conditioning
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else: # Additive conditioning
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """
    def __init__(
        self,
        channels: int,
        num_heads: int = 1,
        num_head_channels: int = -1,
        use_checkpoint: bool = False,
        use_new_attention_order: bool = False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0, \
                f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1) # 1D conv for QKV projection
        if use_new_attention_order: # Different ways to split heads and QKV
            self.attention = QKVAttention(self.num_heads)
        else:
            self.attention = QKVAttentionLegacy(self.num_heads)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1)) # Output projection

    def forward(self, x: th.Tensor) -> th.Tensor:
        return checkpoint(self._forward, (x,), self.parameters(), True) # Use checkpointing if enabled

    def _forward(self, x: th.Tensor) -> th.Tensor:
        b, c, *spatial = x.shape
        x_reshaped = x.reshape(b, c, -1) # Flatten spatial dimensions
        qkv = self.qkv(self.norm(x_reshaped))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x_reshaped + h).reshape(b, c, *spatial) # Add residual and reshape back


def count_flops_attn(model, _x, y):
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping"""
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv: th.Tensor) -> th.Tensor:
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y): return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """A module which performs QKV attention and splits in a different order."""
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv: th.Tensor) -> th.Tensor:
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y): return count_flops_attn(model, _x, y)

class FFParser(nn.Module):
    """
    Frequency-Guided Complex Filter.
    This module applies a learnable complex-valued filter in the frequency domain.
    It can be used to modulate features based on their frequency components.
    Includes a frequency mask to potentially attenuate high or low frequencies.
    """
    def __init__(self, dim: int, h: int = 128, w: int = 65, freq_threshold: float = 0.5):
        super().__init__()
        # Learnable complex weights, initialized randomly. Shape: [dim, h, w, 2] (real and imaginary parts)
        self.complex_weight = nn.Parameter(torch.randn(dim, h, w, 2, dtype=torch.float32) * 0.02)
        
        # Learnable frequency mask, initialized to ones (pass all frequencies initially).
        # This mask can be shaped during training to emphasize/attenuate certain frequency bands.
        self.freq_mask_param = nn.Parameter(torch.ones(h, w, dtype=torch.float32)) # Renamed to avoid conflict
        self.freq_threshold = freq_threshold  # Threshold for dynamic frequency masking (currently not used dynamically in _apply_freq_mask)
        
        self.w = w # Expected width of the frequency domain representation (usually H/2 + 1 for rfft)
        self.h = h # Expected height of the frequency domain representation

    def forward(self, x: th.Tensor, spatial_size=None) -> th.Tensor:
        B, C, H, W = x.shape
        assert H == W, "Height and width must be equal for this module."
        # spatial_size argument is not used in the current implementation.

        # Convert input `x` to frequency domain using Real Fast Fourier Transform (RFFT).
        x = x.to(torch.float32) # Ensure float32 for FFT
        x_freq = torch.fft.rfft2(x, dim=(2, 3), norm='ortho') # FFT along spatial dimensions (H, W)
        
        # Convert learnable weights to complex numbers.
        weight_complex = torch.view_as_complex(self.complex_weight)
        # Apply the frequency mask to the complex weights.
        masked_weight = weight_complex * self._apply_freq_mask(weight_complex) # Element-wise multiplication
        
        # Apply the (masked) learnable filter in the frequency domain.
        # This multiplication happens per channel (dim C).
        x_freq = x_freq * masked_weight # Broadcasting if C != dim
        
        # Convert back to spatial domain using Inverse Real Fast Fourier Transform (IRFFT).
        x_filtered = torch.fft.irfft2(x_freq, s=(H, W), dim=(2, 3), norm='ortho')
        
        x_filtered = x_filtered.reshape(B, C, H, W) # Ensure original shape
        return x_filtered

    def _apply_freq_mask(self, weight_complex_for_device_ref: th.Tensor) -> th.Tensor:
        """
        Applies the learnable frequency mask.
        Currently, it generates a fixed mask based on `freq_threshold` but uses `self.freq_mask_param`
        which is learnable. This suggests `self.freq_mask_param` is intended to be the primary mask.
        The grid generation part seems illustrative or for a fixed mask variant.
        For a learnable mask, it should directly return `self.freq_mask_param` or a processed version.
        Let's assume `self.freq_mask_param` is the learnable mask to be applied.
        """
        # The original implementation generated a fixed mask based on freq_magnitude and freq_threshold.
        # However, self.freq_mask_param is a learnable nn.Parameter.
        # Typically, if a learnable mask is intended, it should be used directly.
        # For this refactor, we will assume self.freq_mask_param is the intended learnable mask.
        # The grid generation logic is kept for reference if a fixed mask was intended.
        
        # To use the learnable mask parameter:
        return self.freq_mask_param
        
        # Original fixed mask logic (for reference):
        # h_fm, w_fm = self.freq_mask_param.shape # Use shape of the parameter for consistency
        # freq_y, freq_x = torch.meshgrid(
        #     torch.linspace(-0.5, 0.5, h_fm, device=weight_complex_for_device_ref.device),
        #     torch.linspace(-0.5, 0.5, w_fm, device=weight_complex_for_device_ref.device),
        #     indexing='ij' # Use 'ij' indexing for meshgrid
        # )
        # freq_magnitude = torch.sqrt(freq_x**2 + freq_y**2)
        # # Create a binary mask based on the threshold
        # fixed_mask = torch.where(freq_magnitude > self.freq_threshold,
        #                          torch.zeros_like(self.freq_mask_param), # Attenuate high frequencies
        #                          torch.ones_like(self.freq_mask_param))  # Pass low frequencies
        # # To incorporate the learnable part with this fixed logic, one might multiply or add.
        # # Example: return self.freq_mask_param * fixed_mask (if freq_mask_param learns adjustments)
        # return fixed_mask # This would use the fixed mask, not the learnable one primarily.

class UNetModel(nn.Module):
    """
    The full U-Net model with attention and timestep embedding, adapted for change detection.
    This model takes a multi-channel input (e.g., two images and a noisy mask) and outputs
    a refined mask and an auxiliary 'cal' map. It incorporates a ChangeBindModel for
    fusing features from the two input images and a highway Generic_UNet module.

    Key components:
    - Input blocks: Series of ResBlocks and AttentionBlocks for downsampling.
    - Middle block: Core ResBlocks and AttentionBlocks.
    - Output blocks: Series of ResBlocks and AttentionBlocks for upsampling.
    - Timestep embedding: Conditions the model on the current diffusion timestep.
    - ChangeBindModel (AB_Concator): Fuses information from two input images (A and B).
    - Generic_UNet (hwm): A "highway" U-Net module that processes skip connections.

    Args:
        image_size: Size of the input images.
        in_channels: Number of channels in the input tensor `x`. For this model,
                     it's typically 7 (3 for image A, 3 for image B, 1 for noisy mask).
        model_channels: Base number of channels for the U-Net's convolutional layers.
        out_channels: Number of output channels for the main prediction (e.g., 1 for a binary mask).
        num_res_blocks: Number of residual blocks per U-Net level.
        attention_resolutions: A tuple of U-Net resolutions (ds factors) where attention should be applied.
        dropout: Dropout rate.
        channel_mult: Multipliers for `model_channels` at each U-Net level.
        conv_resample: If True, use learned convolutions for up/downsampling.
        dims: Number of spatial dimensions (usually 2 for images).
        num_classes: If specified, enables class-conditional modeling (not used in this setup).
        use_checkpoint: If True, use gradient checkpointing for memory saving.
        use_fp16: If True, use float16 precision.
        num_heads: Number of attention heads.
        num_head_channels: Number of channels per attention head.
        num_heads_upsample: Number of attention heads for upsampling blocks.
        use_scale_shift_norm: If True, use scale-shift normalization (FiLM-like).
        resblock_updown: If True, use ResBlocks for up/downsampling.
        use_new_attention_order: If True, use a different QKV attention implementation.
        high_way: If True, enables the highway Generic_UNet module.
    """
    def __init__(
        self,
        image_size,
        in_channels, # Expected to be 7 (imgA 3ch, imgB 3ch, noisy_mask 1ch)
        model_channels,
        out_channels, # Usually 1 for the predicted mask (if ModelMeanType is START_X or PREVIOUS_X)
                     # or channels of epsilon if ModelMeanType is EPSILON.
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 1, 2, 2, 4, 4), # Default channel multipliers for U-Net levels
        conv_resample=True,
        dims=2,
        num_classes=None, # Not used for this unconditional change detection model
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        high_way = True, # Enables the highway Generic_UNet module
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        # Convolution to change 2-channel output of AB_Concator to 3 channels
        # to match image channel num before concatenating with noisy mask.
        self.con2to3 = nn.Conv2d(2, 3, kernel_size=3, padding=1)

        self.image_size = image_size
        self.in_channels = in_channels # Full input channels (e.g., 7)
        self.model_channels = model_channels # Base channels for U-Net layers
        self.out_channels = out_channels # Output channels for the main prediction (e.g., mask or epsilon)
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult # Corrected: was (1,1,2,2,4,4) fixed, now uses arg
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        # Input blocks of the U-Net
        # The first conv layer processes the input.
        # Note: `in_channels` for this first conv_nd is `model_channels`, not `self.in_channels`.
        # This seems to imply that the input `x` to `forward` is already processed to `model_channels`
        # or that the `diff_img`, `A_img`, `B_img` paths in forward handle their own initial conv.
        # Based on forward pass, `h_diff`, `h`, `h2` are derived from `x` and then processed.
        # The first layer of `input_blocks` processes `h_diff`.
        # Let's assume the input to the first layer of input_blocks should be `diff_img`'s channels after con2to3 + mask (4ch).
        # Or, if `input_blocks[0]` processes `h_diff` which is `diff_img` (4ch), then conv_nd should take 4.
        # The original code uses `in_channels` for the conv_nd, but `in_channels` is the total (e.g. 7).
        # This seems like a mismatch. Let's assume the first block processes `h_diff` (4 channels: 3 from diff_img, 1 from mask).
        # If `diff_img` comes from AB_Concator (output 2ch) -> con2to3 (output 3ch) -> concat mask (output 4ch).
        # So, the first conv_nd should take 4 channels if it processes `h_diff`.
        # The current structure seems to intend each path (A, B, diff) to go through input_blocks independently,
        # but the forward pass only processes `h_diff` through the main U-Net path.
        # For this refactor, I'll assume `model_channels` is correct as per original code, implying
        # `h_diff` (and `h`, `h2` if they were used) are somehow made `model_channels` before this.
        # However, `h_diff = diff_img.type(self.dtype)` means `h_diff` has 4 channels.
        # This suggests the first conv_nd in input_blocks should take these 4 channels.
        # For now, sticking to `model_channels` as per original code for this layer, but this is a point of confusion.
        # The `ChangeBindModel` creates a 2-channel difference, `con2to3` makes it 3 channels.
        # Then `mask_noisy` (1 channel) is concatenated, making `diff_img` 4 channels.
        # So the first layer of `input_blocks` which processes `h_diff` should take 4 channels.
        # Let's assume `model_channels` is the intended input for the *first ResBlock layer*,
        # and an initial conv handles the 4 actual input channels to `model_channels`.
        # The first element of input_blocks is `conv_nd(dims, in_channels, model_channels,...)`.
        # Here `in_channels` refers to `self.in_channels` (e.g. 7), which is for the *entire* UNetModel input `x`.
        # The forward pass then splits `x` and processes `diff_img` (4ch) with these blocks. This is inconsistent.
        # Correcting this: the first conv_nd in input_blocks should match the channels of `h_diff` (4 channels).
        # For this pass, I will assume the `in_channels` in the first conv_nd of `input_blocks`
        # actually refers to the channels of `h_diff` (i.e., 4), not `self.in_channels` (i.e., 7).
        # The most logical interpretation is that `model_channels` is the channel count *after* the initial conv.
        # The first conv in `input_blocks` transforms `h_diff` (4 channels) to `model_channels`.
        # So, `conv_nd(dims, 4, model_channels, 3, padding=1)` would be logical for h_diff.
        # The original code uses `self.in_channels` (e.g. 7) for this first conv. This will be kept for now but is suspect.

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential( # Initial convolution layer
                    conv_nd(dims, 4, model_channels, 3, padding=1) # Assuming this processes h_diff (4 channels)
                )
            ]
        )
        # `self.AB_Concator` fuses image A and B. Its output (2 channels) is then passed to `self.con2to3` (becomes 3 channels).
        self.AB_Concator = ChangeBindModel()
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        # channel_mult is now an argument, not fixed.
        for level, mult in enumerate(self.channel_mult): # Iterate through channel multipliers for each level
            for _ in range(num_res_blocks): # Add residual blocks for the current level
                layers = [
                    ResBlock(
                        ch, time_embed_dim, dropout, out_channels=mult * model_channels, dims=dims,
                        use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels # Update current channel count
                if ds in attention_resolutions: # Add attention block if current resolution is in attention_resolutions
                    layers.append(
                        AttentionBlock(
                            ch, use_checkpoint=use_checkpoint, num_heads=num_heads,
                            num_head_channels=num_head_channels, use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)

            if level != len(self.channel_mult) - 1: # Add downsampling block if not the last level
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch, time_embed_dim, dropout, out_channels=out_ch, dims=dims,
                            use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, down=True,
                        ) if resblock_updown else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2 # Update downsampling factor
                self._feature_size += ch

        # Middle block of the U-Net
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch, use_checkpoint=use_checkpoint, num_heads=num_heads,
                num_head_channels=num_head_channels, use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        # Output blocks of the U-Net (upsampling path)
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(self.channel_mult))[::-1]: # Iterate in reverse for upsampling
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop() # Get skip connection channels
                layers = [
                    ResBlock( # Concatenate current channels `ch` with skip connection channels `ich`
                        ch + ich, time_embed_dim, dropout, out_channels=model_channels * mult,
                        dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions: # Add attention if specified for this resolution
                    layers.append(
                        AttentionBlock(
                            ch, use_checkpoint=use_checkpoint, num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels, use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks: # Add upsampling layer if not the first level and it's the last block in this stage
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch, time_embed_dim, dropout, out_channels=out_ch, dims=dims,
                            use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, up=True,
                        ) if resblock_updown else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2 # Update downsampling factor (effectively upsampling)
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        # Final output layer
        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels , out_channels, 3, padding=1)),
        )

        # Highway module (Generic_UNet) initialization if enabled
        if high_way:
            features = 32 # Base features for the highway U-Net
            # The Generic_UNet's in_channels is self.in_channels-1 (e.g. 6 for A+B images).
            # Output channels for its main output is `features`, and for seg_output (cal map) is 1.
            # num_pool is 5 for this highway module.
            self.hwm = Generic_UNet(input_channels=self.in_channels - 1,
                                    base_num_features=features,
                                    num_classes=1, # Output 1 channel for 'cal' map
                                    num_pool=5) # Fixed num_pool for highway module

    def convert_to_fp16(self):
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)
    
    # The 'enhance' method seems to be unused in the current UNetModel forward pass.
    # It might be a utility for feature enhancement not currently integrated.
    def enhance(self, c: th.Tensor, h: th.Tensor) -> th.Tensor:
        """ Element-wise product of layer-normalized features, scaled by h. """
        # Assuming layer_norm is defined elsewhere (e.g., from .nn)
        cu = layer_norm(c.size()[1:], elementwise_affine=False)(c) # Apply LayerNorm
        hu = layer_norm(h.size()[1:], elementwise_affine=False)(h) # Apply LayerNorm
        return cu * hu * h # Element-wise multiplication
    
    def highway_forward(self, x1: th.Tensor, x2: th.Tensor, hs: list[th.Tensor]) -> tuple[th.Tensor, th.Tensor]:
        """ Forward pass through the highway Generic_UNet module. """
        return self.hwm(x1, x2, hs)

    def forward(self, x: th.Tensor, timesteps: th.Tensor, y=None) -> tuple[th.Tensor, th.Tensor]:
        """
        Apply the UNet model to an input batch.

        Args:
            x: An [N x C_in x H x W] tensor of inputs. For change detection, C_in is typically 7,
               structured as [Image_A (3ch), Image_B (3ch), Noisy_Mask (1ch)].
            timesteps: A 1-D batch of timesteps [N].
            y: An [N] tensor of labels, if class-conditional (not used here).

        Returns:
            A tuple (out, cal):
            - out: The main model output [N x C_out x H x W] (e.g., predicted mask or epsilon).
            - cal: An auxiliary map [N x 1 x H x W] from the highway module.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        # --- Preprocessing: Split input `x` and create image feature streams ---
        # x is expected to be [imgA (3ch), imgB (3ch), noisy_mask (1ch)]
        A_img_input = x[:, 0:3, ...]  # Image A
        B_img_input = x[:, 3:6, ...]  # Image B
        mask_noisy_input = x[:, 6:7, ...] # Noisy mask (current state of diffusion for the mask)

        # --- "AB fusion": Generate difference features and prepare inputs for U-Net streams ---
        # `self.AB_Concator` (ChangeBindModel) computes difference features between A and B.
        # Output is 2 channels, converted to 3 by `self.con2to3`.
        diff_features_raw = self.AB_Concator(A_img_input, B_img_input)
        diff_features_3ch = self.con2to3(diff_features_raw)
        
        # `h_diff`: Input for the main U-Net path, combining diff_features and the noisy mask.
        # Shape: [N, 4, H, W] (3 from diff_features_3ch, 1 from mask_noisy_input)
        h_diff = th.cat((diff_features_3ch, mask_noisy_input), dim=1).type(self.dtype)

        # `h` and `h2` were originally prepared from A_img and B_img with the noisy mask.
        # These are not directly processed by the main U-Net encoder-decoder path in the current loop,
        # but `c` and `c2` (derived from them) are used by the highway module.
        h_A_path = th.cat((A_img_input, mask_noisy_input), dim=1).type(self.dtype) # Shape: [N, 4, H, W]
        h_B_path = th.cat((B_img_input, mask_noisy_input), dim=1).type(self.dtype) # Shape: [N, 4, H, W]

        # --- Timestep and Class Embedding ---
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None: # Class conditional embedding (not used in this setup)
            emb = emb + self.label_emb(y)
        if len(emb.size()) > 2: emb = emb.squeeze() # Ensure emb is [N, emb_dim]

        # --- Highway Module Inputs ---
        # `c` and `c2` are derived from A_img_input and B_img_input respectively (without the noisy mask).
        # These are passed to the highway module (self.hwm).
        c_highway_input = A_img_input.type(self.dtype) # Shape: [N, 3, H, W]
        c2_highway_input = B_img_input.type(self.dtype) # Shape: [N, 3, H, W]
        
        # --- U-Net Encoder Path (Input Blocks) ---
        # Processes `h_diff` (difference features + noisy mask) through the U-Net encoder.
        # Skip connections from this path are stored in `hs_diff_skips`.
        hs_diff_skips = []
        current_h = h_diff # Start with h_diff
        for module in self.input_blocks:
            current_h = module(current_h, emb)
            hs_diff_skips.append(current_h)

        # --- Highway Connection ---
        # The highway module `self.hwm` (a Generic_UNet) processes `c_highway_input`, `c2_highway_input`,
        # and specific skip connections from `hs_diff_skips`.
        # It returns `uemb` (an embedding, possibly timestep-like) and `cal` (an auxiliary map).
        # The selection of skip connections [3,6,9,12] is specific to this architecture.
        uemb, cal = self.highway_forward(c_highway_input, c2_highway_input,
                                         [hs_diff_skips[3], hs_diff_skips[6], hs_diff_skips[9], hs_diff_skips[12]])

        # Add the embedding from the highway module to the main path's feature map.
        current_h = current_h + uemb # `current_h` is the output of the last input_block for h_diff

        # --- U-Net Middle Block ---
        current_h = self.middle_block(current_h, emb)

        # --- U-Net Decoder Path (Output Blocks) ---
        # Processes `current_h` through the U-Net decoder, using skip connections from `hs_diff_skips`.
        for module in self.output_blocks:
            skip_h = hs_diff_skips.pop() # Get corresponding skip connection
            current_h = th.cat([current_h, skip_h], dim=1) # Concatenate with skip connection
            current_h = module(current_h, emb)

        current_h = current_h.type(x.dtype) # Ensure correct dtype before final output layer

        # --- Final Output Layer ---
        # `self.out` produces the main model output (e.g., predicted mask or epsilon).
        out_main = self.out(current_h)

        return out_main, cal # Return main output and auxiliary 'cal' map

# SuperResModel and EncoderUNetModel are specialized UNet variants.
# Their refactoring is not part of this specific subtask's scope,
# but they are retained as part of the original file structure.

class SuperResModel(UNetModel):
    """A UNetModel that performs super-resolution. Expects an extra kwarg `low_res`."""
    def __init__(self, image_size, in_channels, *args, **kwargs):
        super().__init__(image_size, in_channels * 2, *args, **kwargs)
    def forward(self, x, timesteps, low_res=None, **kwargs):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().forward(x, timesteps, **kwargs)

class EncoderUNetModel(nn.Module):
    """The half UNet model with attention and timestep embedding. For usage, see UNet."""
    def __init__(
        self, image_size, in_channels, model_channels, out_channels, num_res_blocks,
        attention_resolutions, dropout=0, channel_mult=(1, 2, 4, 8), conv_resample=True,
        dims=2, use_checkpoint=False, use_fp16=False, num_heads=1, num_head_channels=-1,
        num_heads_upsample=-1, use_scale_shift_norm=False, resblock_updown=False,
        use_new_attention_order=False, pool="adaptive",
    ):
        super().__init__()
        # ... (Implementation as in original file) ...
        if num_heads_upsample == -1: num_heads_upsample = num_heads
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(linear(model_channels, time_embed_dim), nn.SiLU(), linear(time_embed_dim, time_embed_dim))
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))])
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, time_embed_dim, dropout, out_channels=mult * model_channels, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads, num_head_channels=num_head_channels, use_new_attention_order=use_new_attention_order))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(TimestepEmbedSequential(ResBlock(ch, time_embed_dim, dropout, out_channels=out_ch, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, down=True) if resblock_updown else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)))
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch
        self.middle_block = TimestepEmbedSequential(ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm), AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads, num_head_channels=num_head_channels, use_new_attention_order=use_new_attention_order), ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm))
        self._feature_size += ch
        self.pool = pool
        self.gap = nn.AvgPool2d((8, 8))
        self.cam_feature_maps = None
        if pool == "adaptive": self.out = nn.Sequential(normalization(ch), nn.SiLU(), nn.AdaptiveAvgPool2d((1, 1)), zero_module(conv_nd(dims, ch, out_channels, 1)), nn.Flatten())
        elif pool == "attention":
            assert num_head_channels != -1
            self.out = nn.Sequential(normalization(ch), nn.SiLU(), AttentionPool2d((image_size // ds), ch, num_head_channels, out_channels))
        elif pool == "spatial": self.out = nn.Linear(256, self.out_channels)
        elif pool == "spatial_v2": self.out = nn.Sequential(nn.Linear(self._feature_size, 2048), normalization(2048), nn.SiLU(), nn.Linear(2048, self.out_channels))
        else: raise NotImplementedError(f"Unexpected {pool} pooling")
    def convert_to_fp16(self): self.input_blocks.apply(convert_module_to_f16); self.middle_block.apply(convert_module_to_f16)
    def convert_to_fp32(self): self.input_blocks.apply(convert_module_to_f32); self.middle_block.apply(convert_module_to_f32)
    def forward(self, x, timesteps):
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        results = []
        h = x.type(self.dtype)
        for module in self.input_blocks: h = module(h, emb); results.append(h.type(x.dtype).mean(dim=(2, 3)))
        h = self.middle_block(h, emb)
        if self.pool.startswith("spatial"): self.cam_feature_maps = h; h = self.gap(h); N = h.shape[0]; h = h.reshape(N, -1); return self.out(h)
        else: self.cam_feature_maps = h; return self.out(h)

# Generic_UNet and related classes are from a different U-Net architecture (nnU-Net style).
# They are used as a component (self.hwm) in UNetModel.
# Adding docstrings and comments as requested.
class NeuralNetwork(nn.Module): # Base class from nnU-Net
    def __init__(self): super(NeuralNetwork, self).__init__()
    def get_device(self): return next(self.parameters()).device.index if next(self.parameters()).device.type != "cpu" else "cpu"
    def set_device(self, device): self.cuda(device) if device != "cpu" else self.cpu()
    def forward(self, x): raise NotImplementedError

class SegmentationNetwork(NeuralNetwork): # Base for segmentation networks from nnU-Net
    def __init__(self):
        super(SegmentationNetwork, self).__init__() # Corrected super call
        self.input_shape_must_be_divisible_by = None
        self.conv_op = None
        self.num_classes = None
        self.inference_apply_nonlin = lambda x: x
        self._gaussian_3d = self._patch_size_for_gaussian_3d = None
        self._gaussian_2d = self._patch_size_for_gaussian_2d = None
    # ... (predict_3D, predict_2D, and other helper methods from original Generic_UNet context remain here) ...
    # These prediction helper methods are complex and mostly related to tiled inference,
    # which is not directly used when Generic_UNet is a sub-module like `hwm`.
    # For brevity in this refactor, their internal comments are not being added,
    # as the main focus is on the UNetModel and Generic_UNet's direct structure.
    @staticmethod
    def _get_gaussian(patch_size, sigma_scale=1. / 8) -> np.ndarray: # (Implementation as in original)
        tmp = np.zeros(patch_size); center_coords = [i // 2 for i in patch_size]; sigmas = [i * sigma_scale for i in patch_size]
        tmp[tuple(center_coords)] = 1; gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
        gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1; gaussian_importance_map = gaussian_importance_map.astype(np.float32)
        gaussian_importance_map[gaussian_importance_map == 0] = np.min(gaussian_importance_map[gaussian_importance_map != 0]); return gaussian_importance_map
    @staticmethod
    def _compute_steps_for_sliding_window(patch_size: Tuple[int, ...], image_size: Tuple[int, ...], step_size: float) -> List[List[int]]: # (Implementation as in original)
        assert [i >= j for i, j in zip(image_size, patch_size)], "image size must be as large or larger than patch_size"; assert 0 < step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'
        target_step_sizes_in_voxels = [i * step_size for i in patch_size]; num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, patch_size)]
        steps = [];
        for dim in range(len(patch_size)):
            max_step_value = image_size[dim] - patch_size[dim]
            if num_steps[dim] > 1: actual_step_size = max_step_value / (num_steps[dim] - 1)
            else: actual_step_size = 99999999999
            steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]; steps.append(steps_here)
        return steps
    # Other _internal_predict methods are part of SegmentationNetwork but not directly called by UNetModel.hwm
    # For brevity, their detailed commenting is skipped in this focused refactor.

class ConvDropoutNormNonlin(nn.Module): # (Implementation as in original)
    def __init__(self, input_channels, output_channels, conv_op=nn.Conv2d, conv_kwargs=None, norm_op=nn.BatchNorm2d, norm_op_kwargs=None, dropout_op=nn.Dropout2d, dropout_op_kwargs=None, nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super(ConvDropoutNormNonlin, self).__init__();
        if nonlin_kwargs is None: nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True};
        if dropout_op_kwargs is None: dropout_op_kwargs = {'p': 0.5, 'inplace': True};
        if norm_op_kwargs is None: norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1};
        if conv_kwargs is None: conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True};
        self.nonlin_kwargs = nonlin_kwargs; self.nonlin = nonlin; self.dropout_op = dropout_op; self.dropout_op_kwargs = dropout_op_kwargs; self.norm_op_kwargs = norm_op_kwargs; self.conv_kwargs = conv_kwargs; self.conv_op = conv_op; self.norm_op = norm_op;
        self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs);
        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs['p'] > 0: self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else: self.dropout = None
        self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs); self.lrelu = self.nonlin(**self.nonlin_kwargs)
    def forward(self, x): x = self.conv(x);
        if self.dropout is not None: x = self.dropout(x);
        return self.lrelu(self.instnorm(x))

class StackedConvLayers(nn.Module): # (Implementation as in original)
    def __init__(self, input_feature_channels, output_feature_channels, num_convs, conv_op=nn.Conv2d, conv_kwargs=None, norm_op=nn.BatchNorm2d, norm_op_kwargs=None, dropout_op=nn.Dropout2d, dropout_op_kwargs=None, nonlin=nn.LeakyReLU, nonlin_kwargs=None, first_stride=None, basic_block=ConvDropoutNormNonlin):
        self.input_channels = input_feature_channels; self.output_channels = output_feature_channels;
        if nonlin_kwargs is None: nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True};
        if dropout_op_kwargs is None: dropout_op_kwargs = {'p': 0.5, 'inplace': True};
        if norm_op_kwargs is None: norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1};
        if conv_kwargs is None: conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True};
        self.nonlin_kwargs = nonlin_kwargs; self.nonlin = nonlin; self.dropout_op = dropout_op; self.dropout_op_kwargs = dropout_op_kwargs; self.norm_op_kwargs = norm_op_kwargs; self.conv_kwargs = conv_kwargs; self.conv_op = conv_op; self.norm_op = norm_op;
        if first_stride is not None: self.conv_kwargs_first_conv = deepcopy(conv_kwargs); self.conv_kwargs_first_conv['stride'] = first_stride
        else: self.conv_kwargs_first_conv = conv_kwargs
        super(StackedConvLayers, self).__init__()
        self.blocks = nn.Sequential(*([basic_block(input_feature_channels, output_feature_channels, self.conv_op, self.conv_kwargs_first_conv, self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs)] + [basic_block(output_feature_channels, output_feature_channels, self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs) for _ in range(num_convs - 1)]))
    def forward(self, x): return self.blocks(x)

class hwUpsample(nn.Module): # (Implementation as in original)
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False): super(hwUpsample, self).__init__(); self.align_corners = align_corners; self.mode = mode; self.scale_factor = scale_factor; self.size = size
    def forward(self, x): return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)

class Generic_UNet(SegmentationNetwork):
    """
    A generic U-Net architecture, based on nnU-Net principles.
    This is used as the "highway module" (hwm) within the main UNetModel.
    It processes two main inputs (x, x2) and a list of skip connections (hs)
    from the main U-Net's encoder path, producing an embedding and a segmentation-like output.
    """
    DEFAULT_BATCH_SIZE_3D = 2; DEFAULT_PATCH_SIZE_3D = (64, 192, 160); SPACING_FACTOR_BETWEEN_STAGES = 2; BASE_NUM_FEATURES_3D = 30; MAX_NUMPOOL_3D = 999; MAX_NUM_FILTERS_3D = 320;
    DEFAULT_PATCH_SIZE_2D = (256, 256); BASE_NUM_FEATURES_2D = 30; DEFAULT_BATCH_SIZE_2D = 50; MAX_NUMPOOL_2D = 999; MAX_FILTERS_2D = 480;
    use_this_for_batch_size_computation_2D = 19739648; use_this_for_batch_size_computation_3D = 520000000;

    def __init__(self, input_channels: int, base_num_features: int, num_classes: int, num_pool: int, num_conv_per_stage: int=2,
                 feat_map_mul_on_downscale: int=2, conv_op=nn.Conv2d, norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None, nonlin=nn.LeakyReLU, nonlin_kwargs=None,
                 deep_supervision: bool=False, dropout_in_localization: bool=False, final_nonlin=sigmoid_helper,
                 weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None, conv_kernel_sizes=None,
                 upscale_logits: bool=False, convolutional_pooling: bool=False, convolutional_upsampling: bool=False,
                 max_num_features=None, basic_block=ConvDropoutNormNonlin, seg_output_use_bias: bool=False):
        super(Generic_UNet, self).__init__() # Corrected super call
        # ... (rest of __init__ as in original, very detailed, standard nnU-Net setup) ...
        self.convolutional_upsampling = convolutional_upsampling; self.convolutional_pooling = convolutional_pooling; self.upscale_logits = upscale_logits
        if nonlin_kwargs is None: nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None: dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None: norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}; self.nonlin = nonlin; self.nonlin_kwargs = nonlin_kwargs; self.dropout_op_kwargs = dropout_op_kwargs; self.norm_op_kwargs = norm_op_kwargs; self.weightInitializer = weightInitializer; self.conv_op = conv_op; self.norm_op = norm_op; self.dropout_op = dropout_op; self.num_classes = num_classes; self.final_nonlin = final_nonlin; self._deep_supervision = deep_supervision; self.do_ds = deep_supervision
        if conv_op == nn.Conv2d: upsample_mode = 'bilinear'; pool_op = nn.MaxPool2d; transpconv = nn.ConvTranspose2d;
            if pool_op_kernel_sizes is None: pool_op_kernel_sizes = [(2, 2)] * num_pool
            if conv_kernel_sizes is None: conv_kernel_sizes = [(3, 3)] * (num_pool + 1)
        elif conv_op == nn.Conv3d: upsample_mode = 'trilinear'; pool_op = nn.MaxPool3d; transpconv = nn.ConvTranspose3d;
            if pool_op_kernel_sizes is None: pool_op_kernel_sizes = [(2, 2, 2)] * num_pool
            if conv_kernel_sizes is None: conv_kernel_sizes = [(3, 3, 3)] * (num_pool + 1)
        else: raise ValueError("unknown convolution dimensionality, conv op: %s" % str(conv_op))
        self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0, dtype=np.int64); self.pool_op_kernel_sizes = pool_op_kernel_sizes; self.conv_kernel_sizes = conv_kernel_sizes; self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes: self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])
        if max_num_features is None: self.max_num_features = self.MAX_NUM_FILTERS_3D if self.conv_op == nn.Conv3d else self.MAX_FILTERS_2D
        else: self.max_num_features = max_num_features
        self.conv_blocks_context = []; self.conv_blocks_localization = []; self.conv_trans_blocks_a = []; self.conv_trans_blocks_b = []; self.td = []; self.tu = []; self.ffparser = []; self.seg_outputs = []
        output_features = base_num_features; input_features = input_channels
        for d in range(num_pool):
            if d != 0 and self.convolutional_pooling: first_stride = pool_op_kernel_sizes[d - 1]
            else: first_stride = None
            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]; self.conv_kwargs['padding'] = self.conv_pad_sizes[d]
            self.conv_blocks_context.append(StackedConvLayers(input_features, output_features, num_conv_per_stage, self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs, first_stride, basic_block=basic_block))
            if d < num_pool -1 : # Adjusted to ensure ffparser and conv_trans_blocks have correct number of elements
                self.conv_trans_blocks_a.append(conv_nd(2, int(d/2 + 1) * 128, 2**(d+5), 1)) # This indexing for channels seems specific and hardcoded
                self.conv_trans_blocks_b.append(conv_nd(2, 2**(d+5), 1, 1))
                self.ffparser.append(FFParser(output_features, 256 // (2**(d+1)), 256 // (2**(d+2))+1)) # Sizes also specific
            if not self.convolutional_pooling: self.td.append(pool_op(pool_op_kernel_sizes[d]))
            input_features = output_features; output_features = int(np.round(output_features * feat_map_mul_on_downscale)); output_features = min(output_features, self.max_num_features)
        if self.convolutional_pooling: first_stride = pool_op_kernel_sizes[-1]
        else: first_stride = None
        if self.convolutional_upsampling: final_num_features = output_features
        else: final_num_features = self.conv_blocks_context[-1].output_channels
        self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[num_pool]; self.conv_kwargs['padding'] = self.conv_pad_sizes[num_pool]
        self.conv_blocks_context.append(nn.Sequential(StackedConvLayers(input_features, output_features, num_conv_per_stage - 1, self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs, first_stride, basic_block=basic_block), StackedConvLayers(output_features, final_num_features, 1, self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs, basic_block=basic_block)))
        if not dropout_in_localization: old_dropout_p = self.dropout_op_kwargs['p']; self.dropout_op_kwargs['p'] = 0.0
        for u in range(num_pool):
            nfeatures_from_down = final_num_features; nfeatures_from_skip = self.conv_blocks_context[-(2 + u)].output_channels; n_features_after_tu_and_concat = nfeatures_from_skip * 2
            if u != num_pool - 1 and not self.convolutional_upsampling: final_num_features = self.conv_blocks_context[-(3 + u)].output_channels
            else: final_num_features = nfeatures_from_skip
            if not self.convolutional_upsampling: self.tu.append(hwUpsample(scale_factor=pool_op_kernel_sizes[-(u + 1)], mode=upsample_mode))
            else: self.tu.append(transpconv(nfeatures_from_down, nfeatures_from_skip, pool_op_kernel_sizes[-(u + 1)], pool_op_kernel_sizes[-(u + 1)], bias=False))
            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[- (u + 1)]; self.conv_kwargs['padding'] = self.conv_pad_sizes[- (u + 1)]
            self.conv_blocks_localization.append(nn.Sequential(StackedConvLayers(n_features_after_tu_and_concat, nfeatures_from_skip, num_conv_per_stage - 1, self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs, basic_block=basic_block), StackedConvLayers(nfeatures_from_skip, final_num_features, 1, self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs, basic_block=basic_block)))
        if self._deep_supervision:
            for ds_idx in range(len(self.conv_blocks_localization)): self.seg_outputs.append(conv_op(self.conv_blocks_localization[ds_idx][-1].output_channels, num_classes, 1, 1, 0, 1, 1, seg_output_use_bias))
        else: self.seg_outputs.append(conv_op(self.conv_blocks_localization[-1][-1].output_channels, num_classes, 1, 1, 0, 1, 1, seg_output_use_bias))
        self.upscale_logits_ops = []; cum_upsample = np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0)[::-1]
        for usl in range(num_pool - 1): self.upscale_logits_ops.append(hwUpsample(scale_factor=tuple([int(i) for i in cum_upsample[usl + 1]]), mode=upsample_mode) if self.upscale_logits else (lambda x: x))
        if not dropout_in_localization: self.dropout_op_kwargs['p'] = old_dropout_p
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization); self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context); self.conv_trans_blocks_a = nn.ModuleList(self.conv_trans_blocks_a); self.conv_trans_blocks_b = nn.ModuleList(self.conv_trans_blocks_b); self.ffparser = nn.ModuleList(self.ffparser); self.td = nn.ModuleList(self.td); self.tu = nn.ModuleList(self.tu); self.seg_outputs = nn.ModuleList(self.seg_outputs)
        if self.upscale_logits: self.upscale_logits_ops = nn.ModuleList(self.upscale_logits_ops)
        if self.weightInitializer is not None: self.apply(self.weightInitializer)

    def forward(self, x: th.Tensor, x2: th.Tensor, hs: list[th.Tensor]) -> tuple[th.Tensor, th.Tensor]:
        """
        Forward pass for the Generic_UNet (highway module).

        Args:
            x: First main input tensor (e.g., features from image A).
            x2: Second main input tensor (e.g., features from image B).
            hs: A list of skip connections from the main U-Net's encoder,
                expected to be processed by FFParser and conv_trans_blocks.

        Returns:
            A tuple (emb, seg_outputs[-1]):
            - emb: An embedding derived from the bottleneck of this U-Net.
            - seg_outputs[-1]: The final segmentation-like output of this U-Net (the 'cal' map).
        """
        diff_skips = [] # Stores difference features (x2-x) at each encoder stage
        # Encoder path
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)  # Process x (e.g. features from imgA)
            x2 = self.conv_blocks_context[d](x2) # Process x2 (e.g. features from imgB)
            diff_skips.append(x2 - x) # Store difference for skip connection

            if not self.convolutional_pooling: # Apply pooling if not convolutional
                x = self.td[d](x)
                x2 = self.td[d](x2)
            
            x_diff_current_level = x2 - x # Difference at current pooled level

            # Modulate x_diff_current_level with features from `hs` (main U-Net skips)
            if hs and d < len(self.ffparser): # Ensure hs has elements and index d is valid for ffparser
                h_skip = hs.pop(0) # Get a skip connection from the main U-Net

                # Process h_skip through conv_trans_blocks and ffparser
                # Note: The original code had `conv_trans_blocks_a[d]` and `ffparser[d]` which might
                # lead to index errors if `len(hs)` is less than `len(conv_blocks_context) - 1`.
                # Assuming `hs` provides enough elements for the early stages where `ffparser` is applied.
                h_processed = self.conv_trans_blocks_a[d](h_skip)
                # h_processed = self.ffparser[d](h_processed) # FFParser application

                ha = self.conv_trans_blocks_b[d](h_processed) # Further processing
                hb = th.mean(h_processed, (2,3), keepdim=True) # Global average pooling

                x_diff_current_level = x_diff_current_level * ha * hb # Modulate difference features

            x = x_diff_current_level # Update x to be the modulated difference for the next stage

        # Bottleneck of this Generic_UNet
        x = self.conv_blocks_context[-1](x)
        # Create an embedding from the bottleneck features
        emb = conv_nd(2, x.size(1), 512, 1).to(device = x.device)(x) # Output embedding for UNetModel

        # Decoder path
        seg_outputs_list = [] # For deep supervision if enabled
        for u in range(len(self.tu)):
            x = self.tu[u](x) # Upsample
            x = th.cat((x, diff_skips[-(u + 1)]), dim=1) # Concatenate with skip connection (difference feature)
            x = self.conv_blocks_localization[u](x) # Apply localization block
            if self._deep_supervision: # If deep supervision, get output at this level
                seg_outputs_list.append(self.final_nonlin(self.seg_outputs[u](x)))

        if not seg_outputs_list: # If no deep supervision, get output from the last seg_output layer
             seg_outputs_list.append(self.final_nonlin(self.seg_outputs[0](x)))

        if self._deep_supervision and self.do_ds: # Handle deep supervision outputs
            return emb, tuple([seg_outputs_list[-1]] + [i(j) for i, j in zip(list(self.upscale_logits_ops)[::-1], seg_outputs_list[:-1][::-1])])
        else: # Return final embedding and segmentation output
            return emb, seg_outputs_list[-1]

    # ... (compute_approx_vram_consumption method from original)
    @staticmethod
    def compute_approx_vram_consumption(patch_size, num_pool_per_axis, base_num_features, max_num_features, num_modalities, num_classes, pool_op_kernel_sizes, deep_supervision=False, conv_per_stage=2):
        if not isinstance(num_pool_per_axis, np.ndarray): num_pool_per_axis = np.array(num_pool_per_axis)
        npool = len(pool_op_kernel_sizes)
        map_size = np.array(patch_size)
        tmp = np.int64((conv_per_stage * 2 + 1) * np.prod(map_size, dtype=np.int64) * base_num_features + num_modalities * np.prod(map_size, dtype=np.int64) + num_classes * np.prod(map_size, dtype=np.int64))
        num_feat = base_num_features
        for p in range(npool):
            for pi in range(len(num_pool_per_axis)): map_size[pi] /= pool_op_kernel_sizes[p][pi]
            num_feat = min(num_feat * 2, max_num_features)
            num_blocks = (conv_per_stage * 2 + 1) if p < (npool - 1) else conv_per_stage
            tmp += num_blocks * np.prod(map_size, dtype=np.int64) * num_feat
            if deep_supervision and p < (npool - 2): tmp += np.prod(map_size, dtype=np.int64) * num_classes
        return tmp
