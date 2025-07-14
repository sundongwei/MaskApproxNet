import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
# Unused imports removed:
# from functools import partial
# import timm
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# import types
# import math


class ConvolutionalLayer(nn.Module):
    """A simple 2D convolutional layer wrapper."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        """
        Initializes the ConvolutionalLayer.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolving kernel.
            stride: Stride of the convolution.
            padding: Zero-padding added to both sides of the input.
        """
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the convolutional layer.

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width).
        Returns:
            Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        out = self.conv2d(x)
        return out


class UpsamplingConvolutionalLayer(nn.Module):
    """A 2D transposed convolutional layer for upsampling."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        """
        Initializes the UpsamplingConvolutionalLayer.
        This layer uses ConvTranspose2d to increase spatial dimensions.
        Padding is fixed to 1, which is common for upsampling by factor of 2
        with kernel_size 4 and stride 2.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolving kernel.
            stride: Stride of the convolution (typically 2 for upsampling).
        """
        super().__init__()
        # Using ConvTranspose2d for upsampling.
        # padding=1, kernel_size=4, stride=2 typically doubles the input size.
        self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the upsampling convolutional layer.

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width).
        Returns:
            Output tensor with increased spatial dimensions.
        """
        out = self.conv2d(x)
        return out

class ResidualBlock(nn.Module):
    """A standard residual block with two convolutional layers."""
    def __init__(self, channels: int):
        """
        Initializes the ResidualBlock.

        Args:
            channels: Number of input and output channels for the block.
        """
        super().__init__()
        self.conv1 = ConvolutionalLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = ConvolutionalLayer(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the residual block.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).
        Returns:
            Output tensor of the same shape as input.
        """
        residual = x
        out = self.relu(self.conv1(x))
        # The output of the second convolution is scaled by 0.1 before adding the residual.
        # This is a common technique, sometimes referred to as "Residual Scaling".
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        return out


def resize(input_tensor: torch.Tensor, size: tuple[int, int] = None, scale_factor: float = None,
           mode: str = 'nearest', align_corners: bool = None, show_warning: bool = True) -> torch.Tensor:
    """
    A wrapper around F.interpolate that optionally issues a warning regarding align_corners.

    This warning is typically relevant when upsampling and precise alignment is critical,
    often suggesting input/output sizes that are multiples of some factor plus one.

    Args:
        input_tensor: The input tensor to resize.
        size: The target output size (height, width).
        scale_factor: Multiplier for spatial size.
        mode: The algorithm used for upsampling ('nearest', 'linear', 'bilinear', 'bicubic', 'trilinear').
        align_corners: Geometrically, we consider the pixels of the input and output as points on a grid.
                       If set to True, the input and output tensors are aligned by the center points of their corner pixels,
                       preserving the values at the corner pixels.
                       If set to False, the input and output tensors are aligned by the corner points of their corner pixels,
                       and the interpolation uses edge value padding for out-of-boundary values.
        show_warning: If True, display the warning message related to align_corners.

    Returns:
        The resized tensor.
    """
    if show_warning and warnings is not None: # Ensure warnings module is available
        if size is not None and align_corners: # Only warn if align_corners is True and size is specified
            input_h, input_w = tuple(int(x) for x in input_tensor.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)

            if output_h > input_h or output_w > input_w: # Warning only for upsampling
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and ((output_h - 1) % (input_h - 1) != 0
                                           or (output_w - 1) % (input_w - 1) != 0)): # Corrected condition with OR
                    warnings.warn(
                        f'When align_corners={align_corners}, the output would be more aligned if '
                        f'input size H, W were of the form `x+1` and output size H, W were of the form `nx+1` for some integer n. '
                        f'Got input_size=({input_h}, {input_w}) and output_size=({output_h}, {output_w}).'
                    )
    return F.interpolate(input_tensor, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)


class Decoder(nn.Module):
    """
    Decoder module for a change detection task.
    It takes an embedding (feature map) from an encoder and upsamples it
    progressively to produce a change probability map. The upsampling is
    done in two stages, each involving an upsampling convolutional layer followed
    by a residual block. The final output is generated by a convolutional layer.
    """
    def __init__(self, embedding_dim: int = 256, num_output_channels: int = 2):
        """
        Initializes the Decoder.

        Args:
            embedding_dim: The number of channels in the input embedding (feature map).
            num_output_channels: The number of output channels for the change map.
                                 The original code used output_nc=2. This could represent
                                 change/no-change probabilities for a softmax, or two
                                 independent channels. If a single probability map is desired,
                                 this should typically be 1.
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_output_channels = num_output_channels

        # Upsampling path
        # First upsample block: embedding_dim -> embedding_dim / 2
        # Increases spatial resolution by 2x.
        self.upsample_conv_reduce_dim_2x = UpsamplingConvolutionalLayer(
            self.embedding_dim, self.embedding_dim // 2, kernel_size=4, stride=2
        )
        self.residual_block_reduce_dim_2x = nn.Sequential(ResidualBlock(self.embedding_dim // 2))

        # Second upsample block: embedding_dim / 2 -> embedding_dim / 4
        # Increases spatial resolution by another 2x (total 4x).
        self.upsample_conv_reduce_dim_1x = UpsamplingConvolutionalLayer(
            self.embedding_dim // 2, self.embedding_dim // 4, kernel_size=4, stride=2
        )
        self.residual_block_reduce_dim_1x = nn.Sequential(ResidualBlock(self.embedding_dim // 4))
        
        # Final prediction layer to produce the change map.
        self.change_probability_head = ConvolutionalLayer(
            self.embedding_dim // 4, self.num_output_channels, kernel_size=3, stride=1, padding=1
        )
        
        # Output activation function.
        # Note: This activation is defined but not explicitly used in the forward pass of this module.
        # It might be applied externally (e.g., in the loss function or during post-processing),
        # especially if num_output_channels > 1 (e.g. for softmax).
        # If num_output_channels == 1 and a probability is expected, this sigmoid would typically be applied.
        self.output_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Decoder.

        Args:
            x: Input tensor (embedding) of shape (batch_size, embedding_dim, height, width).
        Returns:
            Output tensor (change_map) of shape
            (batch_size, num_output_channels, height_out, width_out).
            The spatial dimensions (height_out, width_out) are 4x those of the input.
        """
        # First upsampling stage
        x = self.upsample_conv_reduce_dim_2x(x)
        x = self.residual_block_reduce_dim_2x(x)

        # Second upsampling stage
        x = self.upsample_conv_reduce_dim_1x(x)
        x = self.residual_block_reduce_dim_1x(x)

        # Final prediction
        change_map = self.change_probability_head(x)

        # As noted in __init__, self.output_activation is not applied here.
        # If num_output_channels is 1 and a bounded probability [0,1] is desired from this module,
        # one would typically apply: change_map = self.output_activation(change_map)
        # If num_output_channels > 1 (e.g. for classification logits), activation (like Softmax)
        # is often handled by the loss function (e.g., nn.CrossEntropyLoss).

        return change_map