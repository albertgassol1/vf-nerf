import math
import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels: int, kernel_size: int, sigma: float, dim: int = 2) -> None:
        """
        Initialize the gaussian smoothing module.
        :param channels: The number of channels.
        :param kernel_size: The kernel size.
        :param sigma: The sigma value.
        :param dim: The number of dimensions.
        """
        super().__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= (
                1 /
                (std * math.sqrt(2 * math.pi)) *
                torch.exp(-(((mgrid - mean) / (2 * std)) ** 2))
            )

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                "Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim)
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the gaussian smoothing module.
        :param input: The input tensor.
        :return: The smoothed tensor.
        """
        return self.conv(input, weight=self.weight.to(input.device), groups=self.groups)


def smooth_vf(vf: torch.Tensor, k: int = 3, sigma: float = 1.0) -> torch.Tensor:
    """
    Smooth the vector field.
    :param vf: The vector field.
    :param k: The kernel size.
    :param sigma: The sigma value.
    :return: The smoothed vector field.
    """
    # Create the gaussian smoothing module
    smoothing = GaussianSmoothing(3, k, sigma, 3)
    vf = vf.permute(3, 0, 1, 2).unsqueeze(0)
    input = F.pad(
        vf, (k // 2, k // 2, k // 2, k // 2, k // 2, k // 2), mode="replicate"
    )
    output = smoothing(input).squeeze(0).permute(1, 2, 3, 0)

    return output
