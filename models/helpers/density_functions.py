from abc import abstractmethod
from typing import Dict, Optional, Tuple

import torch
from torch import nn


class Density(nn.Module):
    def __init__(self,
                 params_init: Dict[str, torch.Tensor] = dict()) -> None:
        """
        Density function class.
        :params params_init: The initial parameters of the density function.
        """
        super().__init__()
        for p in params_init:
            param = nn.Parameter(torch.tensor(params_init[p]))
            setattr(self, p, param)

    def forward(self,
                input: torch.Tensor,
                beta: Optional[float] = None,
                scale: Optional[float] = None,
                mean: Optional[float] = None,
                cutoff: float = -0.5) -> torch.Tensor:
        """
        Forward pass of the density function.
        :params input: The input tensor.
        :params beta: The beta parameter.
        :params scale: The scale parameter.
        :params mean: The mean parameter.
        :returns: The output tensor.
        """
        return self.density_func(input, beta=beta, scale=scale, mean=mean)

    @abstractmethod
    def density_func(self,
                     input: torch.Tensor,
                     beta: Optional[float] = None,
                     scale: Optional[float] = None) -> torch.Tensor:
        """
        Forward pass of the density function.
        :params input: The input tensor.
        :params beta: The beta parameter.
        :params scale: The scale parameter.
        :returns: The density.
        """
        pass


class SdfDensity(Density):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    def __init__(self, params_init={}, beta_min=0.0001):
        super().__init__(params_init=params_init)
        self.beta_min = torch.tensor(beta_min)

    def density_func(self,
                     input: torch.Tensor,
                     beta: Optional[float] = None,
                     scale: Optional[float] = None,
                     mean: Optional[float] = None,
                     cutoff: float = -0.5) -> torch.Tensor:
        """
        Forward pass of the density function.
        :params input: The input tensor.
        :params beta: The beta parameter.
        :params scale: The scale parameter.
        :params mean: The mean parameter.
        """
        if beta is None:
            beta = self.get_beta()

        alpha = 1 / beta
        return alpha * (0.5 + 0.5 * input.sign() * torch.expm1(-input.abs() / beta))

    def get_beta(self):
        beta = self.beta.abs() + self.beta_min.to(self.beta.device)
        return beta


class SimpleDensity(Density):  # like NeRF
    def __init__(self,
                 params_init: Dict[str, torch.Tensor] = dict(),
                 noise_std: float = 1.0) -> None:
        """
        Simple density function class.
        :params params_init: The initial parameters of the density function.
        :params noise_std: The standard deviation of the noise.
        """
        super().__init__(params_init=params_init)
        self.noise_std = noise_std

    def density_func(self,
                     input: torch.Tensor,
                     beta: Optional[float] = None,
                     scale: Optional[float] = None,
                     mean: Optional[float] = None) -> torch.Tensor:
        """
        Forward pass of the density function.
        :params input: The input tensor.
        :params beta: The beta parameter.
        :params scale: The scale parameter.
        :params mean: The mean parameter.
        :returns: The density.
        """
        if self.training and self.noise_std > 0.0:
            noise = torch.randn(input.shape).cuda() * self.noise_std
            input = input + noise
        return torch.relu(input)


class LaplaceDensity(Density):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    def __init__(self,
                 params_init: Dict[str, torch.Tensor] = dict(),
                 beta_bounds: Tuple[float, float] = (1e-6, 0.0006),
                 scale_min: float = 1.0,
                 mean_bounds: Tuple[float, float] = (0.5, 1.0)) -> None:
        """
        Laplace density function class.
        :params params_init: The initial parameters of the density function.
        :params beta_min: The minimum value of beta.
        :params scale_min: The minimum value of scale.
        :params mean_bounds: The bounds of the mean.
        """
        super().__init__(params_init=params_init)
        self.beta_bounds = torch.tensor(beta_bounds)
        self.scale_min = torch.tensor(scale_min)
        self.mean_bounds = torch.tensor(mean_bounds)

    def density_func(self,
                     input: torch.Tensor,
                     beta: Optional[float] = None,
                     scale: Optional[float] = None,
                     mean: Optional[float] = None,
                     cutoff: float = -0.5) -> torch.Tensor:
        """
        Forward pass of the density function.
        :params input: The input tensor.
        :params beta: The beta parameter.
        :params scale: The scale parameter.
        :params mean: The mean parameter.
        """
        if beta is None:
            beta = self.get_beta()
        if scale is None:
            scale = self.get_scale()
        if mean is None:
            mean = self.get_mean()

        scaled_cdf = self.laplacian_cdf(input, beta, scale, mean) - \
            self.laplacian_cdf(torch.tensor([cutoff]).to(input.device), beta, scale, mean)
        return nn.ReLU()(scaled_cdf)

    @staticmethod
    def laplacian_cdf(x: torch.Tensor,
                      beta: torch.Tensor,
                      scale: torch.Tensor,
                      mean: torch.Tensor) -> torch.Tensor:
        """
        Laplace CDF.
        :params x: The input tensor.
        :params beta: The beta parameter.
        :params scale: The scale parameter.
        :params mean: The mean parameter.
        :returns: The CDF.
        """
        laplace_cdf = 0.5 + 0.5 * torch.sign(x - mean) * (1 - torch.exp(-torch.abs(x - mean) / beta))
        return scale * laplace_cdf

    def get_beta(self) -> torch.Tensor:
        """
        Beta parameter getter.
        :returns: The beta parameter.
        """
        beta = torch.clamp(self.beta,
                           self.beta_bounds[0].to(self.mean.device),
                           self.beta_bounds[1].to(self.mean.device))
        return beta

    def set_beta(self, beta: torch.Tensor) -> None:
        """
        Beta parameter setter.
        :params beta: The beta parameter.
        """
        self.beta.data = beta

    def get_scale(self) -> torch.Tensor:
        """
        Scale parameter getter.
        :returns: The scale parameter.
        """
        if hasattr(self, 'scale'):
            scale = self.scale.abs()
            return torch.max(scale, self.scale_min.to(self.scale.device))
        else:
            return 1 / self.get_beta()

    def get_mean(self) -> torch.Tensor:
        """
        Mean parameter getter.
        :returns: The mean parameter.
        """
        return torch.clamp(self.mean,
                           self.mean_bounds[0].to(self.mean.device),
                           self.mean_bounds[1].to(self.mean.device))


class ExponentialDensity(Density):
    def __init__(self,
                 params_init: Dict[str, torch.Tensor] = dict(),
                 beta_min: float = 1e-4) -> None:
        """
        Exponential density function class.
        :params params_init: The initial parameters of the density function.
        :params beta_min: The minimum value of beta.
        """
        super().__init__(params_init=params_init)
        self.beta_min = torch.tensor(beta_min)

    def density_func(self,
                     input: torch.Tensor,
                     beta: Optional[float] = None,
                     scale: Optional[float] = None,
                     mean: Optional[float] = None) -> torch.Tensor:
        """
        Forward pass of the density function.
        :params input: The input tensor.
        :params beta: The beta parameter.
        :params scale: The scale parameter.
        :params mean: The mean parameter.
        :returns: The density.
        """
        if beta is None:
            beta = self.get_beta()

        return 1 / beta * (1 - torch.exp(-beta * input))

    def get_beta(self) -> torch.Tensor:
        """
        Beta parameter getter.
        :returns: The beta parameter.
        """
        beta = self.beta.abs() + self.beta_min
        return beta


class SigmoidDensity(Density):
    def __init__(self,
                 params_init: Dict[str, torch.Tensor] = dict(),
                 beta_min: float = 1e-4,
                 scale_min: float = 1.0,
                 ) -> None:
        """
        Sigmoid density function class.
        :params params_init: The initial parameters of the density function.
        :params beta_min: The minimum value of beta.
        :params scale_min: The minimum value of scale.
        """
        super().__init__(params_init=params_init)
        self.beta_min = torch.tensor(beta_min)
        self.scale_min = torch.tensor(scale_min)

    def density_func(self,
                     input: torch.Tensor,
                     beta: Optional[float] = None,
                     scale: Optional[float] = None,
                     mean: Optional[float] = None) -> torch.Tensor:
        """
        Forward pass of the density function.
        :params input: The input tensor.
        :params beta: The beta parameter.
        :params scale: The scale parameter.
        :params mean: The mean parameter.
        :returns: The density.
        """
        if beta is None:
            beta = self.get_beta()

        if scale is None:
            scale = self.get_scale()
        x = -input - 0.5

        return 1.0 / (1.0 + torch.exp(-beta * x)) * scale

    def get_beta(self) -> torch.Tensor:
        """
        Beta parameter getter.
        :returns: The beta parameter.
        """
        beta = torch.max(self.beta.abs(), self.beta_min)
        return beta

    def get_scale(self) -> torch.Tensor:
        """
        Scale parameter getter.
        :returns: The scale parameter.
        """
        scale = self.scale.abs()
        return torch.max(scale, self.scale_min)


class LaplaceDensitySdf(Density):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    def __init__(self, params_init={}, beta_min=0.0001):
        super().__init__(params_init=params_init)
        self.beta_min = torch.tensor(beta_min)

    def density_func(self,
                     input: torch.Tensor,
                     beta: Optional[float] = None,
                     scale: Optional[float] = None,
                     mean: Optional[float] = None):
        if beta is None:
            beta = self.get_beta()

        alpha = 1 / beta
        return alpha * (0.5 + 0.5 * input.sign() * torch.expm1(-input.abs() / beta))

    def get_beta(self):
        beta = self.beta.abs() + self.beta_min
        return beta
