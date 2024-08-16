import abc
import sys
from typing import Tuple

import numpy as np
import torch

sys.path.append('.')  # isort:skip


class RaySampler(metaclass=abc.ABCMeta):
    def __init__(self,
                 near: float,
                 far: float,
                 N_samples: int):
        """
        Ray sampler class.
        :param near: The nearest point.
        :param far: The farthest point.
        :param N_samples: The number of samples.
        """
        self.near = near
        self.far = far
        self._N_samples = N_samples

    @property
    def N_samples(self) -> int:
        """
        Get number of samples per ray.
        :return: number of samples per ray
        """
        return self._N_samples
    
    @N_samples.setter
    def N_samples(self, N_samples: int) -> None:
        """
        Set number of samples per ray.
        :param N_samples: number of samples per ray
        """
        self._N_samples = N_samples

    def active_sampler(self) -> bool:
        """
        Return whether the sampler is active.
        :return: Whether the sampler is active.
        """
        return self.N_samples > 0

    def sample(self,
               ray_dirs: torch.Tensor,
               cam_loc: torch.Tensor,
               additional_depths: torch.Tensor = torch.empty(0),
               device: torch.device = torch.device('cuda'),
               coarse_z_vals: torch.Tensor = torch.empty(0),
               coarse_weights: torch.Tensor = torch.empty(0)) -> Tuple[torch.Tensor,
                                                                       torch.Tensor]:
        """
        Sample the points, neighbors and z values.
        :param ray_dirs: The ray directions.
        :param cam_loc: The camera location.
        :param additional_depths: The additional depths.
        :param device: The device.
        :param coarse_z_vals: The coarse z values.
        :param coarse_weights: The coarse weights.
        :return: The points and z values.
        """
        # Get the z values.
        z_vals = self.get_z_vals(ray_dirs, cam_loc, device=device,
                                 coarse_weights=coarse_weights, coarse_z_vals=coarse_z_vals)
        if additional_depths.shape[0] > 0:
            z_vals = z_vals.cpu()
            z_vals = torch.cat((z_vals, additional_depths), dim=1)
            z_vals, _ = z_vals.sort(dim=1)
            z_vals = z_vals.to(device)

        # Compute the points.
        points = (cam_loc.unsqueeze(1) +
                  z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1))

        return points, z_vals

    @abc.abstractmethod
    def get_z_vals(self,
                   ray_dirs: torch.Tensor,
                   cam_loc: torch.Tensor,
                   device: torch.device = torch.device('cuda'),
                   coarse_z_vals: torch.Tensor = torch.empty(0),
                   coarse_weights: torch.Tensor = torch.empty(0)) -> torch.Tensor:
        """
        Get the z values.
        """
        pass


class UniformSampler(RaySampler):
    def __init__(self,
                 N_samples: int,
                 near: float,
                 far: float,
                 deterministic: bool = False):
        """
        Uniform sampler class.
        :param near: The nearest point.
        :param N_samples: The number of samples.
        :param far: The farthest point.
        :param deterministic: Whether to use deterministic sampling.
        """
        super().__init__(near,
                         far,
                         N_samples)  # default far is 2*R
        self.deterministic = deterministic

    def get_z_vals(self,
                   ray_dirs: torch.Tensor,
                   cam_loc: torch.Tensor,
                   device: torch.device = torch.device('cuda'),
                   coarse_z_vals: torch.Tensor = torch.empty(0),
                   coarse_weights: torch.Tensor = torch.empty(0)) -> torch.Tensor:
        """
        Uniformly sample the z values.
        :param ray_dirs: The ray directions.
        :param cam_loc: The camera location.
        :param device: The device.
        """
        near = self.near * torch.ones(ray_dirs.shape[0], 1).to(device)
        far = self.far * torch.ones(ray_dirs.shape[0], 1).to(device) if isinstance(self.far, float) \
            else self.far.to(device)

        t_vals = torch.linspace(0., 1., steps=self.N_samples).to(device)
        z_vals = near * (1. - t_vals) + far * (t_vals)

        if not self.deterministic:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(device)

            z_vals = lower + (upper - lower) * t_rand

        return z_vals


class FineSampler(RaySampler):
    def __init__(self,
                 N_samples: int,
                 deterministic: bool = False,
                 pytest: bool = False) -> None:
        """
        Fine sampler that uses the samples and weights from the coarse sampler
        to do imverse sampling.
        :param N_samples: The number of samples.
        :param deterministic: Whether to use deterministic sampling.
        :param pytest: Whether to use pytest.
        """

        super().__init__(near=-1.0, far=-1.0, N_samples=N_samples)
        self.deterministic = deterministic
        self.pytest = pytest

    # Taken from https://github.com/yenchenlin/nerf-pytorch/blob/master/run_nerf_helpers.py#L196
    def sample_pdf(self,
                   bins: torch.Tensor,
                   weights: torch.Tensor) -> torch.Tensor:
        """
        Get and sample the pdf.
        :param bins: The bins.
        :param weights: The weights.
        :return: The sampled pdf.
        """
        # Get pdf
        weights = weights + 1e-5  # prevent nans
        pdf = weights / torch.sum(weights, -1, keepdim=True)
        cdf = torch.cumsum(pdf, -1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

        # Take uniform samples
        if self.deterministic:
            u = torch.linspace(0., 1., steps=self.N_samples)
            u = u.expand(list(cdf.shape[:-1]) + [self.N_samples]).to(bins.device)
        else:
            u = torch.rand(list(cdf.shape[:-1]) + [self.N_samples]).to(bins.device)

        # Pytest, overwrite u with numpy's fixed random numbers
        if self.pytest:
            np.random.seed(0)
            new_shape = list(cdf.shape[:-1]) + [self.N_samples]
            if self.deterministic:
                u = np.linspace(0., 1., self.N_samples)
                u = np.broadcast_to(u, new_shape)
            else:
                u = np.random.rand(*new_shape)
            u = torch.tensor(u)

        # Invert CDF
        u = u.contiguous()
        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.max(torch.zeros_like(inds - 1), inds - 1)
        above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
        inds_g = torch.stack([below, above], dim=-1)  # (batch, N_samples, 2)

        # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
        # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
        matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
        bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

        denom = (cdf_g[..., 1] - cdf_g[..., 0])
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

        return samples.detach()

    def get_z_vals(self,
                   ray_dirs: torch.Tensor,
                   cam_loc: torch.Tensor,
                   device: torch.device = torch.device('cuda'),
                   coarse_z_vals: torch.Tensor = torch.empty(0),
                   coarse_weights: torch.Tensor = torch.empty(0)) -> torch.Tensor:
        """
        Perform inverse sampling to get the fine z values.
        :param coarse_z_vals: The coarse z values.
        :param coarse_weights: The coarse weights.
        :return: The fine z values.
        """

        # Get the midpoints.
        midpoints = .5 * (coarse_z_vals[..., 1:] + coarse_z_vals[..., :-1])
        # Sample the pdf.
        z_vals = self.sample_pdf(midpoints, coarse_weights[..., 1:-1])

        # Sort the z values.
        z_vals, _ = torch.sort(torch.cat([coarse_z_vals, z_vals], dim=-1), dim=-1)

        return z_vals


class RangeFineSampler(RaySampler):
    def __init__(self,
                 N_samples: int,
                 near: float,
                 far: float,
                 deterministic: bool = False,
                 range: float = 0.5,
                 max_samples: int = 100,
                 pytest: bool = False) -> None:
        """
        Fine sampler that uses the samples and weights from the coarse sampler
        to do imverse sampling.
        :param N_samples: The number of samples.
        :param deterministic: Whether to use deterministic sampling.
        :param range: The range of the z values.
        :param pytest: Whether to use pytest.
        """

        super().__init__(near=near, far=far, N_samples=N_samples)
        self.deterministic = deterministic
        self.pytest = pytest
        self.range = range
        self.max_samples = max_samples

    def get_z_vals(self,
                   ray_dirs: torch.Tensor,
                   cam_loc: torch.Tensor,
                   device: torch.device = torch.device('cuda'),
                   coarse_z_vals: torch.Tensor = torch.empty(0),
                   coarse_weights: torch.Tensor = torch.empty(0)) -> torch.Tensor:
        """
        Perform inverse sampling to get the fine z values.
        :param coarse_z_vals: The coarse z values.
        :param coarse_weights: The coarse weights.
        :return: The fine z values.
        """
        N_samples = min(self.max_samples, self.N_samples)
        # Get argmax of weights and corresponding z values
        max_indices = torch.argmax(coarse_weights, dim=-1)
        max_z_vals = coarse_z_vals[torch.arange(coarse_z_vals.shape[0]), max_indices]

        # Sample uniformnly N_samples z values around the max_z_vals within the range
        # Assuming max_z_vals and coarse_z_vals are tensors
        # Sample uniformly N_samples z values around the max_z_vals within the range
        z_vals = max_z_vals[:, None] - self.range + 2*self.range / (N_samples - 1) * torch.arange(N_samples).to(device)

        if not self.deterministic:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(device)

            z_vals = lower + (upper - lower) * t_rand
        
        # Add N_samples random z values to the coarse z values
        z_add = torch.rand((z_vals.shape[0], N_samples)).to(device) * (self.far - self.near) + self.near
        z_vals_ret = torch.sort(torch.cat([coarse_z_vals, z_add], dim=-1), dim=-1)[0]

        z_vals_ret[max_indices > 0], _ = torch.sort(torch.cat([coarse_z_vals[max_indices > 0], z_vals[max_indices > 0]], dim=-1), dim=-1)

        return z_vals_ret