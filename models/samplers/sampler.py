from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F


class Sampler(ABC):
    def __init__(self, number_of_samples: int) -> None:
        """
        Sampler class.
        :params number_of_samples: The number of samples to generate.
        """

        # Store the number of samples.
        self._number_of_samples = number_of_samples

    @property
    def number_of_samples(self) -> int:
        """
        Get the number of samples.
        :returns: The number of samples.
        """
        return self._number_of_samples

    @number_of_samples.setter
    def number_of_samples(self, n_samples: int) -> None:
        """
        Set the number of samples.
        :param n_samples: New number of samples.
        """
        self._number_of_samples = n_samples

    @abstractmethod
    def sample(self) -> Union[torch.Tensor, np.ndarray]:
        """
        Sample a set of points.
        :returns: The sampled points as tensor or array.
        """
        pass


class UniformSampler(Sampler):
    def __init__(self,
                 min_bounds: torch.Tensor,
                 max_bounds: torch.Tensor,
                 number_of_samples: int) -> None:
        """
        Uniform sampler class in a 3D rectangular box.
        :params min_bounds: The minimum bounds of the box.
        :params max_bounds: The maximum bounds of the box.
        :params number_of_samples: The number of samples to generate.
        """

        # Initialize the super class.
        super().__init__(number_of_samples)

        # Store the minimum and maximum bounds.
        self._min_bounds = min_bounds
        self._max_bounds = max_bounds

    @property
    def max_bounds(self) -> torch.Tensor:
        """
        The maximum bounds.
        :returns: The maximum bounds.
        """
        return self._max_bounds

    @property
    def min_bounds(self) -> torch.Tensor:
        """
        The minimum bounds.
        :returns: The minimum bounds.
        """
        return self._min_bounds

    def sample(self) -> Union[torch.Tensor, np.ndarray]:
        """
        Sample a set of points.
        :params device: The device to use.
        :returns: The sampled points.
        """

        # Sample the points.
        points = torch.rand((self.number_of_samples, 3), device=self._max_bounds.device)
        points = points * (self._max_bounds - self._min_bounds) + self._min_bounds

        return points


class UnitVectorSampler(Sampler):
    def __init__(self,
                 number_of_samples: int,
                 device: torch.device) -> None:
        """
        Unit vector sampler class.
        :params number_of_samples: The number of samples to generate.
        :params device: The device to use.
        """

        # Initialize the super class.
        super().__init__(number_of_samples)

        # Store the device.
        self.device = device

    def sample(self) -> Union[torch.Tensor, np.ndarray]:
        """
        Sample a set of unit vectors.
        :returns: The sampled points.
        """

        # Sample the unit vectors.
        vectors = torch.rand((self.number_of_samples, 3), device=self.device)

        # Normalize the vectors.
        unit_vectors = F.normalize(vectors, p=2, dim=1)

        return unit_vectors


class UniformSphereSampler(Sampler):
    def __init__(self,
                 number_of_samples: int) -> None:
        """
        Uniform sphere sampler class.
        :params number_of_samples: The number of samples to generate.
        """

        # Initialize the super class.
        super().__init__(number_of_samples)

    def sample(self) -> Union[torch.Tensor, np.ndarray]:
        """
        Sample a uniform set of points on a sphere.
        :returns: The sampled points.
        """

        # Sample phi angle
        phi = np.random.uniform(0.0, 2.0 * np.pi, self.number_of_samples)
        # Sample cosine of theta angle
        cos_theta = np.random.uniform(-1.0, 1.0, self.number_of_samples)
        # Compute theta angle
        theta = np.arccos(cos_theta)
        # Sample radius. Sample from a cube and take the cube root.
        u = np.random.uniform(0.0, 1.0, self.number_of_samples)
        r = np.cbrt(u)

        # Compute the points.
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        # Stack the points and return.
        return np.stack((x, y, z), axis=1)


class SphereSampler(Sampler):
    def __init__(self,
                 number_of_samples: int) -> None:
        """
        Uniform sphere sampler class.
        :params number_of_samples: The number of samples to generate.
        """

        # Initialize the super class.
        super().__init__(number_of_samples)

    def sample(self, r_max: float, r_min: float) -> Union[torch.Tensor, np.ndarray]:
        """
        Sample a set of points in a sphere of radius between r_min and r_max.
        :returns: The sampled points.
        """

        # Sample phi angle
        phi = np.random.uniform(0.0, 2.0 * np.pi, self.number_of_samples)
        # Sample cosine of theta angle
        cos_theta = np.random.uniform(-1.0, 1.0, self.number_of_samples)
        # Compute theta angle
        theta = np.arccos(cos_theta)
        # Sample radius between r_min and r_max.
        u = np.random.uniform(0.0, 1.0, self.number_of_samples)
        r = np.cbrt(u) * (r_max - r_min) + r_min

        # Compute the points.
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        # Stack the points and return.
        return np.stack((x, y, z), axis=1)
