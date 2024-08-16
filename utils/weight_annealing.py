from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class WeightAnnealing(ABC):
    def __init__(self,
                 number_of_weights: int,
                 number_of_epochs) -> None:
        """
        Weight annealing.
        :params number_of_weights: The number of weights.
        :params number_of_epochs: The number of epochs.
        """
        self.number_of_weights = number_of_weights
        self.number_of_epochs = number_of_epochs

    @abstractmethod
    def get_weights(self,
                    epoch: int,
                    device: torch.device = 'cpu') -> torch.Tensor:
        """
        Get the weights for the given epoch.
        :params epoch: The epoch.
        :params device: The device.
        :return: The weights.
        """
        pass


class LinearAnnealing(WeightAnnealing):
    def __init__(self,
                 number_of_weights: int,
                 number_of_epochs: int,
                 soft_anneal: bool = False) -> None:
        """
        Linear weight annealing.
        :params number_of_weights: The number of weights.
        :params number_of_epochs: The number of epochs.
        :params soft_anneal: Whether to use soft annealing.
        """
        super().__init__(number_of_weights, number_of_epochs)

        self.weight_indices = torch.arange(self.number_of_weights) - \
            int((self.number_of_weights - 1) / 2)
        self.relu = nn.ReLU()
        self.mid_value = (self.number_of_weights - 1) / 2
        self.soft_anneal = soft_anneal

    def get_weights(self,
                    epoch: int,
                    device: torch.device = 'cpu') -> torch.Tensor:
        """
        Get the weights for the given epoch.
        :params epoch: The epoch.
        :params device: The device.
        :return: The weights.
        """
        if epoch < 0:
            weights = torch.ones(self.number_of_weights).to(device).float() / self.number_of_weights
            return weights

        self.weight_indices = self.weight_indices.to(device).float()

        linear_weights = -self.mid_value / self.number_of_epochs * epoch * self.weight_indices.abs() + self.mid_value
        relu_weights: torch.Tensor = self.relu(linear_weights)
        weights = relu_weights / relu_weights.sum()

        if weights[int(self.mid_value)] >= 0.8 and self.soft_anneal:
            weights[int(self.mid_value) - 2:int(self.mid_value) + 3] = 0.05
            weights[int(self.mid_value)] = 0.8

        return weights
