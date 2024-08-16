from abc import ABC, abstractmethod

import torch


class ParameterAnnealing(ABC):
    def __init__(self, number_of_epochs: int) -> None:
        """
        Parameter annealing.
        :params number_of_epochs: The number of epochs.
        """
        self.number_of_epochs = number_of_epochs
        self.initial_parameter = torch.tensor(0.0)

    @abstractmethod
    def get_parameter(self, epoch: int, device: torch.device = 'cuda') -> torch.Tensor:
        """
        Get the parameter for the given epoch.
        :params epoch: The epoch.
        :params device: The device.
        :return: The parameter.
        """
        pass

    def set_initial_parameter(self, initial_parameter: torch.Tensor) -> None:
        """
        Set the initial parameter.
        :params initial_parameter: The initial parameter.
        """
        self.initial_parameter = initial_parameter


class ParameterLinearAnnealing(ParameterAnnealing):
    def __init__(self, number_of_epochs: int, final_value: float) -> None:
        """
        Linear parameter annealing.
        :params number_of_epochs: The number of epochs.
        :params final_value: The final value.
        """
        super().__init__(number_of_epochs)
        self.final_value = torch.tensor(final_value)

    def get_parameter(self, epoch: int, device: torch.device = 'cuda') -> torch.Tensor:
        """
        Get the parameter for the given epoch. Apply linear annealing.
        :params epoch: The epoch.
        :params device: The device.
        :return: The parameter.
        """
        if epoch < 0:
            return self.initial_parameter.to(device)
        if epoch >= self.number_of_epochs:
            return self.final_value.to(device)

        new_parameter = self.initial_parameter + \
            (self.final_value - self.initial_parameter) * (epoch / self.number_of_epochs)
        return new_parameter.to(device)
