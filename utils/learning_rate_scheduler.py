from abc import ABC, abstractmethod
from typing import Dict, Union

import torch


class LearningRateSchedule(ABC):

    @abstractmethod
    def get_learning_rate(self, epoch: int) -> float:
        """
        Get the learning rate for the given epoch.
        :params epoch: The epoch.
        :return: The learning rate.
        """
        pass

    @abstractmethod
    def load_state_dict(self, state_dict: Dict[str, Union[float, int]]) -> None:
        """
        Load the state dict.
        :params state_dict: The state dict.
        """
        pass

    def adjust_learning_rate(self,
                             optimizer: torch.optim.Optimizer,
                             epoch: int) -> float:
        """
        Adjust the learning rate.
        :params optimizer: The optimizer.
        :params epoch: The epoch.
        :return: The learning rate.
        """
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.get_learning_rate(epoch)

        return self.get_learning_rate(epoch)


class ConstantLearningRateSchedule(LearningRateSchedule):

    def __init__(self, learning_rate: float) -> None:
        """
        Constant learning rate schedule.
        :params learning_rate: The learning rate.
        """
        self.learning_rate = learning_rate

    def load_state_dict(self, state_dict: Dict[str, Union[float, int]]) -> None:
        """
        Load the state dict.
        :params state_dict: The state dict.
        """
        self.learning_rate = state_dict['learning_rate']

    def get_learning_rate(self, epoch: int) -> float:
        """
        Get the learning rate for the given epoch.
        :params epoch: The epoch.
        :return: The learning rate.
        """
        return self.learning_rate


class StepLearningRateSchedule(LearningRateSchedule):

    def __init__(self, learning_rate: float, frequency: int, decay_rate: float) -> None:
        """
        Step learning rate schedule.
        :params learning_rate: The learning rate.
        :params frequency: The frequency.
        :params decay_rate: The decay rate.
        """
        self.learning_rate = learning_rate
        self.frequency = frequency
        self.decay_rate = decay_rate

    def load_state_dict(self, state_dict: Dict[str, Union[float, int]]) -> None:
        """
        Load the state dict.
        :params state_dict: The state dict.
        """
        self.learning_rate = state_dict['learning_rate']
        self.frequency = state_dict['frequency']
        self.decay_rate = state_dict['decay_rate']

    def get_learning_rate(self, epoch: int) -> float:
        """
        Get the learning rate for the given epoch.
        :params epoch: The epoch.
        :return: The learning rate.
        """
        return self.learning_rate * (self.decay_rate ** (epoch // self.frequency))


class ExponentialRateSchedule(LearningRateSchedule):

    def __init__(self, learning_rate: float, decay_rate: float) -> None:
        """
        Exponential learning rate schedule.
        :params learning_rate: The learning rate.
        :params gamma: The gamma.
        """
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate

    def load_state_dict(self, state_dict: Dict[str, Union[float, int]]) -> None:
        """
        Load the state dict.
        :params state_dict: The state dict.
        """
        self.learning_rate = state_dict['learning_rate']
        self.decay_rate = state_dict['decay_rate']

    def get_learning_rate(self, epoch: int) -> float:
        """
        Get the learning rate for the given epoch.
        :params epoch: The epoch.
        :return: The learning rate.
        """
        return self.learning_rate * (self.decay_rate ** epoch)
