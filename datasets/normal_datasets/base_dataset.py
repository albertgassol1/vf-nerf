from typing import Tuple

import torch


class BaseDataset(torch.utils.data.Dataset):

    def __init__(self,
                 n_images: int,
                 shuffle_views: bool,
                 pixels_per_batch: int,
                 all_pixels: bool = False,
                 white_bkgd: bool = False) -> None:
        """
        Base dataset.
        :param n_images: The number of images.
        :param shuffle_views: Whether to shuffle the views.
        :param pixels_per_batch: The number of pixels per batch.
        :param all_pixels: Whether to use all pixels.
        :param white_bkgd: Whether to use a white background.
        """

        self.n_images = n_images

        # Shuffle the views.
        self._shuffle_views = shuffle_views

        # Save the pixels per batch.
        self._pixels_per_batch = pixels_per_batch
        if self.shuffle_views:
            self._pixels_per_batch = self.pixels_per_batch // self.n_images
        self.total_pixels = self.n_images * self.pixels_per_batch

        # Save wheather to use all pixels.
        self._all_pixels = all_pixels

        self.white_bkgd = False

        # Save the epoch.
        self.epoch = 0

        # Set scale
        self.scale = 3.5

    @property
    def shuffle_views(self) -> bool:
        """
        Whether to shuffle the views.
        :return: Whether to shuffle the views.
        """
        return self._shuffle_views

    @shuffle_views.setter
    def shuffle_views(self, shuffle_views: bool) -> None:
        """
        Set whether to shuffle the views.
        :param shuffle_views: Whether to shuffle the views.
        """
        self._shuffle_views = shuffle_views

    @property
    def pixels_per_batch(self) -> int:
        """
        The number of pixels per batch.
        :return: The number of pixels per batch.
        """
        return self._pixels_per_batch

    @pixels_per_batch.setter
    def pixels_per_batch(self, pixels_per_batch: int) -> None:
        """
        Set the number of pixels per batch.
        :param pixels_per_batch: The number of pixels per batch.
        """
        self._pixels_per_batch = pixels_per_batch
        if self.shuffle_views:
            self._pixels_per_batch = self.pixels_per_batch // self.n_images

    @property
    def all_pixels(self) -> bool:
        """
        Whether to use all pixels.
        :return: Whether to use all pixels.
        """
        return self._all_pixels

    @all_pixels.setter
    def all_pixels(self, all_pixels: bool) -> None:
        """
        Set the all pixels flag.
        :param all_pixels: All pixels flag.
        """
        self._all_pixels = all_pixels

    def __len__(self) -> int:
        """
        Get the length of the dataset.
        :return: The length of the dataset.
        """
        return self.n_images

    def get_bounds(self) -> Tuple[float, float]:
        """
        Get the bounds of the dataset.
        :return: The bounds of the dataset.
        """
        pass

    def get_vf_init_method(self) -> Tuple[str, str]:
        """
        Get the vector field initialization method.
        :return: The vector field initialization method.
        """
        return "center", ""
    
    def sample_new_images(self) -> None:
        """
        Sample new images from the dataset.
        """
        pass

    def get_centroid(self, device: torch.device) -> torch.Tensor:
        """
        Return scene center.
        :return: scene center.
        """
        return torch.zeros(3).to(device)
