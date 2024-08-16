import glob
import os
import sys
from typing import Dict, List, Tuple

sys.path.append('.')  # isort:skip

import cv2
import numpy as np
import torch
import trimesh

from config_parser.vf_nerf_config import DatasetConfig
from datasets.helpers.dataset_output import DatasetOutput
from datasets.normal_datasets.base_dataset import BaseDataset


class ScanNetDataset(BaseDataset):
    def __init__(self, config: DatasetConfig, factor: int = 40, train: bool = True) -> None:
        """
        Dataset class for the ScanNet dataset.
        :param config: The dataset configuration.
        :param factor: The factor to subsample the images.
        :param train: Whether to use the train or test set.
        """

        # Get the data directory.
        self.data_dir = os.path.join(config.data_root_dir, config.data_dir, f"{config.scene}")
        assert os.path.exists(self.data_dir), f"Data directory {self.data_dir} does not exist."

        factor = factor if train else 1

        # Load the image paths
        image_paths = sorted(
            glob.glob(f'{self.data_dir}/color/*.jpg'))[::factor]
        depth_paths = sorted(
            glob.glob(f'{self.data_dir}/depth/*.png'))[::factor]

        # Initialize the base dataset.
        super().__init__(n_images=len(image_paths),
                         shuffle_views=config.shuffle_views,
                         pixels_per_batch=config.pixels_per_batch,
                         all_pixels=config.all_pixels)

        self.config = config

        # Load the depth maps
        # Load the images
        image_paths = np.asarray(image_paths)
        depth_paths = np.asarray(depth_paths)
        self.rgb_images, self.depth_images = self.load_images(image_paths, depth_paths)

        # Get image size and number of pixels
        H, W = cv2.imread(depth_paths[0], cv2.IMREAD_UNCHANGED).shape
        self.image_size = (H - 2 * self.config.crop_edge, W - 2 * self.config.crop_edge)
        self.n_pixels = self.image_size[0] * self.image_size[1]

        # Load the poses
        self.load_poses(factor)

        # Load the intrinsics from txt file
        self.intrinsics = torch.eye(4).float()
        with open(os.path.join(self.data_dir, "intrinsic/intrinsic_depth.txt")) as f:
            lines = f.readlines()
        self.intrinsics = torch.from_numpy(np.array(list(map(float, (lines[0] + lines[1] + lines[2] + lines[3]).split()))).reshape(4, 4)).float()
        self.intrinsics[0, 2] -= self.config.crop_edge
        self.intrinsics[1, 2] -= self.config.crop_edge

        # Compute max depth
        self.max_depth = torch.max(torch.cat(self.depth_images)).item()

        # Get the ground truth mesh centroid.
        mesh = trimesh.load(os.path.join(self.data_dir, f"{config.scene}_vh_clean.ply"))
        self.gt_mesh_centroid = torch.from_numpy(np.asarray(mesh.centroid)).float()
        self.scale = np.abs(mesh.bounds - self.gt_mesh_centroid.numpy()).max() * 1.1

    def load_images(self, image_paths: np.ndarray, depth_paths: np.ndarray) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Load the images.
        :return: The RGB and depth images.
        """
        rgb_images: List[torch.Tensor] = list()
        depth_images: List[torch.Tensor] = list()
        for i in range(len(image_paths)):
            img = cv2.cvtColor(cv2.imread(image_paths[i]), cv2.COLOR_BGR2RGB) / 255.0
            depth = cv2.imread(depth_paths[i], cv2.IMREAD_UNCHANGED).astype(np.float32) / 1e3
            img = cv2.resize(img, (depth.shape[1], depth.shape[0])).transpose(2, 0, 1)
            if self.config.crop_edge > 0:
                img = img[:, self.config.crop_edge:-self.config.crop_edge, self.config.crop_edge:-self.config.crop_edge]
                depth = depth[self.config.crop_edge:-self.config.crop_edge, self.config.crop_edge:-self.config.crop_edge]
            depth = torch.from_numpy(depth).float().reshape(-1, 1)

            img = img.reshape(3, -1).transpose(1, 0)
            rgb_images.append(torch.from_numpy(img).float())
            depth_images.append(depth)

        return rgb_images, depth_images

    def sample_new_images(self) -> None:
        """
        Sample new images from the dataset.
        """
        if not self.config.random_img_sampling:
            return

        # Sample a random permutation of the images.
        idx = np.random.choice(self.n_images, self.n_images // self.config.factor, replace=False)

        # Load the images
        self.rgb_images, self.depth_images = self.load_images(self.image_paths[idx], self.depth_paths[idx])

        # Load the poses
        self.poses = self.all_poses[idx].clone()

    def load_poses(self, factor: int) -> None:
        """
        Load the poses.
        :param factor: The factor to subsample the poses.
        """
        poses_paths = sorted(
            glob.glob(f'{self.data_dir}/pose/*.txt'))[::factor]
        poses: List[torch.Tensor] = list()
        for pose_path in poses_paths:
            with open(pose_path) as f:
                lines = f.readlines()
            c2w = np.array(list(map(float, (lines[0] + lines[1] + lines[2] + lines[3]).split()))).reshape(4, 4)
            c2w = torch.from_numpy(c2w).float()
            poses.append(c2w)
        self.all_poses = torch.stack(poses)
        self.poses = self.all_poses.clone()

    def __len__(self) -> int:
        """
        Return the length of the dataset.
        :return: The length of the dataset.
        """
        return self.n_images // self.config.factor if self.config.random_img_sampling else self.n_images

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get an item from the dataset.
        :param idx: The camera index to sample from if not in shuffle mode.
        :return: The dataset output.
        """
        if self._all_pixels:
            uv = np.mgrid[0:self.image_size[0], 0:self.image_size[1]].astype(np.int32)
            uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
            uv = uv.reshape(2, -1).transpose(1, 0)
            far = self.depth_images[idx] * 1.25 if self.config.far_per_ray else torch.empty(0)

            return DatasetOutput(rgb=self.rgb_images[idx],
                                 uv=uv,
                                 intrinsics=self.intrinsics.repeat(self.n_pixels, 1, 1),
                                 pose=self.poses[idx].repeat(self.n_pixels, 1, 1),
                                 depth=self.depth_images[idx],
                                 far=far).to_dict()

        if self.shuffle_views:
            all_uv = torch.empty((self.total_pixels, 2), dtype=torch.float32)
            all_rgb = torch.empty((self.total_pixels, 3), dtype=torch.float32)
            all_intrinsics = torch.empty((self.total_pixels, 4, 4), dtype=torch.float32)
            all_pose = torch.empty((self.total_pixels, 4, 4), dtype=torch.float32)
            all_depth = torch.empty((self.total_pixels, 1), dtype=torch.float32)
            uv = np.mgrid[0:self.image_size[0], 0:self.image_size[1]].astype(np.int32)
            uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
            uv = uv.reshape(2, -1).transpose(1, 0)

            n_images = self.n_images if not self.config.random_img_sampling else self.n_images // self.config.factor
            for i in range(n_images):
                # Random shuffle the pixels.
                sampling_idx = torch.randperm(self.n_pixels)[:self.pixels_per_batch]

                # Append the data.
                all_uv[i * self.pixels_per_batch:(i + 1) * self.pixels_per_batch, :] = uv[sampling_idx, :]
                all_rgb[i * self.pixels_per_batch:(i + 1) * self.pixels_per_batch,
                        :] = self.rgb_images[i][sampling_idx, :]
                all_depth[i * self.pixels_per_batch:(i + 1) * self.pixels_per_batch,
                          :] = self.depth_images[i][sampling_idx, :]
                all_intrinsics[i * self.pixels_per_batch:(i + 1) * self.pixels_per_batch,
                               :, :] = self.intrinsics.repeat(self.pixels_per_batch, 1, 1)
                all_pose[i * self.pixels_per_batch:(i + 1) * self.pixels_per_batch,
                         :, :] = self.poses[i].repeat(self.pixels_per_batch, 1, 1)

            # Return the dataset output.
            far = all_depth * 1.25 if self.config.far_per_ray else torch.empty(0)
            return DatasetOutput(rgb=all_rgb,
                                 uv=all_uv,
                                 intrinsics=all_intrinsics,
                                 pose=all_pose,
                                 depth=all_depth,
                                 far=far).to_dict()

        uv = np.mgrid[0:self.image_size[0], 0:self.image_size[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)
        # Random shuffle the pixels.
        sampling_idx = torch.randperm(self.n_pixels)[:self.pixels_per_batch]

        far = self.depth_images[idx][sampling_idx, :] * 1.25 if self.config.far_per_ray else torch.empty(0)
        return DatasetOutput(rgb=self.rgb_images[idx][sampling_idx, :],
                             uv=uv[sampling_idx, :],
                             intrinsics=self.intrinsics.repeat(self.pixels_per_batch, 1, 1),
                             pose=self.poses[idx].repeat(self.pixels_per_batch, 1, 1),
                             depth=self.depth_images[idx][sampling_idx, :],
                             far=far).to_dict()

    def get_bounds(self) -> Tuple[float, float]:
        """
        Get the bounds of the dataset.
        :return: The bounds of the dataset.
        """
        return 0.0, self.max_depth * 1.25

    def get_vf_init_method(self) -> Tuple[str, str]:
        """
        Get the vector field initialization method.
        :return: The vector field initialization method.
        """
        return f"exterior_{self.config.scene}", os.path.join(self.data_dir, f"{self.config.scene}.pth")

    def get_centroid(self, device: torch.device) -> torch.Tensor:
        """
        Return scene center.
        :return: scene center.
        """
        return self.gt_mesh_centroid.to(device)
