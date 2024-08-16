from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def quat_to_rot(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to rotation matrix.
    :param q: The quaternion.
    :return: The rotation matrix.
    """
    batch_size, _ = q.shape
    q = F.normalize(q, dim=1)
    qr = q[:, 0]
    qi = q[:, 1]
    qj = q[:, 2]
    qk = q[:, 3]

    rotation = torch.ones((batch_size, 3, 3)).cuda()
    rotation[:, 0, 0] = 1 - 2 * (qj**2 + qk**2)
    rotation[:, 0, 1] = 2 * (qj * qi - qk * qr)
    rotation[:, 0, 2] = 2 * (qi * qk + qr * qj)
    rotation[:, 1, 0] = 2 * (qj * qi + qk * qr)
    rotation[:, 1, 1] = 1 - 2 * (qi**2 + qk**2)
    rotation[:, 1, 2] = 2 * (qj * qk - qi * qr)
    rotation[:, 2, 0] = 2 * (qk * qi - qj * qr)
    rotation[:, 2, 1] = 2 * (qj * qk + qi * qr)
    rotation[:, 2, 2] = 1 - 2 * (qi**2 + qj**2)

    return rotation


def pixel2camera(u: torch.Tensor,
                 v: torch.Tensor,
                 z: torch.Tensor,
                 intrinsics: torch.Tensor,
                 device: torch.device = torch.device('cuda')) -> torch.Tensor:
    """
    Convert pixel coordinates to camera coordinates.
    :param u: The u coordinates.
    :param v: The v coordinates.
    :param z: The z coordinates.
    :param intrinsics: The intrinsics.
    :param device: The device.
    :return: The camera coordinates.
    """
    # Get the intrinsics.
    fx = intrinsics[:, 0, 0].to(device)
    fy = intrinsics[:, 1, 1].to(device)
    cx = intrinsics[:, 0, 2].to(device)
    cy = intrinsics[:, 1, 2].to(device)
    skew = intrinsics[:, 0, 1].to(device)

    # Pinhole projection.
    x = (u - cx + cy * skew / fy -
         skew * v / fy) / fx * z.abs()
    y = (v - cy) / fy * z.abs()

    # Return homogeneous coordinates.
    return torch.stack([x, y, z, torch.ones_like(z).to(device)], dim=-1)


def load_K_Rt_from_P(filename: str,
                     projection: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the camera matrix and decompose it to K, R and t.
    :param filename: The filename.
    :param projection: The projection matrix.
    :return: The intrinsics and the pose.
    """
    # Load the camera matrix.
    if projection is None:
        with open(filename) as f:
            lines = f.read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        projection = np.asarray(lines).astype(np.float32).squeeze()

    # Decompose the camera matrix.
    decomposition = cv2.decomposeProjectionMatrix(projection)
    intrinsics = decomposition[0]
    rotation = decomposition[1]
    translation = decomposition[2]

    intrinsics = intrinsics / intrinsics[2, 2]
    intrinsics_hom = np.eye(4)
    intrinsics_hom[:3, :3] = intrinsics

    # Convert to homogeneous coordinates.
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = rotation.transpose()
    pose[:3, 3] = (translation[:3] / translation[3])[:, 0]

    return intrinsics_hom, pose


class Projector():
    def __init__(self, device) -> None:
        """
        Projector class.
        :param device: The device.
        """
        self.device = device

    def inbound(self, pixel_locations: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        Check if the pixel locations are in valid range
        :param pixel_locations: [..., 2]
        :param h: height
        :param w: width
        :return: mask, bool, [...]
        """
        return (pixel_locations[..., 0] <= w - 1.) & \
               (pixel_locations[..., 0] >= 0) & \
               (pixel_locations[..., 1] <= h - 1.) &\
               (pixel_locations[..., 1] >= 0)

    def normalize(self, pixel_locations: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        Normalize pixel locations to [-1, 1]
        :param pixel_locations: [..., 2]
        :param h: height
        :param w: width
        :return: normalized pixel locations [..., 2]
        """
        resize_factor = torch.tensor([w-1., h-1.]).to(pixel_locations.device)[None, None, :]
        normalized_pixel_locations = 2 * pixel_locations / resize_factor - 1.  # [n_views, n_points, 2]
        return normalized_pixel_locations

    def compute_projections(self, xyz: torch.Tensor, train_cameras: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project 3D points into cameras
        :param xyz: [..., 3]
        :param train_cameras: [n_views, 34], 34 = img_size(2) + intrinsics(16) + extrinsics(16)
        :return: pixel locations [..., 2], mask [...]
        """
        original_shape = xyz.shape[:2]
        xyz = xyz.reshape(-1, 3)
        num_views = len(train_cameras)
        train_intrinsics = train_cameras[:, 2:18].reshape(-1, 4, 4)  # [n_views, 4, 4]
        train_poses = train_cameras[:, -16:].reshape(-1, 4, 4)  # [n_views, 4, 4]
        xyz_h = torch.cat([xyz, torch.ones_like(xyz[..., :1])], dim=-1)  # [n_points, 4]
        projections = train_intrinsics.bmm(torch.inverse(train_poses)) \
            .bmm(xyz_h.t()[None, ...].repeat(num_views, 1, 1))  # [n_views, 4, n_points]
        projections = projections.permute(0, 2, 1)  # [n_views, n_points, 4]
        pixel_locations = projections[..., :2] / torch.clamp(projections[..., 2:3], min=1e-8)  # [n_views, n_points, 2]
        pixel_locations = torch.clamp(pixel_locations, min=-1e6, max=1e6)
        mask = projections[..., 2] > 0   # a point is invalid if behind the camera
        return pixel_locations.reshape((num_views, ) + original_shape + (2, )), \
               mask.reshape((num_views, ) + original_shape)

    def compute_angle(self, xyz: torch.Tensor, query_camera: torch.Tensor,
                      train_cameras: torch.Tensor) -> torch.Tensor:
        """
        Compute the angle between the ray from query camera to xyz and the ray from each train camera to xyz
        :param xyz: [..., 3]
        :param query_camera: [34, ]
        :param train_cameras: [n_views, 34]
        :return: [n_views, ..., 4]; The first 3 channels are unit-length vector of the difference between
        query and target ray directions, the last channel is the inner product of the two directions.
        """
        original_shape = xyz.shape[:2]
        xyz = xyz.reshape(-1, 3)
        train_poses = train_cameras[:, -16:].reshape(-1, 4, 4)  # [n_views, 4, 4]
        num_views = len(train_poses)
        query_pose = query_camera[-16:].reshape(-1, 4, 4).repeat(num_views, 1, 1)  # [n_views, 4, 4]
        ray2tar_pose = (query_pose[:, :3, 3].unsqueeze(1) - xyz.unsqueeze(0))
        ray2tar_pose /= (torch.norm(ray2tar_pose, dim=-1, keepdim=True) + 1e-6)
        ray2train_pose = (train_poses[:, :3, 3].unsqueeze(1) - xyz.unsqueeze(0))
        ray2train_pose /= (torch.norm(ray2train_pose, dim=-1, keepdim=True) + 1e-6)
        ray_diff = ray2tar_pose - ray2train_pose
        ray_diff_norm = torch.norm(ray_diff, dim=-1, keepdim=True)
        ray_diff_dot = torch.sum(ray2tar_pose * ray2train_pose, dim=-1, keepdim=True)
        ray_diff_direction = ray_diff / torch.clamp(ray_diff_norm, min=1e-6)
        ray_diff = torch.cat([ray_diff_direction, ray_diff_dot], dim=-1)
        ray_diff = ray_diff.reshape((num_views, ) + original_shape + (4, ))
        return ray_diff

    def compute(self,  xyz: torch.Tensor, query_camera: torch.Tensor,
                 train_imgs: torch.Tensor, train_cameras: torch.Tensor,
                 featmaps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the rgb and deep feature samples, ray directions and mask
        :param xyz: [n_rays, n_samples, 3]
        :param query_camera: [1, 34], 34 = img_size(2) + intrinsics(16) + extrinsics(16)
        :param train_imgs: [1, n_views, h, w, 3]
        :param train_cameras: [1, n_views, 34]
        :param featmaps: [n_views, d, h, w]
        :return: rgb_feat_sampled: [n_rays, n_samples, 3+n_feat],
                 ray_diff: [n_rays, n_samples, 4],
                 mask: [n_rays, n_samples, 1]
        """
        assert (train_imgs.shape[0] == 1) \
               and (train_cameras.shape[0] == 1) \
               and (query_camera.shape[0] == 1), 'only support batch_size=1 for now'

        train_imgs = train_imgs.squeeze(0)  # [n_views, h, w, 3]
        train_cameras = train_cameras.squeeze(0)  # [n_views, 34]
        query_camera = query_camera.squeeze(0)  # [34, ]

        train_imgs = train_imgs.permute(0, 3, 1, 2)  # [n_views, 3, h, w]

        h, w = train_cameras[0][:2]

        # compute the projection of the query points to each reference image
        pixel_locations, mask_in_front = self.compute_projections(xyz, train_cameras)
        normalized_pixel_locations = self.normalize(pixel_locations, h, w)   # [n_views, n_rays, n_samples, 2]

        # rgb sampling
        rgbs_sampled = F.grid_sample(train_imgs, normalized_pixel_locations, align_corners=True)
        rgb_sampled = rgbs_sampled.permute(2, 3, 0, 1)  # [n_rays, n_samples, n_views, 3]

        # deep feature sampling
        feat_sampled = F.grid_sample(featmaps, normalized_pixel_locations, align_corners=True)
        feat_sampled = feat_sampled.permute(2, 3, 0, 1)  # [n_rays, n_samples, n_views, d]
        rgb_feat_sampled = torch.cat([rgb_sampled, feat_sampled], dim=-1)   # [n_rays, n_samples, n_views, d+3]

        # mask
        inbound = self.inbound(pixel_locations, h, w)
        ray_diff = self.compute_angle(xyz, query_camera, train_cameras)
        ray_diff = ray_diff.permute(1, 2, 0, 3)
        mask = (inbound * mask_in_front).float().permute(1, 2, 0)[..., None]   # [n_rays, n_samples, n_views, 1]
        return rgb_feat_sampled, ray_diff, mask
