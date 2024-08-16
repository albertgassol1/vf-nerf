import sys
from typing import Optional, Tuple

sys.path.append('.')  # isort:skip

import torch
import torch.nn.functional as F

import utils.pinhole_model as pinhole_model


def get_ray_directions_and_cam_location(uv: torch.Tensor,
                                        pose: torch.Tensor,
                                        intrinsics: torch.Tensor,
                                        device: torch.device = torch.device('cuda')) -> Tuple[torch.Tensor,
                                                                                              torch.Tensor,
                                                                                              torch.Tensor]:
    """
    Compute the ray directions and the camera locations.
    :param uv: The uv coordinates.
    :param pose: The camera pose in quatertion or euler angle representation.
    :param intrinsics: The intrinsics.
    :param device: The device.
    :return: The ray directions and the camera locations.
    """
    # Get the camera location and rotation.
    if pose.shape[1] == 7:
        # In case of quaternion vector representation
        cam_loc = pose[:, 4:]
        R = pinhole_model.quat_to_rot(pose[:, :4])
        p = torch.eye(4).repeat(pose.shape[0], 1, 1).to(device).float()
        p[:, :3, :3] = R
        p[:, :3, 3] = cam_loc
    else:
        # In case of pose matrix representation
        cam_loc = pose[:, :3, 3]
        p = pose

    num_samples, _ = uv.shape

    # Reshape the pixel coordinates and the depth. Set the depth to 1.
    depth = torch.ones(num_samples).to(device) * torch.sign(intrinsics[0, 1, 1])
    x_cam = uv[:, 0].view(-1)
    y_cam = uv[:, 1].view(-1)
    z_cam = depth.view(-1)

    # Convert the pixel coordinates to camera coordinates.
    pixel_points_cam = pinhole_model.pixel2camera(x_cam, y_cam, z_cam, intrinsics=intrinsics, device=device)

    # Permute for batch matrix product
    pixel_points_cam = pixel_points_cam.unsqueeze(-1)

    # Convert the camera coordinates to world coordinates.
    world_coords = torch.bmm(p, pixel_points_cam).permute(0, 2, 1)[:, :, :3]

    # Compute the ray directions.
    directions = world_coords - cam_loc[:, None, :]
    ray_dirs = F.normalize(directions, dim=2)

    return directions, ray_dirs, cam_loc


def convert_to_ndc(origins: torch.Tensor,
                   directions: torch.Tensor,
                   intrinsics: torch.Tensor,
                   near: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a set of rays to NDC coordinates.
    :param origins: The ray origins.
    :param directions: The ray directions.
    :param intrinsics: The intrinsics.
    :param near: The near plane.
    """
    # Get the focal length.
    focal = intrinsics[0, 0, 0]
    # Get the width and height.
    w = (intrinsics[0, 0, 2] + 0.5) * 2
    h = (intrinsics[0, 1, 2] + 0.5) * 2

    # Shift ray origins to near plane
    t = -(near + origins[..., 2]) / directions[..., 2]
    origins = origins + t[..., None] * directions

    # Projection
    o0 = -1. / (w / (2. * focal)) * origins[..., 0] / origins[..., 2]
    o1 = -1. / (h / (2. * focal)) * origins[..., 1] / origins[..., 2]
    o2 = 1. + 2. * near / origins[..., 2]

    d0 = -1. / (w / (2. * focal)) * (directions[..., 0] / directions[..., 2] - origins[..., 0] / origins[..., 2])
    d1 = -1. / (h / (2. * focal)) * (directions[..., 1] / directions[..., 2] - origins[..., 1] / origins[..., 2])
    d2 = -2. * near / origins[..., 2]

    origins = torch.stack([o0, o1, o2], -1)
    directions = torch.stack([d0, d1, d2], -1)

    return origins, directions

def nerf_volume_rendering(sigma: torch.Tensor, z_vals: torch.Tensor, normalize: bool = False) -> torch.Tensor:
    """
    Render weights for volume rendering.
    :params sigma: The sigma.
    :params z_vals: The z values.
    :param normalize: Whether to normalize the weights.
    :returns: The weights.
    """

    # Compute distances between the samples.
    dists = z_vals[:, 1:] - z_vals[:, :-1]
    dists = torch.cat([dists, torch.tensor([1e10]).to(z_vals.device).repeat(z_vals.shape[0], 1)], dim=-1)

    # LOG SPACE
    free_energy = dists * sigma.squeeze()

    alpha = 1.0 - torch.exp(-free_energy)  # probability of it is not empty here
    weights = alpha * torch.cumprod(1. - alpha + 1e-10, dim=-1, dtype=torch.float32)

    if normalize:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-5)
    return weights


def volsdf_volume_rendering(z_vals: torch.Tensor,
                            density: torch.Tensor,
                            normalize: bool = True) -> torch.Tensor:
    """
    Compute the weights for volume rendering.
    :params z_vals: The z values.
    :params density: The density.
    :params ray_dirs: The ray directions.
    :params vector_field: The vector field.
    :returns: The weights.
    """

    # Compute distances between the samples.
    dists = z_vals[:, 1:] - z_vals[:, :-1]
    dists = torch.cat([dists, torch.tensor([1e10]).to(z_vals.device).repeat(z_vals.shape[0], 1)], dim=-1)

    # LOG SPACE
    free_energy = dists * density
    shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1).to(dists.device),
                                     free_energy[:, :-1]], dim=-1)
    transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))
    alpha = 1.0 - torch.exp(-free_energy)  # probability of it is not empty here
    weights = alpha * transmittance

    if normalize:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-5)
    return weights


def get_rgb_and_depth(weights: torch.Tensor,
                      rgb: torch.Tensor,
                      z_vals: torch.Tensor,
                      white_back: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the rgb and depth.
    :param weights: The weights.
    :param rgb: The rgb.
    :param z_vals: The z values.
    :param white_back: Whether to use a white background.
    :return: The rgb and depth.
    """
    rgb_map = torch.sum(weights.unsqueeze(2) * rgb, dim=1)  # [N_rays, 3]
    if white_back:
        rgb_map = rgb_map + (1. - torch.sum(weights, dim=-1, keepdim=True))
    depth_map = torch.sum(weights * z_vals, dim=-1)

    return rgb_map, depth_map
