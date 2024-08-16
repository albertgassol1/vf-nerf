import numpy as np
import torch
import torch.nn.functional as F
import trimesh


def recenter_poses(poses: np.ndarray) -> np.ndarray:
    """
    Recenter poses according to the original NeRF code.
    :param poses: Poses to recenter.
    :return: Recentered poses.
    """
    poses_ = poses.copy()
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)
    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses


def poses_avg(poses: np.ndarray) -> np.ndarray:
    """
    Average poses according to the original NeRF code.
    :param poses: Poses to average.
    :return: Averaged poses.
    """
    hwf = poses[0, :3, -1:]
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([look_at(vec2, up, center), hwf], 1)
    return c2w


def normalize(x: np.ndarray) -> np.ndarray:
    """
    Normalization helper function.
    :param x: Input array.
    :return: Normalized array.
    """
    return x / np.linalg.norm(x)


def look_at(z: np.ndarray, up: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """
    Construct look at view matrix
    https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/lookat-function
    :param z: z vector.
    :param up: up vector.
    :param pos: position vector.
    :return: View matrix.
    """
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def to_float(img: np.ndarray) -> np.ndarray:
    """
    Convert an image to float.
    :param img: Input image.
    :return: Float image.
    """
    if len(img.shape) >= 3:
        return np.array([to_float(i) for i in img])
    else:
        return (img / 255.).astype(np.float32)


def sample_poses_z(sphere_radious: float, num_poses: int) -> torch.Tensor:
    """
    Sample poses from the sphere of radius sphere_radious. The poses contain
    orientation and position. The poses z orientation points towards the center.
    :param sphere_radious: The sphere radius.
    :param num_poses: The number of poses.
    :return: The poses.
    """

    # Get a sphere mesh.
    sphere_mesh = trimesh.primitives.Sphere(radius=sphere_radious)
    # Transform to a mesh
    sphere_mesh = trimesh.Trimesh(vertices=sphere_mesh.vertices, faces=sphere_mesh.faces)

    # Sample points from the mesh evenly
    surface_points, _ = trimesh.sample.sample_surface_even(sphere_mesh, num_poses)
    surface_points = torch.from_numpy(surface_points).float()

    # Create the poses
    poses = torch.eye(4).float().unsqueeze(0).repeat(num_poses, 1, 1)

    # Set the positions
    poses[:, :3, 3] = surface_points

    # Set the z orientation
    poses[:, :3, 2] = -F.normalize(surface_points, dim=1)

    # Define the up vector
    up = torch.tensor([0, 1, 0]).float().unsqueeze(0).repeat(num_poses, 1)

    # Set the Y orientation
    poses[:, :3, 1] = F.normalize(torch.cross(up, poses[:, :3, 3]), dim=1)

    # Set the X orientation
    poses[:, :3, 0] = F.normalize(torch.cross(poses[:, :3, 1], poses[:, :3, 2]), dim=1)

    return poses
