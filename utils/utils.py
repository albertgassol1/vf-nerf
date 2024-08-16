import datetime
import os
import socket
from glob import glob
from typing import List, Union, Tuple

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import requests
import skimage
from sklearn.cluster import KMeans
import torch
import trimesh
import torch.nn.functional as F
import lpips
from torchvision.transforms import ToTensor, Normalize, Compose
from scipy.spatial import cKDTree as KDTree


def get_timestamp() -> str:
    """
    Create a timestamp.
    :return: The timestamp.
    """
    return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def compute_sphere_intersections(cam_loc: torch.Tensor,
                                 ray_directions: torch.Tensor,
                                 radius: float = 1.0,
                                 device: torch.device = torch.device('cuda')) -> torch.Tensor:
    """
    Compute the sphere intersections.
    :param cam_loc: The camera location.
    :param ray_directions: The ray directions.
    :param radius: The radius of the sphere.
    :return: The sphere intersections.
    """

    ray_cam_dot = torch.bmm(ray_directions.view(-1, 1, 3),
                            cam_loc.view(-1, 3, 1)).squeeze(-1)
    under_sqrt = ray_cam_dot ** 2 - (cam_loc.norm(2, 1, keepdim=True) ** 2 - radius ** 2)

    # sanity check
    if (under_sqrt <= 0).sum() > 0:
        raise ValueError('BOUNDING SPHERE PROBLEM!')

    sphere_intersections = torch.sqrt(under_sqrt) * torch.tensor([-1, 1]).to(device).float() - ray_cam_dot
    sphere_intersections = sphere_intersections.clamp_min(0.0)

    return sphere_intersections


def mkdir_ifnotexists(directory: str) -> None:
    """
    Create a directory if it does not exist.
    :param directory: The directory.
    """
    if not os.path.exists(directory):
        os.mkdir(directory)


def glob_imgs(path: str) -> List[str]:
    imgs: List[str] = list()
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs


def load_rgb(path: str,
             normalize_rgb: bool = False,
             transpose: bool = True) -> np.ndarray:
    """
    Load an RGB image.
    :param path: The path to the image.
    :param normalize_rgb: Whether to normalize the RGB values.
    :param transpose: Whether to transpose the image.
    :return: The RGB image.
    """
    # Load and convert the image.
    img = imageio.imread(path)
    img = skimage.img_as_float32(img)

    # Normalize the RGB values betweeon [0,1].
    if normalize_rgb:  # [-1,1] --> [0,1]
        img -= 0.5
        img *= 2.
    if transpose:
        img = img.transpose(2, 0, 1)
    return img


def save_rgb(path: str,
             image: np.ndarray) -> None:
    """
    Save an RGB image.
    :param path: The path to the image.
    :param image: The RGB image.
    """

    # Denormalize the RGB values betweeon [-1,1].
    # image = np.clip(image, 0, 1).astype(np.uint8)

    # Save the image.
    imageio.imwrite(path, (image * 255).astype(np.uint8))


def save_depth(path: str, depth: np.ndarray) -> None:
    """
    Save a depth map.
    :param path: The path to the depth map.
    :param depth: The depth map.
    """
    fig, ax = plt.subplots()
    cax = ax.imshow(depth, cmap="plasma")
    fig.colorbar(cax, ax=ax, label='Depth value')
    ax.axis('off')
    fig.savefig(path + ".png", bbox_inches='tight', pad_inches=0)
    np.save(path, depth)
    plt.close(fig)


def load_depth(path: str) -> torch.Tensor:
    """
    Read a depth map.
    :param path: The path to the depth map.
    :return: The depth map.
    """
    depth_data = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    return torch.from_numpy(depth_data).float()


def save_pcl(points: np.ndarray, path: str, colors: np.ndarray = None) -> None:
    """
    Save a point cloud.
    :param points: The point cloud.
    :param path: The path to the point cloud.
    """
    # Save the point cloud.
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(points)

    if colors is not None:
        pcl.colors = o3d.utility.Vector3dVector(colors)

    # Save the point cloud.
    o3d.io.write_point_cloud(path, pcl)


def de_parallel(model: Union[torch.nn.Module, torch.nn.parallel.DistributedDataParallel]) -> torch.nn.Module:
    return model.module if hasattr(model, 'module') else model


def set_seed_and_default() -> None:
    # Set pytorch default type.
    torch.set_default_dtype(torch.float32)

    # Set the random seed.
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Set the anomaly detection.
    # torch.autograd.set_detect_anomaly(True)


def wandb_mode() -> str:
    """
    Get the wandb mode. If we have internet connection, use online mode, otherwise use offline mode.
    :return: The wandb mode.
    """
    # Check if we are on the local machine.
    if socket.gethostname() == "albert":
        print("Local test. Offline mode.")
        return "offline"
    # Check if we have internet connection.
    try:
        requests.get('https://www.google.com/')
        print("Online mode.")
        return "online"
    except:
        print("Offline mode.")
        return "offline"
    
def project_to_plane(points: torch.Tensor, vectors: torch.Tensor, 
                     u_plane: torch.Tensor, v_plane: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project points to a plane.
    :param points: The points.
    :param vectors: The vectors.
    :param u_plane: The u vector of the plane.
    :param v_plane: The v vector of the plane.
    :return: The projected points and vectors.
    """

    # Normalize the vectors.
    u_plane = u_plane / torch.norm(u_plane)
    v_plane = v_plane / torch.norm(v_plane)

    # Take an arbitrary origin on the plane.
    origin = points[0]

    # Project point on the plane.
    points_u = torch.matmul(points - origin, u_plane)
    points_v = torch.matmul(points - origin, v_plane)

    # Project vectors on the plane.
    vectors_u = torch.matmul(vectors, u_plane)
    vectors_v = torch.matmul(vectors, v_plane)

    return torch.stack([points_u, points_v], dim=1), torch.stack([vectors_u, vectors_v], dim=1)

def get_dominant_bases(num_bases: int, decimation: float, path_to_bases: str) -> np.ndarray:
    """
    Compute the dominant bases from the mesh.
    :param num_bases: The number of bases.
    :param decimation: The decimation factor.
    :param path_to_bases: The path to mesh.
    """
    # Load the mesh.
    mesh = trimesh.load(path_to_bases)

    # Decimate the mesh.
    mesh = mesh.simplify_quadratic_decimation(decimation * len(mesh.faces))

    # Cluster the normals.
    kmeans = KMeans(n_clusters=num_bases)
    kmeans.fit(mesh.vertex_normals)

    return kmeans.cluster_centers_

def get_psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute the PSNR.
    :param prediction: The predicted image.
    :param target: The target image.
    :return: The PSNR.
    """
    mse = torch.mean((prediction - target) ** 2)
    if mse == 0:
        return float('inf')
    return (-10 * torch.log10(mse)).item()


def get_ssim(prediction: torch.Tensor, target: torch.Tensor, window_size: int = 11, 
             size_average: bool = True, C1: float = 1e-4, C2:float = 9e-4) -> float:
    """
    Compute the SSIM.
    :param prediction: The predicted image.
    :param target: The target image.
    :param window_size: The window size.
    :param size_average: Whether to average the SSIM.
    :param C1: The C1 constant.
    :param C2: The C2 constant.
    :return: The SSIM.
    """
    # Convert (H, W, 3) to (3, H, W)
    prediction = prediction.permute(2, 0, 1).unsqueeze(0)  # Add batch dimension
    target = target.permute(2, 0, 1).unsqueeze(0)  # Add batch dimension

    # Define a window for local statistics computation
    window = torch.ones(window_size, window_size).unsqueeze(0).unsqueeze(0)
    window /= window.sum()

    # Expand the window to match the number of channels of the images
    window = window.expand(prediction.size(1), 1, window_size, window_size).to(prediction.device)

    # Compute local means using convolution
    mu1 = F.conv2d(prediction, window, padding=window_size // 2, groups=prediction.size(1))
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=target.size(1))

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(prediction * prediction, window, padding=window_size // 2, groups=prediction.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=target.size(1)) - mu2_sq
    sigma12 = F.conv2d(prediction * target, window, padding=window_size // 2, groups=prediction.size(1)) - mu1_mu2

    # Compute SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.mean(1).mean(1).mean(1).item()
    
def get_lpips(prediction: torch.Tensor, target: torch.Tensor, net: str = 'vgg') -> float:
    """
    Compute the LPIPS.
    :param prediction: The predicted image.
    :param target: The target image.
    :param net: The network.
    :return: The LPIPS.
    """
    # Convert images to tensor format and normalize
    transform = Compose([
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    prediction_t = transform(prediction.numpy()).unsqueeze(0)
    target_t = transform(target.numpy()).unsqueeze(0)
    
    # Calculate LPIPS distance
    distance = lpips.LPIPS(net=net)(prediction_t, target_t).item()
    return distance

def get_l1_cm(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """
    Computes the L1 loss between two depth maps, the result is in cm. The depth maps are assumed to be in meters.
    :param prediction: The predicted depth map.
    :param target: The target depth map.
    :return: The L1 loss in cm.
    """

    # Convert the depth maps to cm.
    prediction = prediction * 100
    target = target * 100

    # Compute the L1 loss.
    return torch.mean(torch.abs(prediction - target)).item()

def get_chamfer_distance(pred_mesh: trimesh.Trimesh, ref_mesh: trimesh.Trimesh, num_points: int = 2500000) -> Tuple[float, float, float, float]:
    """
    Compute the chamfer distance between two meshes.
    :param pred_mesh: The predicted mesh.
    :param ref_mesh: The reference mesh.
    :param num_points: The number of points to sample.
    :return: The mean chamfer distance, the median chamfer distance, the min chamfer distance, and the max chamfer distance.
    """

    def sample_mesh(m: trimesh.Trimesh, n: int) -> np.ndarray:
        """
        Sample the mesh.
        :param m: The mesh.
        :param n: The number of points.
        :return: The sampled points.
        """
        vpos, _ = trimesh.sample.sample_surface(m, n)
        return vpos
    
    # Sample the meshes.
    pred_points = sample_mesh(pred_mesh, num_points)
    ref_points = sample_mesh(ref_mesh, num_points)

    # one direction
    pred_points_kd_tree = KDTree(pred_points)
    one_distances, _ = pred_points_kd_tree.query(ref_points)
    gt_to_gen_chamfer_mean = np.mean(np.square(one_distances))
    gt_to_gen_chamber_median = np.median(np.square(one_distances))
    min_gt_to_gen_chamfer = np.min(np.square(one_distances))
    max_gt_to_gen_chamfer = np.max(np.square(one_distances))

    # other direction
    ref_points_kd_tree = KDTree(ref_points)
    two_distances, _ = ref_points_kd_tree.query(pred_points)
    gen_to_gt_chamfer_mean = np.mean(np.square(two_distances))
    gen_to_gt_chamber_median = np.median(np.square(two_distances))
    min_gen_to_gt_chamfer = np.min(np.square(two_distances))
    max_gen_to_gt_chamfer = np.max(np.square(two_distances))

    return gt_to_gen_chamfer_mean + gen_to_gt_chamfer_mean, gt_to_gen_chamber_median + gen_to_gt_chamber_median, \
           min(min_gt_to_gen_chamfer, min_gen_to_gt_chamfer), max(max_gt_to_gen_chamfer, max_gen_to_gt_chamfer)
    