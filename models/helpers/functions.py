import sys
from typing import Tuple

sys.path.append('.')  # isort:skip

import torch
import torch.nn.functional as F

from models.samplers.sampler import SphereSampler

def forward_window_cosine_similarity(x: torch.Tensor,
                                     y: torch.Tensor,
                                     weights: torch.Tensor) -> torch.Tensor:
    """
    Compute the cosine similarities for each vector in x and y using a window.
    :params x: The first tensor, of size (batch_size, num_samples, num_features).
    :params y: The second tensor, of size (batch_size, num_samples, num_features).
    :params weights: The weights for the window, of size (window_size).
    :returns: The cosine similarities, of size (batch_size, window_size).
    """
    shape = weights[0]

    # Compute sum of weights to use as normalization factor
    normalizer = torch.tensor(0.0).to(x.device)
    for i in range(shape):
        normalizer = normalizer + weights[i].abs()

    # Compute the cosine similarities.
    cosine_similarities = F.cosine_similarity(x, y, dim=2)
    cosine_similarities[:, shape:-shape] = cosine_similarities[:, shape:-shape] * weights[0].abs() / normalizer

    # Compute the windowed cosine similarities.
    for i in range(1, shape):
        cosine_similarities[:, shape:-shape] = cosine_similarities[:, shape:-shape] + \
            F.cosine_similarity(x[:, shape:-shape, :],
                                y[:, (shape + i):(-shape + i), :], dim=2) * weights[i].abs() / normalizer

    return cosine_similarities


def window_cosine_similarity(x: torch.Tensor,
                             y: torch.Tensor,
                             weights: torch.Tensor) -> torch.Tensor:
    """
    Compute the cosine similarities for each vector in x and y using a window. Use forwad and backward window.
    :params x: The first tensor, of size (batch_size, num_samples, num_features).
    :params y: The second tensor, of size (batch_size, num_samples, num_features).
    :params weights: The weights for the window, of size (window_size).
    :returns: The cosine similarities, of size (batch_size, window_size).
    """
    start = int((weights.shape[0] + 1) / 2 + 1)
    middle = int((weights.shape[0] - 1) / 2)

    # Compute sum of weights to use as normalization factor
    normalizer = torch.tensor(0.0).to(x.device)
    for i in range(weights.shape[0]):
        normalizer = normalizer + weights[i].abs()

    # Compute the cosine similarities.
    cosine_similarities = F.cosine_similarity(x, y, dim=2)
    cosine_similarities[:, start:-start] = cosine_similarities.clone()[:, start:-start] * \
        weights[middle] / normalizer

    # Compute the windowed cosine similarities.
    for i in range(1, start - 1):
        cosine_similarities[:, start:-start] = cosine_similarities.clone()[:, start:-start] + \
            F.cosine_similarity(x[:, start:-start, :],
                                y[:, (start + i):(-start + i), :], dim=2) * weights[middle + i].abs() / normalizer + \
            F.cosine_similarity(x[:, start:-start, :],
                                y[:, (start - i - 1):(-start - i - 1), :], dim=2) * weights[middle - i].abs() / normalizer

    return cosine_similarities


def get_border_indices_and_gt(points: torch.Tensor, normals: torch.Tensor, far: float, radius: float,
                              centroid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get the border indices and ground truth for the points.
    :param points: The points, of size (num_rays, num_samples, 3).
    :param normals: The normals, of size (num_rays, num_samples, 3).
    :param far: The far value.
    :param radius: The radius.
    :param centroid: The centroid, of size (3).
    :return: The normals of the border points and ground truth.
    """
    # Compute the distances.
    distances = torch.norm(points - centroid, dim=2)

    # Compute the border indices.
    border_points_condition = distances > (far/2 - radius)
    border_points = points[border_points_condition]
    border_normals = normals[border_points_condition]

    # Compute the ground truth.
    gt = F.normalize(centroid - border_points, dim=1)

    return border_normals, gt

def sample_border_points(r_min: float, r_max: float,
                         num_samples: int, centroid: torch.Tensor,
                         device: torch.device = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample border points.
    :param r_min: The minimum radius.
    :param r_max: The maximum radius.
    :param num_samples: The number of samples.
    :param centroid: The centroid.
    :return: The points and ground truth normals.
    """
    # Sample the points.
    points = torch.from_numpy(SphereSampler(num_samples).sample(r_max, r_min)).float().to(device) + centroid

    # Compute the ground truth.
    gt = F.normalize(centroid - points, dim=1)

    return points, gt

def sample_center_points(centroid: torch.Tensor, radius: float, 
                         num_samples: int, device: torch.device = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample center points.
    :param centroid: The centroid.
    :param radius: The radius.
    :param num_samples: The number of samples.
    :return: The points and ground truth normals.
    """
    # Sample the points.
    points = torch.from_numpy(SphereSampler(num_samples).sample(radius, 0.0)).float().to(device) + centroid

    # Compute the ground truth.
    gt = F.normalize(points - centroid, dim=1)

    return points, gt


def get_center_indices_and_gt(points: torch.Tensor, normals: torch.Tensor, centroid: torch.Tensor, radius: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get the center indices and ground truth for the points.
    :param points: The points, of size (num_rays, num_samples, 3).
    :param normals: The normals, of size (num_rays, num_samples, 3).
    :param centroid: The centroid, of size (3).
    :param radius: The radius.
    :return: The center normals and ground truth.
    """

    # Compute the distances.
    distances = torch.norm(points - centroid, dim=2)

    # Compute the center indices.
    center_points_condition = distances < radius
    center_points = points[center_points_condition]
    center_normals = normals[center_points_condition]

    # Compute the ground truth.
    gt = F.normalize(center_points - centroid, dim=1)

    return center_normals, gt

def get_cosine_losses(cosine_similarity: torch.Tensor,
                      weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the cosine losses.
    :param cosine_similarity: cosine similarities
    :param weights: weights
    :return: cosine losses
    """
    # Get the maximums of the coarse weights which are not zero and minimize the cosine similarity of those points.
    maxs, argmaxs = torch.max(weights, dim=1)
    # TODO: Remove this hack
    real_maxs = (maxs > 0.0) & (argmaxs < 70)
    minimizable_cosines = cosine_similarity[torch.arange(cosine_similarity.shape[0]), argmaxs][real_maxs]
    # Get the cosine similarities until the maximums.
    if torch.any(real_maxs):
        maximizable_cosines = torch.cat([cosine_similarity[i, :argmaxs[i]]
                                         for i in range(cosine_similarity.shape[0]) if real_maxs[i]], dim=0)
    else:
        maximizable_cosines = torch.tensor(0.0).to(cosine_similarity.device)
    # Compute the cosine_loss
    max_cosine_sim = -torch.mean(maximizable_cosines)
    min_cosine_sim = torch.mean(minimizable_cosines)
    return min_cosine_sim, max_cosine_sim

def get_similarity_loss(x1: torch.Tensor,
                        x2: torch.Tensor,
                        v1: torch.Tensor,
                        v2: torch.Tensor) -> torch.Tensor:
    """
    Compute the similarity loss.
    :params x1: The first point.
    :params x2: The second point.
    :params v1: The first vector.
    :params v2: The second vector.
    :returns: The similarity loss.
    """

    # Compute the distance between x2 and x1. The shape of the points is (batch_size, n_samples, 3).
    distance = torch.norm(x2 - x1, dim=1)

    # Compute the estimated x1 and x2 points.
    x1_estimated = x2 + F.normalize(v2, dim=1) * distance.unsqueeze(-1)
    x2_estimated = x1 + F.normalize(v1, dim=1) * distance.unsqueeze(-1)

    # Compute L2 norm of the difference between the estimated and the ground truth points.
    x1 = x1.reshape(-1, 3)
    x2 = x2.reshape(-1, 3)
    x1_estimated = x1_estimated.reshape(-1, 3)
    x2_estimated = x2_estimated.reshape(-1, 3)

    x_1_diff = torch.norm(x1 - x1_estimated, dim=1)
    x_2_diff = torch.norm(x2 - x2_estimated, dim=1)

    diffrence_loss = x_1_diff + x_2_diff

    # Compute the weighted loss. Zero out the loss for the points which have a similar direction.
    with torch.no_grad():
        cosine_similarity = F.cosine_similarity(v1, v2, dim=1).reshape(-1)
        indices = torch.where((cosine_similarity < 0.5) & (diffrence_loss > 0.5 * diffrence_loss.max()))[0]
    # with torch.no_grad():
    #     difference_loss[cosine_similarity > 0] = 0
    if indices.shape[0] == 0:
        weighted_loss = torch.zeros(1).to(diffrence_loss.device)
    else:
        weighted_loss = torch.mean(diffrence_loss[indices] * (1 - cosine_similarity[indices]))

    return weighted_loss
