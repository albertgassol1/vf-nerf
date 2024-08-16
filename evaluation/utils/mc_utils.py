import math

import numpy as np
import torch
import torch.nn.functional as F


def get_easy_convergence_points(points, vts, N=64, size=2.0):

    vts_first = vts[:, :, :3]
    vts_second = vts[:, :, 3:]
    convergence_points = (vts_first * vts_second).sum(dim=-1)
    convergence_points[convergence_points > -0.1] = 0
    convergence_points[convergence_points <= -0.1] = 1
    convergence_points = convergence_points

    points_first = points[:, :, :3]
    points_second = points[:, :, 3:]
    distance_before = 2 * torch.norm(points_first - points_second, dim=-1)

    new_points_first = points_first + vts_first * (size / N)
    new_points_second = points_second + vts_second * (size / N)
    distance_after = torch.norm(new_points_first - points_second, dim=-1) + torch.norm(
        points_first - new_points_second, dim=-1
    )

    convergence_points[
        (convergence_points == 1) & (distance_after > distance_before)
    ] = 0

    return convergence_points


def extract_divergence(vt_values, N):

    # Threshold set to determine which voxel blocks have a surface
    threshold = -0.5

    internal_vt = (
        F.normalize(vt_values.clone(), dim=1).reshape(N, N, N, 3).permute(3, 0, 1, 2)
    )

    # Section to create a (2, 2, 2) box with coordinates around the origin
    # It is used to compute the directions for the divergence (reused code from mesh.py)
    N_box = 2
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N_box - 1)
    overall_index = torch.arange(0, N_box ** 3, 1, out=torch.LongTensor())
    filter = torch.zeros(N_box ** 3, 3)
    filter[:, 2] = overall_index % N_box
    filter[:, 1] = (overall_index.long() // N_box) % N_box
    filter[:, 0] = ((overall_index.long() // N_box) // N_box) % N_box
    filter[:, 0] = (filter[:, 0] * voxel_size) + voxel_origin[2]
    filter[:, 1] = (filter[:, 1] * voxel_size) + voxel_origin[1]
    filter[:, 2] = (filter[:, 2] * voxel_size) + voxel_origin[0]
    filter = (
        F.normalize(filter, dim=1)
        .reshape(N_box, N_box, N_box, 3)
        .permute(3, 0, 1, 2)
        .unsqueeze(0)
    )
    scatter_indices = (
        torch.arange(N_box ** 3)
        .reshape(N_box, N_box, N_box)
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat_interleave(3, dim=1)
    )
    complete_filter = torch.zeros_like(filter).repeat_interleave(N_box ** 3, dim=0)
    complete_filter.scatter_(dim=0, index=scatter_indices, src=filter)
    face_area = math.sqrt(3.0) / 4.0
    shape_volume = math.sqrt(2.0) / 3.0

    divergence = F.conv3d(internal_vt.unsqueeze(0), complete_filter).squeeze()
    divergence = (divergence * torch.abs(divergence) * face_area).sum(
        dim=0
    ) / shape_volume

    divergence_values = torch.zeros((N, N, N)).type_as(divergence)
    divergence_values[:-1, :-1, :-1] += divergence

    divergence_values[divergence_values > threshold] = 0
    divergence_values[divergence_values <= threshold] = 1

    return divergence_values


def get_set_predictions(decoder, samples, max_batch, device):

    samples.requires_grad = False
    num_samples = samples.shape[0]
    predicions = torch.zeros_like(samples[:, :3])

    head = 0

    while head < num_samples:
        sample_subset = samples[head: min(head + max_batch, num_samples)]
        sample_subset = sample_subset.to(device)
        predicions[head: min(head + max_batch, num_samples)] = (
            decoder(sample_subset).detach().cpu()[:, :3]
        )
        head += max_batch

    return predicions


def unify_direction(divergence_grid, vt_grid, N=64):

    n_reduction = 2

    selection_filter = torch.zeros(
        (n_reduction ** 3, 1, n_reduction, n_reduction, n_reduction)
    ).type_as(vt_grid)
    selection_filter[0, 0, 0, 0, 0] = 1
    selection_filter[1, 0, 0, 1, 0] = 1
    selection_filter[2, 0, 1, 1, 0] = 1
    selection_filter[3, 0, 1, 0, 0] = 1
    selection_filter[4, 0, 0, 0, 1] = 1
    selection_filter[5, 0, 0, 1, 1] = 1
    selection_filter[6, 0, 1, 1, 1] = 1
    selection_filter[7, 0, 1, 0, 1] = 1

    temp_vt_grid = F.conv3d(
        vt_grid.clone().unsqueeze(1),
        selection_filter,
        padding=1,
    )
    temp_vt_grid = temp_vt_grid.permute(2, 3, 4, 1, 0)[1:, 1:, 1:]
    surface_vt = temp_vt_grid[divergence_grid == 1]
    distance_matrix = 1.0 - (
        torch.bmm(surface_vt[:, :, 0].unsqueeze(-1), surface_vt[:, :, 0].unsqueeze(-2)) +
        torch.bmm(
            surface_vt[:, :, 1].unsqueeze(-1), surface_vt[:, :, 1].unsqueeze(-2)
        ) +
        torch.bmm(
            surface_vt[:, :, 2].unsqueeze(-1), surface_vt[:, :, 2].unsqueeze(-2)
        )
    ).reshape(surface_vt.shape[0], n_reduction ** 6)
    extreme_indices = torch.argmax(distance_matrix, dim=-1)
    first = (
        (extreme_indices // (n_reduction ** 3))
        .unsqueeze(-1)
        .unsqueeze(-1)
        .repeat_interleave(3, dim=-1)
    )
    second = (
        (extreme_indices % (n_reduction ** 3))
        .unsqueeze(-1)
        .unsqueeze(-1)
        .repeat_interleave(3, dim=-1)
    )
    first_vec = torch.gather(surface_vt, dim=1, index=first)
    second_vec = torch.gather(surface_vt, dim=1, index=second)

    first_vec = first_vec.repeat_interleave(n_reduction ** 3, dim=1)
    second_vec = second_vec.repeat_interleave(n_reduction ** 3, dim=1)

    first_distance = torch.norm(first_vec - surface_vt, dim=-1)
    second_distance = torch.norm(second_vec - surface_vt, dim=-1)
    choice = torch.argmin(
        torch.stack((first_distance, second_distance), dim=-1), dim=-1
    )

    direction_choice = torch.zeros((N, N, N, 8)).type_as(choice)
    direction_choice[divergence_grid == 1] = choice

    return direction_choice.reshape(-1, 8)


def make_comb_format(choice_side, norms, N):

    n_reduction = 2

    selection_filter = torch.zeros(
        (n_reduction ** 3, 1, n_reduction, n_reduction, n_reduction)
    ).type_as(norms)
    selection_filter[0, 0, 0, 0, 0] = 1
    selection_filter[1, 0, 0, 1, 0] = 1
    selection_filter[2, 0, 1, 1, 0] = 1
    selection_filter[3, 0, 1, 0, 0] = 1
    selection_filter[4, 0, 0, 0, 1] = 1
    selection_filter[5, 0, 0, 1, 1] = 1
    selection_filter[6, 0, 1, 1, 1] = 1
    selection_filter[7, 0, 1, 0, 1] = 1

    norms = F.conv3d(
        norms.clone().reshape(N, N, N).unsqueeze(0).unsqueeze(0),
        selection_filter,
        padding=1,
    )
    norms = norms.permute(2, 3, 4, 1, 0)[1:, 1:, 1:].reshape(N ** 3, 8)

    different_side = torch.zeros((N ** 3, 28)).type_as(norms)
    different_side_norms = torch.zeros((N ** 3, 28, 2)).type_as(norms)

    inc = np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
            [1, 0, 1],
        ]
    )

    combs = []
    comb_to_idx = [0] * 64
    dist = [0] * 64
    for i in range(7):
        for j in range(i + 1, 8):
            comb_to_idx[i * 8 + j] = len(combs)
            dist[i * 8 + j] = np.linalg.norm(inc[i] - inc[j])
            combs.append([i, j])

    for i, indices in enumerate(combs):
        different_side[:, i] = choice_side[:, indices[0]] != choice_side[:, indices[1]]
        different_side_norms[:, i, 0] = norms[:, indices[0]]
        different_side_norms[:, i, 1] = norms[:, indices[1]]

    return different_side, different_side_norms
