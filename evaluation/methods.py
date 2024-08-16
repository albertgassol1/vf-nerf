import gc
import json
import os
import sys
import shutil
from typing import Optional

sys.path.append('.')  # isort:skip

import imageio
import numpy as np
import open3d as o3d
import plyfile
import skimage
import torch
import torch.nn.functional as F
import trimesh
from evaluate_3d_reconstruction import run_evaluation
from torch import nn

import datasets.helpers.poses_utils as poses_utils
import evaluation.utils.marching_cubes_vt as marching_cubes_vt
import evaluation.utils.mc_utils as mc_utils
import evaluation.utils.plots as plots
import utils.utils as utils
from config_parser.vf_nerf_config import DatasetConfig
from datasets.normal_datasets import dataset_dict
from evaluation.utils.guassian_smoothing import smooth_vf
from evaluation.utils.renderer import Renderer
from models.nerf.vector_field_nerf import VectorFieldNerf


def refuse(mesh: trimesh.Trimesh, dataset_config: DatasetConfig) -> trimesh.Trimesh:
    """
    Use tsdf mesh to remove artifacts.
    :param mesh: The mesh.
    :param dataset_config: The dataset configuration.
    :return: The mesh.
    """
    # Create the dataset and dataloader.
    dataset = dataset_dict[dataset_config.dataset_name](dataset_config)
    dataset.all_pixels = True
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    renderer = Renderer(dataset.image_size[0], dataset.image_size[1])
    mesh_opengl = renderer.mesh_opengl(mesh)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=4 / 512.0,
                                                          sdf_trunc=0.04,
                                                          color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)


    for batch in dataloader:
        h, w = dataset.image_size
        intrinsic = batch["intrinsics"].squeeze(0).numpy()[0, :]
        pose = batch["pose"].squeeze(0).numpy()[0, :]
        rgb = batch["rgb"].squeeze(0).numpy().reshape(h, w, 3)  
        rgb = (rgb * 255).astype(np.uint8)
        rgb = o3d.geometry.Image(rgb)
        _, depth_pred = renderer(h, w, intrinsic, pose, mesh_opengl)
        depth_pred = o3d.geometry.Image(depth_pred)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb, depth_pred, depth_scale=1.0, depth_trunc=5.0, convert_rgb_to_intensity=False)
        fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width=w, height=h, fx=fx,  fy=fy, cx=cx, cy=cy)
        extrinsic = np.linalg.inv(pose)
        volume.integrate(rgbd, intrinsic, extrinsic)
    
    mesh = volume.extract_triangle_mesh()

    # Convert open3d mesh to trimesh.
    mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles))
    return mesh
       
@torch.no_grad()
def quadrant_marching_cubes(model: nn.Module,
                            resolution: int,
                            path: str,
                            checkpoint: int,
                            max_batch: int = 100000,
                            scale: float = 1.0,
                            centroid: torch.Tensor = torch.zeros(3),
                            device: torch.device = torch.device("cuda"),
                            num_quadrants: int = 35,
                            smooth_after: bool = False,
                            smooth_all:bool = False) -> None:
    """
    Generate a mesh from the trained vector field network.
    :param model: The model.
    :param resolution: The resolution.
    :param path: The path.
    :param checkpoint: The checkpoint.
    :param max_batch: The maximum batch size.
    :param scale: The scale.
    :param device: The device.
    :param centroid: The centroid.
    :param num_quadrants: The number of quadrants.
    :param smooth_after: Whether to smooth after.
    :param smooth_all: Whether to smooth all.
    """

    assert num_quadrants == 64 or num_quadrants == 8 or num_quadrants == 35

    meshes = []

    if num_quadrants == 35:
        s = scale / 3
        values = torch.tensor([-2*s, 0, 2*s])
        translations = torch.cartesian_prod(values, values, values).float()
        for t in translations:
            meshes.append(marching_cubes_mesh(model, resolution, path, checkpoint, max_batch, s, device, t, 
                                              centroid, False, smooth_after=smooth_after, smooth_all=smooth_all))

    if num_quadrants == 8 or num_quadrants == 35:
        s = scale / 2
        values = torch.tensor([-s, s]).float()
    else:
        s = scale / 4
        values = torch.tensor([-3*s, -1*s, 3*s, 1*s]).float()   

    translations = torch.cartesian_prod(values, values, values).float()

    for t in translations:
        meshes.append(marching_cubes_mesh(model, resolution, path, checkpoint, max_batch, s, device, t, centroid, False,
                                          smooth_after=smooth_after, smooth_all=smooth_all))
        
    # Merge the meshes.
    merged_mesh = trimesh.util.concatenate(meshes[0], meshes[1])
    for i in range(2, len(meshes)):
        merged_mesh = trimesh.util.concatenate(merged_mesh, meshes[i])


    # Save the mesh.
    merged_mesh.export(os.path.join(path, f"merged-mesh-scaled-{checkpoint}.ply"))
    merged_mesh.apply_scale([1/scale, 1/scale, 1/scale])
    merged_mesh.apply_translation(-centroid)
    # Save the mesh
    merged_mesh.export(os.path.join(path, f"merged-mesh-{checkpoint}.ply"))

@torch.no_grad()
def marching_cubes_mesh(model: nn.Module,
                        resolution: int,
                        path: str,
                        checkpoint: int,
                        max_batch: int = 100000,
                        scale: float = 1.0,
                        device: torch.device = torch.device("cuda"),
                        translation: torch.Tensor = torch.zeros(3),
                        centroid: torch.Tensor = torch.zeros(3),
                        save: bool = True,
                        alternative: bool = False,
                        smooth_after: bool = False,
                        smooth_all:bool = False) -> trimesh.Trimesh:
    """
    Generate a mesh from the trained vector field network.
    :param model: The model.
    :param resolution: The resolution.
    :param path: The path.
    :param checkpoint: The checkpoint.
    :param max_batch: The maximum batch size.
    :param scale: The scale.
    :param device: The device.
    :param translation: The translation.
    :param centroid: The centroid.
    :param save: Whether to save the mesh.
    :param alternative: Whether to use the alternative method.
    :param smooth_after: Whether to smooth after.
    :param smooth_all: Whether to smooth all.
    """
    model.eval()

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

    selected_indices = np.mgrid[: int(resolution / 2),
                                : int(resolution / 2),
                                : int(resolution / 2)]
    selected_indices = np.moveaxis(selected_indices, 0, -1).reshape(-1, 3)
    selected_indices = (selected_indices[:, None] * 2 + inc[None]).reshape(-1, 3)

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-scale, -scale, -scale]
    voxel_size = scale * 2.0 / (resolution - 1)

    overall_index = torch.arange(0, resolution ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(resolution ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % resolution
    samples[:, 1] = (overall_index.long() // resolution) % resolution
    samples[:, 0] = ((overall_index.long() // resolution) //
                     resolution) % resolution

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2] + translation[0] + centroid[0]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1] + translation[1] + centroid[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0] + translation[2] + centroid[2]

    prediction = mc_utils.get_set_predictions(model, samples, max_batch, device)
    # del samples
    # gc.collect()
    if smooth_all:
        prediction = smooth_vf(
            prediction.reshape(resolution, resolution, resolution, 3), k=3, sigma=1
        ).reshape(resolution**3, 3)
    divergence_values = mc_utils.extract_divergence(prediction, resolution)
    if smooth_after or smooth_all:
        prediction = smooth_vf(
            prediction.reshape(resolution, resolution, resolution, 3), k=9, sigma=2
        ).reshape(resolution**3, 3)

    norms = torch.norm(prediction.clone(), dim=1)
    vt_values = F.normalize(prediction, dim=1).reshape(resolution,
                                                       resolution,
                                                       resolution, 3)

    # Uncomment these lines to try an alternative way of getting the mesh (might be more similar to MeshUDF)
    if alternative:
        reordered_vts = torch.from_numpy(
            marching_cubes_vt.get_grid_comb_div(
                res=resolution,
                selected_indices=selected_indices,
                mgrid=F.pad(vt_values, (0, 0, 0, 1, 0, 1, 0, 1)).numpy(),
            )
        )
        reordered_points = torch.from_numpy(
            marching_cubes_vt.get_grid_comb_div(
                res=resolution,
                selected_indices=selected_indices,
                mgrid=F.pad(samples.reshape(resolution, resolution, resolution, 3), (0, 0, 0, 1, 0, 1, 0, 1)).numpy(),
            )
        )
        convergence_points = mc_utils.get_easy_convergence_points(
            reordered_points, reordered_vts, N=resolution, size=2.0
        ).reshape(resolution, resolution, resolution, 28)

    chosen_direction = mc_utils.unify_direction(
        divergence_values, vt_values.permute(3, 0, 1, 2), N=resolution
    )
    del divergence_values
    gc.collect()
    comb_values, norms = mc_utils.make_comb_format(chosen_direction, norms, resolution)
    comb_values = comb_values.reshape(resolution, resolution,
                                      resolution, 28)
    del chosen_direction
    gc.collect()
    norms = norms.reshape(resolution, resolution,
                          resolution, 28, 2)
    comb_values = comb_values[
        selected_indices[:, 0], selected_indices[:, 1], selected_indices[:, 2]
    ].reshape(resolution, resolution,
              resolution, 28)
    norms = norms[
        selected_indices[:, 0], selected_indices[:, 1], selected_indices[:, 2]
    ]

    if alternative:
        comb_values[convergence_points == 1] = 1
        comb_values[convergence_points == 0] = 0

    udf = norms.clone().cpu().numpy()
    del norms
    gc.collect()
    comb_values = comb_values.clone().cpu().numpy().reshape(-1, 28)
    mask = comb_values.sum(-1)
    selected_indices = selected_indices[mask > 0]
    udf = udf[mask > 0].reshape(-1, 2)
    comb_values = comb_values[mask > 0].reshape(-1)

    if len(comb_values) > 0:
        vs, fs = marching_cubes_vt.contrastive_marching_cubes(
            comb_values, isovalue=0.0, selected_indices=selected_indices,
            res=resolution, udf=udf
        )
        del comb_values, udf, mask, selected_indices
        gc.collect()

        vs, fs = np.array(list(vs.keys())), np.array(fs) - 1
        num_verts = vs.shape[0]
        num_faces = fs.shape[0]

        verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

        for i in range(0, num_verts):
            verts_tuple[i] = tuple(vs[i, :])

        faces_building = []
        for i in range(0, num_faces):
            faces_building.append((fs[i, :].tolist(),))

        faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

        el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
        el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

        ply_data = plyfile.PlyData([el_verts, el_faces])

        mesh = trimesh.Trimesh(vertices=[[vertex[0], vertex[1], vertex[2]] for vertex in ply_data['vertex']], 
                            faces=[face[0] for face in ply_data['face']])
    else:
        mesh = trimesh.Trimesh()
        
    mesh.apply_scale([scale, scale, scale])
    mesh.apply_translation(translation)
    mesh.apply_translation(centroid)
    if save:
        ply_data.write(os.path.join(path, f"mesh-{checkpoint}.ply"))
        # Save the mesh
        mesh.export(os.path.join(path, f"mesh-scaled-{checkpoint}.ply"))

    return mesh


@torch.no_grad()
def plot_2d_slices(model: nn.Module,
                   centroid: torch.Tensor = torch.zeros(3),
                   scale: float = 1.0,
                   path: Optional[str] = None,
                   device: torch.device = torch.device("cuda"),
                   smooth: bool = False) -> None:
    """
    Plot 2D slices of the vector field.
    :param model: The model.
    :param path: The path.
    :param device: The device.
    :param smooth: Whether to smooth the vector field.
    """
    lower_bound = -scale + centroid
    upper_bound = scale + centroid

    model.eval()

    y_values = torch.linspace(lower_bound[1], upper_bound[1], 20).to(device)
    x_values = torch.linspace(lower_bound[0], upper_bound[0], 20).to(device)
    z_values = torch.linspace(lower_bound[2], upper_bound[2], 20).to(device)

    if smooth and path is not None:
        path = os.path.join(path, "smooth_2d_plots")
        utils.mkdir_ifnotexists(path)
    elif path is not None:
        path = os.path.join(path, "2d_plots")
        utils.mkdir_ifnotexists(path)

    if smooth and path is not None:
        path = os.path.join(path, "smooth_2d_plots")
        utils.mkdir_ifnotexists(path)
    elif path is not None:
        path = os.path.join(path, "2d_plots")
        utils.mkdir_ifnotexists(path)

    # Plot the vector fields.
    for z in z_values:
        # Create the grid.
        grid = torch.stack(torch.meshgrid(x_values, y_values, z, indexing='ij'), dim=-1).reshape(-1, 3)

        # Compute the vector field.
        vector_field: torch.Tensor = model(grid)[:, :3]
        if smooth:
            vector_field = smooth_vf(vector_field.reshape(20, 20, 1, 3)).reshape(20*20*1, 3)

        plots.plot_2d_slices(grid[:, 0].cpu().numpy(), grid[:, 1].cpu().numpy(),
                             vector_field.cpu().numpy(), z.cpu().item(), path)

    plots.show()

@torch.no_grad()
def plot_overall_scene(model: nn.Module,
                       centroid: torch.Tensor = torch.zeros(3),
                       scale: float = 1.0,
                       path: Optional[str] = None,
                       device: torch.device = torch.device("cuda"),
                       smooth: bool = False) -> None:
    """
    Plot 2D slices of the vector field.
    :param model: The model.
    :param centroid: The centroid.
    :param scale: The scale.
    :param path: The path.
    :param device: The device.
    :param smooth: Whether to smooth the vector field.
    """
    lower_bound = -scale + centroid
    upper_bound = scale + centroid

    model.eval()
    # Fix the z coordinate.
    z_values = torch.linspace(lower_bound[2], upper_bound[0], 15).to(device)

    # Create x and z coordinates.
    y_values = torch.linspace(lower_bound[1], upper_bound[1], 15).to(device)
    x_values = torch.linspace(lower_bound[0], upper_bound[0], 15).to(device)

    if smooth and path is not None:
        path = os.path.join(path, "smooth_overall")
        utils.mkdir_ifnotexists(path)
    elif path is not None:
        path = os.path.join(path, "overall")
        utils.mkdir_ifnotexists(path)

    # Plot the vector fields.
    overall_grid = []
    overall_vector_field = []
    for z in z_values:
        # Create the grid.
        grid = torch.stack(torch.meshgrid(x_values, y_values, z, indexing='ij'), dim=-1).reshape(-1, 3)
        vector_field: torch.Tensor = model(grid)[:, :3]
        overall_grid.append(grid)
        overall_vector_field.append(vector_field)

    overall_grid = torch.cat(overall_grid, dim=0)
    overall_vector_field = torch.cat(overall_vector_field, dim=0)
    if smooth:
        overall_vector_field = smooth_vf(overall_vector_field.reshape(15, 15, 15, 3)).reshape(15**3, 3)

    plots.plot_overall_scene(overall_grid[:, 0].cpu().numpy(), overall_grid[:, 1].cpu().numpy(),
                             overall_grid[:, 2].cpu().numpy(), overall_vector_field.cpu().numpy(), path)

    plots.show()

@torch.no_grad()
def plot_3d_slices(model: nn.Module,
                   path: Optional[str] = None,
                   device: torch.device = torch.device("cuda"),
                   smooth: bool = False) -> None:
    """
    Plot 2D slices of the vector field.
    :param model: The model.
    :param path: The path.
    :param device: The device.
    """
    model.eval()
    # Fix the z coordinate.
    z_values = torch.linspace(-2, 2, 10).to(device)

    # Create x and z coordinates.
    y_values = torch.linspace(-4, 2.5, 20).to(device)
    x_values = torch.linspace(-2.5, 3.0, 20).to(device)

    if smooth and path is not None:
        path = os.path.join(path, "smooth_3d_plots")
        utils.mkdir_ifnotexists(path)
    elif path is not None:
        path = os.path.join(path, "3d_plots")
        utils.mkdir_ifnotexists(path)

    # Plot the vector fields.
    for z in z_values:
        # Create the grid.
        grid = torch.stack(torch.meshgrid(x_values, y_values, z, indexing='ij'), dim=-1).reshape(-1, 3)

        # Compute the vector field.
        vector_field: torch.Tensor = model(grid)[:, :3]
        if smooth:
            vector_field = smooth_vf(vector_field.reshape(20, 20, 1, 3)).reshape(20*20*1, 3)

        # Plot the vector field.
        plots.plot_3d_slices(grid[:, 0].cpu().numpy(), grid[:, 1].cpu().numpy(),
                             vector_field.cpu().numpy(), z.cpu().item(), path, scale=5e-2)

    plots.show()

@torch.no_grad()
def render_images(model: VectorFieldNerf,
                  eval_path: str,
                  dataset_config: DatasetConfig,
                  epoch: int, 
                  split_size: int = 512,
                  device: torch.device = torch.device("cuda")) -> None:
    """
    Render all images using the model.
    :param model: The model.
    :param eval_path: The evaluation path.
    :param dataset_config: The dataset configuration.
    :param epoch: The epoch.
    :param split_size: The split size.
    :param device: The device.
    """

    # Create the dataset.
    dataset = dataset_dict[dataset_config.dataset_name](dataset_config)
    dataset.all_pixels = True
    model.ray_sampler.near, model.ray_sampler.far = dataset.get_bounds()
    if model.config.ray_sampler_config.fine_sampling():
            model.fine_sampler.near, model.fine_sampler.far = dataset.get_bounds()
    
    # Create the dataloader.
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    path = os.path.join(eval_path, "rendered_images")
    utils.mkdir_ifnotexists(path)

    # Render all images.
    for i, batch in enumerate(dataloader):
        # Get data
        all_pose = batch["pose"].squeeze(0)
        all_pixels = batch["uv"].squeeze(0)
        all_intrinsics = batch["intrinsics"].squeeze(0)

        # Split the pixels, pose, and intrinsics into batches.
        num_pixels = all_pixels.shape[0]
        num_batches = int(np.ceil(num_pixels / split_size))
        pixels_split = torch.split(all_pixels, split_size, dim=0)
        pose_split = torch.split(all_pose, split_size, dim=0)
        intrinsics_split = torch.split(all_intrinsics, split_size, dim=0)

        # Render the images.
        rgb = np.zeros((dataset.image_size[0], dataset.image_size[1], 3))
        rgb_fine = np.zeros((dataset.image_size[0], dataset.image_size[1], 3))
        depth_map = np.zeros((dataset.image_size[0], dataset.image_size[1], 1))
        depth_map_fine = np.zeros((dataset.image_size[0], dataset.image_size[1], 1))
        for j in range(num_batches):
            # Get the pixels, pose, and intrinsics for the current batch.
            pixels = pixels_split[j].to(device)
            pose = pose_split[j].to(device)
            intrinsics = intrinsics_split[j].to(device)
            # Forward pass
            output = model.render(pose, pixels, intrinsics, epoch, dataset.white_bkgd)
            rgb_values = output.coarse_rgb_values.cpu().numpy()
            predicted_depth = output.coarse_depth_map.cpu().numpy()
            if model.config.ray_sampler_config.fine_sampling() and output.fine_normals is not None:
                rgb_values_fine = output.fine_rgb_values.cpu().numpy()
                predicted_depth_fine = output.fine_depth_map.cpu().numpy()
                rgb_fine[pixels[:, 1].long().cpu().numpy(), pixels[:, 0].long().cpu().numpy(), :] = rgb_values_fine
                depth_map_fine[pixels[:, 1].long().cpu().numpy(), pixels[:, 0].long().cpu().numpy(),
                               :] = predicted_depth_fine

            # Update the rgb image.
            rgb[pixels[:, 1].long().cpu().numpy(), pixels[:, 0].long().cpu().numpy(), :] = rgb_values
            depth_map[pixels[:, 1].long().cpu().numpy(), pixels[:, 0].long().cpu().numpy(), :] = predicted_depth

        # Save the image.
        utils.save_rgb(os.path.join(path, f"image-{i}.png"), rgb)
        utils.save_depth(os.path.join(path, f"depth-{i}"), depth_map)
        if model.config.ray_sampler_config.fine_sampling() and output.fine_normals is not None:
            utils.save_rgb(os.path.join(path, f"image-fine-{i}.png"), rgb_fine)
            utils.save_depth(os.path.join(path, f"depth-fine-{i}"), depth_map_fine)

@torch.no_grad()
def metrics(model: VectorFieldNerf,
            eval_path: str,
            dataset_config: DatasetConfig,
            epoch: int, 
            split_size: int = 200,
            device: torch.device = torch.device("cuda")) -> None:
    """
    Compute the PSNR for images and MSE for depth maps.
    :param model: The model.
    :param eval_path: The evaluation path.
    :param dataset_config: The dataset configuration.
    :param epoch: The epoch.
    :param split_size: The split size.
    :param device: The device.
    """
    # Create the dataset.
    dataset = dataset_dict[dataset_config.dataset_name](dataset_config)
    dataset.all_pixels = True
    images_path = os.path.join(eval_path, "rendered_images")

    num_images = len(dataset)
    # num_images=89

    # Check if all images and depths maps exist.
    image_paths = [os.path.join(images_path, f"image-{i}.png") for i in range(0, num_images)]
    depth_paths = [os.path.join(images_path, f"depth-{i}.npy") for i in range(0, num_images)]
    if not all([os.path.exists(path) for path in image_paths]) or \
            not all([os.path.exists(path) for path in depth_paths]):
        print("Not all images and depth maps exist. Rendering images and depth maps.")
        render_images(model, eval_path, dataset_config, epoch, split_size, device)

    # Create the dataloader.
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # Compute the PSNR, SSIM, LPIPS, and MAE.
    psnr = torch.zeros(num_images)

    metrics_dict = dict()
    
    for i, batch in enumerate(dataloader):
        if i >= num_images:
            break
        target_image = batch["rgb"].squeeze(0).reshape(dataset.image_size[0], dataset.image_size[1], 3)
        target_depth = batch["depth"].squeeze(0)

        # Load the predicted image
        predicted_image = torch.from_numpy(utils.load_rgb(os.path.join(images_path, f"image-{i}.png"))).float().permute(1, 2, 0)
        # Load the predicted depth map.
        predicted_depth = torch.from_numpy(np.load(os.path.join(images_path, f"depth-{i}.npy"))).float().reshape(-1, 1)

        # Compute the PSNR
        psnr[i] = utils.get_psnr(predicted_image, target_image)

        # Update the metrics dictionary.
        metrics_dict[f"image-{i}"] = {"psnr": psnr[i].item()}

    # Save the metrics in a json file. Save values for all images and save the mean.
    metrics_dict.update({"mean_psnr": psnr.mean().item()})
    
    with open(os.path.join(eval_path, "metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=4)


def tsdf_mesh(eval_path: str, dataset_config: DatasetConfig) -> None:
    """
    Transform the depth maps into a TSDF mesh.
    :param eval_path: The evaluation path.
    :param dataset_config: The dataset configuration.
    """

    
    # Get the dataset.
    dataset = dataset_dict[dataset_config.dataset_name](dataset_config)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=4 / 512.0,
                                                          sdf_trunc=0.04,
                                                          color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    # Get the number of depth maps.
    images_path = os.path.join(eval_path, "rendered_images")
    file_names = os.listdir(images_path)
    num_depth_maps = len([file_name for file_name in file_names if file_name.endswith(".npy") and
                         file_name.startswith("depth")])

    for i in range(num_depth_maps):
        # Load the depth map.
        depth_map = np.load(os.path.join(images_path, f"depth-{i}.npy"))
        rgb_image = skimage.img_as_uint(imageio.imread(os.path.join(images_path, f"image-{i}.png")))

        # Get pose and intrinsics
        if dataset_config.dataset_name in ["dtu", "deepfashion"]:
            intrinsic = dataset.intrinsics[i].clone()
        else:
            intrinsic = dataset.intrinsics.clone()
        pose = dataset.poses[i].clone()

        # Construct the open3d intrinsic matrix.
        intrinsics_o3d = o3d.camera.PinholeCameraIntrinsic(depth_map.shape[1], depth_map.shape[0],
                                                           intrinsic[0, 0], intrinsic[1, 1],
                                                           intrinsic[0, 2], intrinsic[1, 2])
        # Generate open3d depth image in millimeters.
        depth_image_o3d = o3d.geometry.Image((depth_map * 1000).astype(np.uint16))

        # Generate rgbd image.
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(rgb_image.astype(np.uint8)),
                                                                        depth_image_o3d,
                                                                        convert_rgb_to_intensity=False,
                                                                        depth_trunc=10.0)
        volume.integrate(rgbd_image, intrinsics_o3d, np.linalg.inv(pose))

    # Save the volume.
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    mesh_path = os.path.join(eval_path, "tsdf-mesh")
    utils.mkdir_ifnotexists(mesh_path)
    o3d.io.write_triangle_mesh(os.path.join(mesh_path, "tsdf.ply"), mesh, False, True)

def metrics_3d(eval_path: str,
               dataset_config: DatasetConfig,
               num_points: int = 1000000,
               icp_align: bool = False,
               distance_thresh: float = 0.01) -> None:
    """
    Compute the 3D metrics.
    :param eval_path: The evaluation path.
    :param dataset_config: The dataset configuration.
    :param num_points: The number of points.
    :param icp_align: Whether to use ICP alignment.
    :param distance_thresh: The distance threshold.
    """

    # Check if the TSDF mesh exists.
    if not os.path.exists(os.path.join(eval_path, "tsdf-mesh", "tsdf.ply")):
        print("TSDF mesh does not exist. Generating mesh.")
        tsdf_mesh(eval_path, dataset_config)

    if not os.path.exists(os.path.join(eval_path, "tsdf-mesh", "tsdf-smoothed.ply")):
        print("TSDF smooth mesh does not exist. Generating mesh.")
        tsdf_o3d = o3d.io.read_triangle_mesh(os.path.join(eval_path, "tsdf-mesh", "tsdf.ply"))
        tsdf_o3d.compute_vertex_normals()
        tsdf_o3d = tsdf_o3d.filter_smooth_laplacian(number_of_iterations=10)
        o3d.io.write_triangle_mesh(os.path.join(eval_path, "tsdf-mesh", "tsdf-smoothed.ply"), tsdf_o3d)

    tsdf_msh = trimesh.load(os.path.join(eval_path, "tsdf-mesh", "tsdf.ply"))
    tsdf_smoothed_msh = trimesh.load(os.path.join(eval_path, "tsdf-mesh", "tsdf-smoothed.ply"))
    gt_msh = trimesh.load(os.path.join(dataset_config.data_root_dir, dataset_config.data_dir, f"{dataset_config.scene}_mesh.ply"))

    if not os.path.exists(os.path.join(eval_path, "tsdf-mesh", f"refused-tsdf.ply")):
        # Create postprocessed mesh.
        refused_mesh = refuse(tsdf_msh, dataset_config)
        refused_mesh.export(os.path.join(eval_path, "tsdf-mesh", f"refused-tsdf.ply"))
    else:
        refused_mesh = trimesh.load(os.path.join(eval_path, "tsdf-mesh", f"refused-tsdf.ply"))

    if not os.path.exists(os.path.join(eval_path, "tsdf-mesh", f"refused-tsdf-smoothed.ply")):
        # Create postprocessed mesh.
        refused_mesh_smoothed = refuse(tsdf_smoothed_msh, dataset_config)
        refused_mesh_smoothed.export(os.path.join(eval_path, "tsdf-mesh", f"refused-tsdf-smoothed.ply"))
    else:
        refused_mesh_smoothed = trimesh.load(os.path.join(eval_path, "tsdf-mesh", f"refused-tsdf-smoothed.ply"))

    # Compute the Chamfer distances.
    mean_tsdf_chf, median_tsdf_chf, min_tsdf_chf, max_tsdf_chf = utils.get_chamfer_distance(tsdf_msh, gt_msh, num_points)
    mean_refused_chf, median_refused_chf, min_refused_chf, max_refused_chf = utils.get_chamfer_distance(refused_mesh, gt_msh, num_points)
    mean_tsdf_chf_smoothed, median_tsdf_chf_smoothed, min_tsdf_chf_smoothed, max_tsdf_chf_smoothed = utils.get_chamfer_distance(tsdf_smoothed_msh, gt_msh, num_points)
    mean_refused_chf_smoothed, median_refused_chf_smoothed, min_refused_chf_smoothed, max_refused_chf_smoothed = utils.get_chamfer_distance(refused_mesh_smoothed, gt_msh, num_points)

    # Compute the precision, recall, and f-score.
    scene = os.path.join(os.getcwd(),
                         dataset_config.data_root_dir, 
                         dataset_config.data_dir, 
                         f"{dataset_config.scene}_mesh.ply")
    results_tsdf: dict = run_evaluation("tsdf.ply", os.path.join(os.getcwd(), eval_path, "tsdf-mesh"), 
                                        f"{dataset_config.scene}_mesh", full_path_to_gt_ply=scene, icp_align=icp_align, distance_thresh=distance_thresh)
    results_tsdf_smoothed: dict = run_evaluation("tsdf-smoothed.ply", os.path.join(os.getcwd(), eval_path, "tsdf-mesh"),
                                                 f"{dataset_config.scene}_mesh", full_path_to_gt_ply=scene, icp_align=icp_align, distance_thresh=distance_thresh)
    
    results_refused: dict = run_evaluation(f"refused-tsdf.ply", os.path.join(os.getcwd(), eval_path, "tsdf-mesh"),
                                           f"{dataset_config.scene}_mesh", full_path_to_gt_ply=scene, icp_align=icp_align, distance_thresh=distance_thresh)
    results_refused_smoothed: dict = run_evaluation(f"refused-tsdf-smoothed.ply", os.path.join(os.getcwd(), eval_path, "tsdf-mesh"),
                                                    f"{dataset_config.scene}_mesh", full_path_to_gt_ply=scene, icp_align=icp_align, distance_thresh=distance_thresh)

    metrics_dict = {"tsdf": {"chamfer distance": {"mean": mean_tsdf_chf, "median": median_tsdf_chf, "min": min_tsdf_chf, "max": max_tsdf_chf}},
                    # "mc_smoothed": {"chamfer distance": {"mean": mean_mc_chf_smoothed, "median": median_mc_chf_smoothed, "min": min_mc_chf_smoothed, "max": max_mc_chf_smoothed}},
                    "refused_tsdf": {"chamfer distance": {"mean": mean_refused_chf, "median": median_refused_chf, "min": min_refused_chf, "max": max_refused_chf}},
                    "tsdf_smoothed": {"chamfer distance": {"mean": mean_tsdf_chf_smoothed, "median": median_tsdf_chf_smoothed, "min": min_tsdf_chf_smoothed, "max": max_tsdf_chf_smoothed}},
                    "refused_tsdf_smoothed": {"chamfer distance": {"mean": mean_refused_chf_smoothed, "median": median_refused_chf_smoothed, "min": min_refused_chf_smoothed, "max": max_refused_chf_smoothed}}}
    
    metrics_dict["tsdf"].update(results_tsdf)
    metrics_dict["tsdf_smoothed"].update(results_tsdf_smoothed)
    metrics_dict["refused_tsdf"].update(results_refused)
    metrics_dict["refused_tsdf_smoothed"].update(results_refused_smoothed)

    with open(os.path.join(eval_path, "3d-metrics.json"), "w") as f:
         json.dump(metrics_dict, f, indent=4)                               


def metrics_3d_no_vf(eval_path: str,
                     checkpoint: int,
                     dataset_config: DatasetConfig,
                     num_points: int = 1000000,
                     icp_align: bool = True,
                     distance_thresh: float = 0.05) -> None:
    """
    Compute the 3D metrics.
    :param eval_path: The evaluation path.
    :param checkpoint: The checkpoint.
    :param dataset_config: The dataset configuration.
    :param num_points: The number of points.
    :param icp_align: Whether to use ICP alignment.
    :param distance_thresh: The distance threshold.
    """

    # Check if marchin cubes mesh exists.
    if not os.path.exists(os.path.join(eval_path, "mesh", f"mesh-scaled-{checkpoint}.ply")):
        raise FileExistsError("Marching cubes mesh does not exist. Please generate the mesh first.")
    
    # Load the predicted meshes and the ground truth mesh.
    mc_msh = trimesh.load(os.path.join(eval_path, "mesh", f"mesh-scaled-{checkpoint}.ply"))
    gt_msh = trimesh.load(os.path.join(dataset_config.data_root_dir, dataset_config.data_dir, f"{dataset_config.scene}_mesh.ply"))

    if not os.path.exists(os.path.join(eval_path, "mesh", f"refused-mesh-{checkpoint}.ply")):
        # Create postprocessed mesh.
        refused_mesh = refuse(mc_msh, dataset_config)
        refused_mesh.export(os.path.join(eval_path, "mesh", f"refused-mesh-{checkpoint}.ply"))
    else:
        refused_mesh = trimesh.load(os.path.join(eval_path, "mesh", f"refused-mesh-{checkpoint}.ply"))

    # Compute the Chamfer distances.
    mean_mc_chf, median_mc_chf, min_mc_chf, max_mc_chf = utils.get_chamfer_distance(mc_msh, gt_msh, num_points)
    mean_refused_chf, median_refused_chf, min_refused_chf, max_refused_chf = utils.get_chamfer_distance(refused_mesh, gt_msh, num_points)

    # Compute the precision, recall, and f-score.
    scene = os.path.join(os.getcwd(),
                         dataset_config.data_root_dir, 
                         dataset_config.data_dir, 
                         f"{dataset_config.scene}_mesh.ply")
    
    results_mc: dict = run_evaluation(f"mesh-scaled-{checkpoint}.ply", os.path.join(os.getcwd(), eval_path, "mesh"),
                                      f"{dataset_config.scene}_mesh", full_path_to_gt_ply=scene, icp_align=icp_align, distance_thresh=distance_thresh)    
    results_refused: dict = run_evaluation(f"refused-mesh-{checkpoint}.ply", os.path.join(os.getcwd(), eval_path, "mesh"),
                                           f"{dataset_config.scene}_mesh", full_path_to_gt_ply=scene, icp_align=icp_align, distance_thresh=distance_thresh)

    # Save the metrics.
    metrics_dict = {"mc": {"chamfer distance": {"mean": mean_mc_chf, "median": median_mc_chf, "min": min_mc_chf, "max": max_mc_chf}},
                    "refused": {"chamfer distance": {"mean": mean_refused_chf, "median": median_refused_chf, "min": min_refused_chf, "max": max_refused_chf}}}
    
    metrics_dict["mc"].update(results_mc)
    metrics_dict["refused"].update(results_refused)

    with open(os.path.join(eval_path, "3d-metrics.json"), "w") as f:
         json.dump(metrics_dict, f, indent=4)                               
