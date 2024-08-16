import os
import sys
from typing import List, Optional, Tuple

sys.path.append('.')  # isort:skip

import torch
import torch.nn.functional as F
from torch import nn

from config_parser.vf_nerf_config import VFNerfConfig
import models.helpers.functions as functions
from models.helpers.density_functions import LaplaceDensity
from models.nerf.output import NerfOutput
from models.samplers.ray_sampler import UniformSampler, RangeFineSampler
from models.vector_field.rendering_network import RenderingNetwork
from models.vector_field.vector_field_network import VectorFieldNetwork
import utils.rendering as rendering
from utils.weight_annealing import LinearAnnealing



class VectorFieldNerf():
    def __init__(self, config: VFNerfConfig) -> None:
        """
        VectorFieldNerf class.
        :param config: The config.
        """ 
        self.config = config

        # Initialize the vf network
        self.vector_field_network = VectorFieldNetwork(config.vf_net_config)
        # If we use the fine sampler, initialize the fine vf network
        if config.ray_sampler_config.fine_sampling():
            # self.fine_vector_field_network = VectorFieldNetwork(config.vf_net_config)
            self.fine_vector_field_network = self.vector_field_network
        
        # Initialize the rendering network.
        self.rendering_network = RenderingNetwork(config.rendering_net_config)

        # Create the ray samplers
        self.ray_sampler = UniformSampler(config.ray_sampler_config.n_samples,
                                          config.ray_sampler_config.near,
                                          config.ray_sampler_config.far,
                                          (not config.ray_sampler_config.perturb))
        if config.ray_sampler_config.fine_sampling():
            self.fine_sampler = RangeFineSampler(config.ray_sampler_config.n_importance,
                                                 config.ray_sampler_config.near,
                                                 config.ray_sampler_config.far,
                                                 (not config.ray_sampler_config.perturb),
                                                 range=config.ray_sampler_config.fine_range,
                                                 max_samples=config.ray_sampler_config.max_samples
                                                 )
            
        # Initialize the density function
        self.density = LaplaceDensity(**config.density_config.todict())
        if self.config.cos_sim_weights_anneal != "none":
            self.annealing = LinearAnnealing(self.config.cos_sim_weights.shape[0],
                                             self.config.anneal_end - self.config.anneal_start,
                                             self.config.cos_sim_weights_anneal == "soft")
                                             
        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.scheduler_config.lr, weight_decay=config.scheduler_config.weight_decay)
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                                self.config.scheduler_config.lr_decay_factor **
                                                                (1. / self.config.scheduler_config.lr_decay_steps)) 

        # Make networks parallel
        if self.config.cuda_config.num_gpus > 1:
            self.vector_field_network = nn.DataParallel(self.vector_field_network)
            self.rendering_network = nn.DataParallel(self.rendering_network)
            self.density = nn.DataParallel(self.density)
            if self.config.ray_sampler_config.fine_sampling():
                self.fine_vector_field_network = nn.DataParallel(self.fine_vector_field_network)
        
        # Move to device.
        self.vector_field_network.to(config.cuda_config.device)
        self.rendering_network.to(config.cuda_config.device)
        self.density.to(config.cuda_config.device)
        if self.config.ray_sampler_config.fine_sampling():
            self.fine_vector_field_network.to(config.cuda_config.device)

    def cpu(self) -> None:
        """
        Move the model to cpu.
        """
        self.vector_field_network.cpu()
        self.rendering_network.cpu()
        self.density.cpu()
        if self.config.ray_sampler_config.fine_sampling():
            self.fine_vector_field_network.cpu()

    def to(self, device: torch.device) -> None:
        """
        Move the model to a device.
        :param device: The device.
        """
        self.vector_field_network.to(device)
        self.rendering_network.to(device)
        self.density.to(device)
        if self.config.ray_sampler_config.fine_sampling():
            self.fine_vector_field_network.to(device)

    def new_scheduler(self, num_steps: int) -> None:
        """
        Create a new scheduler.
        :param num_steps: The number of steps.
        """
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                                self.config.scheduler_config.lr_decay_factor **
                                                                (1. / num_steps))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config.scheduler_config.lr)

    def reset_scheduler(self, num_steps: Optional[int] = None) -> None:
        """
        Reset the scheduler.
        :param num_steps: The number of steps.
        """
        if num_steps is None:
            num_steps = self.config.scheduler_config.lr_decay_steps
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                                self.config.scheduler_config.lr_decay_factor **
                                                                (1. / num_steps))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config.scheduler_config.lr)

    def parameters(self) -> List[nn.Parameter]:
        """
        Get the parameters of the model.
        :return: The parameters of the model.
        """
        params = list(self.vector_field_network.parameters()) + \
                    list(self.rendering_network.parameters()) + \
                    list(self.density.parameters())
        if self.config.ray_sampler_config.fine_sampling():
            params += list(self.fine_vector_field_network.parameters())
        return params

    def train(self) -> None:
        """
        Set the model to training mode.
        """
        if self.config.numerical_jacobian:
            self.vector_field_network.eval()
        else:
            self.vector_field_network.train()
        self.rendering_network.train()
        self.density.train()
        if self.config.ray_sampler_config.fine_sampling():
            self.fine_vector_field_network.train()

    def eval(self) -> None:
        """
        Set the model to evaluation mode.
        """
        self.vector_field_network.eval()
        self.rendering_network.eval()
        self.density.eval()
        if self.config.ray_sampler_config.fine_sampling():
            self.fine_vector_field_network.eval()

    def load(self, path: str) -> int:
        """
        Load the model from a path.
        :param path: The path to load the model from.
        :return: The epoch of the model.
        """

        # Load the model.
        checkpoint = torch.load(path, map_location=self.config.cuda_config.device)

        # Load the sdf network.
        self.vector_field_network.load_state_dict(checkpoint['vf_net'])

        # Load the sdf rendering network.
        self.rendering_network.load_state_dict(checkpoint['rendering_net'])

        # Load the density function.
        self.density.load_state_dict(checkpoint['density'])

        # Load the epoch.
        epoch = checkpoint['epoch'] + 1

        # Load the optimizer.
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        # Load the scheduler.
        self.scheduler.load_state_dict(checkpoint['scheduler'])

        # If we use the fine sampler, load the fine vf network
        if self.config.ray_sampler_config.fine_sampling() and 'fine_vf_net' in checkpoint.keys():
            self.fine_vector_field_network.load_state_dict(checkpoint['fine_vf_net'])

        return epoch
    
    def save(self, epoch: int, path: str) -> None:
        """
        Save the model to a path.
        :param epoch: The epoch of the model.
        :param path: The path to save the model to.
        """
        state_dict = {
            'vf_net': self.vector_field_network.state_dict(),
            'rendering_net': self.rendering_network.state_dict(),
            'density': self.density.state_dict(),
            'epoch': epoch,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        } 
        if self.config.ray_sampler_config.fine_sampling():
            state_dict['fine_vf_net'] = self.fine_vector_field_network.state_dict()

        torch.save(state_dict, os.path.join(path, f"{epoch}.pth"))
        torch.save(state_dict, os.path.join(path, "latest.pth"))

    def render(self,
               pose: torch.Tensor,
               pixels: torch.Tensor,
               intrinsics: torch.Tensor,
               epoch: int,
               white: bool = False) -> NerfOutput:
        """
        Render the scene.
        :params pose: The pose.
        :params pixels: The pixels.
        :params intrinsics: The intrinsics.
        :params white: Whether to use white background.
        :returns: The rendered output.
        """

        # Anneal the weights
        if self.config.cos_sim_weights_anneal != "none" and epoch > self.config.anneal_start:
            self.config.cos_sim_weights = self.annealing.get_weights(epoch - self.config.anneal_start,
                                                                     pose.device)
        # Get the number of pixels.
        num_pixels = pixels.shape[0]

        # Get the ray directions and the camera locations.
        directions, ray_dirs, cam_loc = rendering.get_ray_directions_and_cam_location(
                pixels, pose, intrinsics, device=pose.device)
        ray_dirs = ray_dirs.reshape(-1, 3)
        cam_loc = cam_loc.reshape(-1, 3)
        directions = directions.reshape(-1, 3)
        # Sample the rays.
        points_coarse, z_vals = self.ray_sampler.sample(directions, cam_loc, device=pose.device)
        points_coarse_flat = points_coarse.reshape(-1, 3)

        # Repeat the dirs to match the number of samples.
        repeated_ray_dirs = ray_dirs.unsqueeze(1).repeat(1, self.ray_sampler.N_samples, 1).reshape(-1, 3)

        # Compute the normal vectors and feature vectors.
        with torch.no_grad():
            vf_output = self.vector_field_network(points_coarse_flat)
            normals_coarse_flat, feature_vectors_coarse, jacobian = \
                vf_output[:, :3], vf_output[:, 3:(self.config.vf_net_config.feature_vector_dims+3)], vf_output[:, self.config.vf_net_config.feature_vector_dims+3:]
            normals_coarse = normals_coarse_flat.reshape(z_vals.shape[0], z_vals.shape[1], 3)

            # Compute directional derivatives
            if self.config.numerical_jacobian:
                dir_derivatives = self.compute_numerical_directional_derivatives(points_coarse_flat, normals_coarse_flat).reshape(-1, 3)
            elif self.vector_field_network.training:
                dir_derivatives = self.compute_directional_derivatives(points_coarse_flat, normals_coarse_flat, jacobian).reshape(-1, 3)
            else:
                dir_derivatives = None

            density = self.get_density(normals_coarse, repeated_ray_dirs)
            # Compute the weights.
            if self.config.rendering == "volsdf":
                weights_coarse = rendering.volsdf_volume_rendering(z_vals, density, self.config.normalize_rendering)
            elif self.config.rendering == "nerf":
                weights_coarse = rendering.nerf_volume_rendering(z_vals, density, self.config.normalize_rendering)

            # If white background, set the background color.
            if white:
                acc_map_coarse = torch.sum(weights_coarse, -1)
                rgb_values_coarse = rgb_values_coarse + \
                    (1. - acc_map_coarse[..., None])

        # FINE NETWORK PART
        rgb_values_fine = None
        depth_map_fine = None
        normals_fine = None
        points_fine = None
        # If we use the fine sampler, sample the fine points.
        if self.config.ray_sampler_config.fine_sampling():
            N_fine_samples = min(self.fine_sampler.N_samples, self.fine_sampler.max_samples)
            points_coarse, z_vals = self.fine_sampler.sample(directions, cam_loc, device=pose.device,
                                                             coarse_z_vals=z_vals, coarse_weights=weights_coarse)
            points_coarse_flat = points_coarse.reshape(-1, 3)
            repeated_ray_dirs = ray_dirs.unsqueeze(1).repeat(
                1, N_fine_samples + self.ray_sampler.N_samples, 1).reshape(-1, 3)

            # Predict the fine normals and feature vectors.
            vf_output = self.fine_vector_field_network(points_coarse_flat)
            normals_coarse_flat, feature_vectors_coarse, jacobian_coarse = \
                vf_output[:, :3], vf_output[:, 3:(self.config.vf_net_config.feature_vector_dims+3)], vf_output[:, self.config.vf_net_config.feature_vector_dims+3:]
            normals_coarse = normals_coarse_flat.reshape(z_vals.shape[0], N_fine_samples + self.ray_sampler.N_samples, 3)
            
            # Compute directional derivatives
            if self.config.numerical_jacobian:
                coarse_dir_derivatives = self.compute_numerical_directional_derivatives(points_coarse_flat, normals_coarse_flat, fine=True).reshape(-1, 3)
                dir_derivatives = torch.cat([dir_derivatives, coarse_dir_derivatives], dim=0)
            elif self.vector_field_network.training:
                coarse_dir_derivatives = self.compute_directional_derivatives(points_coarse_flat, normals_coarse_flat, jacobian_coarse).reshape(-1, 3)
                dir_derivatives = torch.cat([dir_derivatives, dir_derivatives], dim=0)

            # Render the fine weights.
            density = self.get_density(normals_coarse, repeated_ray_dirs, True)
            if self.config.rendering == "volsdf":
                weights_coarse= rendering.volsdf_volume_rendering(z_vals, density, self.config.normalize_rendering)
            elif self.config.rendering == "nerf":
                weights_coarse = rendering.nerf_volume_rendering(z_vals, density, self.config.normalize_rendering)

            repeated_ray_dirs = repeated_ray_dirs 
            predicted_rgb_coarse = self.rendering_network(
                points_coarse_flat, normals_coarse_flat, repeated_ray_dirs, feature_vectors_coarse)
            N = self.ray_sampler.N_samples + N_fine_samples
            predicted_rgb_coarse = predicted_rgb_coarse.reshape(num_pixels, N, 3)

            # Compute the final RGB values.
            # Compute the depth map.
            rgb_values_coarse = torch.sum(weights_coarse.unsqueeze(-1) * predicted_rgb_coarse, dim=1)
            depth_map_coarse = torch.sum(weights_coarse.unsqueeze(-1) * z_vals.unsqueeze(-1), dim=1)

            # If white background, set the background color.
            if white:
                acc_map_coarse = torch.sum(weights_coarse, -1)
                rgb_values_coarse = rgb_values_coarse + \
                    (1. - acc_map_coarse[..., None])

        return NerfOutput(points_coarse=points_coarse, points_fine=points_fine,
                          coarse_normals=normals_coarse, coarse_rgb_values=rgb_values_coarse,
                          coarse_depth_map=depth_map_coarse, fine_normals=normals_fine,
                          fine_rgb_values=rgb_values_fine, fine_depth_map=depth_map_fine,
                          z_vals=z_vals,
                          directional_derivtives=dir_derivatives.norm(dim=-1) if dir_derivatives is not None else None,
                          ray_dirs=repeated_ray_dirs,
                          coarse_colors=predicted_rgb_coarse.reshape(-1, 3))
    

    def get_colors(self, pose: torch.Tensor, pixels: torch.Tensor, 
                   intrinsics: torch.Tensor, epoch: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the colors.
        :params pose: The pose.
        :params pixels: The pixels.
        :params intrinsics: The intrinsics.
        :returns: The colors.
        """

        # Anneal the weights
        if self.config.cos_sim_weights_anneal != "none" and epoch > self.config.anneal_start:
            self.config.cos_sim_weights = self.annealing.get_weights(epoch - self.config.anneal_start,
                                                                     pose.device)
                  
        # Get the ray directions and the camera locations.
        directions, ray_dirs, cam_loc = rendering.get_ray_directions_and_cam_location(
                pixels, pose, intrinsics, device=pose.device)
        ray_dirs = ray_dirs.reshape(-1, 3)
        cam_loc = cam_loc.reshape(-1, 3)
        directions = directions.reshape(-1, 3)
        # Sample the rays.
        points_coarse, _ = self.ray_sampler.sample(directions, cam_loc, device=pose.device)
        points_coarse_flat = points_coarse.reshape(-1, 3)

        # Repeat the dirs to match the number of samples.
        repeated_ray_dirs = ray_dirs.unsqueeze(1).repeat(1, self.ray_sampler.N_samples, 1).reshape(-1, 3)

        # Compute the normal vectors and feature vectors.
        vf_output = self.vector_field_network(points_coarse_flat)
        normals_coarse_flat, feature_vectors_coarse = \
            vf_output[:, :3], vf_output[:, 3:(self.config.vf_net_config.feature_vector_dims+3)]

        # Predict the coarse RGB values.
        predicted_rgb_coarse = self.rendering_network(
            points_coarse_flat, normals_coarse_flat, repeated_ray_dirs, feature_vectors_coarse)
        
        return predicted_rgb_coarse, points_coarse_flat, repeated_ray_dirs
    
    def get_vector_field(self, pose: torch.Tensor, pixels: torch.Tensor, 
                         intrinsics: torch.Tensor) -> torch.Tensor:
        """
        Compute the vector field.
        :params pose: The pose.
        :params pixels: The pixels.
        :params intrinsics: The intrinsics.
        :returns: The vector field.
        """

        # Get the ray directions and the camera locations.
        directions, ray_dirs, cam_loc = rendering.get_ray_directions_and_cam_location(
                pixels, pose, intrinsics, device=pose.device)
        ray_dirs = ray_dirs.reshape(-1, 3)
        cam_loc = cam_loc.reshape(-1, 3)
        directions = directions.reshape(-1, 3)
        # Sample the rays.
        points_coarse, _ = self.ray_sampler.sample(directions, cam_loc, device=pose.device)
        points_coarse_flat = points_coarse.reshape(-1, 3)

        # Compute the normal vectors and feature vectors.
        vf_output = self.vector_field_network(points_coarse_flat)

        return vf_output[:, :3]
    
    def get_weights_and_color(self, points: torch.Tensor, 
                              repeated_ray_dirs: torch.Tensor, 
                              z_vals: torch.Tensor, epoch: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the weights.
        :params pose: The pose.
        :params pixels: The pixels.
        :params intrinsics: The intrinsics.
        :returns: The weights.
        """

        # Anneal the weights
        if self.config.cos_sim_weights_anneal != "none" and epoch > self.config.anneal_start:
            self.config.cos_sim_weights = self.annealing.get_weights(epoch - self.config.anneal_start,
                                                                     points.device)
                  
        points_coarse_flat = points.reshape(-1, 3)

        # Compute the normal vectors and feature vectors.
        vf_output = self.vector_field_network(points_coarse_flat)
        normals_coarse_flat, feature_vectors_coarse = \
            vf_output[:, :3], vf_output[:, 3:(self.config.vf_net_config.feature_vector_dims+3)]
        normals_coarse = normals_coarse_flat.reshape(z_vals.shape[0], z_vals.shape[1], 3)

        density = self.get_density(normals_coarse, repeated_ray_dirs, True)
        # Compute the weights.
        if self.config.rendering == "volsdf":
            weights_coarse = rendering.volsdf_volume_rendering(z_vals, density, self.config.normalize_rendering)
        elif self.config.rendering == "nerf":
            weights_coarse = rendering.nerf_volume_rendering(z_vals, density, self.config.normalize_rendering)

        # Predict the coarse RGB values.
        predicted_rgb_coarse = self.rendering_network(
            points_coarse_flat, normals_coarse_flat, repeated_ray_dirs, feature_vectors_coarse)

        return weights_coarse, predicted_rgb_coarse

    def get_density(self,
                    normals: torch.Tensor,
                    ray_dirs: torch.Tensor,
                    fine: bool = False) -> torch.Tensor:
        """
        Get the density.
        :params normals: The normals.
        :params ray_dirs: The ray directions.
        :returns: The density.
        """
        # Compute the cosine similarity between the vectors.
        cos_sim_weights = torch.ones_like(self.config.cos_sim_weights) / self.config.cos_sim_weights.shape[0]
        if self.config.cos_sim_weights_anneal == "anneal_fine" and fine:
            cos_sim_weights = self.config.cos_sim_weights
        cosine_similarity = functions.window_cosine_similarity(normals[:, :-1, :],
                                                               normals[:, 1:, :],
                                                               cos_sim_weights)
        
        # Compute the cosine similarity between the normal vectors and the ray directions.
        cosine_similarity_ray_dirs = F.cosine_similarity(normals[:, :-1, :],
                                                         ray_dirs.reshape(normals.shape[0], normals.shape[1], 3)[:, :-1, :], 
                                                         dim=2)
        indices = \
            torch.where(torch.logical_and(cosine_similarity_ray_dirs < self.config.dir_to_normal_th,
                                          cosine_similarity < 0))
        
        density = self.density(-cosine_similarity.reshape(-1, 1), 
                               cutoff=self.config.density_config.cutoff).reshape(normals.shape[0], normals.shape[1] - 1).clone()
        density[indices[0], indices[1]] = 0.0
        # Consider last point to have cosine similarity of 1 and compute last density, which is 0
        density = torch.cat([density, torch.zeros(density.shape[0], 1).to(density.device)], dim=-1)

        return density
    
    def compute_directional_derivatives(self, points: torch.Tensor, normals: torch.Tensor, jac: torch.Tensor) -> torch.Tensor:
        """
        Compute the directional derivatives.
        :params points: The points.
        :params normals: The normals.
        :params jac: The jacobian (num_points, 9).
        :returns: The directional derivatives.
        """
        # Reshape jac to (num_points, 3, 3).
        jac = jac.reshape(-1, 3, 3)
        
        # Compute two normal vectors wrt to each normal.
        normals_1 = torch.zeros_like(normals)
        normals_1[:, 0] = normals[:, 1]
        normals_1[:, 1] = -normals[:, 0]
        normals_2 = torch.cross(normals, normals_1, dim=-1)

        # Compute the directional derivatives.
        dir_derivatives = torch.zeros((points.shape[0], 2, 3)).float().to(points.device)
        dir_derivatives[:, 0, :] = torch.bmm(jac, F.normalize(normals_1, dim=-1).unsqueeze(-1)).squeeze(-1)
        dir_derivatives[:, 1, :] = torch.bmm(jac, F.normalize(normals_2, dim=-1).unsqueeze(-1)).squeeze(-1)

        return dir_derivatives
    
    def compute_numerical_directional_derivatives(self, points: torch.Tensor, normals: torch.Tensor,
                                                  epsilon: float = 1e-5, fine: bool = False) -> torch.Tensor:
        """
        Compute the numerical directional derivatives.
        :params points: The points.
        :params normals: The normals.
        :params epsilon: The epsilon.
        :returns: The numerical directional derivatives.
        """

        # Compute the numerical jacobian.
        jacobian = torch.zeros(points.shape[0], 3, 3).to(points.device)
        for i in range(3):
            pos_h = points.clone()
            pos_h[:, i] += epsilon
            neg_h = points.clone()
            neg_h[:, i] -= epsilon
            
            if fine:
                jacobian[:, i] = (self.fine_vector_field_network(pos_h)[:, :3] - \
                                  self.fine_vector_field_network(neg_h)[:, :3]) / (2.0 * epsilon)
            else:
                jacobian[:, :, i] = (self.vector_field_network(pos_h)[:, :3] - \
                                     self.vector_field_network(neg_h)[:, :3]) / (2.0 * epsilon)

        # Compute the directional derivatives.
        return self.compute_directional_derivatives(points, normals, jacobian)
        