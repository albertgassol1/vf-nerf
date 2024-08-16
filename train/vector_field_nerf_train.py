import os
import shutil
import sys
from typing import Dict, Optional

sys.path.append('.')  # isort:skip

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data.distributed
import tqdm

import models.helpers.functions as functions
import utils.utils as utils
import wandb
from config_parser.vf_nerf_config import VFRunnerConfig
from datasets.normal_datasets import dataset_dict
from models.losses.vf_loss import VFLoss
from models.nerf.vector_field_nerf import VectorFieldNerf


class VectorFieldNerfRunner():
    def __init__(self, config: VFRunnerConfig) -> None:
        """
        VectorFieldNerf class.
        :params config: The config.
        """
        # Set the config.
        self.config = config

        # Set seed and default type.
        utils.set_seed_and_default()

        # Create the dataset.
        self.dataset = dataset_dict[config.dataset_config.dataset_name](self.config.dataset_config)
        if self.config.dataset_config.dataset_name == "deepfashion":
            self.config.vf_nerf_config.center_supervision = False
        config.vf_nerf_config.scheduler_config.lr_decay_steps = config.num_epochs * len(self.dataset)

        # Create the IBRNerf
        self.model = VectorFieldNerf(config.vf_nerf_config)
        self.model.ray_sampler.near, self.model.ray_sampler.far = self.dataset.get_bounds()
        if self.model.config.ray_sampler_config.fine_sampling():
            self.model.fine_sampler.near, self.model.fine_sampler.far = self.dataset.get_bounds()

        self.model.vector_field_network.init, load_path = self.dataset.get_vf_init_method()
        self.model.vector_field_network.load_init(load_path, self.config.vf_nerf_config.cuda_config.device)

        # Create the dataloader.
        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                      batch_size=1,
                                                      shuffle=True)

        # Create the losses.
        self.loss = VFLoss(self.config.vf_loss_config, self.config.vf_loss_weights)

        # Create the output folders.
        self.create_output_folders()

        # Load the model.
        self.load_model()

        # Set to train mode.
        self.model.train()

        # Print number of detected GPUs.
        print(f"Number of GPUs: {self.config.vf_nerf_config.cuda_config.num_gpus}")

        # Create the wandb run.
        if not self.config.offline:
            self.wandb_run = wandb.init(project=self.config.wandb_project,
                                        name=self.config.expname,
                                        config=self.config,
                                        resume="allow",
                                        id=self.config.timestamp,
                                        mode=utils.wandb_mode())

    def create_output_folders(self) -> None:
        """
        Create the output folders.
        """
        # Create the experiment folder.
        utils.mkdir_ifnotexists(self.config.exps_folder)

        utils.mkdir_ifnotexists(os.path.join(self.config.exps_folder,
                                             self.config.expname))
        if self.config.timestamp == "":
            self.config.timestamp = utils.get_timestamp()

        utils.mkdir_ifnotexists(os.path.join(self.config.exps_folder,
                                             self.config.expname,
                                             self.config.timestamp))

        utils.mkdir_ifnotexists(os.path.join(self.config.exps_folder,
                                             self.config.expname,
                                             self.config.timestamp,
                                             "checkpoints"))
        utils.mkdir_ifnotexists(os.path.join(self.config.exps_folder,
                                             self.config.expname,
                                             self.config.timestamp,
                                             "checkpoints",
                                             "vf_nerf"))

        # Copy the configuration folder.
        conf_path = os.path.join(self.config.exps_folder,
                                 self.config.expname,
                                 self.config.timestamp, 'vf_nerf.conf')
        if not os.path.exists(conf_path):
            shutil.copy2(self.config.config_path, conf_path)

        # Copy the shell command.
        print(f"shell command : {sys.argv}")

    def load_model(self) -> None:
        """
        Load the model.
        """
        if self.config.checkpoint != "":
            path = os.path.join(self.config.exps_folder,
                                self.config.expname,
                                self.config.timestamp,
                                "checkpoints",
                                "vf_nerf",
                                f"{self.config.checkpoint}.pth")
            if os.path.exists(path):
                self.config.start_epoch = self.model.load(path) + 1
                if self.model.config.ray_sampler_config.fine_sampling():
                    self.model.fine_sampler.N_samples = \
                        min(self.model.fine_sampler.N_samples + int(5*(self.config.start_epoch // self.model.config.ray_sampler_config.increase_every)), 
                            self.model.fine_sampler.max_samples)
                print(f"Loaded model from {self.config.checkpoint}")
            else:
                raise FileExistsError(f"Checkpoint path: {path} does not exist.")

    def train(self) -> None:
        """
        Train the model.
        """
        if self.config.vf_loss_weights.directional_derivatives == 0.0:
            self.model.eval()
        with tqdm.tqdm(range(self.config.start_epoch, self.config.num_epochs), desc="Epochs") as pbar:
            for epoch in pbar:
                # Train the model for one epoch.
                self.dataset.sample_new_images()
                if self.model.config.ray_sampler_config.fine_sampling() and epoch % self.model.config.ray_sampler_config.increase_every == 0:
                    self.model.fine_sampler.N_samples = self.model.fine_sampler.N_samples + 5
                loss = self.train_epoch(epoch)

                # Save the model.
                if epoch % self.config.save_frequency == 0:
                    self.model.save(epoch,
                                    os.path.join(self.config.exps_folder,
                                                self.config.expname,
                                                self.config.timestamp,
                                                "checkpoints",
                                                "vf_nerf"))
                pbar.set_description(f"Epoch {epoch}:  Loss {loss}")
        self.config.start_epoch = self.config.num_epochs + 1

    def train_epoch(self, epoch: int) -> float:
        """
        Train the model for one epoch.
        :param epoch: The epoch.
        :return: The average loss.
        """
        average_losses: Optional[Dict[str, float]] = None

        for i, train_data in enumerate(self.dataloader):

            # Get the pixels, intrinsics and pose
            pixels = train_data["uv"].squeeze(0).to(self.config.vf_nerf_config.cuda_config.device)
            intrinsics = train_data["intrinsics"].squeeze(0).to(self.config.vf_nerf_config.cuda_config.device)
            pose = train_data["pose"].squeeze(0).to(self.config.vf_nerf_config.cuda_config.device)

            # Render the rays.
            outputs = self.model.render(pose, pixels, intrinsics, epoch, self.dataset.white_bkgd)

            # Get border points and normals
            if self.dataset.get_vf_init_method()[0] == "center" and self.config.dataset_config.dataset_name != "deepfashion":
                supervised_normals, gt_normals = functions.get_border_indices_and_gt(outputs.points_coarse,
                                                                                     outputs.coarse_normals,
                                                                                     self.dataset.get_bounds()[1],
                                                                                     self.config.dataset_config.border_radius,
                                                                                     self.dataset.get_centroid(self.config.vf_nerf_config.cuda_config.device))
                border_points, border_gt_normals = functions.sample_border_points(self.dataset.get_bounds()[1] / 2 - self.config.dataset_config.border_radius,
                                                                                  self.dataset.get_bounds()[1] / 2 ,
                                                                                  (outputs.points_coarse.shape[0] * outputs.points_coarse.shape[1]) // 10,
                                                                                  self.dataset.get_centroid(self.config.vf_nerf_config.cuda_config.device),
                                                                                  outputs.points_coarse.device)
                border_normals = self.model.vector_field_network(border_points)[:, :3]
                supervised_normals = torch.cat([supervised_normals, border_normals], dim=0)
                gt_normals = torch.cat([gt_normals, border_gt_normals], dim=0)
            else:
                supervised_normals = torch.empty(0, 3).to(self.config.vf_nerf_config.cuda_config.device)
                gt_normals = torch.empty(0).to(self.config.vf_nerf_config.cuda_config.device)
                if self.config.vf_nerf_config.border_supervision:
                    border_points, border_gt_normals  = functions.sample_border_points(self.dataset.get_bounds()[1] - 5*self.config.dataset_config.border_radius,
                                                                                       self.dataset.get_bounds()[1],
                                                                                       (outputs.points_coarse.shape[0] * outputs.points_coarse.shape[1]) // 10,
                                                                                       self.dataset.get_centroid(self.config.vf_nerf_config.cuda_config.device),
                                                                                       outputs.points_coarse.device)
                    supervised_normals = torch.cat([supervised_normals, self.model.vector_field_network(border_points)[:, :3]], dim=0)
                    gt_normals = torch.cat([gt_normals, border_gt_normals], dim=0)
                if self.config.vf_nerf_config.center_supervision:
                    ray_center_normals, ray_center_gt_normals = functions.get_center_indices_and_gt(outputs.points_coarse,
                                                                                                    outputs.coarse_normals,
                                                                                                    self.dataset.get_centroid(self.config.vf_nerf_config.cuda_config.device),
                                                                                                    self.config.dataset_config.border_radius)
                    center_points, center_gt_normals = functions.sample_center_points(self.dataset.get_centroid(self.config.vf_nerf_config.cuda_config.device),
                                                                                      self.config.dataset_config.border_radius,
                                                                                      (outputs.points_coarse.shape[0] * outputs.points_coarse.shape[1]) // 10,
                                                                                      outputs.points_coarse.device)

                    supervised_normals = torch.cat([supervised_normals, ray_center_normals, self.model.vector_field_network(center_points)[:, :3]], dim=0)
                    gt_normals = torch.cat([gt_normals, ray_center_gt_normals, center_gt_normals], dim=0)

            # Get the predictions and ground truth.
            predictions = {
                "rgb": outputs.coarse_rgb_values,
                "depth": outputs.coarse_depth_map,
                "normals": outputs.coarse_normals.reshape(-1, 3),
                "supervised_normals": supervised_normals,
                "directional_derivatives": outputs.directional_derivtives
            }
            ground_truth = {
                "rgb": train_data["rgb"].reshape(-1, 3).to(self.config.vf_nerf_config.cuda_config.device),
                "depth": train_data["depth"].squeeze(0).to(self.config.vf_nerf_config.cuda_config.device),
                "supervised_normals": gt_normals
            }
            # Compute the loss.
            loss, losses_dict = self.loss(predictions, ground_truth, epoch)

            # Compute fine loss.
            if self.config.vf_nerf_config.ray_sampler_config.fine_sampling() and outputs.fine_normals is not None:
                predictions = {
                    "rgb": outputs.fine_rgb_values,
                    "depth": outputs.fine_depth_map,
                    "normals": outputs.fine_normals.reshape(-1, 3),
                    "supervised_normals": torch.empty(0),
                    "directional_derivatives": None
                }
                fine_loss, fine_losses_dict = self.loss(predictions, ground_truth, epoch)
                fine_losses_dict = {f"fine_{key}": value for key, value in fine_losses_dict.items()}

            # Compute total loss
            total_loss = loss + \
                fine_loss if self.config.vf_nerf_config.ray_sampler_config.fine_sampling() and outputs.fine_normals else loss

            # Backpropagate the loss.
            self.model.optimizer.zero_grad()
            total_loss.backward()

            # Clip the gradients.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(
            ), self.config.vf_nerf_config.scheduler_config.clip_norm)

            # Update the parameters.
            self.model.optimizer.step()
            self.model.scheduler.step()

            # Compute average loss.
            if average_losses is None:
                average_losses = losses_dict
                average_losses["loss"] = loss.item()
                if self.config.vf_nerf_config.ray_sampler_config.fine_sampling() and outputs.fine_normals:
                    average_losses.update(fine_losses_dict)
                    average_losses["fine_loss"] = fine_loss.item()
            elif loss is not None:
                average_losses["loss"] += loss.item()
                for key in losses_dict.keys():
                    average_losses[key] += losses_dict[key]
                if self.config.vf_nerf_config.ray_sampler_config.fine_sampling() and outputs.fine_normals:
                    average_losses["fine_loss"] += fine_loss.item()
                    for key in fine_losses_dict.keys():
                        average_losses[key] += fine_losses_dict[key]

        # Average the losses.
        for key in average_losses.keys():
            average_losses[key] /= len(self.dataloader)
        print(average_losses)

        # Log the loss.
        if not self.config.offline:
            # Log beta, mean, scale, learning rate, and cosine similarity weights.
            average_losses["beta"] = self.model.density.get_beta().item()
            average_losses["mean"] = self.model.density.get_mean().item()
            average_losses["scale"] = self.model.density.get_scale().item()
            average_losses["learning_rate"] = self.model.optimizer.param_groups[0]["lr"]
            average_losses.update(self.model.config.cos_sim_weights_dict())
            self.wandb_run.log(average_losses)
        return (average_losses["loss"] + average_losses["fine_loss"]) if self.config.vf_nerf_config.ray_sampler_config.fine_sampling() and outputs.fine_normals else average_losses["loss"]
