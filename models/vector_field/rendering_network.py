import sys
from typing import List, Optional

sys.path.append('.')  # isort:skip

import torch
from torch import nn

from config_parser.vf_nerf_config import RenderingNetConfig
from models.helpers.embedder import Embedder, get_embedder


class RenderingNetwork(nn.Module):
    def __init__(self, config: RenderingNetConfig) -> None:
        """
        Rendering network class to predic the RGB color.
        :param config: The config.
        """
        # Save the config.
        self.config = config

        # Initialize the super class.
        super().__init__()

        # Compute input_dimensions from the mode.
        input_dims = 3
        if self.config.mode == "idr":
            input_dims += 6
        elif self.config.mode in ["no_view_dir", "no_normals"]:
            input_dims += 3

        # Add input and output dimensions to the dimensions list.
        dimensions = [input_dims + self.config.feature_vector_dims] + config.dimensions + [self.config.output_dims]

        # Create the embedder.
        self.embedder: Optional[Embedder] = None
        if self.config.embedder_multires > 0:
            self.embedder, input_channels = get_embedder(self.config.embedder_multires, 3)
            dimensions[0] += input_channels - 3

        # Create the layers.
        self.num_layers = len(dimensions) - 1
        self.layers = nn.ModuleList()

        # Create the layers.
        for i in range(self.num_layers):
            # Create layers
            layer = nn.Linear(dimensions[i], dimensions[i + 1])
            if self.config.weight_norm:
                layer = nn.utils.weight_norm(layer)
            elif self.config.batch_norm and i < self.num_layers - 1:
                layer = nn.Sequential(layer, nn.BatchNorm1d(dimensions[i + 1]))
            self.layers.append(layer)

        # Create the activation functions.
        self.activation = nn.ReLU()
        self.output_activation = nn.Sigmoid()

        # Save the mode.
        self.mode = self.config.mode

    def forward(self,
                points: torch.Tensor,
                normals: torch.Tensor,
                view_dirs: torch.Tensor,
                feature_vectors: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the rendering network.
        :param points: The 3D points.
        :param normals: The surface normals.
        :param view_dirs: The view directions.
        :param feature_vectors: The feature vectors.
        :return: The output tensor containing the predicted RGB values.
        """
        # Detach normals if necessary.
        if self.config.detach_normals:
            normals = normals.detach()

        # Embed the view directions.
        if self.embedder is not None:
            view_dirs = self.embedder(view_dirs)

        # Concatenate the input.
        if (feature_vectors.numel() > 0) and (self.config.feature_vector_dims > 0) and (feature_vectors.shape[1] == self.config.feature_vector_dims):
            if self.mode == "idr":
                input_tensor = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
            elif self.mode == "no_view_dir":
                input_tensor = torch.cat([points, normals, feature_vectors], dim=-1)
            elif self.mode == "no_normals":
                input_tensor = torch.cat([points, view_dirs, feature_vectors], dim=-1)
        else:
            if self.mode == "idr":
                input_tensor = torch.cat([points, view_dirs, normals], dim=-1)
            elif self.mode == "no_view_dir":
                input_tensor = torch.cat([points, normals], dim=-1)
            elif self.mode == "no_normals":
                input_tensor = torch.cat([points, view_dirs], dim=-1)

        # Forward pass.
        for i in range(self.num_layers):
            input_tensor = self.layers[i](input_tensor)
            if i < self.num_layers - 1:
                input_tensor = self.activation(input_tensor)

        # Apply the output activation function.
        output_tensor = self.output_activation(input_tensor)

        return output_tensor
