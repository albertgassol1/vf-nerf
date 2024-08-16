import sys
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from config_parser.vf_nerf_config import VFNetConfig
from models.helpers.embedder import Embedder, get_embedder

sys.path.append('.')  # isort:skip


class VectorFieldNetwork(nn.Module):
    def __init__(self, config: VFNetConfig) -> None:
        """
        VectorField class for the vector field network.
        :params config: The configuration.
        """

        # Initialize the super class.
        super().__init__()

        # Save the config.
        self.config = config

        # Add input and output dimensions to the dimensions list.
        dimensions = [self.config.input_dims] + config.dimensions + [self.config.output_dims + self.config.feature_vector_dims]

        # Create the embedder.
        self.embedder: Optional[Embedder] = None
        if self.config.embedder_multires > 0:
            self.embedder, input_channels = get_embedder(self.config.embedder_multires, self.config.input_dims)
            dimensions[0] = input_channels

        # Create the layers.
        self.num_layers = len(dimensions) - 1
        self.layers = nn.ModuleList()

        # Create the skip connections.
        if self.config.skip_connection_in is None:
            self.skip_connection_in: List[int] = list()
        else:
            self.skip_connection_in = self.config.skip_connection_in

        # Create the layers.
        for i in range(self.num_layers):
            # Create layers
            if i + 1 in self.skip_connection_in:
                out_dimensions = dimensions[i + 1] - dimensions[0]
            else:
                out_dimensions = dimensions[i + 1]

            self.layers.append(nn.Linear(dimensions[i], out_dimensions))

            # Weight normalization and batch normalization.
            if self.config.weight_norm:
                self.layers[-1] = nn.utils.weight_norm(self.layers[-1])
            elif self.config.batch_norm and i < self.num_layers - 1:
                self.layers[-1] = nn.Sequential(self.layers[-1], nn.BatchNorm1d(out_dimensions))

            # Xavier initialization.
            if self.config.xavier_init:
                nn.init.xavier_uniform_(self.layers[-1].weight)
                nn.init.constant_(self.layers[-1].bias, self.config.bias_init)

        # Create the activation function.
        self.activation = nn.ReLU()

        # Create activation function for the last layer.
        self.last_activation = nn.Tanh()

        # Save init to center.
        assert (self.config.init in ["center", "exterior", ""]) or ("exterior" in self.config.init), \
            "init must be one of [center, exterior, '']"

    def get_outputs(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass and return the normal vectors and feature vectors.
        :params input_tensor: The input tensor.
        :returns: The normal vectors and feature vectors.
        """
        # Perform a forward pass.
        output_tensor = self.forward(input_tensor)

        # Split the output tensor.
        normal_vectors = output_tensor[:, :3]
        feature_vectors = output_tensor[:, 3:]

        # Return the normal vectors and feature vectors.
        return normal_vectors, feature_vectors

    @property
    def init(self) -> str:
        """
        Get the initialization method.
        :return: The initialization method.
        """
        return self.config.init

    @init.setter
    def init(self, init: str) -> None:
        """
        Set the initialization method.
        :param init: The initialization method.
        """
        self.config.init = init

    def load_init(self, init_path: str, device: torch.device = torch.device('cpu')) -> None:
        """
        Initialize the network to point to the center.
        :params init_path: The path to the initialization.
        :params device: The device to use.
        """

        # Get the network state dict.
        if self.config.init == "center":
            if self.embedder is not None:
                state_dict = torch.load('exps_vf_nerf/point_to_center/embedding.pth',
                                        map_location=torch.device('cpu'))
            else:
                state_dict = torch.load('exps_vf_nerf/point_to_center/no_embedding.pth',
                                        map_location=torch.device('cpu'))
        elif self.config.init == "exterior":
            if self.embedder is not None:
                state_dict = torch.load('exps_vf_nerf/point_exterior/embedding.pth',
                                        map_location=torch.device('cpu'))
            else:
                state_dict = torch.load('exps_vf_nerf/point_exterior/no_embedding.pth',
                                        map_location=torch.device('cpu'))
        elif "exterior" in self.config.init:
            state_dict = torch.load(init_path, map_location=torch.device('cpu'))

        # Load the state dict.
        self.load_state_dict(state_dict)

        # Move the network to the device.
        self.to(device)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the vector field network.
        :params input_tensor: The input tensor.
        :returns: The output tensor.
        """
        if self.training:
            with torch.enable_grad():
                points = points.requires_grad_(True)
                y = self._forward(points)
                d_output= torch.ones_like(y[:, 0], requires_grad=False, device=y.device)
                grad_x = torch.autograd.grad(
                    outputs=y[:, 0],
                    inputs=points,
                    grad_outputs=d_output,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True)[0]
                grad_y = torch.autograd.grad(
                    outputs=y[:, 1],
                    inputs=points,
                    grad_outputs=d_output,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True)[0]
                grad_z = torch.autograd.grad(
                    outputs=y[:, 2],
                    inputs=points,
                    grad_outputs=d_output,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True)[0]
                flat_jacobian = torch.cat([grad_x, grad_y, grad_z], dim=-1)
                return torch.cat([y, flat_jacobian], dim=-1)
        else:
            return self._forward(points)
    
    def _forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the vector field network.
        :params input_tensor: The input tensor.
        :returns: The output tensor.
        """

        # Embed the input.
        if self.embedder is not None:
            input_tensor = self.embedder(input_tensor)

        # Pass through the layers.
        x = input_tensor.clone()
        for i in range(self.num_layers):
            # Pass through the layer.
            if i in self.skip_connection_in:
                x = torch.cat([x, input_tensor], 1) / torch.sqrt(torch.tensor([2]).to(x.device).float())

            x = self.layers[i](x)

            # Apply the activation function.
            if i < self.num_layers - 1:
                x = self.activation(x)
            else:
                x = self.last_activation(x)

            # Apply dropout.
            if self.config.dropout and i < self.num_layers - 1:
                x = F.dropout(x, p=self.config.dropout_probability, training=self.training)

        # Return the output.
        return x
