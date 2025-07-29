from typing import Union

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import get_device

from components.agents.OCZero.model import DynamicsNetwork, RepresentationNetwork


class OCZeroFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, n_features, hidden_dim, top_k):
        super().__init__(observation_space, features_dim=hidden_dim)
        self.encoder = RepresentationNetwork(n_features, hidden_dim, top_k)

    def forward(self, observations):
        return self.encoder(observations)


class OCZeroMlpExtractor(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        projector_dim: int,
        action_space: int,
        net_arch: Union[list[int], dict[str, list[int]]],
        activation_fn: type[nn.Module],
        device: Union[torch.device, str] = "auto",
    ) -> None:
        super().__init__()
        device = get_device(device)

        # --- Dynamics Network ---
        self.dynamics_network = DynamicsNetwork(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            proj_dim=projector_dim,
            action_space_size=action_space,
        ).to(device)

        # --- Actor/Critic Network ---
        policy_net: list[nn.Module] = []
        value_net: list[nn.Module] = []
        last_layer_dim_pi = feature_dim
        last_layer_dim_vf = feature_dim

        if isinstance(net_arch, dict):
            # Note: if key is not specified, assume linear network
            pi_layers_dims = net_arch.get("pi", [])  # Layer sizes of the policy network
            vf_layers_dims = net_arch.get("vf", [])  # Layer sizes of the value network
        else:
            pi_layers_dims = vf_layers_dims = net_arch
        # Iterate through the policy layers and build the policy net
        for curr_layer_dim in pi_layers_dims:
            policy_net.append(nn.Linear(last_layer_dim_pi, curr_layer_dim))
            policy_net.append(activation_fn())
            last_layer_dim_pi = curr_layer_dim
        # Iterate through the value layers and build the value net
        for curr_layer_dim in vf_layers_dims:
            value_net.append(nn.Linear(last_layer_dim_vf, curr_layer_dim))
            value_net.append(activation_fn())
            last_layer_dim_vf = curr_layer_dim

        # Save dim, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.policy_net = nn.Sequential(*policy_net).to(device)
        self.value_net = nn.Sequential(*value_net).to(device)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)

    def forward_dynamics(
        self,
        hidden_state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_hidden_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the dynamics network.
        :return: projection_loss, action_prediction_loss, reward_prediction_loss
        """
        return self.dynamics_network(hidden_state, action, reward, next_hidden_state)


class OCZeroPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        n_features=6,
        hidden_dim=64,
        projector_dim=512,
        top_k=16,
        **kwargs,
    ):
        self.hidden_dim = hidden_dim
        self.projector_dim = projector_dim

        kwargs["ortho_init"] = True
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=OCZeroFeaturesExtractor,
            features_extractor_kwargs=dict(
                n_features=n_features,
                hidden_dim=hidden_dim,
                top_k=top_k,
            ),
            **kwargs,
        )

        # Create auxiliary optimizer for dynamics network and feature extractor parameters
        aux_params = list(self.mlp_extractor.dynamics_network.parameters()) + list(
            self.features_extractor.parameters()
        )
        aux_lr = kwargs.get("learning_rate", 1e-4)
        self.auxiliary_optimizer = torch.optim.Adam(aux_params, lr=aux_lr)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = OCZeroMlpExtractor(
            self.features_dim,
            self.hidden_dim,
            self.projector_dim,
            self.action_space.n,
            self.net_arch,
            self.activation_fn,
            device=self.device,
        )

    def evaluate_dynamics(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
    ):
        hidden_states = self.features_extractor(states)
        next_hidden_states = self.features_extractor(next_states)

        return self.mlp_extractor.forward_dynamics(
            hidden_states, actions, rewards, next_hidden_states
        )

    def reinitialize_weights(self):
        """Reinitialize all weights in the policy network"""
        # Reinitialize the relational network encoder
        if hasattr(self.features_extractor, "relational_network_encoder"):
            self.features_extractor.relational_network_encoder.init_weights()

        # Reinitialize all linear layers in the policy
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.orthogonal_(
                    module.weight, gain=torch.sqrt(torch.tensor(2.0))
                )
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
