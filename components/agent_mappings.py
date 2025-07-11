from components.encoders.breakout_encoder import BreakoutEncoder
from components.encoders.pong_encoder import PongEncoder
from components.encoders.object_discovery_encoder import ObjectDiscoveryEncoder
from components.policies.transformer import CustomTransformerPolicy
from components.policies.deep_sets import CustomDeepSetPolicy
from components.policies.lstm import CustomLSTMPolicy
from components.policies.gnn import CustomGNNPolicy
from components.policies.relational_network import CustomRelationalNetworkPolicy


def get_agent_mapping(key, config, n_envs, game_id, model_name, model_extension=""):
    """
    Get agent mappings configuration for different agent types.

    Args:
        config: Configuration dictionary loaded from config.yaml
        n_envs: Number of environments
        game_name: Name of the Atari game
        model_name: Name of the model (e.g., "PPO", "A2C")

    Returns:
        Agent mapping with encoder, policy, and other configurations
    """
    rb_encoder = {
        "Breakout": BreakoutEncoder,
        "Pong": PongEncoder,
    }

    game_name = "Breakout" if "Breakout" in game_id else "Pong"
    model_extension = f"_{model_extension}" if model_extension else ""

    if key == "player+ball":
        agent_mapping = {
            "encoder": rb_encoder[game_name](
                encoding_method="paddle+ball",
                speed_scale=config["encoder"]["speed_scale"],
                num_envs=n_envs,
            ),
            "n_features": 5 if "Breakout" in game_name else 6,
            "name": model_name + "_rb_player_ball" + model_extension,
            "policy": "MlpPolicy",
            "use_feature_kwargs": False,
            "n_stack": None,
            "wrapper_kwargs": {"screen_size": -1, "max_pool": False},
        }
    elif key == "player-ball":
        agent_mapping = {
            "encoder": rb_encoder[game_name](
                encoding_method="paddle-ball",
                speed_scale=config["encoder"]["speed_scale"],
                num_envs=n_envs,
            ),
            "n_features": 4,
            "name": model_name + "_rb_player_ball_egocentric" + model_extension,
            "policy": "MlpPolicy",
            "use_feature_kwargs": False,
            "n_stack": None,
            "wrapper_kwargs": {"screen_size": -1, "max_pool": False},
        }
    elif key == "player+ball+bricks":
        agent_mapping = {
            "encoder": rb_encoder[game_name](
                encoding_method="bricks+paddle+ball",
                speed_scale=config["encoder"]["speed_scale"],
                num_envs=n_envs,
            ),
            "n_features": 113,
            "name": model_name + "_rb_player_ball_bricks" + model_extension,
            "policy": "MlpPolicy",
            "use_feature_kwargs": False,
            "n_stack": None,
            "wrapper_kwargs": {"screen_size": -1, "max_pool": False},
        }
    elif key == "player+ball+trajectory":
        agent_mapping = {
            "encoder": rb_encoder[game_name](
                encoding_method="paddle+ball+trajectory",
                speed_scale=config["encoder"]["speed_scale"],
                num_envs=n_envs,
            ),
            "n_features": 13,
            "name": model_name + "_rb_player_ball_trajectory" + model_extension,
            "policy": "MlpPolicy",
            "use_feature_kwargs": False,
            "n_stack": None,
            "wrapper_kwargs": {"screen_size": -1, "max_pool": False},
        }
    
    elif key == "deepsets":
        agent_mapping = {
            "encoder": rb_encoder[game_name](
                encoding_method="bricks+paddle+ball+discovery",
                speed_scale=config["encoder"]["speed_scale"],
                num_envs=n_envs,
            ),
            "n_features": 6,
            "name": model_name + "_rb_deepsets" + model_extension,
            "policy": CustomDeepSetPolicy,  # TODO: Try CustomTransformerPolicy
            "use_feature_kwargs": True,
            "n_stack": None,
            "wrapper_kwargs": {"screen_size": -1, "max_pool": False},
        }
    elif key == "relational_network":
        agent_mapping = {
            "encoder": rb_encoder[game_name](
                encoding_method="bricks+paddle+ball+discovery",
                speed_scale=config["encoder"]["speed_scale"],
                num_envs=n_envs,
            ),
            "n_features": 6,
            "name": model_name + "_relational_net" + model_extension,
            "policy": CustomRelationalNetworkPolicy,  # TODO: Try CustomTransformerPolicy
            "use_feature_kwargs": True,
            "n_stack": None,
            "wrapper_kwargs": {"screen_size": -1, "max_pool": False},
        }
    elif key == "transformer":
        agent_mapping = {
            "encoder": rb_encoder[game_name](
                encoding_method="bricks+paddle+ball+discovery",
                speed_scale=config["encoder"]["speed_scale"],
                num_envs=n_envs,
            ),
            "n_features": 6,
            "name": model_name + "_rb_transformers" + model_extension,
            "policy": CustomTransformerPolicy,  # TODO: Try CustomTransformerPolicy
            "use_feature_kwargs": True,
            "n_stack": None,
            "wrapper_kwargs": {"screen_size": -1, "max_pool": False},
        }
    elif key == "cnn":
        agent_mapping = {
            "encoder": None,  # CNN does not require a custom encoder
            "n_features": -1,
            "name": model_name + "_cnn" + model_extension,
            "policy": "CnnPolicy",
            "use_feature_kwargs": False,
            "n_stack": 4,  # Stack frames for CNN
            "wrapper_kwargs": {},
        }
    elif key == "naive":
        agent_mapping = {
            "encoder": rb_encoder[game_name](
                encoding_method="paddle+ball",
                speed_scale=config["encoder"]["speed_scale"],
                num_envs=n_envs,
            ),
            "n_features": 5,
            "name": None,  # No model to load for rule-based agent
            "policy": None,
            "use_feature_kwargs": False,
            "n_stack": None,  # No stacking for rule-based agent
            "wrapper_kwargs": {"screen_size": -1, "max_pool": False},
        }
    elif key == "player+ball+bricks+deepsets":
        agent_mapping = {
            "encoder": rb_encoder[game_name](
                encoding_method="bricks+paddle+ball+object_vectors",
                speed_scale=config["encoder"]["speed_scale"],
                num_envs=n_envs,
            ),
            "n_features": 8,
            "name": model_name + "_rb_player_ball_bricks_deepsets" + model_extension,
            "policy": CustomDeepSetPolicy,
            "use_feature_kwargs": True,
            "n_stack": None,
            "wrapper_kwargs": {"screen_size": -1, "max_pool": False},
        }
    elif key == "player+ball+bricks+lstm":
        agent_mapping = {
            "encoder": rb_encoder[game_name](
                encoding_method="bricks+paddle+ball+object_vectors",
                speed_scale=config["encoder"]["speed_scale"],
                num_envs=n_envs,
            ),
            "n_features": 8,
            "name": model_name + "_rb_player_ball_bricks_lstm" + model_extension,
            "policy": CustomLSTMPolicy,
            "use_feature_kwargs": True,
            "n_stack": None,
            "wrapper_kwargs": {"screen_size": -1, "max_pool": False},
        }
    elif key == "player+ball+bricks+gnn":
        agent_mapping = {
            "encoder": rb_encoder[game_name](
                encoding_method="bricks+paddle+ball+object_vectors",
                speed_scale=config["encoder"]["speed_scale"],
                num_envs=n_envs,
            ),
            "n_features": 8,
            "name": model_name + "_rb_player_ball_bricks_gnn" + model_extension,
            "policy": CustomGNNPolicy,
            "use_feature_kwargs": True,
            "n_stack": None,
            "wrapper_kwargs": {"screen_size": -1, "max_pool": False},
        }
    elif key == "transformer+discovery":
        agent_mapping = {
            "encoder": ObjectDiscoveryEncoder(
                speed_scale=config["encoder"]["speed_scale"],
                num_envs=n_envs,
                max_objects=config["encoder"]["max_objects"],
            ),
            "n_features": 8,
            "name": model_name + "_rb_transformer",
            "policy": CustomTransformerPolicy,
            "use_feature_kwargs": True,
            "n_stack": 2,  # Stack frames for temporal encoding
            "wrapper_kwargs": {"screen_size": -1, "max_pool": False},
        }
    elif key == "deepsets+discovery":
        agent_mapping = {
            "encoder": ObjectDiscoveryEncoder(
                speed_scale=config["encoder"]["speed_scale"],
                num_envs=n_envs,
                max_objects=config["encoder"]["max_objects"],
            ),
            "n_features": 8,
            "name": model_name + "_rb_deep_sets" + model_extension,
            "policy": CustomDeepSetPolicy,
            "use_feature_kwargs": True,
            "n_stack": 2,  # Stack frames for temporal encoding
            "wrapper_kwargs": {"screen_size": -1, "max_pool": False},
        }

    else:
        raise ValueError(f"Agent mapping for '{key}' not found.")

    return agent_mapping
