from components.policies.transformer import CustomTransformerPolicy
from components.policies.deep_sets import CustomDeepSetPolicy
from components.policies.lstm import CustomLSTMPolicy
from components.policies.gnn import CustomGNNPolicy


def get_agent_mapping(key, game_name, model_name, model_extension=""):
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

    model_extension = f"_{model_extension}" if model_extension else ""
    game_name = game_name[4:-3]

    if key == "transformer":
        agent_mapping = {
            "encoder": True,
            "name": model_name + "_" + game_name + "_transformer" + model_extension,
            "policy": CustomTransformerPolicy,
        }
    elif key == "deepsets":
        agent_mapping = {
            "encoder": True,
            "name": model_name + "_" + game_name + "_deep_sets" + model_extension,
            "policy": CustomDeepSetPolicy,
        }
    elif key == "lstm":
        agent_mapping = {
            "encoder": True,
            "name": model_name + "_" + game_name + "_lstm" + model_extension,
            "policy": CustomLSTMPolicy,
        }
    elif key == "gnn":
        agent_mapping = {
            "encoder": True,
            "name": model_name + "_" + game_name + "_gnn" + model_extension,
            "policy": CustomGNNPolicy,
        }
    elif key == "cnn":
        agent_mapping = {
            "encoder": None,
            "name": model_name + "_" + game_name + "_cnn" + model_extension,
            "policy": "CnnPolicy",
        }
    elif key == "naive":
        raise NotImplementedError(
            "Naive agent is not implemented in this version. Please use a different agent type."
        )
        agent_mapping = {
            "encoder": True,
            "name": None,  # No model to load for rule-based agent
            "policy": None,
        }
    else:
        raise ValueError(f"Agent mapping for '{key}' not found.")

    return agent_mapping
