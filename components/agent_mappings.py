from components.policies.transformer import CustomTransformerPolicy
from components.policies.deep_sets import CustomDeepSetPolicy
from components.policies.lstm import CustomLSTMPolicy
from components.policies.gnn import CustomGNNPolicy
from components.policies.sa_deepsets import CustomSelfAttentionDeepSetsPolicy
from components.policies.set_transformer import CustomSetTransformerPolicy
from components.policies.deepsets_ext import CustomDeepSetExtensionPolicy
from components.policies.relational_network import CustomRelationalNetworkPolicy


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
            "use_feature_kwargs": True,
        }
    elif key == "deepsets":
        agent_mapping = {
            "encoder": True,
            "name": model_name + "_" + game_name + "_deep_sets" + model_extension,
            "policy": CustomDeepSetPolicy,
            "use_feature_kwargs": True,
        }
    elif key == "deepsets_extension":
        agent_mapping = {
            "encoder": True,
            "name": model_name + "_" + game_name + "_deep_sets_extension" + model_extension,
            "policy": CustomDeepSetExtensionPolicy,
            "use_feature_kwargs": True,
        }
    elif key == "relational_network":
        agent_mapping = {
            "encoder": True,
            "name": model_name + "_" + game_name + "_relational_network" + model_extension,
            "policy": CustomRelationalNetworkPolicy,
            "use_feature_kwargs": True,
        }
    elif key == "set_transformer":
        agent_mapping = {
            "encoder": True,
            "name": model_name + "_" + game_name + "_set_transformer" + model_extension,
            "policy": CustomSetTransformerPolicy,
            "use_feature_kwargs": True,
        }
    elif key == "sa_deepsets":
        agent_mapping = {
            "encoder": True,
            "name": model_name + "_" + game_name + "_sa_deep_sets" + model_extension,
            "policy": CustomSelfAttentionDeepSetsPolicy,
            "use_feature_kwargs": True,
        }
    elif key == "lstm":
        agent_mapping = {
            "encoder": True,
            "name": model_name + "_" + game_name + "_lstm" + model_extension,
            "policy": CustomLSTMPolicy,
            "use_feature_kwargs": True,
        }
    elif key == "gnn":
        agent_mapping = {
            "encoder": True,
            "name": model_name + "_" + game_name + "_gnn" + model_extension,
            "policy": CustomGNNPolicy,
            "use_feature_kwargs": True,
        }
    elif key == "cnn":
        agent_mapping = {
            "encoder": None,
            "name": model_name + "_" + game_name + "_cnn" + model_extension,
            "policy": "CnnPolicy",
            "n_stack": 4,  # Number of frames to stack for CNN input
            "use_feature_kwargs": False,
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
