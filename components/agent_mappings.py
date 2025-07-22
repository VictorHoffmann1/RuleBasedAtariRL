from components.policies.deep_sets import CustomDeepSetPolicy
from components.policies.gnn import CustomGNNPolicy
from components.policies.lstm import CustomLSTMPolicy
from components.policies.relational_network import CustomRelationalNetworkPolicy
from components.policies.sa_deepsets import CustomSelfAttentionDeepSetsPolicy
from components.policies.set_transformer import CustomSetTransformerPolicy
from components.policies.transformer import CustomTransformerPolicy
from components.policies.lstm_relational_network import LSTMRelationalNetworkPolicy


def get_agent_mapping(key, game_name, model_extension=""):
    """
    Get agent mappings configuration for different agent types.

    Args:
        config: Configuration dictionary loaded from config.yaml
        n_envs: Number of environments
        game_name: Name of the Atari game
    Returns:
        Agent mapping with encoder, policy, and other configurations
    """
    model_extension = f"_{model_extension}" if model_extension else ""
    # Remove "ALE" and "-v5" from game_name
    version = "_v5" if "v5" in game_name else "_v4"
    game_name = (
        game_name.replace("ALE/", "")
        .replace("-v5", "")
        .replace("-v4", "")
        .replace("NoFrameskip", "")
    )

    if key == "transformer":
        agent_mapping = {
            "encoder": True,
            "name": "OC_" + game_name + "_transformer" + version + model_extension,
            "policy": CustomTransformerPolicy,
            "n_stack": None,
            "use_feature_kwargs": True,
            "method": "discovery",
        }
    elif key == "deepsets":
        agent_mapping = {
            "encoder": True,
            "name": "OC_" + game_name + "_deep_sets" + version + model_extension,
            "policy": CustomDeepSetPolicy,
            "n_stack": None,
            "use_feature_kwargs": True,
            "method": "discovery",
        }
    elif key == "relational_network":
        agent_mapping = {
            "encoder": True,
            "name": "OC_"
            + game_name
            + "_relational_network"
            + version
            + model_extension,
            "policy": CustomRelationalNetworkPolicy,
            "n_stack": None,
            "use_feature_kwargs": True,
            "method": "discovery",
        }
    elif key == "lstm_relational_net":
        agent_mapping = {
            "encoder": True,
            "name": "OC_"
            + game_name
            + "_lstm_relational_network"
            + version
            + model_extension,
            "policy": LSTMRelationalNetworkPolicy,
            "n_stack": 20,
            "use_feature_kwargs": True,
            "method": "discovery",
        }
    elif key == "set_transformer":
        agent_mapping = {
            "encoder": True,
            "name": "OC_" + game_name + "_set_transformer" + version + model_extension,
            "policy": CustomSetTransformerPolicy,
            "n_stack": None,
            "use_feature_kwargs": True,
            "method": "discovery",
        }
    elif key == "sa_deepsets":
        agent_mapping = {
            "encoder": True,
            "name": "OC_" + game_name + "_sa_deep_sets" + version + model_extension,
            "policy": CustomSelfAttentionDeepSetsPolicy,
            "use_feature_kwargs": True,
            "n_stack": None,
            "method": "discovery",
        }
    elif key == "lstm":
        agent_mapping = {
            "encoder": True,
            "name": "OC_" + game_name + "_lstm" + version + model_extension,
            "policy": CustomLSTMPolicy,
            "use_feature_kwargs": True,
            "n_stack": None,
            "method": "discovery",
        }
    elif key == "gnn":
        agent_mapping = {
            "encoder": True,
            "name": "OC_" + game_name + "_gnn" + version + model_extension,
            "policy": CustomGNNPolicy,
            "n_stack": None,
            "use_feature_kwargs": True,
            "method": "discovery",
        }
    elif key == "mlp":
        agent_mapping = {
            "encoder": True,
            "name": "OC_" + game_name + "_mlp" + version + model_extension,
            "policy": "MlpPolicy",
            "n_stack": None,
            "use_feature_kwargs": False,
            "method": "expert",
        }
    elif key == "cnn":
        agent_mapping = {
            "encoder": False,
            "name": "OC_" + game_name + "_cnn" + version + model_extension,
            "policy": "CnnPolicy",
            "n_stack": 4,  # Number of frames to stack for CNN input
            "use_feature_kwargs": False,
            "method": None,
        }
    elif key == "random":
        agent_mapping = {
            "encoder": True,
            "name": "random",
            "policy": "Random",
            "n_stack": None,
            "use_feature_kwargs": False,
            "method": "discovery",
        }
    elif key == "naive":
        raise NotImplementedError(
            "Naive agent is not implemented in this version. Please use a different agent type."
        )
        agent_mapping = {
            "encoder": True,
            "name": "naive",
            "policy": "Naive",
            "n_stack": None,
            "method": "discovery",
        }
    else:
        raise ValueError(f"Agent mapping for '{key}' not found.")

    return agent_mapping
