import torch
import torch.nn as nn
from stable_baselines3 import PPO


def reinitialize_ppo_weights(model: PPO):
    """
    Reinitialize all weights in a PPO model.
    
    Args:
        model: PPO model with CustomRelationalNetworkPolicy
    """
    # Method 1: Use the custom reinitialize_weights method if available
    if hasattr(model.policy, 'reinitialize_weights'):
        model.policy.reinitialize_weights()
    else:
        # Method 2: Manual reinitialization
        _manual_reinitialize_weights(model)
    
    # Reset optimizer states (important!)
    model.policy.optimizer = model.policy.optimizer.__class__(
        model.policy.parameters(), 
        **model.policy.optimizer.defaults
    )


def _manual_reinitialize_weights(model: PPO):
    """Manual weight reinitialization for PPO model"""
    # Reinitialize the features extractor (RelationalNetwork)
    if hasattr(model.policy.features_extractor, 'relational_network_encoder'):
        model.policy.features_extractor.relational_network_encoder.init_weights()
    
    # Reinitialize all other linear layers
    for module in model.policy.modules():
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=torch.sqrt(torch.tensor(2.0)))
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def create_fresh_ppo_model(env, config, agent_mapping, **model_kwargs):
    """
    Create a fresh PPO model with properly initialized weights.
    
    Args:
        env: Environment
        config: Configuration dictionary
        agent_mapping: Agent mapping configuration
        **model_kwargs: Additional model parameters
        
    Returns:
        PPO model with fresh weights
    """
    model_params = config["model"]
    
    model = PPO(
        agent_mapping["policy"],
        env,
        verbose=2,
        learning_rate=model_kwargs.get("learning_rate", model_params["learning_rate"]),
        batch_size=model_kwargs.get("batch_size", model_params["ppo_batch_size"]),
        n_epochs=model_kwargs.get("n_epochs", model_params["n_epochs"]),
        n_steps=model_kwargs.get("n_steps", model_params["n_steps"]),
        gamma=model_kwargs.get("gamma", model_params["gamma"]),
        gae_lambda=model_kwargs.get("gae_lambda", model_params["gae_lambda"]),
        ent_coef=model_kwargs.get("ent_coef", model_params["ent_coef"]),
        vf_coef=model_kwargs.get("vf_coef", model_params["vf_coef"]),
        clip_range=model_kwargs.get("clip_range", model_params["clip_range"]),
        max_grad_norm=model_kwargs.get("max_grad_norm", model_params["max_grad_norm"]),
        tensorboard_log=model_kwargs.get("tensorboard_log", "./logs/"),
        seed=model_kwargs.get("seed", 0),
        policy_kwargs=model_kwargs.get("policy_kwargs", {})
    )
    
    # Weights are already initialized during model creation
    return model


# Example usage functions
def reinitialize_for_new_trial(model: PPO):
    """
    Convenience function to reinitialize model for hyperparameter search trials
    """
    print("Reinitializing model weights for new trial...")
    reinitialize_ppo_weights(model)
    print("Model weights reinitialized successfully!")


def reset_model_state(model: PPO, seed: int = None):
    """
    Reset model to initial state with optional new seed
    """
    if seed is not None:
        model.set_random_seed(seed)
    
    reinitialize_ppo_weights(model)
    
    # Reset any internal counters
    model.num_timesteps = 0
    model._num_timesteps_at_start = 0
