import os
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from environment import make_atari_env

# Create directory to save model
model_path = "./weights"
os.makedirs(model_path, exist_ok=True)

# Create vectorized Atari environment
env = make_atari_env("BreakoutNoFrameskip-v4", n_envs=4, seed=0)

# Stack frames (to capture motion)
env = VecFrameStack(env, n_stack=4)

# Initialize A2C model
model = A2C("CnnPolicy", 
            env, 
            verbose=2,
            learning_rate=0.0007,
            n_steps=20,
            ent_coef=0.01,
            gamma=0.99,
            gae_lambda=0.95,
            vf_coef=0.5)

# Train the model
model.learn(total_timesteps=1_000_000)

# Save the model
model.save(os.path.join(model_path, "a2c_breakout"))

# Close training environment
env.close()
