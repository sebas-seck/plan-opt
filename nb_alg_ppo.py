# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
from plan_opt.demand import Demand
from plan_opt.envs.rampup1 import RampupEnv

# %%
D = Demand(seed=3348)
D.generate_demand()
D.add_sudden_change()
D.info()
D.show()

# %% [markdown]
# ### Create the environment
# The action space is descrete, only categorical changes of equipment are allowed.

# %%
env = RampupEnv(demand=D.data)

# %% [markdown]
# ### Train the model

# %%
# %%time
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=2500)

# %% [markdown]
# ### Predict

# %%
obs = env.reset()
for i in range(10000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        env.render()
        obs = env.reset()

# env.close()
# env.render()

# %%
def evaluate(model, num_steps=1000):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_steps: (int) number of timesteps to evaluate it
    :return: (float) Mean reward for the last 100 episodes
    """
    episode_rewards = [0.0]
    obs = env.reset()
    for i in range(num_steps):
        # _states are only useful when using LSTM policies
        action, _states = model.predict(obs)

        obs, reward, done, info = env.step(action)

        # Stats
        episode_rewards[-1] += reward
        if done:
            obs = env.reset()
            episode_rewards.append(0.0)
    # Compute mean reward for the last 100 episodes
    mean_100ep_reward = round(np.mean(episode_rewards[-100:]), 1)
    print("Mean reward:", mean_100ep_reward, "Num episodes:", len(episode_rewards))

    return mean_100ep_reward


# %%
# Evaluate the trained agent
mean_reward = evaluate(model, num_steps=10000)

# %%
model.policy

# %%
model.learning_rate

# %%
