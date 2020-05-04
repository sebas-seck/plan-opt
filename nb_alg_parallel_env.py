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
import os
import subprocess
import numpy as np
import webbrowser
import gym
from gym import spaces
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from plan_opt.demand import Demand
from plan_opt.demand_small_samples import four_weeks_uprising
from plan_opt.envs.rampup1 import RampupEnv

# %% [markdown]
# ### Create the vectorized environment

# %%
import plan_opt
env_id = "plan-opt-v0"
num_cpu = 4  # Number of processes to use
env = make_vec_env(env_id, n_envs=num_cpu)

# %%
timesteps = 20000
tensorboard_log = "logs/rampup_tensorboard/"
tb_log_suffix = f'0-255_{str(timesteps)[:-3]}k_vec_env{num_cpu}'
tb_log_suffix

# %% [markdown]
# ### Train the model

# %%
# %%time
deterministic = False
model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=tensorboard_log)
model.learn(total_timesteps=timesteps, eval_freq=100,
            tb_log_name=f"A2C_train_run_{tb_log_suffix}") #, reset_num_timesteps=False)

# %% [markdown]
# ### Evaluation
# Using an overseeable demand for four weeks only.

# %%
demand = Demand(period=len(four_weeks_uprising), data=four_weeks_uprising)
demand.show(only_data=True)

# %%
# Separate evaluation env
eval_env = RampupEnv(demand.data)
eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=100,
                             deterministic=deterministic, render=False)

eval_model = A2C('MlpPolicy', eval_env, verbose=1, tensorboard_log=tensorboard_log)
eval_model.learn(total_timesteps=timesteps, callback=eval_callback,
                 tb_log_name=f"A2C_eval_run_{tb_log_suffix}")

# %%
eval_env.fill_table = True
obs = eval_env._set_initial_state(initial_state_status=3)
while not eval_env.done:
    action, _states = eval_model.predict(obs, deterministic=True)
    obs, reward, done, info = eval_env.step(action)
eval_env.render()
eval_env.table

# %%
evaluate_policy(eval_model, eval_env)

# %% [markdown]
# ### Tensorboard
# Start Tensorboard on port 6006 and open it in a browser.

# %%
if 1==0:
    pid = subprocess.Popen(['tensorboard', '--logdir', f'./{tensorboard_log}', '--port', '6006'])
    os.system('sleep 5')
    webbrowser.open('http://localhost:6006')

# %%
# Alternatively, load the TensorBoard notebook extension
# # %load_ext tensorboard
# # %tensorboard --logdir ./rampup_tensorboard/

# %% [markdown]
# To wrap up, kill the Tensorboard process.

# %%
if 1==0:
    os.system('kill -9 $(lsof -t -i:6006)')

# %%
