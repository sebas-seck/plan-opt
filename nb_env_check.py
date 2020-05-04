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

# %% [markdown]
# ## Environment Check
# Stable Baselines3 offers an [environment checker](https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html?highlight=check_env) to test an environment for conformity with the Gym API. Warnings are returned if the environment does not conform.

# %%
import numpy as np
from plan_opt.demand import Demand
from plan_opt.envs.rampup1 import RampupEnv
from stable_baselines3.common.env_checker import check_env

# %%
D = Demand(seed=3348)
D.generate_demand()
D.add_sudden_change()

# %%
env = RampupEnv(demand=D.data)

# %%
a = env.observation_space.sample()
a

# %%
a.shape

# %%
b = env.reset()
b

# %%
b.shape

# %%
len(env.obs_demand)

# %%
np.set_printoptions(threshold=np.inf)
env.obs_demand

# %%
env.observation_space

# %%
check_env(env)

# %%
