# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
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
# # 11 `rampup-v3` Check
# Brief notebook to review variables returned after each step.

# %%
import random
import gym

from plan_opt.demand import Demand
from plan_opt.envs.rampup3 import LEGAL_CHANGES, DEFAULT_CONFIG
from plan_opt.demand_small_samples import four_weeks_uprising
from plan_opt.env_health import print_step_details

# %%
demand = Demand(period=len(four_weeks_uprising), data=four_weeks_uprising)
demand.show(only_data=True)
env = gym.make("rampup-v3").create(DEFAULT_CONFIG, demand)

# %% [markdown]
# ### First step

# %%
obs = env._set_initial_state(initial_state_status=3)
obs, reward, done, info = env.step(2)
print_step_details(env, obs, reward, done, info)

# %% [markdown]
# ### Step at random point in time

# %%
for i in range(5):
    print(f"Random step {i}")
    a = env.reset()
    action = random.sample(LEGAL_CHANGES[env.obs_last_legal_status], 1)[0]
    obs, reward, done, info = env.step(action)
    print_step_details(env, obs, reward, done, info)

# %%
env.observation_space.__dict__

# %%
