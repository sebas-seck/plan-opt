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
# ## Step Test
# How does a single step affect all parameters?

# %%
from plan_opt.demand import Demand
from plan_opt.envs.rampup1 import RampupEnv
from plan_opt.demand_small_samples import four_weeks_uprising

# %%
demand = Demand(period=len(four_weeks_uprising), data=four_weeks_uprising)
demand.show(only_data=True)
env = RampupEnv(demand.data)

# %% [markdown]
# ### First steps

# %%
obs = env._set_initial_state(initial_state_status=3)
obs, reward, done, info = env.step(2)
print(reward, done, info, obs.shape)
print(obs)

# %% [markdown]
# ### Step at random point in time

# %%
for i in range(5):
    a = env.reset()
    print(env.state_time, a)

# %%
