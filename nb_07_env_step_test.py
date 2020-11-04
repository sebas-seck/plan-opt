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
# # 7 `rampup-v1` Step Test
# Brief notebook to review variables returned after each step.

# %%
import random

from plan_opt.demand import Demand
from plan_opt.envs.rampup1 import RampupEnv1, LEGAL_CHANGES
from plan_opt.demand_small_samples import four_weeks_uprising

# %%
demand = Demand(period=len(four_weeks_uprising), data=four_weeks_uprising)
demand.show(only_data=True)
env = RampupEnv1(demand)


# %% [markdown]
# ### Step Details Helper Function
# Displays the current timestep, whether the episode if done after the timestep, the info dictionary, the shape of the observation and the observation.

# %%
def print_step_details(o, r, d, i):
    print(
        "Timestep:\t",
        env.state_time,
        "\nReward:\t\t",
        r,
        "\nDone:\t\t",
        d,
        "\nInfo:\t\t",
        i,
        "\nShape:\t\t",
        o.shape,
        "\nObservation:\n",
        o,
        "\n",
    )


# %% [markdown]
# ### First step

# %%
obs = env._set_initial_state(initial_state_status=3)
obs, reward, done, info = env.step(2)
print_step_details(obs, reward, done, info)

# %% [markdown]
# ### Step at random point in time

# %%
for i in range(5):
    print(f"Random step {i}")
    a = env.reset()
    action = random.sample(LEGAL_CHANGES[env.obs_last_legal_status], 1)[0]
    obs, reward, done, info = env.step(action)
    print_step_details(obs, reward, done, info)

# %%
