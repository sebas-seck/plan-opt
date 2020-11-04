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
# # 4 Finding a suitable observation space

# %%
import numpy as np
from gym import spaces

# %% [markdown]
# Definition of observation space s1:
#
# `gym.spaces.Box(<min of values>, <max of values>, [<fleet size>, <horizon (visible timeframe)>, <dimension of features>])`

# %%
s1 = spaces.Box(0, 1, [1, 28, 2])
print(s1)
s1.sample()

# %%
s2 = spaces.Tuple((spaces.Discrete(4), spaces.Box(0, 1, [28])))
print(s2)
s2.sample()

# %%
s3 = spaces.Tuple((spaces.Discrete(4), spaces.Discrete(28), spaces.Box(0, 1, [28])))
print(s3)
s3.sample()

# %%
s4 = spaces.Box(0, 1, [1])
print(s4)
s4.sample()

# %%
s5 = spaces.Box(low=np.array([0, 1, 2, 3, 4]), high=np.array([10, 20, 30, 40, 50]))
print(s5)
s5.sample()

# %%
# 2 resources, 5 features (normalized demand and dummy encoded status), horizon of 28
s6 = spaces.Box(0, 1, [5, 2, 28], dtype=np.float32)
print(s6)
s6.sample()

# %%
# 2 resources, 5 features (normalized demand and dummy encoded status), horizon of 28
s7 = spaces.Box(0, 1, [2, 4, 28], dtype=np.float32)
print(s7)
s7.sample()

# %%
# 2 resources, 5 features (normalized demand and dummy encoded status), horizon of 28
s7 = spaces.Box(0, 1, [2, 4, 28], dtype=np.float32)
print(s7)
s7.sample()

# %%
# 2 resources, 5 features (normalized demand and dummy encoded status), horizon of 28
# different bounds for each dimension...
s8 = spaces.Box(low=0, high=1, shape=[1, 5, 5])
print(s8)
s8.sample()

# %%
low = np.array([[[0], [0], [0], [0], [0]]])
print("LOW\n", low.shape, "\n", low)
high = np.array([[255], [1], [1], [1], [1]])
print("\nHIGH\n", high.shape, "\n", high)

# %% [markdown]
# ### Hard-coded observation space for shape (1, 5, 5)

# %%
low = np.array(
    [
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    ],
    dtype=np.uint8,
)
print("LOW\n", low.shape, "\n", low)
high = np.array([255, 1, 1, 1, 1])
high = np.array(
    [
        [
            [255, 255, 255, 255, 255],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ]
    ],
    dtype=np.uint8,
)
print("\nHIGH\n", high.shape, "\n", high)

# %%
# 2 resources, 5 features (normalized demand and dummy encoded status), horizon of 28
# different bounds for each dimension...
s9 = spaces.Box(low=low, high=high, shape=[1, 5, 5], dtype=np.uint8)
print(s9)
s9.sample()

# %% [markdown]
# ### Generated observation space highs for shape (Resources, Timeframe, Features)
# The below logic made its way to `_generate_obs_space_HL()` in `rampup-v1`.

# %%
fleet_size, horizon, features = 1, 8, 6
bound_hl = "high"

if bound_hl == "low":
    bound = 0
else:
    bound = 1

obs_hl = [[]] * fleet_size
l = [[]] * features
for i in range(fleet_size):

    for j in range(features):
        # only the first feature goes up to 255
        if j == 0 and bound_hl == "high":
            bound = 255
        else:
            bound = 1
        l[j] = [bound] * horizon
    obs_hl[i] = l

np.array(obs_hl, dtype=np.uint8)

# %%
fleet_size, horizon, features = 1, 8, 6
bound_hl = "low"

if bound_hl == "low":
    bound = 0
else:
    bound = 1

obs_hl = [[]] * fleet_size
l = [[]] * features
for i in range(fleet_size):

    for j in range(features):
        # only the first feature goes up to 255
        if j == 0 and bound_hl == "high":
            bound = 255
        elif bound_hl == "high":
            bound = 1
        l[j] = [bound] * horizon
    obs_hl[i] = l

np.array(obs_hl, dtype=np.uint8)

# %%
