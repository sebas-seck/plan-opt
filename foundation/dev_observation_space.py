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
from gym import spaces

# %% [markdown]
# Definition of observation space s1:
#
# `gym.spaces.Box(<min of values>, <max of values>, [<no of resources>, <horizon/timeframe>, <dimension of features>])`

# %%
s1 = spaces.Box(0, 1, [1, 28, 2])
print(s1)
s1.sample()

# %%
s2 = spaces.Tuple((spaces.Discrete(4), spaces.Box(0,1,[28])))
print(s2)
s2.sample()

# %%
s3 = spaces.Tuple((spaces.Discrete(4), spaces.Discrete(28), spaces.Box(0,1,[28])))
print(s3)
s3.sample()

# %%
s4 = spaces.Box(0,1,[1])
print(s4)
s4.sample()

# %%
s5 = spaces.Box(low=np.array([0, 1, 2, 3, 4]), high=np.array([10, 20, 30, 40, 50]))
print(s5)
s5.sample()

# %%
# 2 resources, 5 features (normalized demand and dummy encoded status), horizon of 28
s6 = spaces.Box(0,1,[5, 2, 28], dtype=np.float32)
print(s6)
s6.sample()

# %%
# 2 resources, 5 features (normalized demand and dummy encoded status), horizon of 28
s7 = spaces.Box(0,1,[2, 4, 28], dtype=np.float32)
print(s7)
s7.sample()
