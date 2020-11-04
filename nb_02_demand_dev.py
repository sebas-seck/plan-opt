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
# # 2 Development of Synthetic Demand
# Building a synthetic demand function from an assumed near-standstill is approached by stacking various streched cosines, data points are then randomly chosen through stacking various distributions around the demand function.

# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
# %pylab inline
pylab.rcParams["figure.figsize"] = (15, 6)

# %% [markdown]
# ## Demand Curve
# ### General Parameters

# %%
years = 3
N = 365 * years
x = np.arange(1, N)
np.random.seed(416)

# %% [markdown]
# ### Weeks

# %%
y_weekly = 15 * -np.cos(2 * x) + 15
plt.plot(x, y_weekly)
plt.show()

# %% [markdown]
# This is what weekly looks like for 4 weeks

# %%
x_weekly_demo = np.arange(1, 28)
y_weekly_demo = 15 * -np.cos(2 * x_weekly_demo) + 15
plt.plot(x_weekly_demo, y_weekly_demo)
plt.show()

# %% [markdown]
# ### Summer

# %%
y_summer = 15 * -np.cos(1 / (360 / 2) * np.pi * x)
plt.plot(x, y_summer)
plt.show()

# %% [markdown]
# ### Spring & Autumn Peaks

# %%
y_peaks = 8 * -np.cos((1 / (360 / 4)) * np.pi * x)
plt.plot(x, y_peaks)
plt.show()

# %% [markdown]
# ### Short-term recovery from a near-standstill

# %%
y_recover_short = x ** (1 / 3) * 2
plt.plot(x, y_recover_short)
plt.show()

# %% [markdown]
# ### Long-term recovery

# %%
y_recover_long = 1 / (1 + np.exp(-(x - 365) / 80)) * 20
plt.plot(x, y_recover_long)
plt.show()

# %% [markdown]
# ### Growth

# %%
y_growth = x * 0.025
plt.plot(x, y_growth)
plt.show()

# %% [markdown]
# ### Combine all to create reference curve

# %%
y_all = y_weekly + y_summer + y_peaks + y_recover_short + y_recover_long + y_growth + 20
plt.plot(x, y_all)
plt.show()

# %% [markdown]
# ## Data around Demand Curve
# ### Poisson
# Sensible with fluctuations at both ends

# %%
# cumulative poisson distribution
y_poisson = np.random.poisson(40, N - 1) * 4 - 100
plt.subplot(1, 2, 1)
plt.plot(np.sort(y_poisson), x)

y = y_all + y_poisson
y = y.clip(min=0)  # just in case a negative number occurs
plt.subplot(1, 2, 2)
plt.scatter(x, y, alpha=0.5)

plt.show()

# %% [markdown]
# ### Gamma
# Baseline is a bit stronger here

# %%
# cumulative poisson distribution around values
y_gamma = np.random.gamma(2, 20, N - 1)  # shape, scale, size
plt.subplot(1, 2, 1)
plt.plot(np.sort(y_gamma), x)

y = y_all + y_gamma
y = y.clip(min=0)  # just in case a negative number occurs
plt.subplot(1, 2, 2)
plt.scatter(x, y, alpha=0.5)

plt.show()

# %% [markdown]
# ### Cauchy
# Add some extreme outlier every now and then

# %%
y_cauchy = np.random.standard_cauchy(N - 1)
plt.subplot(1, 2, 1)
plt.plot(np.sort(y_cauchy), x)

y = y_all + y_cauchy - np.mean(y_cauchy)
y = y.clip(min=0)  # just in case a negative number occurs
plt.subplot(1, 2, 2)
plt.scatter(x, y, alpha=0.5)

plt.show()

# %% [markdown]
# ### Combined Synthetic Demand

# %%
y = y_all + y_poisson + y_gamma + y_cauchy - np.mean(y_cauchy)
y = y.clip(min=0)  # just in case a negative number occurs
plt.scatter(x, y, alpha=0.5)
plt.show()

# %% [markdown]
# ### Sudden Changes
# At a random point in time there is a spike or dip with mean 100 and standard deviation of 50, negative values are clipped and

# %% [markdown]
# Variety of sudden changes

# %%
mu, sigma = 1000, 50
ndist = np.random.normal(mu, sigma, 1000)
ndist = ndist.clip(min=0)
count, bins, ignored = plt.hist(ndist, 30, density=True)
plt.plot(
    bins,
    1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-((bins - mu) ** 2) / (2 * sigma ** 2)),
    linewidth=2,
    color="r",
)
plt.show()

# %%
# start at a random point in time
start = np.random.choice(N)
magnitude = np.random.normal(1000, 50)  # .clip(min=0)
if np.random.rand() < 0.5:  # sudden change up or down
    magnitude = magnitude * -1
steepness = np.random.normal(
    50, 25
)  # random value from normal distribution within 25 standard deviations and mean 50
x_sudden = np.arange(1, N - start)
y_sudden = 1 / (1 + np.exp(-(x_sudden - 365) / steepness)) * magnitude
plt.subplot(1, 2, 1)
plt.plot(x_sudden, y_sudden)

y = y_all
y[start:] += y_sudden
y = y.clip(min=0)  # just in case a negative number occurs
plt.subplot(1, 2, 2)
plt.scatter(x, y, alpha=0.5)
plt.show()

# %% [markdown]
# ### Combined Synthetic Demand with Sudden Change

# %%
y = y_all + y_poisson + y_gamma + y_cauchy - np.mean(y_cauchy)
y[start:] += y_sudden
y = y.clip(min=0)  # just in case a negative number occurs
plt.scatter(x, y, alpha=0.5)
plt.show()

# %% [markdown]
# All ideas from above are taken to `demand.py` into the class `Demand` which easily generates synthtic demand arrays.

# %%
