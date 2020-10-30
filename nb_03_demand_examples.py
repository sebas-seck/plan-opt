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
# # 3 Demand Examples
# Selection of Demand instances with different seeds, sudden changes and some completely random instances.

# %%
# %load_ext autoreload
# %autoreload 2
# %pylab inline
pylab.rcParams["figure.figsize"] = (15, 6)

# %%
from plan_opt.demand import Demand

# %% [markdown]
# ## Seeded Sample Demands

# %%
a = Demand(seed=612)
a.generate_demand()
a.info()
a.show()

# %%
b = Demand(seed=7)
b.generate_demand()
b.info()
b.show()

# %%
c = Demand(seed=165)
c.generate_demand()
c.info()
c.show()

# %%
d = Demand(seed=1256)
d.generate_demand()
d.info()
d.show()

# %%
e = Demand(seed=3348)
e.generate_demand()
e.info()
e.show()

# %% [markdown]
# ## Seeded Sample Demand with Sudden Change

# %%
w = Demand(seed=1256)
w.generate_demand()
w.add_sudden_change()
w.info()
w.show()

# %%
v = Demand(seed=3348)
v.generate_demand()
v.add_sudden_change()
v.info()
v.show()

# %% [markdown]
# ## Multiple Random Sample Demands

# %%
demand_dict = {}
for i in range(10):
    demand_dict[i] = Demand()
    demand_dict[i].generate_demand()
    demand_dict[i].apply_sudden(probability=0.8)
    demand_dict[i].info()
    demand_dict[i].show()

# %%
