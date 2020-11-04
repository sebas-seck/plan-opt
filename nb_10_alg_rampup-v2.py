# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
# # 10 Review of rampup-v2 with A2C

# %%
from plan_opt.demand_small_samples import four_weeks_uprising
from plan_opt.train_eval import train_and_evaluate
from plan_opt.create import env_cb_creator
from plan_opt.demand import Demand

# %%
config = {
    "tensorboard_log": "logs/rampup_tensorboard/",
    "timesteps": 25000,
    "eval_episodes": 50,
    "repetitions": 2,
    "show_table": True,
}

# %%
demand_4W = Demand(period=len(four_weeks_uprising), data=four_weeks_uprising)
env_4W, eval_callback_4W, demand_4W = env_cb_creator(demand=demand_4W)

# %%
train_and_evaluate(
    config=config,
    train_env=env_4W,
    eval_env=env_4W,
    eval_callback=eval_callback_4W,
    tb_log_name=f"A2C_{str(config['timesteps'])[:-3]}k_train4W_eval4W",
)

# %%
