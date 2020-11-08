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
# # 12 Review of rampup-v3 with A2C
#
# The observation in v3 is changed because it is not neccessary to provide the entire demand array. A preprocessed datum can compress all required information and discard irrelevant details. Information on precise demand is moved from the observation space to the info bit of `step()`.

# %%
import os
import subprocess
import webbrowser

from plan_opt.create import env_cb_creator
from plan_opt.demand import Demand
from plan_opt.demand_small_samples import four_weeks_uprising
from plan_opt.env_health import env_health
from plan_opt.train_eval import train_and_evaluate

# %%
config = {
    # ENVIRONMENT CONFIGURATION
    "ENV_ID": "rampup-v3",
    "REWARD_THRESHOLD": 80,
    "PUNISH_ILLEGAL": True,
    # WORKFLOW CONFIGURATION
    "TENSORBOARD_LOG": "logs/rampup_tensorboard/",
    "TIMESTEPS": 150000,
    "REPETITIONS": 15,
    "EVAL_EPISODES": 50,
    "SHOW_TABLE": False,
    "LEARNING_RATE": 0.0007,
}

# %%
tb_suffix = ""
tb_suffix += f"_{str(config['TIMESTEPS'])[:-3]}k"
if config["PUNISH_ILLEGAL"]:
    tb_suffix += f"_legal_chg"
tb_suffix

# %%
demand_4W = Demand(period=len(four_weeks_uprising), data=four_weeks_uprising)
env_4W, eval_callback_4W, demand_4W = env_cb_creator(config, demand_4W)

# %% [markdown]
# ### Quick Health Check
# - The observation is much neater compared to earlier versions

# %%
env_health(config, first_step=False, random_steps=1, verbose=0)

# %% [markdown]
# ### Train and Evaluate
# Results look much more promising, as illegal moves are clearly learned and avoided. There are significant differences between repetitions!

# %%
best_model, train_env, eval_env = train_and_evaluate(
    config=config,
    train_env=env_4W,
    eval_env=env_4W,
    eval_callback=eval_callback_4W,
    tb_log_name=f"A2C_train4W_eval4W_{tb_suffix}",
)

# %% [markdown]
# 15 repetitions over 150k episodes show results with variance remaining when applying the models!
#
# ![Evaluation](docs/nb12-eval.png)

# %% [markdown]
# ### Tensorboard

# %%
if 1 == 0:
    pid = subprocess.Popen(
        ["tensorboard", "--logdir", f"./{config['TENSORBOARD_LOG']}", "--port", "6006"]
    )
    os.system("sleep 5")
    webbrowser.open("http://localhost:6006")

# %%
