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
# Starting with `rampup-v3`, a configuration is mandatory for the creation of an environment to have a consistent way of configuring.

# %%
import os
import subprocess
import webbrowser

from plan_opt.create import env_cb_creator
from plan_opt.demand import Demand
from plan_opt.demand_small_samples import four_weeks_uprising
from plan_opt.env_health import env_health
from plan_opt.train_eval import train_and_evaluate

# %% [markdown]
# config = {
#     "tensorboard_log": "logs/rampup_tensorboard/",
#     "timesteps": 50000,
#     "eval_episodes": 20,
#     "repetitions": 5,
#     "show_table": False,
# }

# %%
config = {
    # ENVIRONMENT CONFIGURATION
    "ENV_ID": "rampup-v2",
    "PUNISH_ILLEGAL": False,
    # WORKFLOW CONFIGURATION
    "TENSORBOARD_LOG": "logs/rampup_tensorboard/",
    "TIMESTEPS": 50000,
    "REPETITIONS": 5,
    "EVAL_EPISODES": 50,
    "SHOW_TABLE": False,
    "LEARNING_RATE": 0.0007,
}

# %%
demand_4W = Demand(period=len(four_weeks_uprising), data=four_weeks_uprising)
env_4W, eval_callback_4W, demand_4W = env_cb_creator(config, demand_4W)

# %% [markdown]
# ### Quick Health Check

# %%
env_health(config, first_step=False, random_steps=0, verbose=0)

# %% [markdown]
# ### Train and Evaluate
# Results are foul, the environment cannot be trained to understand simple rules, which actions may follow other actions. The observation seems overly complicated, including the human view on upcoming demand.

# %%
best_model, train_env, eval_env = train_and_evaluate(
    config=config,
    train_env=env_4W,
    eval_env=env_4W,
    eval_callback=eval_callback_4W,
    tb_log_name=f"A2C_{str(config['TIMESTEPS'])[:-3]}k_train4W_eval4W_legal",
)

# %% [markdown]
# ### Tensorboard

# %%
if 1 == 1:
    pid = subprocess.Popen(
        ["tensorboard", "--logdir", f"./{config['TENSORBOARD_LOG']}", "--port", "6006"]
    )
    os.system("sleep 5")
    webbrowser.open("http://localhost:6006")

# %% [markdown]
# ### Interpretation

# %% [markdown]
# With punishment of illegal action transitions when training and evaluating on 4W, results are foul! Evaluation, most often, gets stuck and remains negative. Only some auspicious occurences turn out positive.
#
# ![Evaluation](docs/nb10-eval.png)

# %%
