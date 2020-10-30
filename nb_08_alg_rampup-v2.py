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
# # 8 Review of `rampup-v2` with A2C

# %%
import os
import subprocess
import numpy as np
import webbrowser
import gym
from gym import spaces
from stable_baselines3 import A2C
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

from plan_opt.demand import Demand
from plan_opt.demand_small_samples import four_weeks_uprising

# %% [markdown]
#
#
# What I do here is to create training environments using the simple 4-weeks demand and a random 3-years demand. Then, performance per model is evaluated for the 4-weeks demand.

# %% [markdown]
# ## Environments and Evaluation callbacks
# ### 4W -> `four_weeks_uprising` demand
# Using an overseeable demand for four weeks only.
# The environment is created from the Gym registry with the custom 4 weeks demand.

# %%
demand_4W = Demand(period=len(four_weeks_uprising), data=four_weeks_uprising)
demand_4W.show(only_data=True)
env_4W = gym.make("rampup-v2").create(demand_4W)

# %%
# create eval callback
eval_callback_4W = EvalCallback(
    env_4W,
    best_model_save_path="./logs/",
    log_path="./logs/",
    eval_freq=5000,
    deterministic=True,
    render=False,
    verbose=0,
)

# %% [markdown]
# ### 3YS1256 -> 3 years demand at seed 1256

# %%
demand_3YS1256 = Demand(seed=1256)
demand_3YS1256.generate_demand()
demand_3YS1256.info()
demand_3YS1256.show()
env_3YS1256 = gym.make("rampup-v2").create(demand_3YS1256)

# %%
# create eval callback
eval_callback_3YS1256 = EvalCallback(
    env_3YS1256,
    best_model_save_path="./logs/",
    log_path="./logs/",
    eval_freq=5000,
    deterministic=True,
    render=False,
    verbose=0,
)


# %%
def env_cb_creator(seed):
    demand = Demand(seed=seed)
    demand.generate_demand()
    env = gym.make("rampup-v2").create(demand)
    callback = EvalCallback(
        env,
        best_model_save_path="./logs/",
        log_path="./logs/",
        eval_freq=5000,
        deterministic=True,
        render=False,
        verbose=0,
    )
    return env, callback


# %% [markdown]
# ## Training and Evaluation

# %% [markdown]
# The procedure to train and evaluate models on different environments is defined in `train_and_evaluate()`. Due to unstable training observed, as [recommended by SB3](https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html?highlight=learn#stable_baselines3.a2c.A2C.learn), I change the optimizer to RMSpropTFLike for more stable training.
#
# The environment is not vectorized for parallel use, as rewards during the evaluation callback do not make it to the tensorboard.

# %%
def train_and_evaluate(train_env, eval_env, eval_callbacks, tb_log_name, episodes=10):

    model = A2C(
        "MlpPolicy",
        train_env,
        policy_kwargs=dict(optimizer_class=RMSpropTFLike),
        tensorboard_log=tensorboard_log,
        verbose=0,
    )
    model.learn(
        total_timesteps=timesteps, callback=eval_callbacks, tb_log_name=tb_log_name
    )

    # eval_env.fill_table = True
    # obs = eval_env._set_initial_state(initial_state_status=3)
    # while not eval_env.done:
    #    action, _states = model.predict(obs, deterministic=True)
    #    obs, reward, done, info = eval_env.step(action)
    # eval_env.render()

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=episodes)
    economic_potential = eval_env.demand.economic_potential()
    lost_potential = economic_potential - max(mean_reward, 0)
    lost_potential_perc = round(lost_potential / economic_potential * 100, 4)

    summary = "POLICY EVALUATION RESULTS"
    summary += f"\nEvaluated episodes:\t{episodes}"
    summary += f"\nMean reward:\t\t{mean_reward}"
    summary += f"\nStandard deviation:\t{std_reward}"
    summary += f"\nEconomic potential:\t{economic_potential}"
    summary += f"\nLost potential:\t\t{lost_potential} ({lost_potential_perc}%)"
    print(summary)
    # display(eval_env.episode_table)


# %%
timesteps = 50000
tensorboard_log = "logs/rampup_tensorboard/"
tb_log_suffix = f"{str(timesteps)[:-3]}k"
print(f"Tensorboard logs saved with suffix {tb_log_suffix}")

# %% [markdown]
# #### Training 4W (4 weeks), Evaluation 4W

# %%
train_and_evaluate(
    env_4W, env_4W, eval_callback_4W, f"A2C_{tb_log_suffix}_train4W_eval4W"
)

# %% [markdown]
# #### Training 4W (4 weeks), Evaluation 3YS1256

# %%
train_and_evaluate(
    env_4W,
    env_3YS1256,
    eval_callback_3YS1256,
    f"A2C_{tb_log_suffix}_train4W_eval3YS1256",
)

# %% [markdown]
# #### Training 3YR (3 years random), Evaluation 4W

# %%
train_and_evaluate(
    "rampup-v2", env_4W, eval_callback_4W, f"A2C_{tb_log_suffix}_train3YR_eval4W"
)

# %% [markdown]
# #### Training 3YR, Evaluation 3YS1256

# %%
train_and_evaluate(
    "rampup-v2",
    env_3YS1256,
    eval_callback_3YS1256,
    f"A2C_{tb_log_suffix}_train3YR_eval3YS1256",
)

# %% [markdown]
# #### Training 3YS1256, Evaluation 3YS1256

# %%
train_and_evaluate(
    env_3YS1256,
    env_3YS1256,
    eval_callback_3YS1256,
    f"A2C_{tb_log_suffix}_train3YS1256_eval3YS1256",
)

# %% [markdown]
# #### Training 3YS1256, Evaluation 3YS3348

# %%
env_3YS3348, eval_callback_3Y3348 = env_cb_creator(seed=3348)
env_3YS3348.demand.show(only_data=True)
train_and_evaluate(
    env_3YS1256,
    env_3YS3348,
    eval_callback_3Y3348,
    f"A2C_{tb_log_suffix}_train3YS1256_eval3YS3348",
    20,
)

# %% [markdown]
# ## Results (move to env that cares for punishing illegal moves)
#
# With illegal punishment, training and evaluation on 4W! Evaluation, most often, gets stuck and remains negative. Only some evaluations turn out positive.
#
# ![Evaluation](docs/nb08-eval.png)

# %% [markdown]
# ### Tensorboard
# Start Tensorboard on port 6006 and open it in a browser.

# %%
if 1 == 0:
    pid = subprocess.Popen(
        ["tensorboard", "--logdir", f"./{tensorboard_log}", "--port", "6006"]
    )
    os.system("sleep 5")
    webbrowser.open("http://localhost:6006")

# %%
# Alternatively, load the TensorBoard notebook extension
# # %load_ext tensorboard
# # %tensorboard --logdir ./rampup_tensorboard/

# %% [markdown]
# To wrap up, kill the Tensorboard process.

# %%
if 1 == 0:
    os.system("kill -9 $(lsof -t -i:6006)")
