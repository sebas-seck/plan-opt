# -*- coding: utf-8 -*-

import numpy as np
import math

from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import A2C
from IPython.core.display import display


def train_and_evaluate(config, train_env, eval_env, eval_callback, tb_log_name):
    """Trains and evaluates on separate environments with config parameters.

    Args:
        config (Dict): Relevant parameters.
        train_env (RampupEnv2): Environment to train with.
        eval_env (RampupEnv2): Environment to evaluate with.
        eval_callback (EvalCallback): Callback wrapper for evaluation
            callbacks during training.
        tb_log_name (str): Log name to identify metrics on Tensorboard.
    """
    best_model = None
    best_mean = -np.inf

    # sched_LR = LinearSchedule(config["TIMESTEPS"], 0.005, 0.00001)

    for i in range(config["REPETITIONS"]):
        print(f"\nRunning repetition {i+1}/{config['REPETITIONS']}...")
        model = A2C(
            "MlpPolicy",
            train_env,
            learning_rate=config["LEARNING_RATE"],
            # learning_rate=[lrate],
            # learning_rate=0.1 * 1/(1 + 0.0 * 1),
            policy_kwargs=dict(optimizer_class=RMSpropTFLike),
            tensorboard_log=config["TENSORBOARD_LOG"],
            verbose=0,
        )
        model.learn(
            total_timesteps=config["TIMESTEPS"],
            callback=eval_callback,
            tb_log_name=tb_log_name,
        )

        if config["SHOW_TABLE"]:
            eval_env.fill_table = True
            obs = eval_env._set_initial_state(initial_state_status=3)
            while not eval_env.done:
                action, _states = model.predict(obs, deterministic=False)
                obs, reward, done, info = eval_env.step(action)

        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=config["EVAL_EPISODES"]
        )
        if mean_reward > best_mean:
            best_mean = mean_reward
            best_model = model
        if config["PUNISH_ILLEGAL"]:
            economic_potential = eval_env.demand.economic_potential_no_illegal()
        else:
            economic_potential = eval_env.demand.economic_potential()
        lost_potential = economic_potential - max(mean_reward, 0)
        lost_potential_perc = round(lost_potential / economic_potential * 100, 4)

        summary = "POLICY EVALUATION RESULTS"
        summary += f"\nEvaluated episodes:\t{config['EVAL_EPISODES']}"
        summary += f"\nMean reward:\t\t{mean_reward}"
        summary += f"\nStandard deviation:\t{std_reward}"
        summary += f"\nEconomic potential:\t{economic_potential}"
        summary += f"\nLost potential:\t\t{lost_potential} ({lost_potential_perc}%)"
        print(summary)
        if config["SHOW_TABLE"]:
            print("Sample Episode Table")
            display(eval_env.episode_table)

    return best_model, train_env, eval_env
