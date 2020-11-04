# -*- coding: utf-8 -*-

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
    for i in range(config["repetitions"]):
        print(f"\nRunning repetition {i+1}/{config['repetitions']}...")
        model = A2C(
            "MlpPolicy",
            train_env,
            policy_kwargs=dict(optimizer_class=RMSpropTFLike),
            tensorboard_log=config["tensorboard_log"],
            verbose=0,
        )
        model.learn(
            total_timesteps=config["timesteps"],
            callback=eval_callback,
            tb_log_name=tb_log_name,
        )

        if config["show_table"]:
            eval_env.fill_table = True
            obs = eval_env._set_initial_state(initial_state_status=3)
            while not eval_env.done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)

        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=config["eval_episodes"]
        )
        economic_potential = eval_env.demand.economic_potential()
        lost_potential = economic_potential - max(mean_reward, 0)
        lost_potential_perc = round(lost_potential / economic_potential * 100, 4)

        summary = "POLICY EVALUATION RESULTS"
        summary += f"\nEvaluated episodes:\t{config['eval_episodes']}"
        summary += f"\nMean reward:\t\t{mean_reward}"
        summary += f"\nStandard deviation:\t{std_reward}"
        summary += f"\nEconomic potential:\t{economic_potential}"
        summary += f"\nLost potential:\t\t{lost_potential} ({lost_potential_perc}%)"
        print(summary)
        if config["show_table"]:
            print("Sample Episode Table")
            display(eval_env.episode_table)
