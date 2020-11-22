# -*- coding: utf-8 -*-

import math

import numpy as np
from IPython.core.display import display
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, tb_log_name, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.tb_log_name = tb_log_name

    def _on_step(self) -> bool:
        if "alpha" in self.model.policy_kwargs["optimizer_kwargs"]:
            value = self.model.policy_kwargs["optimizer_kwargs"]["alpha"]
            self.logger.record("train/policy_alpha", value)

        if "eps" in self.model.policy_kwargs["optimizer_kwargs"]:
            value = self.model.policy_kwargs["optimizer_kwargs"]["eps"]
            self.logger.record("train/policy_eps", value)

        # if "weight_decay" in self.model.policy_kwargs["optimizer_kwargs"]:
        #     value = self.model.policy_kwargs["optimizer_kwargs"]["weight_decay"]
        #     self.logger.record("train/weight_decay", value)

        # values always exist
        value = self.model.gamma
        self.logger.record("train/gamma", value)

        return True

    def _on_training_end(self) -> None:
        # ADD HPARAMS
        hparams = {
            "policy_weight_decay": self.model.policy_kwargs["optimizer_kwargs"][
                "weight_decay"
            ]
        }

        # self.logger.TensorBoardOutputFormat.writer.add_hparams(hparams, {"hparam/test1":3})
        # print(self.logger.get_dir())
        from torch.utils.tensorboard import SummaryWriter

        x = self.logger.get_dir().rsplit("/", 1)[1]

        metric_dict = {}
        for k, v in hparams.items():
            metric_dict["hparam/" + k] = v

        # writer = SummaryWriter(log_dir=self.logger.get_dir())
        writer = SummaryWriter(log_dir="logs/rampup_tensorboard/")
        writer.add_hparams(hparam_dict=hparams, metric_dict=metric_dict, run_name=x)


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
            policy_kwargs=config["POLICY_KWARGS"],
            tensorboard_log=config["TENSORBOARD_LOG"],
            verbose=0,
        )

        model.learn(
            total_timesteps=config["TIMESTEPS"],
            callback=[eval_callback, TensorboardCallback(tb_log_name)],
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
