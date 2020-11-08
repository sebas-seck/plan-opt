# -*- coding: utf-8 -*-

import gym
from stable_baselines3.common.callbacks import EvalCallback

from plan_opt.demand import Demand


def env_cb_creator(config, demand=None, seed=None):
    """Creates all instances needed for Rampup RL.

    To work, exactly one argument must remain None, the other must be
    provided (XOR logical operation).

    Args:
        demand (Demand, optional): [description]. Defaults to None.
        seed (int, optional): Seed to initialize random number
            generator. Defaults to None.

    Returns:
        Tuple(RampupEnv, EvalCallback, Demand): Instances ready to use
            with RL.
    """
    # if exactly one argument is not none
    if (demand is None) ^ (seed is None):
        if demand is None:
            demand = Demand(seed=seed)
            demand.generate_demand()
            demand.show()
        else:
            demand.show(only_data=True)
        env = gym.make(config["ENV_ID"]).create(config=config, demand=demand)
        callback = EvalCallback(
            env,
            best_model_save_path="../logs/",
            log_path="../logs/",
            eval_freq=100,
            deterministic=True,
            render=False,
            verbose=0,
        )
        return env, callback, demand
    else:
        print('Please provide either "demand" or a "seed"!')
