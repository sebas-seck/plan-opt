# -*- coding: utf-8 -*-

import random

import gym
import numpy as np
from stable_baselines3.common.env_checker import check_env

from plan_opt.demand import Demand
from plan_opt.envs.rampup2 import LEGAL_CHANGES


def env_health(config, env=None, first_step=False, random_steps=0, verbose=0):

    if env is None:
        demand = Demand()
        demand.generate_demand()
        demand.add_sudden_change()
        demand.data = np.around(demand.data)
        demand.data = demand.data.astype("int32")
        env = gym.make(config["ENV_ID"]).create(config, demand)

    if verbose > 0:
        a = env.observation_space.sample()
        print("Observation Shape:", a.shape)
        print("Observation Length:", len(env.obs_demand))
        print("Observation Sample:\n", a)

    check_env(env)

    if first_step:
        obs = env._set_initial_state(initial_state_status=3)
        obs, reward, done, info = env.step(2)
        print_step_details(env, obs, reward, done, info)

    if random_steps > 0:
        for i in range(random_steps):
            print("Random step:\t", i)
            a = env.reset()
            action = random.sample(LEGAL_CHANGES[env.obs_last_legal_status], 1)[0]
            obs, reward, done, info = env.step(action)
            print_step_details(env, obs, reward, done, info)


def print_step_details(env, o, r, d, i):
    """Prints all available detail about a single step.

    Action surroundings range from one timestep back and `attr`:horizon into the future.
    Args:
        env (RampupEnv): [description]
        o (np.array): [description]
        r (int): [description]
        d (bool): [description]
        i (Dict): [description]
    """
    i_long = ""
    for k, v in i.items():
        i_long += "\n  {:<25}{}".format(k, v)
    print(
        "Timestep:\t",
        env.state_time,
        # f"{env.state_time-1} -> {env.state_time}",
        "\nAction:\t\t",
        env.obs_last_legal_status,
        "\nDemand:\t\t",
        env.demand.data[env.state_time],
        "\nReward:\t\t",
        r,
        "\nDone:\t\t",
        d,
        "\nInfo:\t\t",
        i_long,
        "\nShape:\t\t",
        o.shape,
        "\nObservation:\n",
        o,
        "\n",
    )
