# -*- coding: utf-8 -*-

import gym
import numpy as np
import pandas as pd
from gym import spaces

action_list = [
    "OPERATE",  # 0
    "PREPARE",  # 1
    "PARK",  # 2
    "STORE",  # 3
]

legal_changes = {
    0: {0, 1, 2},  # from OPERATE to OPERATE, PREPARE, or PARK
    1: {1, 0, 2},  # from PREPARE to PREPARE, OPERATE, or PARK
    2: {2, 1, 3},  # from PARK to PARK, PREPARE, or STORE
    3: {3, 2},  # from STORE to STORE or PARK
}
demand = [
    0,
    0,
    0,
    0,
    0,
    0,
    0,  # week 1
    0,
    0,
    0,
    2,
    0,
    3,
    1,  # week 2
    0,
    0,
    60,
    25,
    32,
    87,
    56,  # week 3
    92,
    83,
    29,
    40,
    86,
    70,
    45,  # week 4
]


class RampupEnv0(gym.Env):
    """Creates an environment to run ramp-up simulations.

    Environment follows gym interface53
    """

    action_list = [
        "OPERATE",  # 0
        "PREPARE",  # 1
        "PARK",  # 2
        "STORE",  # 3
    ]

    def __init__(self, verbose=False, horizon=10):
        super(RampupEnv0, self).__init__()

        self.fleet_size = 1
        self.horizon = horizon
        self.state_time = 0  # time component of state
        self.state_status = np.zeros(self.horizon)  # status component of state
        # action_space: The Space object corresponding to valid actions
        # observation_space: The Space object corresponding to valid observations
        self.action_space = spaces.Discrete(len(action_list))
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(self.fleet_size), spaces.Discrete(self.horizon))
        )

        self.total_reward = 0
        self.verbose = verbose

    def translate_action(self, action):
        if action == 0:
            action_description = "OPERATE_IN_REV_SERVICE"
            if demand[self.state_time] > 30:
                reward = 15000
            else:
                reward = -3000
        elif action == 1:
            action_description = "PREPARE_FOR_FLIGHT"
            reward = -2000
        elif action == 2:
            action_description = "PARK_AT_BASE"
            reward = -1000
        elif action == 3:
            action_description = "STORE_AT_BASE_ST"
            reward = -500

        if self.verbose:
            print(f"Action {action} leads to state {action_description}")

        return action_description, reward

    def step(self, action):
        # Increment time component of state
        self.state_time += 1

        action_description, reward = self.translate_action(action)

        # Adjust the status component of state
        self.state_status[self.state_time] = action

        obs = self.state_status[0 : self.state_time]
        done = bool(self.state_time == self.horizon - 1)
        info = {}

        if self.verbose:
            print(f"Action taken: {action_description}")

        self.total_reward += reward

        return obs, reward, done, info

    def undo_step(self, o, r, d, i):
        self.state_time -= 1
        self.total_reward -= r

        return o, r, d, i

    def render(self):
        if self.verbose:
            print(f"Reward so far: {self.total_reward}")

    def reset(self):
        # Initialize the agent in the first field
        self.state_time = 0
        self.total_reward = 0

    def set_state(self, state):
        state_time, state_status = state
        self.state_time = state_time
        self.state_status[self.state_time] = state_status
