# -*- coding: utf-8 -*-
"""Module with latest environment"""

import random

import gym
import numpy as np
import pandas as pd
from gym import spaces
from plan_opt.demand import Demand

# from IPython.core.display import HTML, display

ACTION_LIST = [
'OPERATE',  # 0
'PREPARE',  # 1
'PARK',  # 2
'STORE',  # 3
]

LEGAL_CHANGES = {
    0: {0, 1, 2},
    1: {1, 0, 2},
    2: {2, 1, 3},
    3: {3, 2},
}


class RampupEnv(gym.Env):
    """Creates an environment to run ramp-up simulations.

    Environment follows the gym interface.

    Parameters:
        demand (Demand): Instance of demand data
        horizon (int): Length of future observation to be regarded
        fleet_size (int): Size of the fleet, one unit is equivalent to a
            carrier capable of holding multiple pieces of cargo
    """
    def __init__(self, demand=None, timeframe=0, verbose=False):
        super(RampupEnv, self).__init__()

        if demand is None:
            demand = Demand()
            demand.generate_demand()
            demand.apply_sudden(probability=0.3)
            self.demand = demand.data
        else:
            self.demand = demand
        self.horizon = 5
        self.fleet_size = 1
        self.done = None

        self.fill_table = False
        self.table = pd.DataFrame()

        if timeframe == 0:
            timeframe = len(self.demand)
        self.timeframe = timeframe
        self.total_reward = 0
        self.reward_range = (-3000, 15000)
        self.action_features = len(ACTION_LIST)
        self.features = self.action_features + 1

        self.state_time = 0  # time component of state
        self.state_status = np.array(np.zeros(timeframe),
                                     dtype=np.uint8,
                                     ndmin=2)  # status component of state

        # action_space: The Space object corresponding to valid actions
        self.action_space = spaces.Discrete(self.action_features)
        # observation_space: The Space object corresponding to valid observations
        self.observation_space = spaces.Box(
            low=0,
            high=254, shape=[self.fleet_size, self.features, self.horizon],
            dtype=np.uint8)
        # Normalized demand on a scale between 0 and 255!
        # self.obs_demand = self.demand / np.linalg.norm(self.demand)
        # self.obs_demand = self.demand / np.linalg.norm(self.demand) * 255
        self.obs_demand = (self.demand / np.linalg.norm(self.demand) *
                           255).astype('uint8')
        # self.obs_demand = (self.obs_demand * 255).astype('uint8')
        # Observation space needs to extend beyond the end to cover the horizon
        self.obs_demand = np.append(
            self.obs_demand, np.array([0] * self.horizon, dtype=np.uint8))

        self.obs_dummy_status = None
        self._generate_dummy_status()

        self.obs_last_legal_status = None
        self.total_reward = 0
        self.verbose = verbose

    def _generate_dummy_status(self):
        self.obs_dummy_status = {}
        for i in range(self.action_features):
            self.obs_dummy_status[i] = [0] * (self.timeframe + self.horizon)

    def _retrieve_obs(self):
        """Generate current observation window from full observation.

        A rolling window of the length of :attr:`horizon`
        """
        obs = []
        start = self.state_time
        end = self.state_time + self.horizon
        for i in range(self.fleet_size):
            # other flight must be in separate square brackets!
            obs.append(self.obs_demand[start:end])
            for j in range(self.action_features):
                obs.append(self.obs_dummy_status[j][start:end])
        obs = np.asarray([obs], dtype=np.uint8)
        return obs

    def _translate_action(self, action):
        if action == 0:
            action_description = 'OPERATE'
            if self.demand[self.state_time] > 80:
                reward = 15000
            else:
                reward = -3000
        elif action == 1:
            action_description = 'PREPARE'
            reward = -2000
        elif action == 2:
            action_description = 'PARK'
            reward = -1000
        elif action == 3:
            action_description = 'STORE'
            reward = -500

        if self.verbose:
            print(f'Action {action} leads to state {action_description}')

        return action_description, reward

    def step(self, action):

        # if action not in LEGAL_CHANGES[self.obs_last_legal_status]:
        #     # Illegal action, pick another random legal action!
        #     action = random.sample(LEGAL_CHANGES[self.obs_last_legal_status],
        #                            1)[0]

        # Increment time component of state
        self.state_time += 1
        action_description, reward = self._translate_action(action)
        # if action not in LEGAL_CHANGES[self.obs_last_legal_status]:
        #     reward = -250000
        self.obs_last_legal_status = action
        # Adjust the status component of state
        self.obs_dummy_status[action][self.state_time] = 1

        obs = self._retrieve_obs()

        self.done = bool(self.state_time == self.timeframe - 1)
        info = {}
        if self.verbose:
            print(f'Action taken: {action_description}')
        self.total_reward += reward

        if self.fill_table:
            self.table.at[self.state_time,
                          'Current demand'] = self.demand[self.state_time]
            try:
                self.table.at[self.state_time,
                              'Next demand'] = self.demand[self.state_time + 1]
            except IndexError:
                pass
            self.table.at[self.state_time, 'Action'] = action
            self.table.at[self.state_time,
                          'Action description'] = action_description
            self.table.at[self.state_time, 'Reward'] = reward

        return obs, reward, self.done, info

    def render(self):
        # if self.verbose:
        print(f'Reward so far: {self.total_reward}')

    def reset(self):
        # Blank reset
        self._set_initial_state()
        # Choose a random start point
        self.state_time = np.random.randint(0, self.timeframe - 1)
        self.done = bool(self.state_time == self.timeframe - 1)
        random_status = np.random.randint(0, self.action_features)
        self.obs_dummy_status[random_status][self.state_time] = 1
        action_description, reward = self._translate_action(random_status)
        self.total_reward = reward
        self.obs_last_legal_status = random_status

        return self._retrieve_obs()

        # TODO Do all states prior to the random start point need to be
        # set in the dummy vars?

    def _set_initial_state(self, initial_state_status: list = None):

        self.state_time = 0
        self.done = bool(self.state_time == self.timeframe - 1)
        self.obs_dummy_status = None
        self._generate_dummy_status()
        # self.obs_dummy_status = {}
        # for i in range(self.action_features):
        #     self.obs_dummy_status[i] = [0] * self.timeframe
        if initial_state_status is not None:
            # Dummy var to 1 for chosen initial status
            self.obs_dummy_status[initial_state_status][0] = 1
            action_description, reward = self._translate_action(
                initial_state_status)
            self.obs_last_legal_status = initial_state_status
        else:
            reward = 0

        self.total_reward = reward

        if self.fill_table:
            self.table.at[self.state_time,
                          'Current demand'] = self.demand[self.state_time]
            try:
                self.table.at[self.state_time,
                              'Next demand'] = self.demand[self.state_time + 1]
            except IndexError:
                pass
            self.table.at[self.state_time, 'Action'] = initial_state_status
            if initial_state_status is not None:
                self.table.at[self.state_time,
                              'Action description'] = action_description
            self.table.at[self.state_time, 'Reward'] = self.total_reward

        return self._retrieve_obs()
