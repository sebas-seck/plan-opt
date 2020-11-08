# -*- coding: utf-8 -*-

import random
import textwrap
import gym
from gym.core import ActionWrapper
import numpy as np
import pandas as pd
from gym import spaces
from torch import dtype
from plan_opt.demand import Demand

ACTION_LIST = [
    "OPERATE",  # 0
    "PREPARE",  # 1
    "PARK",  # 2
    "STORE",  # 3
]

LEGAL_CHANGES = {
    0: {0, 1, 2},
    1: {1, 0, 2},
    2: {2, 1, 3},
    3: {3, 2},
}

DEFAULT_CONFIG = {
    # ENVIRONMENT CONFIGURATION
    "ENV_ID": "rampup-v3",
    "REWARD_THRESHOLD": 80,
    "PUNISH_ILLEGAL": True,
    # WORKFLOW CONFIGURATION
    "TENSORBOARD_LOG": "logs/rampup_tensorboard/",
    "TIMESTEPS": 50000,
    "REPETITIONS": 5,
    "EVAL_EPISODES": 50,
    "SHOW_TABLE": False,
    "LEARNING_RATE": 0.0007,
}


class RampupEnv3(gym.Env):
    """Implements an environment to run ramp-up simulations.

    Environment follows the gym interface.

    Parameters:
        config (Dict, optional): Configuration used by the environment
            (in parts, other bits are relevant for processes around
            training and evaluation).
        demand (Demand, optional): Instance of demand data. Defaults to
            None.
        verbose (bool, optional): Defaults to False.

    Attributes:
        punish_illegal (bool): Feature flag to prevent and punish trials
            of illegal actions.
        horizon (int): Required future view depending on the current
            status.
        fleet_size (int, optional): Size of the fleet, one unit is
            equivalent to a carrier capable of holding multiple pieces of
            cargo.
        done (bool): True after the final step within the timeframe has
            been made.
        fill_table (bool): If true, :attr:`episode_table` is populated.
            taken, on demand detail for the current and next timesteps.
        episode_table (pandas.DataFrame): Table covering details per
            timestep. Information captured includes a record per timestep
            for current demand, next demand, action, action description and
            the reward. Useful for evaluation, populated only if
            `fill_table` is True.
        reward_threshold (int): Threshold, after which a profit can be made.
        total_reward (int): Reward accumulated up to the current timestep.
        reward_range (Tuple[int, int]): Range of possible rewards.
        action_features (int): Number of unique actions.
        features (int): Total number of features: Demand feature plus all actions.
        state_time (int): Time component of state. Logically, also
            referred to as (current) timestep.
        state_status (numpy.ndarray): Status component of state (current
            status).
        action_space (gym.spaces.Discrete): The Space object
            corresponding to valid actions.
        observation_space (gym.spaces.Box): The Space object
            corresponding to valid observations.
        obs_dummy_status (Dict[int, list[int]]): Dummy encoded status
            observation. Each possible action is a key in the dictionary,
            the value associated is a list of length len(timeframe).
        obs_last_legal_status (int): Action key of the status at
            state_time - 1.
        verbose (bool):

    Example:
        demand is [0,0,100] at timesteps [0,1,2]
    """

    @classmethod
    def create(cls, config=DEFAULT_CONFIG, demand=None, verbose=False):
        """Initializes object with specified parameters.

        Args:
            demand (Demand, optional): Instance of demand data.
                Defaults to None.
            verbose (bool, optional): Defaults to False.

        Returns:
            RampupEnv3: Instance of `RampupEnv3`.
        """
        return RampupEnv3(config, demand, verbose)

    def __init__(self, config=DEFAULT_CONFIG, demand=None, verbose=False):
        super(RampupEnv3, self).__init__()

        if demand is None:
            demand = Demand()
            demand.generate_demand()
            demand.apply_sudden(probability=0.3)
            demand.data = demand.data.astype("int32")
        self.demand = demand
        self.punish_illegal = config["PUNISH_ILLEGAL"]
        self.reward_threshold = config[
            "REWARD_THRESHOLD"
        ]  # TODO Change to dict for multiple resources

        self.timeframe = len(self.demand.data)
        self.horizon = 3
        self.past = 0
        self.visibility = self.horizon + self.past
        self.fleet_size = 1
        self._done = None
        self.fill_table = False
        self.episode_table = pd.DataFrame()
        self.total_reward = 0
        self.reward_range = (-28000, 15000)
        self.action_features = len(ACTION_LIST)
        self.features = self.action_features + 1
        self.state_time = 0
        self.state_status = np.array(np.zeros(self.timeframe), dtype=np.uint8, ndmin=2)
        self.action_space = spaces.Discrete(self.action_features)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=[self.features,],
            dtype=np.float32,
            # low=0, high=1, shape=[self.fleet_size, self.features, 1], dtype=np.float32,
        )
        self.obs_dummy_status = None
        self._generate_dummy_status()
        self.obs_last_legal_status = None
        self.total_reward = 0
        self.verbose = verbose

    def _generate_dummy_status(self):
        # create dict with timeframe plus padding per action feature
        self.obs_dummy_status = {}
        for i in range(self.action_features):
            self.obs_dummy_status[i] = [0] * (self.past + self.timeframe + self.horizon)

    def _retrieve_obs(self):
        # retrieve current observation and info
        obs = []
        start = self.state_time
        end = start + 1

        try:
            next_profitable_demand = (
                int(
                    np.argwhere(
                        self.demand.data[self.state_time + 1 :] > self.reward_threshold
                    )[0]
                )
                + 1
            )
        except IndexError:
            next_profitable_demand = 0
        try:
            obs_append = 1 / next_profitable_demand
        except ZeroDivisionError:
            obs_append = 0
        # obs_append = obs_append*255
        obs.append(obs_append)

        for j in range(self.action_features):
            obs.append(self.obs_dummy_status[j][self.state_time])

        obs = np.asarray(obs, dtype=np.float32)

        # Info
        demand_surrounding = ""
        demand_surrounding += f"{self.demand.data[self.state_time-1:self.state_time]}"
        demand_surrounding += f"-NOW({self.demand.data[self.state_time]})-"
        # add future state status
        demand_surrounding += (
            f"{self.demand.data[self.state_time+1:self.state_time+1+self.horizon]}"
        )

        info = {
            "timestep_change": f"{self.state_time-1} -> {self.state_time}",
            "action_change": f"{self.state_status[0][self.state_time-1]} -> {self.obs_last_legal_status}",
            "demand_surrounding": demand_surrounding,
            "next_profitable_demand": next_profitable_demand,
            "demand_observation": obs_append,
        }

        return obs, info

    def _translate_action(self, action):
        # maps actions with explanations and rewards
        if action == 0:
            action_description = "OPERATE"
            if self.demand.data[self.state_time] > self.reward_threshold:
                reward = 15000
            else:
                reward = -3000
        elif action == 1:
            action_description = "PREPARE"
            reward = -2000
        elif action == 2:
            action_description = "PARK"
            reward = -1000
        elif action == 3:
            action_description = "STORE"
            reward = -500

        if self.verbose:
            print(f"Action {action} leads to state {action_description}")

        return action_description, reward

    def step(self, action):
        """Transitions the environment from one state time to the next.

        Args:
            action (int): Action to be taken at the current status.

        Returns:
            Tuple(np.array, int, bool, Dict): Returns all changes made
                by the step.
        """
        if self.punish_illegal:
            action_illegal = action not in LEGAL_CHANGES[self.obs_last_legal_status]
            if action_illegal:
                # Repeat last action if illegal action is chosen
                action = self.obs_last_legal_status

        # Increment time component of state
        self.state_time += 1
        action_description, reward = self._translate_action(action)
        if self.punish_illegal:
            if action_illegal:
                reward += -25000
        self.obs_last_legal_status = action
        # Adjust the status component of state (dummy view)
        self.obs_dummy_status[action][self.state_time] = 1
        # Adjust the status component of state (plain view)
        self.state_status[0][self.state_time] = action

        obs, info = self._retrieve_obs()

        if self.verbose:
            print(f"Action taken: {action_description}")
        self.total_reward += reward

        if self.fill_table:
            self.episode_table.at[self.state_time, "Current demand"] = self.demand.data[
                self.state_time
            ]
            try:
                self.episode_table.at[
                    self.state_time, "Next demand"
                ] = self.demand.data[self.state_time + 1]
            except IndexError:
                pass
            self.episode_table.at[self.state_time, "Action"] = action
            self.episode_table.at[
                self.state_time, "Action description"
            ] = action_description
            self.episode_table.at[self.state_time, "Reward"] = reward

        return obs, reward, self.done, info

    def render(self):
        print(f"Reward so far: {self.total_reward}")

    def reset(self):
        # Blank reset
        self._set_initial_state()
        # Choose a random start point
        self.state_time = np.random.randint(1, self.timeframe - 2)
        random_status = np.random.randint(0, self.action_features)
        self.obs_dummy_status[random_status][self.state_time] = 1
        action_description, reward = self._translate_action(random_status)
        self.total_reward = reward
        self.obs_last_legal_status = random_status
        obs, _ = self._retrieve_obs()

        return obs

        # TODO Do all states prior to the random start point need to be
        # set in the dummy vars?

    @property
    def done(self):
        self._done = bool(self.state_time == (self.timeframe - self.past - 1))
        return self._done

    def _set_initial_state(self, initial_state_status=None):
        # set status and time on an initialized environment
        self.state_time = 0
        self.obs_dummy_status = None
        self._generate_dummy_status()

        if initial_state_status is not None:
            # set the status in the dummy view
            self.obs_dummy_status[initial_state_status][self.state_time] = 1
            # set the status in the plain view
            self.obs_last_legal_status = initial_state_status
            self.state_status[0][self.state_time] = initial_state_status
            action_description, reward = self._translate_action(initial_state_status)
        else:
            self.obs_last_legal_status = 3
            reward = 0

        self.total_reward = reward

        if self.fill_table:
            self.episode_table.at[self.state_time, "Current demand"] = self.demand.data[
                self.state_time
            ]
            try:
                self.episode_table.at[
                    self.state_time, "Next demand"
                ] = self.demand.data[self.state_time + 1]
            except IndexError:
                pass
            self.episode_table.at[self.state_time, "Action"] = initial_state_status
            if initial_state_status is not None:
                self.episode_table.at[
                    self.state_time, "Action description"
                ] = action_description
            self.episode_table.at[self.state_time, "Reward"] = self.total_reward
        obs, _ = self._retrieve_obs()

        return obs
