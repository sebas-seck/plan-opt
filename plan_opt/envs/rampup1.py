# -*- coding: utf-8 -*-

import random
import textwrap
import gym
from gym.core import ActionWrapper
import numpy as np
import pandas as pd
from gym import spaces
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


class RampupEnv1(gym.Env):
    """Implements an environment to run ramp-up simulations.

    Environment follows the gym interface.

    Parameters:
        demand (Demand, optional): Instance of demand data. Defaults to
            None.
        timeframe (int, optional): Total timeframe (in days) under
            observation. Defaults to 0.
        verbose (bool, optional): Defaults to False.

    Attributes:
        horizon (int): Length of future observation to be regarded.
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
        total_reward (int): Reward accumulated up to the current timestep.
        reward_range (Tuple[int, int]): Range of possible rewards.
        action_features (int): Number of unique actions.
        features (int): Total number of features: Actions and demand.
        state_time (int): Time component of state. Logically, also
            referred to as (current) timestep.
        state_status (numpy.ndarray): Status component of state (current
            status).
        action_space (gym.spaces.Discrete): The Space object
            corresponding to valid actions.
        observation_space (gym.spaces.Box): The Space object
            corresponding to valid observations.
        obs_demand (numpy.ndarray): Observation over the entire
            timeframe. The observation space needs to extend beyond the
            end as the horizon extends the consideration beyond the
            available data. Per default, demand of 0 is added for the
            length of the horizon.
        obs_dummy_status (Dict[int, list[int]]): Dummy encoded status
            observation. Each possible action is a key in the dictionary,
            the value associated is a list of length len(timeframe).
        obs_last_legal_status (int): Action key of the status at
            state_time - 1.
        verbose (bool):
    """

    @classmethod
    def create(cls, demand=None, timeframe=0, verbose=False):
        """Initializes object with specified parameters.

        Args:
            demand (Demand, optional): Instance of demand data.
                Defaults to None.
        timeframe (int, optional): Total timeframe (in days) under
            observation. Defaults to 0.
        verbose (bool, optional): Defaults to False.

        Returns:
            RampupEnv1: Instance of `RampupEnv1`.
        """
        return RampupEnv1(demand, timeframe, verbose)

    def __init__(self, demand=None, timeframe=0, verbose=False):
        super(RampupEnv1, self).__init__()

        if demand is None:
            demand = Demand()
            demand.generate_demand()
            demand.apply_sudden(probability=0.3)
        self.demand = demand
        if timeframe == 0:
            timeframe = len(self.demand.data)
        self.timeframe = timeframe

        self.horizon = 3
        self.fleet_size = 1
        self.done = None
        self.fill_table = False
        self.episode_table = pd.DataFrame()
        self.total_reward = 0
        self.reward_range = (-28000, 15000)
        self.action_features = len(ACTION_LIST)
        self.features = self.action_features + 1
        self.state_time = 0
        self.state_status = np.array(np.zeros(timeframe), dtype=np.uint8, ndmin=2)
        self.action_space = spaces.Discrete(self.action_features)
        self.observation_space = spaces.Box(
            low=self._generate_obs_space_HL("low"),
            high=self._generate_obs_space_HL("high"),
            shape=[self.fleet_size, self.features, self.horizon],
            dtype=np.uint8,
        )
        self.obs_demand = (
            self.demand.data / np.linalg.norm(self.demand.data) * 255
        ).astype("uint8")
        self.obs_demand = np.append(
            self.obs_demand, np.array([0] * self.horizon, dtype=np.uint8)
        )
        self.obs_dummy_status = None
        self._generate_dummy_status()
        self.obs_last_legal_status = None
        self.total_reward = 0
        self.verbose = verbose

    def _generate_obs_space_HL(self, bound_hl):
        if bound_hl == "low":
            bound = 0
        else:
            bound = 1
        obs_hl = [[]] * self.fleet_size
        l = [[]] * self.features
        for i in range(self.fleet_size):

            for j in range(self.features):
                # only the first feature goes up to 255
                if j == 0 and bound_hl == "high":
                    bound = 255
                elif bound_hl == "high":
                    bound = 1
                l[j] = [bound] * self.horizon
            obs_hl[i] = l
        return np.array(obs_hl, dtype=np.uint8)

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
            action_description = "OPERATE"
            if self.demand.data[self.state_time] > 80:
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

        # OLD THOUGHTS
        # if action not in LEGAL_CHANGES[self.obs_last_legal_status]:
        #     # Illegal action, pick another random legal action!
        #     action = random.sample(LEGAL_CHANGES[self.obs_last_legal_status],
        #                            1)[0]

        # action_illegal = action not in LEGAL_CHANGES[self.obs_last_legal_status]
        # if action_illegal:
        #     # Repeat last action if illegal action is chosen :)
        #     action = self.obs_last_legal_status

        # Increment time component of state
        self.state_time += 1
        action_description, reward = self._translate_action(action)
        # if action_illegal:
        #     reward += -25000
        self.obs_last_legal_status = action
        # Adjust the status component of state
        self.obs_dummy_status[action][self.state_time] = 1

        obs = self._retrieve_obs()

        self.done = bool(self.state_time == self.timeframe - 1)
        info = {}
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
        # if self.verbose:
        economic_potential = self.demand.economic_potential_no_illegal()
        lost_potential = economic_potential - max(self.total_reward, 0)
        lost_potential_perc = round(lost_potential / economic_potential * 100, 4)
        print(
            textwrap.dedent(
                f"""            Reward so far: {self.total_reward}
            Economic potential: {economic_potential}
            Lost potential: {lost_potential} ({lost_potential_perc}%)
            """
            )
        )

    def reset(self):
        # Blank reset
        self._set_initial_state()
        # Choose a random start point
        self.state_time = np.random.randint(0, self.timeframe - 2)
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
            action_description, reward = self._translate_action(initial_state_status)
            self.obs_last_legal_status = initial_state_status
        else:
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

        return self._retrieve_obs()
