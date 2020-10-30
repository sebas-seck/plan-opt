# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


class Demand:
    """Provides demand generation and interaction.

    Stacks multiple trigonometric functions on top of each other to
    imitate demand cycles starting from a slump. Functions represent
    weeks, summer time, season peaks, general growth and the short and
    long term recovery from the slump.

    If :attr:`data` is provided during instantiation, functionality is
    limited as no demand curve is available.

    Arguments:
        period (int, optional): Period (given in days) under
            considertation for demand. Defaults to 365*3.
        seed (int, optional): Seed for reproducibility. Set to
            np.random.seed(np.random.randint(0, 10000)) if no value
            provided. Defaults to None.
        data (np.ndarray, optional): Deterministic demand data can be
            supplied. Generated of no data is supplied. Defaults to
            None.

    Attributes:
        has_sudden_change (bool): Existance of a sudden change in
            demand.
        sc_magnitude (int): Magnitude of sudden change if it exists.
        sc_direction (int): Direction of sudden change if it exists.
        sc_steepness (int): Steepness of sudden change if it exists.
        sc_start (int): Start of sudden change if it exists.
        y_all (np.ndarray): y values of the demand curve
        data (np.ndarray): y values of generated data scattered around
            demand curve
    """

    def __init__(self, period=365 * 3, seed=None, data=None, verbose=0):
        self.period = period
        self.x = np.arange(1, period + 1)
        if seed is None:
            self.seed = np.random.randint(0, 10000)
        else:
            self.seed = seed
        np.random.seed(self.seed)

        self.has_sudden_change = None
        self.sc_magnitude = None
        self.sc_direction = None
        self.sc_steepness = None
        self.sc_start = None

        self.y_all = None
        self.data = data
        self.verbose = verbose

    def generate_demand(self):
        # create multiple curves for different seasonalities & patterns
        y_weekly = 15 * -np.cos(2 * self.x) + 15
        y_summer = 15 * -np.cos(1 / (360 / 2) * np.pi * self.x)
        y_peaks = 8 * -np.cos((1 / (360 / 4)) * np.pi * self.x)
        y_recover_short = self.x ** (1 / 3) * 2
        y_recover_long = 1 / (1 + np.exp(-(self.x - 365) / 80)) * 20
        y_growth = self.x * 0.025
        # stack the curves
        self.y_all = (
            y_weekly
            + y_summer
            + y_peaks
            + y_recover_short
            + y_recover_long
            + y_growth
            + 20
        )
        # create multiple distributions around the stacked curve
        y_poisson = np.random.poisson(40, self.period) * 4 - 100
        y_gamma = np.random.gamma(2, 20, self.period)  # shape, scale, size
        y_cauchy = np.random.standard_cauchy(self.period)
        # create the synthetic demand by adding all distributions up
        self.data = self.y_all + y_poisson + y_gamma + y_cauchy - np.mean(y_cauchy)
        self.data = self.data.clip(min=0)

    def add_sudden_change(
        self, start=None, magnitude=None, steepness=None, direction=None,
    ):
        """Adds a sudden change of demand to an existing demand array.

        Args:
            start (int, optional): Point in time during period when the
                sudden change begins. If it is None, a random point in
                time from :attr:`period` is picked. Defaults to None.
            magnitude (int, optional): Magnitude of sudden change. Set to np.random.normal(1000, 50) if no value is provided. Defaults to None.
            steepness (int, optional): [description]. Defaults to np.random.normal(50, 25). Defaults to None.
            direction (int, optional): Growth is indicated by 1, decline by 0. Defaults to np.random.randint(2). Defaults to None.
        """
        if start is None:
            self.sc_start = np.random.choice(self.period)
        else:
            self.sc_start = start
        self.sc_magnitude = magnitude
        self.sc_steepness = steepness
        self.sc_direction = direction

        if self.sc_start is None:
            self.sc_start = np.random.choice(self.period)
        if self.sc_magnitude is None:
            self.sc_magnitude = np.random.normal(1000, 50)
        if self.sc_steepness is None:
            self.sc_steepness = np.random.normal(50, 25)
        if self.sc_direction is None:
            self.sc_direction = np.random.randint(2)
        if self.sc_direction == 0:  # sudden change up or down
            self.sc_magnitude = self.sc_magnitude * -1
            self.sc_direction = "DOWN"
        else:
            self.sc_direction = "UP"

        x_sudden = np.arange(1, self.period + 1 - self.sc_start)
        y_sudden = (
            1 / (1 + np.exp(-(x_sudden - 365) / self.sc_steepness)) * self.sc_magnitude
        )

        self.data[self.sc_start :] += y_sudden
        self.data = self.data.clip(min=0)

        self.y_all[self.sc_start :] += y_sudden
        self.y_all = self.y_all.clip(min=0)

    def apply_sudden(self, probability=0.5):
        """Applies sudden changes in some cases"""
        if np.random.rand() <= probability:
            self.has_sudden_change = True
            self.add_sudden_change()
        else:
            self.has_sudden_change = False

    def show(self, only_data=False):
        """Plots demand curve and data scattered around demand curve.

        The demand curve is shown as line diagramm, the data (which is
        scattered around the demand curve), is shown as a scatter plot.

        Args:
            only_data (bool, optional): If `True`, only the data is
                plotted. Set to `True` if data was not generated
                within instance but has been provided during
                instantiation. Defaults to False.
        """
        try:
            if only_data:
                plt.scatter(self.x, self.data, alpha=0.5)
            else:
                fig, axs = plt.subplots(1, 2)
                axs[0].plot(self.x, self.y_all)
                axs[0].set_title("Demand Curve")
                axs[1].scatter(self.x, self.data, alpha=0.5)
                axs[1].set_title("Scattered Data around Demand Curve")
            plt.show()
        except ValueError:
            print("Demand has not been created yet!")

    def info(self):
        print(f"Seed is {self.seed}")
        print(f"Period is {self.period}")
        if self.sc_magnitude is not None:
            print(
                f"Sudden change:\n    Start: {self.sc_start}\n    Magnitude: {self.sc_magnitude}\n    Steepness: {self.sc_steepness}\n    Direction: {self.sc_direction}"
            )

    def economic_potential(
        self, profit_threshold=80, static_profit=15000, static_loss=500
    ):
        """Derives economic potential from demand data.

        This simple calculation merely takes data points above the
        profit_threshold and multiplies those with static_profit. Data
        points below the threshold are assigned the static_loss.
        Combined, this make the economic potential.

        Args:
            profit_threshold (int, optional): Profit is assumed to be
                made at demand greater or equal to the
                profit_threshold. Defaults to 80.
            static_profit (int, optional): Profit assumed if threshold
                is reached. Defaults to 15000.
            static_loss (int, optional): Loss assumed if threshold is
                not reached. Defaults to 500.

        Returns:
            economic_potential (int): Sales potential offset by the loss
            potential.
        """
        event_over_threshold = sum([i >= profit_threshold for i in self.data])
        event_below_threshold = len(self.data) - event_over_threshold
        sales_potential = event_over_threshold * static_profit
        loss_potential = event_below_threshold * static_loss
        economic_potential = sales_potential - loss_potential

        return economic_potential

    def economic_potential_no_illegal(self):

        reward_per_action = {0: 15000, 1: -2000, 2: -1000, 3: -500}

        reward = -500
        action = 3
        data = np.append(self.data, [0, 0, 0])
        for i in range(len(data) - 3):
            #     prepare_in_3 = a[i+3]>80
            prepare_in_2 = data[i + 2] > 80
            prepare_in_1 = data[i + 1] > 80
            prepare_now = data[i] > 80

            # In anticipation of demand
            if prepare_in_2 and action == 3:
                action = 2

            elif prepare_in_1 and action == 2:
                action = 1

            elif prepare_now and action == 1:
                action = 0

            # After demand
            elif not prepare_now and action == 0:
                if prepare_in_1:
                    action = 1
                else:
                    action = 2

            reward += reward_per_action[action]

            if self.verbose == 2:
                print(data[i], prepare_in_2, prepare_in_1, prepare_now, action)

        return reward
