# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 1 Initial Approach with Dynamic Programming
# This notebook contains some intial trials of hands-on, basic reinforcement learning algorithms to familiarize with the topic and eventually arrive at the bare frame of the environment for ramp-up scenarios. The set-up follows the discourse in [Artificial Intelligence: Reinforcement Learning in Python](https://www.udemy.com/share/1013kmBEQbdF1XTXQ=/) by Lazy Programmer Inc roughly, altough the environment conforms with OpenAI's toolkit for RL, [Gym](https://gym.openai.com/).

# %%
import gym
import numpy as np
import pandas as pd
from gym import spaces
from IPython.core.display import HTML, display


# %% [markdown]
# ## Review helper functions

# %% [markdown]
# - With `display_side_by_side()`, tables are displayed side by side to ease interpretations and save space as the table index is similar.
# - Future value at each possible combination of state and episode is generated and displayed with `pivot_v()`
# - The policy at each possible combination of state and episode is generated and displayed with `pivot_p()`

# %%
def display_side_by_side(dfs: list, captions: list):
    output = ""
    combined = dict(zip(captions, dfs))
    for caption, df in combined.items():
        output += (
            df.style.set_table_attributes("style='display:inline'")
            .set_caption(caption)
            ._repr_html_()
        )
        output += "\xa0\xa0\xa0"
    display(HTML(output))


def pivot_v(V):
    v_table = pd.Series(V).reset_index()
    v_table.columns = ["Episode", "Action", "Value"]
    v_table.sort_values(by=["Episode", "Action"], inplace=True)
    v_table_p = v_table.pivot(index="Episode", columns="Action")
    return v_table, v_table_p


def pivot_p(policy):
    p_table = pd.Series(policy).reset_index()
    p_table.columns = ["Episode", "State", "Policy_S"]
    p_table.sort_values(by=["Episode"], inplace=True)
    p_table = p_table.pivot(index="Episode", columns="State")
    p_table = p_table.assign(demand=demand)
    return p_table


# %% [markdown]
# ## Definitions of constants
#
# All constants needed for the environment include
# - the **action list**, a discrete set of actions to move between states
# - the **legal changes**, a constraining factor which actions may follow upon which state
# - **demand**, for now just a list which is the trajectory that needs to be followed during an episode, only the choice of actions is open to the agent

# %%
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


# %% [markdown]
# ## Environment Definition
# The environment is in accordance with Gym. It's a sequence of X days, on each day a single action can be performed. That's the entire logic encoded in the environment!

# %%
class RampupEnv(gym.Env):
    def __init__(self, horizon=10, verbose=False):
        super(RampupEnv, self).__init__()

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
            action_description = "OPERATE"
            if demand[self.state_time] > 80:
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


# %% [markdown]
# ## Random Walk Episode
# The length of the episode is determined by `horizon`, actions are taken at random. We observe, that games mostly lead to a negative total reward, thus it takes a policy better than a random policy to perform well.

# %%
horizon = 7 * 4
env = RampupEnv(horizon=horizon)

for step in range(horizon):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        print(
            f"Random walk episode is complete.\nTrajectory:\n{obs}\nTotal reward: {env.total_reward}"
        )
        break

# %% [markdown]
# ## Value Iteration
# All possible state transitions (transitions between possible combinations of state time and state actions) are iteratively evaluated for their value, until the improvement in value converges to a defined threshold. It takes 23 iterations for the sample environment with the provided data. The resulting table shows the value for each state-action.

# %%
h = 7 * 4  # time horizon
env = RampupEnv(horizon=h, verbose=False)
all_states = set(
    [(s, a) for s in range(len(env.state_status)) for a in range(env.action_space.n)]
)

SMALL_ENOUGH = 1e-3  # threshold for convergence
GAMMA = 1.0  # discount factor

V = {}
for state in all_states:
    V[state] = 0

biggest_change = 0
iters = 0

while True:
    iters += 1
    biggest_change = 0
    for state in all_states:
        old_v = V[state]
        env.set_state(state)
        state_time, state_status = state
        if env.state_time < env.horizon - 1:
            new_v = 0  # answer is accumulated
            p_a = 1.0 / len(legal_changes[state_status])  # equal probability
            for a in legal_changes[state_status]:
                env.set_state(state)
                o, r, d, i = env.step(a)
                new_v += p_a * (r + GAMMA * V[(env.state_time, a)])
                o, r, d, i = env.undo_step(o, r, d, i)
            V[state] = new_v
            biggest_change = max(biggest_change, np.abs(old_v - V[state]))

        if env.verbose:
            print(f"State: {state}, Old V: {old_v}")

    if biggest_change < SMALL_ENOUGH:
        print(f"Done in {iters} iterations")
        v_table, v_table_p = pivot_v(V)
        display(v_table_p)
        break

# %% [markdown]
# ## Policy Iteration
# Value iteration stops as the improvement of value converges, policy iteration then adds a policy improvement step. In the case of the example, there is no improvement visible, but we do get the policy for the values from value iterations inside of the attached policy improvement step.

# %%
h = 7 * 4  # time horizon
env = RampupEnv(horizon=h, verbose=False)
all_states = set(
    [(s, a) for s in range(len(env.state_status)) for a in range(env.action_space.n)]
)

SMALL_ENOUGH = 1e-3
GAMMA = 1

policy = {}
for state in all_states:
    policy[state] = env.action_space.sample()

V = {}
for state in all_states:
    V[state] = 0

iters1, iters2 = 0, 0
rep1, rep2 = "", ""

while True:
    iters1 = 0
    # Policy Evaluation Step
    while True:
        iters1 += 1
        biggest_change = 0
        for state in all_states:
            old_v = V[state]
            env.set_state(state)
            state_time, state_status = state
            if env.state_time < env.horizon - 1:
                new_v = 0  # answer is accumulated
                p_a = 1.0 / len(legal_changes[state_status])  # equal probability
                for a in legal_changes[state_status]:
                    env.set_state(state)
                    o, r, d, i = env.step(a)
                    new_v += p_a * (r + GAMMA * V[(env.state_time, a)])
                    o, r, d, i = env.undo_step(o, r, d, i)
                V[state] = new_v
                biggest_change = max(biggest_change, np.abs(old_v - V[state]))

            if env.verbose:
                print(f"State: {state}, Old V: {old_v}")

        if biggest_change < SMALL_ENOUGH:
            rep1 += f"{str(iters1)}, "
            iters2 += 1
            break

    # Policy Improvement Step
    is_policy_converged = True
    for state in all_states:
        for state in policy:
            old_a = policy[state]
            new_a = None
            best_value = float("-inf")
            env.set_state(state)
            state_time, state_status = state
            if env.state_time < env.horizon - 1:
                # loop through all possible actions to find the best current action
                for a in legal_changes[state_status]:
                    env.set_state(state)
                    o, r, d, i = env.step(a)
                    v = r + GAMMA * V[(env.state_time, a)]
                    # print(v)
                    if v > best_value:
                        best_value = v
                        new_a = a
                policy[state] = new_a
                if new_a != old_a:
                    is_policy_converged = False
    if is_policy_converged:
        print(
            f"Policy evaluation and improvement steps were entered {iters2} times. \
Policy evaluation iterated {rep1[:-2]} times."
        )
        v_table, v_table_p = pivot_v(V)
        p_table = pivot_p(policy)
        display_side_by_side([v_table_p, p_table], ["Value Table", "Policy Table"])
        break

# %%
