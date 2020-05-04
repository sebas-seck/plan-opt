# ---
# jupyter:
#   jupytext:
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
# # Dynamic Programming approaches
# Follows dynamic programming from scratch to develop a suitable environment along the way

# %%
import gym
import numpy as np
import pandas as pd
from gym import spaces
from IPython.core.display import HTML, display


# %%
def display_side_by_side(dfs:list, captions:list):
    """Display tables side by side to save vertical space
    Input:
        dfs: list of pandas.DataFrame
        captions: list of table captions
    """
    output = ""
    combined = dict(zip(captions, dfs))
    for caption, df in combined.items():
        output += df.style.set_table_attributes("style='display:inline'").set_caption(caption)._repr_html_()
        output += "\xa0\xa0\xa0"
    display(HTML(output))


# %%
def pivot_v(V):
    v_table = pd.Series(V).reset_index()
    v_table.columns = ['Episode', 'Action', 'Value']
    v_table.sort_values(by=['Episode', 'Action'], inplace=True)
    v_table_p = v_table.pivot(index='Episode', columns='Action')
    return v_table, v_table_p

def pivot_p(policy):
    p_table = pd.Series(policy).reset_index()
    p_table.columns = ['Episode', 'State', 'Policy_S']
    p_table.sort_values(by=['Episode'], inplace=True)
    p_table = p_table.pivot(index='Episode', columns='State')
    p_table = p_table.assign(demand = demand)
    return p_table


# %% [markdown]
# ### Constant definitions for the environment

# %%
action_list = [
    'OPERATE',  # 0
    'PREPARE',  # 1
    'PARK',  # 2
    'STORE',  # 3
]
# DEEP_STORAGE_TERUEL

legal_changes = {
    0: {0, 1, 2},  # from OPERATE_IN_REV_SERVICE to SELF, PREPARE_FOR_FLIGHT, PARK_AT_BASE
    1: {1, 0, 2},  # from PREPARE_FOR_FLIGHT to SELF, OPERATE_IN_REV_SERVICE, PARK_AT_BASE
    2: {2, 1, 3},  # from PARK_AT_BASE to SELF, PREPARE_FOR_FLIGHT, STORE_AT_BASE_ST
    3: {3, 2}      # from STORE_AT_BASE_ST to SELF, PARK_AT_BASE
}

demand = [
    0,   0,   0,   0,   0,   0,   0,  # week 1
    0,   0,   0,   2,   0,   3,   1,  # week 2
    0,   0,  60,  25,  32,  87,  56,  # week 3
   92,  83,  29,  40,  86,  70,  45,  # week 4
]


# %% [markdown]
# def get_action_delay(action, states_status):
#     prop = np.random.randint(1, 100)
#     some_dict = {
#         # 2% of the time the plane requires unplanned service
#         0: (0, 0.98) if prop <=98 else (1, 0.02),
#         # 5% of transitions from 2 to 1 fail and remain at 2
#         1: (2, 0.05) if prop <=5 and state_status == 2 else (1, 0.95),
#         # 20% of transition from 3 to 2 fail and remain at 3
#         2: (3, 0.2) if prop <= 20 and state_status == 3 else (2, 0.8),
#         3: (3, 1)
#     }
#     
#     return some_dict[action]

# %%
def get_action_delay(action, states_status, random=True):
    prop = np.random.randint(1, 100) if random else 1  # 1 always returns tuple of intended action
    some_dict = {
        # 2% of the time the plane requires unplanned service
        0: (0, 0.98) if prop <=98 else (1, 0.02),
        # 5% of transitions from 2 to 1 fail and remain at 2
        1: (2, 0.05) if prop >=95 and state_status == 2 else (1, 0.95),
        # 20% of transition from 3 to 2 fail and remain at 3
        2: (3, 0.2) if prop >= 80 and state_status == 3 else (2, 0.8),
        3: (3, 1)
    }
    
    return some_dict[action]


# %%
if False:
    for i in range(100):
        print(get_action_delay(2, 3))


# %% [markdown]
# ## Definition of the Ramp Up Environment

# %%
class RampupEnv(gym.Env):
    """Creates an environment to run ramp-up simulations.
    
    Environment follows gym interface53
    """
    def __init__(self, verbose=False, horizon=10):
        
        self.fleet_size = 1
        self.horizon = horizon
        
        self.state_time = 0  # time component of state
        self.state_status = np.zeros(self.horizon)  # status component of state

        # action_space: The Space object corresponding to valid actions
        # observation_space: The Space object corresponding to valid observations
        self.action_space = spaces.Discrete(len(action_list))
        self.observation_space = spaces.Tuple((
                spaces.Discrete(self.fleet_size),
                spaces.Discrete(self.horizon)
                ))
        
        self.total_reward = 0
        self.verbose = verbose
        
        
    def translate_action(self, action):
        if action == 0:
            action_description = 'OPERATE_IN_REV_SERVICE'
            if demand[self.state_time] > 30:
                reward = 15000
            else:
                reward = -3000
        elif action == 1:
            action_description = 'PREPARE_FOR_FLIGHT'
            reward = -2000
        elif action == 2:
            action_description = 'PARK_AT_BASE'
            reward = -1000
        elif action == 3:
            action_description = 'STORE_AT_BASE_ST'
            reward = -500
            
        if self.verbose:
            print(f'Action {action} leads to state {action_description}')
            
        return action_description, reward
        
            

    def step(self, action):
        # Increment time component of state
        self.state_time += 1

        action_description, reward = self.translate_action(action)
        
        # Adjust the status component of state
        self.state_status[self.state_time] = action
        
        obs = self.state_status[0:self.state_time]
        done = bool(self.state_time == self.horizon - 1)
        info = {}
        
        if self.verbose:
            print(f'Action taken: {action_description}')
        
        self.total_reward += reward
        
        return obs, reward, done, info
    
    
    def undo_step(self, o, r, d, i):
        self.state_time -= 1
        self.total_reward -= r
        
        return o, r, d, i
            
    def render(self):
        if self.verbose:
            print(f'Reward so far: {self.total_reward}')
    
    def reset(self):
        # Initialize the agent in the first field
        self.state_time = 0
        self.total_reward = 0
        
    
    def set_state(self, state):
        state_time, state_status = state
        self.state_time = state_time
        self.state_status[self.state_time] = state_status


# %%
h = 7*4
env = RampupEnv(horizon=h, verbose=False)

for step in range(h):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        print("We're done")
        print(obs)
        print(f"Total reward is {env.total_reward}")
        break

# %% [markdown]
# ## Iterative Policy Evaluation (Value Iteration?)

# %% [markdown]
# #### Going over all possible states (combination of state time, i.e. episodes, and all actions)

# %%
h = 7*4  # time horizon
env = RampupEnv(horizon=h, verbose=False)
all_states = set([(s, a) for s in range(len(env.state_status)) for a in range(env.action_space.n)])

SMALL_ENOUGH = 1e-3 # threshold for convergence
GAMMA = 1.  # discount factor

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
        if env.state_time < env.horizon -1:
            new_v = 0  # answer is accumulated
            p_a = 1. / len(legal_changes[state_status])  # equal probability
            for a in legal_changes[state_status]:
                env.set_state(state)
                o, r, d, i = env.step(a)
                new_v += p_a * (r + GAMMA * V[(env.state_time, a)])
                o, r, d, i = env.undo_step(o, r, d, i)
            V[state] = new_v
            biggest_change = max(biggest_change, np.abs(old_v - V[state]))
            
        if env.verbose:
            print(f'State: {state}, Old V: {old_v}')
            
    if biggest_change < SMALL_ENOUGH:
        print(f'Done in {iters} iterations')
        break

# %%
v_table, v_table_p = pivot_v(V)
v_table_p

# %% [markdown]
# #### Evaluate fixed policy
# This down here may just be bullshit as it evaluates just the random  policy that is being assigned.. but if a fixed policy is plugged in it starts to make sense..!

# %%
policy = {}
for state in all_states:
    policy[state] = env.action_space.sample()
    
V = {}
for state in all_states:
    V[state] = 0
    
gamma = .9

iters = 0

while True:
    iters += 1
    if iters%100000 == 0:
        print(f'Working on iteration {iters}')
    biggest_change = 0
    for state in all_states:
        old_v = V[state]
        env.set_state(state)
        if env.state_time < env.horizon -1:
            a = policy[state]
            env.set_state(state)
            o, r, d, i = env.step(a)
            V[state] = r + gamma * V[(env.state_time, a)]
            biggest_change = max(biggest_change, np.abs(old_v - V[state]))
            o, r, d, i = env.undo_step(o, r, d, i)
    if biggest_change < SMALL_ENOUGH:
        print(f'Done in {iters} iterations')
        break

# %%
p_table = pivot_p(policy)
# p_table

# %% [markdown]
# ## Policy Iteration

# %%
# %%time
h = 7*4  # time horizon
env = RampupEnv(horizon=h, verbose=False)
all_states = set([(s, a) for s in range(len(env.state_status)) for a in range(env.action_space.n)])

SMALL_ENOUGH = 1e-3
GAMMA = 1

policy = {}
for state in all_states:
    policy[state] = env.action_space.sample()
    
V = {}
for state in all_states:
    V[state] = 0
    
iters = 0

while True:
    # Policy Evaluation Step
    while True:
        iters += 1
        biggest_change = 0
        for state in all_states:
            old_v = V[state]
            env.set_state(state)
            state_time, state_status = state
            if env.state_time < env.horizon -1:
                new_v = 0  # answer is accumulated
                p_a = 1. / len(legal_changes[state_status])  # equal probability
                for a in legal_changes[state_status]:
                    env.set_state(state)
                    o, r, d, i = env.step(a)
                    new_v += p_a * (r + GAMMA * V[(env.state_time, a)])
                    o, r, d, i = env.undo_step(o, r, d, i)
                V[state] = new_v
                biggest_change = max(biggest_change, np.abs(old_v - V[state]))

            if env.verbose:
                print(f'State: {state}, Old V: {old_v}')

        if biggest_change < SMALL_ENOUGH:
            print(f'Done in {iters} iterations')
            break

    # Policy Improvement Step
    is_policy_converged = True
    for state in all_states:
        for state in policy:
            old_a = policy[state]
            new_a = None
            best_value = float('-inf')
            env.set_state(state)
            state_time, state_status = state
            if env.state_time < env.horizon -1:
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
        break

# %%
v_table, v_table_p = pivot_v(V)
p_table = pivot_p(policy)
display_side_by_side([v_table_p, p_table], ['Value Table', 'Policy Table'])

# %% [markdown]
# ## Policy Iteration with Possible Delays
# Delay changes need to be defined for all state transitions

# %%
delay_changes = {
    (0, 0): {0: 0.98, 1: 0.02, 2: 0},  # Staying in 0 works in 98% of cases, 2% of the time we are pushed to 1
    (0, 1): {0: 0,    1: 1,    2: 0},
    (0, 2): {0: 0,    1: 0,    2: 1},

    (1, 0): {0: 1,    1: 0,    2: 0},
    (1, 1): {0: 0,    1: 1,    2: 0},
    (1, 2): {0: 0,    1: 0,    2: 1},
    
    (2, 1): {1: 1,    2: 0,    3: 0},
    (2, 2): {1: 0,    2: 1,    3: 0},
    (2, 3): {1: 0,    2: 0,    3: 1},
    
    (3, 2): {2: 1,    3: 0},
    (3, 3): {2: 0,    3: 1},
    
}

# %% [markdown]
# Activate the next cell to check if the code leads to exactly the same results as the previous section (the cell simply imitates no existing action delays).

# %%
# Activate cell to pretend action delays do not exist
if True:
    delay_changes = {
    (0, 0): {0: 1,    1: 0,    2: 0},
    (0, 1): {0: 0,    1: 1,    2: 0},
    (0, 2): {0: 0,    1: 0,    2: 1},

    (1, 0): {0: 1,    1: 0,    2: 0},
    (1, 1): {0: 0,    1: 1,    2: 0},
    (1, 2): {0: 0,    1: 0,    2: 1},
    
    (2, 1): {1: 1,    2: 0,    3: 0},
    (2, 2): {1: 0,    2: 1,    3: 0},
    (2, 3): {1: 0,    2: 0,    3: 1},
    
    (3, 2): {2: 1,    3: 0},
    (3, 3): {2: 0,    3: 1},
    }

# %%
# %%time
h = 7*4  # time horizon
env = RampupEnv(horizon=h, verbose=False)
all_states = set([(s, a) for s in range(len(env.state_status)) for a in range(env.action_space.n)])

SMALL_ENOUGH = 1e-3
GAMMA = 1

policy = {}
for state in all_states:
    policy[state] = env.action_space.sample()
    policy[state] = 1
    
V = {}
for state in all_states:
    V[state] = 0
    
iters1, iters2 = 0, 0

while True:
    # Policy Evaluation Step
    while True:
        iters1 += 1
        biggest_change = 0
        for state in all_states:
            old_v = V[state]
            env.set_state(state)
            state_time, state_status = state
            if env.state_time < env.horizon -1:
                new_v = 0  # answer is accumulated
                pa = len(legal_changes[state_status])  # equal probability
                for a in legal_changes[state_status]:
                    # Account for possible delays
                    p = delay_changes[(state_status, a)][a]
                    p = p / pa
                    env.set_state(state)
                    o, r, d, i = env.step(a)
                    new_v += p * (r + GAMMA * V[(env.state_time, a)])
                    o, r, d, i = env.undo_step(o, r, d, i)
                V[state] = new_v
                biggest_change = max(biggest_change, np.abs(old_v - V[state]))

            if env.verbose:
                print(f'State: {state}, Old V: {old_v}')

        if biggest_change < SMALL_ENOUGH:
            break

    iters2 += 1
    # Policy Improvement Step
    is_policy_converged = True
    for state in all_states:
        for state in policy:
            old_a = policy[state]
            new_a = None
            best_value = float('-inf')
            env.set_state(state)
            state_time, state_status = state
            if env.state_time < env.horizon -1:
                pa = len(legal_changes[state_status])  # equal probability
                # loop through all possible actions to find the best current action
                for a in legal_changes[state_status]:
                    v = 0
                    for a2 in legal_changes[state_status]:
                        p = delay_changes[(state_status, a)][a2]
                        env.set_state(state)
                        o, r, d, i = env.step(a2)
                        v += p * (r + GAMMA * V[(env.state_time, a2)])
                    if v > best_value:
                        best_value = v
                        new_a = a
                policy[state] = new_a
                if new_a != old_a:
                    is_policy_converged = False

    if is_policy_converged:
        print(f'Policy Evaluation done in {iters1} iterations')
        print(f'Policy Improvement done in {iters2} iterations')
        break

# %% [markdown]
# Disadvantage of policy iteration is the fact that there is an iterative algorithm inside another iterative algorithm

# %%
v_table, v_table_p = pivot_v(V)
p_table = pivot_p(policy)
display_side_by_side([v_table_p, p_table], ['Value Table', 'Policy Table'])

# %% [markdown]
# ### Value Iteration
# We do not need to wait for policy evaluation to finish (V to converage), policy improvement will find the policy after a few steps of policy evaluation

# %%
# %%time
h = 7*4  # time horizon
env = RampupEnv(horizon=h, verbose=False)
all_states = set([(s, a) for s in range(len(env.state_status)) for a in range(env.action_space.n)])

SMALL_ENOUGH = 1e-3
GAMMA = .99

policy = {}
for state in all_states:
    # policy[state] = env.action_space.sample()
    policy[state] = 1
    
V = {}
for state in all_states:
    V[state] = 0
    
iters = 0

# repeat until convergence
while True:
    iters += 1
    biggest_change = 0
    for state in all_states:
        old_v = V[state]
        env.set_state(state)
        state_time, state_status = state
        if env.state_time < env.horizon -1:  # excludes terminal states
            new_v = float('-inf')
            for a in legal_changes[state_status]:
                env.set_state(state)
                o, r, d, i = env.step(a)
                v = r + GAMMA * V[(env.state_time, a)]
                if v > new_v:
                    new_v = v
            V[state] = new_v
            biggest_change = max(biggest_change, np.abs(old_v - V[state]))
    if biggest_change < SMALL_ENOUGH:
        print(f'Converged in {iters} iterations')
        break
        
for state in policy.keys():
    best_a = None
    best_value = float('-inf')
    env.set_state(state)
    state_time, state_status = state
    if env.state_time < env.horizon -1:  # excludes terminal states
        for a in legal_changes[state_status]:
            env.set_state(state)
            o, r, d, i = env.step(a)
            v = r + GAMMA * V[(env.state_time, a)]
            if v > best_value:
                best_value = v
                best_a = a
            policy[state] = best_a

# %%
v_table, v_table_p = pivot_v(V)
p_table = pivot_p(policy)
display_side_by_side([v_table_p, p_table], ['Value Table', 'Policy Table'])

# %%
