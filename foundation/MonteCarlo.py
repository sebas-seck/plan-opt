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

# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# # Monte Carlo approaches

# %%
import numpy as np
import pandas as pd
from IPython.core.display import display, HTML
import matplotlib.pyplot as plt
import gym
from gym import spaces


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
    p_table = p_table.assign(demand = demand[:len(p_table)])
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
        
    def current_state(self):
        return self.state_time, int(self.state_status[self.state_time])

# %% [markdown]
# ## MC Prediction Problem only
# This section solves the prediction problem, taking a policy it uses Monte Carlo to obtain V(s). To find the estimate Q(s, a) at time t, we would have a quadratically growing combination between state set and action set sizes.

# %%
SMALL_ENOUGH = 1e-3 # threshold for convergence
GAMMA = 1.  # discount factor

def pred_prob(env, policy):
    
    start_states = [state for state in all_states]
    start_idx = np.random.choice(len(start_states))
    env.set_state(start_states[start_idx])

    state = env.current_state()
    states_and_rewards = [(state, 0)]  # tuples with reward per state
    while env.state_time < env.horizon -1:
        a = policy[state]
        o, r, d, i  = env.step(a)
        state = env.current_state()
        states_and_rewards.append((state, r))
    # calculate the returns by working backwards from the terminal state
    G = 0
    states_and_returns = []
    first = True
    for state, r in reversed(states_and_rewards):
        # the value of the terminal state is 0 by definition
        # we should ignore the first state we encounter
        # and ignore the last G, which is meaningless since it doesn't correspond to any move
        if first:
            first = False
        else:
            states_and_returns.append((state, G))
        G = r + GAMMA*G
    states_and_returns.reverse() # we want it to be in order of state visited
    return states_and_returns


# %%
h = 7*4  # time horizon
env = RampupEnv(horizon=h, verbose=False)
all_states = set([(s, a) for s in range(len(env.state_status)) for a in range(env.action_space.n)])
policy = {}
for state in all_states:
    policy[state] = env.action_space.sample()    
V = {}
returns = {}

for state in all_states:
    env.set_state(state)
    if env.state_time < env.horizon -1:
        returns[state] = []
    else:
        V[state] = 0
        
for t in range(100):

    # generate an episode
    states_and_returns = pred_prob(env, policy)
    seen_states = set()
    for state, G in states_and_returns:
        # Is it the first visit to the state?
        if state not in seen_states:
            returns[state].append(G)
            V[state] = np.mean(returns[state])
            seen_states.add(state)   

# %%
v_table, v_table_p = pivot_v(V)
p_table = pivot_p(policy)
display_side_by_side([v_table_p, p_table], ['Value Table', 'Policy Table'])

# %% [markdown]
# ## Monte Carlo with exploring starts
# Instead of going over all possible combinations, we use the exploring-starts methods to ensure each state finds coverage.

# %%
GAMMA = .9  # discount factor

def pred_prob_es(env, policy):
    
    start_states = [s for s in states]
    start_idx = np.random.choice(len(start_states))
    env.set_state(start_states[start_idx])

    s = env.current_state()
    a = env.action_space.sample()
    states_actions_rewards = [(s, a, 0)]  # tuples with reward per state
    seen_states = set()
    seen_states.add(env.current_state())
    num_steps = 0
    
    while True:
        o, r, d, i  = env.step(a)
        num_steps += 1
        s = env.current_state()
        
        if s in seen_states:
            reward = -200000. / num_steps
            states_actions_rewards.append((s, None, reward))
            break
        elif not env.state_time < env.horizon -1:  # terminal state
            states_actions_rewards.append((s, None, r))
            break
        else:
            a = policy[s]
            state_time, state_status = s
            if a in legal_changes[state_status]:
                states_actions_rewards.append((s, a, r))

        seen_states.add(s)
    
    G = 0
    states_actions_returns = []
    first = True
    for s, a, r in reversed(states_actions_rewards):
        # first state needs to be irgnored
        if first:
            first = False
        else:
            states_actions_returns.append((s, a, G))
        G = r + GAMMA * G
    states_actions_returns.reverse()
    return states_actions_returns

def max_dict(d):
    max_key = None
    max_val = float('-inf')
    for k, v in d.items():
        if v > max_val:
            max_val = v
            max_key = k
    return max_key, max_val


# %%
h = 7*4  # time horizon
env = RampupEnv(horizon=h, verbose=False)
states = set([(s, a) for s in range(len(env.state_status)-1) for a in range(env.action_space.n)])
policy = {}
for s in states:
    policy[s] = env.action_space.sample()    
Q = {}
returns = {}

for s in states:
    #env.set_state(s)
    state_time, state_status = s
    if state_time < env.horizon -1:
        Q[s] = {}
        for a in legal_changes.keys():#[state_status]:
            Q[s][a] = 0
            returns[(s, a)] = []
    else:
        # no care for terminal states
        pass
    
deltas = []
for t in range(2000):
    if t % 100 == 0:
        print(t)
        
    biggest_change = 0
    states_actions_returns = pred_prob_es(env, policy)
    seen_state_action_pairs = set()
    for s, a, G in states_actions_returns:
        sa = (s, a)
        state_time, state_status = s
        if state_time < env.horizon - 1:
            if sa not in seen_state_action_pairs:
                old_q = Q[s][a]
                returns[sa].append(G)
                Q[s][a] = np.mean(returns[sa])
                biggest_change = max(biggest_change, np.abs(old_q - Q[s][a]))
                seen_state_action_pairs.add(sa)
    deltas.append(biggest_change)
    
    # update policy
    for s in policy.keys():
        state_time, state_status = s
        if state_time < env.horizon - 1:
            policy[s] = max_dict(Q[s])[0]
        
plt.plot(deltas)
plt.show

V = {}
for s, Qs in Q.items():
    V[s] = max_dict(Q[s])[1]

# %%
v_table, v_table_p = pivot_v(V)
p_table = pivot_p(policy)
display_side_by_side([v_table_p, p_table], ['Value Table', 'Policy Table'])

# %% [markdown]
# The optimal policy is infeasible as we find many situations which to not work with legal state transitions!

# %% [markdown]
# ### Monte Carlo without exploring starts

# %%
GAMMA = .9  # discount factor

def go_random(a, eps=0.1):
    p = np.random.random()
    if p < (1 - eps):
        return a
    else:
        return env.action_space.sample()
    
def mc_no_es(env, policy):
    s = (0, 0)
    env.set_state(s)
    a = go_random(policy[s])    
    states_actions_rewards = [(s, a, 0)]

    while True:
        o, r, d, i = env.step(a)
        s = env.current_state()
        if not env.state_time < env.horizon -1:  # terminal state
            states_actions_rewards.append((s, None, r))
            break
        else:
            a = go_random(policy[s]) # the next state is stochastic
            states_actions_rewards.append((s, a, r))

  # calculate the returns by working backwards from the terminal state
    G = 0
    states_actions_returns = []
    first = True
    for s, a, r in reversed(states_actions_rewards):
        # the value of the terminal state is 0 by definition
        # we should ignore the first state we encounter
        # and ignore the last G, which is meaningless since it doesn't correspond to any move
        if first:
            first = False
        else:
            states_actions_returns.append((s, a, G))
        G = r + GAMMA*G
    states_actions_returns.reverse() # we want it to be in order of state visited
    return states_actions_returns


# %%
h = 7*4  # time horizon
env = RampupEnv(horizon=h, verbose=False)
states = set([(s, a) for s in range(len(env.state_status)-1) for a in range(env.action_space.n)])
policy = {}
for s in states:
    policy[s] = env.action_space.sample()  
Q = {}
returns = {}

for s in states:
    #env.set_state(s)
    state_time, state_status = s
    if state_time < env.horizon -1:
        Q[s] = {}
        for a in legal_changes.keys():#[state_status]:
            Q[s][a] = 0
            returns[(s, a)] = []
    else:
        # no care for terminal states
        pass

deltas = []
for t in range(5000):
    if t % 1000 == 0:
        print(t)
        
    biggest_change = 0
    states_actions_returns = mc_no_es(env, policy)
    seen_state_action_pairs = set()
    for s, a, G in states_actions_returns:
        sa = (s, a)
        state_time, state_status = s
        if state_time < env.horizon - 1:
            if sa not in seen_state_action_pairs:
                old_q = Q[s][a]
                returns[sa].append(G)
                Q[s][a] = np.mean(returns[sa])
                biggest_change = max(biggest_change, np.abs(old_q - Q[s][a]))
                seen_state_action_pairs.add(sa)
    deltas.append(biggest_change)
    
    # update policy
    for s in policy.keys():
        a, _ = max_dict(Q[s])
        policy[s] = a
        
        
        #state_time, state_status = s
        #if state_time < env.horizon - 1:
        #    policy[s] = max_dict(Q[s])[0]
        
plt.plot(deltas)
plt.show

V = {}
for s in policy.keys():
    V[s] = max_dict(Q[s])[1]

# %%
v_table, v_table_p = pivot_v(V)
p_table = pivot_p(policy)
display_side_by_side([v_table_p, p_table], ['Value Table', 'Policy Table'])

# %%

# %% [markdown]
# down here I am trying to improve..

# %%
h = 7*4  # time horizon
env = RampupEnv(horizon=h, verbose=False)
states = set([(s, a) for s in range(len(env.state_status)-1) for a in range(env.action_space.n)])
policy = {}

#for s in states:
#    policy[s] = 0  #env.action_space.sample()

# initial policy: stay in same status (cause that is always legal)
for (state_time, state_status) in states:
    policy[(state_time, state_status)] = state_status  #env.action_space.sample()
    
u = mc_no_es(env, policy)

# %%
GAMMA = .9  # discount factor
np.random.seed(123)
def go_random(s, intended_a, eps=0.1):
    p = np.random.random()
    state_time, state_status = s
    if p < (1 - eps):
        if intended_a in legal_changes[state_status]:
            return intended_a
        else:
            return np.random.choice(tuple(legal_changes[state_status]), 1)[0]
    else:
        # if the above does not catch, return a random legal action
        return np.random.choice(tuple(legal_changes[state_status]), 1)[0]
    
def mc_no_es(env, policy):
    s = (0, 0)
    env.set_state(s)
    a = go_random(s, policy[s])    
    states_actions_rewards = [(s, a, 0)]
    #print(f'{s} and {a}')

    while True:
        o, r, d, i = env.step(a)
        s = env.current_state()
        if not env.state_time < env.horizon -1:  # terminal state
            states_actions_rewards.append((s, None, r))
            break
        else:
            a = go_random(s, policy[s]) # the next state is stochastic
            states_actions_rewards.append((s, a, r))

  # calculate the returns by working backwards from the terminal state
    G = 0
    states_actions_returns = []
    first = True
    for s, a, r in reversed(states_actions_rewards):
        # the value of the terminal state is 0 by definition
        # we should ignore the first state we encounter
        # and ignore the last G, which is meaningless since it doesn't correspond to any move
        if first:
            first = False
        else:
            states_actions_returns.append((s, a, G))
        G = r + GAMMA*G
    states_actions_returns.reverse() # we want it to be in order of state visited
    #print('here we go')
    #print(states_actions_returns)
    return states_actions_returns


# %%
h = 7*4  # time horizon
env = RampupEnv(horizon=h, verbose=False)
states = set([(s, a) for s in range(len(env.state_status)-1) for a in range(env.action_space.n)])
policy = {}

#for s in states:
#    policy[s] = 0  #env.action_space.sample()

# initial policy: stay in same status (cause that is always legal)
for (state_time, state_status) in states:
    policy[(state_time, state_status)] = state_status  #env.action_space.sample()
    

Q = {}
returns = {}

for s in states:
#     env.set_state(s)
    state_time, state_status = s
    if state_time < env.horizon -1:
        Q[s] = {}
        # for a in legal_changes.keys():#[state_status]:
        for a in legal_changes[state_status]:
            Q[s][a] = 0
            returns[(s, a)] = []
    else:
        # no care for terminal states
        pass

deltas = []
for t in range(5000):
    if t % 1000 == 0:
        print(t)
        
    biggest_change = 0
    states_actions_returns = mc_no_es(env, policy)
    seen_state_action_pairs = set()
    for s, a, G in states_actions_returns:
        sa = (s, a)
        state_time, state_status = s
        if state_time < env.horizon - 1:
            if sa not in seen_state_action_pairs:
                # print(sa)
                old_q = Q[s][a]
                returns[sa].append(G)
                Q[s][a] = np.mean(returns[sa])
                biggest_change = max(biggest_change, np.abs(old_q - Q[s][a]))
                seen_state_action_pairs.add(sa)
    deltas.append(biggest_change)
    
    # update policy
    for s in policy.keys():
        a, _ = max_dict(Q[s])
        policy[s] = a
        
        
        #state_time, state_status = s
        #if state_time < env.horizon - 1:
        #    policy[s] = max_dict(Q[s])[0]
        
plt.plot(deltas)
plt.show

V = {}
for s in policy.keys():
    V[s] = max_dict(Q[s])[1]

# %%
v_table, v_table_p = pivot_v(V)
p_table = pivot_p(policy)
display_side_by_side([v_table_p, p_table], ['Value Table', 'Policy Table'])

# %%
v_table_p.max(axis=1)
