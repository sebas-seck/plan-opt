# RL for Planning Optimization

This project takes a shot at creating a custom environment to be used with Reinforcement Learning to tackle a planning optimization problem. The environment is put to test with [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/index.html).

https://stable-baselines3.readthedocs.io/en/master/index.html

- Requirements are pinned in `requirements.txt`
- Notebooks are saved in the format `nb_<file>.py` not ending in .ipynb due to the use of [Jupytext](https://jupytext.readthedocs.io/en/latest/).

- PPO (on-policy method) does not produce good results
- A2C (off-policy method) results are more promising