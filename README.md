# RL for Planning Optimization

This project takes a shot at creating a custom environment to be used with Reinforcement Learning to tackle a planning optimization problem. The environment is put to test with [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/index.html).

The choronological evolution of the `rampup` environment can be followed in the notebooks starting with `nb_##_ ... .py/ipynb`. The environment with the highest version number is most advanced, though I keep the progress and things that did not work around for reference.

### `rampup-v1`
Very basic concept of the environment, observation space is created quite literally. **Notebooks 01-09** refer to the creation and evaluation of this environment. PPO (on-policy method) does not produce valuable results, results with A2C (off-policy method) are more promising. Also, training and evaluation combinations between the simple 4-weeks demand and various 3 years demands are evaluated.

### `rampup-v2`
Similar environment to `rampup-v1`, but it shows that illegal action transitions can not be learned from this observation space, except for some auspicious occasions! See **notebook 10**. Adds mandatory config for environment creation and workflows around it.

### `rampup-v3`
Redesigned observation space. Illegal moves successfully integrated.

## Work with this repo
- Requirements are pinned in `requirements.txt`
- Notebooks are saved in the format `nb_<file>.py` (for better versioning) in addition to `.ipynb` (to document results) using [Jupytext](https://jupytext.readthedocs.io/en/latest/).
