# -*- coding: utf-8 -*-

from gym.envs.registration import register

register(
    id="rampup-v1", entry_point="plan_opt.envs:RampupEnv1",
)

register(
    id="rampup-v2", entry_point="plan_opt.envs:RampupEnv2",
)

register(
    id="rampup-v3", entry_point="plan_opt.envs:RampupEnv3",
)
