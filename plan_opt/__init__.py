from gym.envs.registration import register

register(
    id="plan-opt-v0", entry_point="plan_opt.envs:RampupEnv",
)
