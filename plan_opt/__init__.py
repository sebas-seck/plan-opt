from gym.envs.registration import register

register(
    id="rampup-v0", entry_point="plan_opt.envs:RampupEnv0",
)

register(
    id="rampup-v1", entry_point="plan_opt.envs:RampupEnv1",
)

# register(
#     id="rampup-v1", entry_point="plan_opt.envs:RampupEnv2",
# )
