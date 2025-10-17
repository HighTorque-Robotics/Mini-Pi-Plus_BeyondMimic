import gymnasium as gym

from . import agents, flat_env_cfg

##
# Register Gym environments.
##

# 用于训练的环境（保持自适应采样）
gym.register(
    id="Tracking-Flat-Hi-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.HiFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HiFlatPPORunnerCfg",
    }, 
)

gym.register(
    id="Tracking-Flat-Hi-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.HiFlatWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HiFlatPPORunnerCfg",
    },
)
