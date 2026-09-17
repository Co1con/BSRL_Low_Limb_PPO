# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents
from .agents.g1_rsl_rl_ppo_cfg import G1PPORunnerCfg

##
# Register Gym environments.
##


# baseline —— gait phase
gym.register(
    id="bsrl-baseline-velocity-train",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bsrl_baseline_velocity_env_cfg:RobotEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)
gym.register(
    id="bsrl-baseline-velocity-play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bsrl_baseline_velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

# CPG —— reward constraint
gym.register(
    id="bsrl-cpg-velocity-train",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bsrl_cpg_velocity_env_cfg:RobotCPGEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="bsrl-cpg-velocity-play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bsrl_cpg_velocity_env_cfg:RobotCPGPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)


gym.register(
    id="g1-29dof-cpg-velocity-train",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_29dof_cpg_velocity_env_cfg:G1CPGEnvCfg",
        "rsl_rl_cfg_entry_point": f"{G1PPORunnerCfg.__module__}:{G1PPORunnerCfg.__name__}",
    },
)

gym.register(
    id="g1-29dof-cpg-velocity-play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_29dof_cpg_velocity_env_cfg:G1CPGPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{G1PPORunnerCfg.__module__}:{G1PPORunnerCfg.__name__}",
    },
)
