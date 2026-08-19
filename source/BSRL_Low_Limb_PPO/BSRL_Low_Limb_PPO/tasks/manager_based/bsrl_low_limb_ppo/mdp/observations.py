from __future__ import annotations  # noqa: I001

import torch
from typing import TYPE_CHECKING

from .hopf_generator import LowLimbGenerator

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


def _get_hopf_generator(env: ManagerBasedRLEnv) -> LowLimbGenerator:
    if not hasattr(env, "hopf_generator") or env.hopf_generator.num_envs != env.num_envs:
        env.hopf_generator = LowLimbGenerator(num_envs=env.num_envs, device=env.device)
        env.hopf_generator_step_counter = -1
        env.hopf_generator_reset_counter = -1
        env.hopf_reference_buf = torch.zeros(env.num_envs, 4, device=env.device)
        env.hopf_state_buf = torch.zeros(env.num_envs, 12, device=env.device)
    return env.hopf_generator


def _step_hopf_generator(env: ManagerBasedRLEnv, command_name: str) -> None:
    generator = _get_hopf_generator(env)
    step_counter = getattr(env, "common_step_counter", 0)

    # 同一个仿真步内可能被多个 observation/reward 调用，只允许 Hopf 前进一步。
    if env.hopf_generator_step_counter == step_counter:
        return

    # reset 后 episode_length_buf 为 0，只重置对应并行环境的 Hopf 内部状态。
    if env.hopf_generator_reset_counter != step_counter:
        reset_env_ids = torch.nonzero(env.episode_length_buf == 0, as_tuple=False).flatten()
        if len(reset_env_ids) > 0:
            generator.reset(reset_env_ids)
        env.hopf_generator_reset_counter = step_counter

    command = env.command_manager.get_command(command_name)
    vx = command[:, 0]
    env.hopf_reference_buf = generator.step_velocity(vx, env.step_dt)
    env.hopf_state_buf = generator.state
    env.hopf_generator_step_counter = step_counter


def hopf_reference(env: ManagerBasedRLEnv, command_name: str = "base_velocity") -> torch.Tensor:
    """输出左右 hip/knee 的 Hopf 参考角。"""
    _step_hopf_generator(env, command_name)
    return env.hopf_reference_buf


def hopf_state(env: ManagerBasedRLEnv, command_name: str = "base_velocity") -> torch.Tensor:
    """输出 master、left、right 三个 Hopf 振荡器的内部状态。"""
    _step_hopf_generator(env, command_name)
    return env.hopf_state_buf
