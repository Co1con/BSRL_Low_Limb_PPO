from __future__ import annotations  # noqa: I001

import torch
from typing import TYPE_CHECKING

from .HALO import HALO

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


def _get_low_limb_halo(env: ManagerBasedRLEnv) -> HALO:
    """获取批量 HALO，并初始化参考角和状态缓存。"""
    halo = getattr(env, "low_limb_halo", None)
    if not isinstance(halo, HALO) or halo.num_envs != env.num_envs:
        env.low_limb_halo = HALO(num_envs=env.num_envs, device=env.device)
        env.low_limb_halo_step_counter = -1
        env.low_limb_halo_reset_counter = -1
        env.low_limb_halo_reference_buf = torch.zeros(env.num_envs, 4, device=env.device)
        env.low_limb_halo_state_buf = torch.zeros(env.num_envs, 14, device=env.device)
    return env.low_limb_halo


def _halo_angles_to_robot_order(halo_angles: torch.Tensor) -> torch.Tensor:
    """将人体角度约定转换为机器人关节顺序和极性。

    HALO 输入：  [左髋, 右髋, 左膝, 右膝]，人体屈曲为正。
    机器人输出：[-左髋, 左膝, -右髋, 右膝]。
    """
    return torch.stack(
        (-halo_angles[:, 0], halo_angles[:, 2], -halo_angles[:, 1], halo_angles[:, 3]),
        dim=1,
    )


def _step_low_limb_halo(env: ManagerBasedRLEnv, command_name: str) -> None:
    """每个环境步只推进一次新版 HALO，并刷新独立输出缓存。"""
    halo = _get_low_limb_halo(env)
    step_counter = getattr(env, "common_step_counter", 0)

    # reward 可能在环境 reset 前调用，observation 则在 reset 后调用。即使 HALO
    # 本步已经推进，也必须先检查后出现的 episode 起点，避免漏掉局部 reset。
    if env.low_limb_halo_reset_counter != step_counter:
        reset_env_ids = torch.nonzero(
            env.episode_length_buf == 0, as_tuple=False
        ).flatten()
        if reset_env_ids.numel() > 0:
            halo.reset(reset_env_ids)
            reset_angles = _halo_angles_to_robot_order(halo.angles[reset_env_ids])
            env.low_limb_halo_reference_buf[reset_env_ids].copy_(reset_angles)
            env.low_limb_halo_state_buf[reset_env_ids] = halo.state[reset_env_ids]
            env.low_limb_halo_reset_counter = step_counter

    if env.low_limb_halo_step_counter == step_counter:
        return

    command = env.command_manager.get_command(command_name)
    halo_angles = halo.step(command[:, 0], env.step_dt)

    env.low_limb_halo_reference_buf.copy_(_halo_angles_to_robot_order(halo_angles))
    env.low_limb_halo_state_buf.copy_(halo.state)
    env.low_limb_halo_step_counter = step_counter


def low_limb_halo_reference(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """读取新版 HALO 的四关节参考角，形状为 ``[num_envs, 4]``。"""
    _step_low_limb_halo(env, command_name)
    return env.low_limb_halo_reference_buf


def low_limb_halo_state(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """读取新版 HALO 的完整状态，形状为 ``[num_envs, 14]``。"""
    _step_low_limb_halo(env, command_name)
    return env.low_limb_halo_state_buf


def rhythm_halo_xy(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """读取 HALO 主节律的 x、y，形状为 ``[num_envs, 2]``。"""
    _step_low_limb_halo(env, command_name)
    rhythm = env.low_limb_halo.rhythm
    return torch.stack((rhythm.x, rhythm.y), dim=1)
