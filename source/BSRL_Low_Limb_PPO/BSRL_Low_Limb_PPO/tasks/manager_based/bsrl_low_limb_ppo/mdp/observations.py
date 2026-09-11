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


def hopf_master_xy(env: ManagerBasedRLEnv, command_name: str = "base_velocity") -> torch.Tensor:
    """输出 MasterHopf 的 x、y，同时刷新 hopf_reference_buf。"""
    _step_hopf_generator(env, command_name)
    return env.hopf_state_buf[:, :2]


# =========================== 新版 LowLimbCPG ===========================

from .LowLimbCPG import LowLimbCPG  # noqa: E402


def _get_low_limb_cpg(env: ManagerBasedRLEnv) -> LowLimbCPG:
    """获取新版批量 CPG；其对象和缓存均与旧 Hopf 完全独立。"""
    cpg = getattr(env, "low_limb_cpg", None)
    if not isinstance(cpg, LowLimbCPG) or cpg.num_envs != env.num_envs:
        env.low_limb_cpg = LowLimbCPG(num_envs=env.num_envs, device=env.device)
        env.low_limb_cpg_step_counter = -1
        env.low_limb_cpg_reset_counter = -1
        env.low_limb_cpg_reference_buf = torch.zeros(env.num_envs, 4, device=env.device)
        env.low_limb_cpg_state_buf = torch.zeros(env.num_envs, 14, device=env.device)
    return env.low_limb_cpg


def _cpg_angles_to_robot_order(cpg_angles: torch.Tensor) -> torch.Tensor:
    """将人体角度约定转换为机器人关节顺序和极性。

    CPG 输入：  [左髋, 右髋, 左膝, 右膝]，人体屈曲为正。
    机器人输出：[-左髋, 左膝, -右髋, 右膝]。
    """
    return torch.stack(
        (-cpg_angles[:, 0], cpg_angles[:, 2], -cpg_angles[:, 1], cpg_angles[:, 3]),
        dim=1,
    )


def _step_low_limb_cpg(env: ManagerBasedRLEnv, command_name: str) -> None:
    """每个环境步只推进一次新版 CPG，并刷新独立输出缓存。"""
    cpg = _get_low_limb_cpg(env)
    step_counter = getattr(env, "common_step_counter", 0)

    # reward 可能在环境 reset 前调用，observation 则在 reset 后调用。即使 CPG
    # 本步已经推进，也必须先检查后出现的 episode 起点，避免漏掉局部 reset。
    if env.low_limb_cpg_reset_counter != step_counter:
        reset_env_ids = torch.nonzero(
            env.episode_length_buf == 0, as_tuple=False
        ).flatten()
        if reset_env_ids.numel() > 0:
            cpg.reset(reset_env_ids)
            reset_angles = _cpg_angles_to_robot_order(cpg.angles[reset_env_ids])
            env.low_limb_cpg_reference_buf[reset_env_ids].copy_(reset_angles)
            env.low_limb_cpg_state_buf[reset_env_ids] = cpg.state[reset_env_ids]
            env.low_limb_cpg_reset_counter = step_counter

    if env.low_limb_cpg_step_counter == step_counter:
        return

    command = env.command_manager.get_command(command_name)
    cpg_angles = cpg.step(command[:, 0], env.step_dt)

    env.low_limb_cpg_reference_buf.copy_(_cpg_angles_to_robot_order(cpg_angles))
    env.low_limb_cpg_state_buf.copy_(cpg.state)
    env.low_limb_cpg_step_counter = step_counter


def low_limb_cpg_reference(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """读取新版 CPG 的四关节参考角，形状为 ``[num_envs, 4]``。"""
    _step_low_limb_cpg(env, command_name)
    return env.low_limb_cpg_reference_buf


def low_limb_cpg_state(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """读取新版 CPG 的完整状态，形状为 ``[num_envs, 14]``。"""
    _step_low_limb_cpg(env, command_name)
    return env.low_limb_cpg_state_buf


def rhythm_hopf_xy(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """读取新版 RhythmHopf 的 x、y，形状为 ``[num_envs, 2]``。"""
    _step_low_limb_cpg(env, command_name)
    rhythm = env.low_limb_cpg.rhythm
    return torch.stack((rhythm.x, rhythm.y), dim=1)
