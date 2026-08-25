"""Left-right symmetry augmentation for the BSRL 12-DoF lower-limb robot."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_states"]


# Isaac Lab runtime joint/action order for this asset:
#   0 left_hip_yaw      1 right_hip_yaw
#   2 left_hip_roll     3 right_hip_roll
#   4 left_hip_pitch    5 right_hip_pitch
#   6 left_knee_pitch   7 right_knee_pitch
#   8 left_ankle_pitch  9 right_ankle_pitch
#  10 left_ankle_roll  11 right_ankle_roll

BSRL_JOINT_DIM = 12
BSRL_POLICY_OBS_DIM = 59
BSRL_LEFT_RIGHT_PERM = torch.tensor([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10], dtype=torch.long)
BSRL_JOINT_SIGNS = torch.tensor(
    [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0],
    dtype=torch.float32,
)


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Return original and left-right mirrored policy observations/actions."""
    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        obs_aug["policy"][:batch_size] = obs["policy"][:]
        obs_aug["policy"][batch_size : 2 * batch_size] = _transform_policy_obs_left_right(obs["policy"][:])
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device, dtype=actions.dtype)
        actions_aug[:batch_size] = actions[:]
        actions_aug[batch_size : 2 * batch_size] = _transform_actions_left_right(actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug


def _transform_policy_obs_left_right(obs: torch.Tensor) -> torch.Tensor:
    """Mirror the current 59-D BSRL policy observation layout."""
    obs = obs.clone()
    if obs.shape[-1] != BSRL_POLICY_OBS_DIM:
        raise ValueError(f"Expected BSRL policy obs dim {BSRL_POLICY_OBS_DIM}, got {obs.shape[-1]}.")

    device = obs.device
    dtype = obs.dtype

    ang_vel_signs = torch.tensor([-1.0, 1.0, -1.0], device=device, dtype=dtype)
    vector_signs = torch.tensor([1.0, -1.0, 1.0], device=device, dtype=dtype)
    command_signs = torch.tensor([1.0, -1.0, -1.0], device=device, dtype=dtype)

    end_idx = 0

    start_idx = end_idx
    end_idx = start_idx + 3
    obs[:, start_idx:end_idx] *= ang_vel_signs

    start_idx = end_idx
    end_idx = start_idx + 3
    obs[:, start_idx:end_idx] *= vector_signs

    start_idx = end_idx
    end_idx = start_idx + 3
    obs[:, start_idx:end_idx] *= command_signs

    for _ in range(4):
        start_idx = end_idx
        end_idx = start_idx + BSRL_JOINT_DIM
        obs[:, start_idx:end_idx] = _switch_bsrl_12dof_joints_left_right(obs[:, start_idx:end_idx])
        
    obs[:, end_idx : end_idx + 2] *= -1.0

    return obs


def _transform_actions_left_right(actions: torch.Tensor) -> torch.Tensor:
    return _switch_bsrl_12dof_joints_left_right(actions)


def _switch_bsrl_12dof_joints_left_right(joint_data: torch.Tensor) -> torch.Tensor:
    perm = BSRL_LEFT_RIGHT_PERM.to(device=joint_data.device)
    signs = BSRL_JOINT_SIGNS.to(device=joint_data.device, dtype=joint_data.dtype)
    return joint_data[..., perm] * signs
