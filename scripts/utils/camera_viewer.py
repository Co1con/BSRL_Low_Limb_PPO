from __future__ import annotations

import torch


def quat_rotate_wxyz(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_w = q[0]
    q_xyz = q[1:4]
    uv = torch.cross(q_xyz, v, dim=0)
    uuv = torch.cross(q_xyz, uv, dim=0)
    return v + 2.0 * (q_w * uv + uuv)


def camera_follow(env):
    if not hasattr(camera_follow, "smooth_camera_positions"):
        camera_follow.smooth_camera_positions = []

    robot_pos = env.unwrapped.scene["robot"].data.root_pos_w[0]
    robot_quat = env.unwrapped.scene["robot"].data.root_quat_w[0]

    camera_offset = torch.tensor([-3.0, 0.0, 0.5], dtype=torch.float32, device=env.device)
    camera_pos = robot_pos + quat_rotate_wxyz(robot_quat, camera_offset)

    window_size = 50
    camera_follow.smooth_camera_positions.append(camera_pos)
    if len(camera_follow.smooth_camera_positions) > window_size:
        camera_follow.smooth_camera_positions.pop(0)

    smooth_camera_pos = torch.mean(torch.stack(camera_follow.smooth_camera_positions), dim=0)

    env.unwrapped.viewport_camera_controller.set_view_env_index(env_index=0)
    env.unwrapped.viewport_camera_controller.update_view_location(
        eye=smooth_camera_pos.cpu().numpy(),
        lookat=robot_pos.cpu().numpy(),
    )
    