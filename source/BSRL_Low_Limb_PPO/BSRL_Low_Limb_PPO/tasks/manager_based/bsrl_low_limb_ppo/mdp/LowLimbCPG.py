"""适用于 Isaac Lab 并行环境的下肢 CPG。

全部动态状态均以 ``num_envs`` 为第一维并保存在同一计算设备上。关节维
顺序固定为：左髋、右髋、左膝、右膝。
"""

from __future__ import annotations

import math

import torch

TWO_PI = 2.0 * math.pi
JOINT_NAMES = ("left_hip", "right_hip", "left_knee", "right_knee")


def _check_dt(dt: float) -> float:
    dt = float(dt)
    if dt <= 0.0:
        raise ValueError("dt must be greater than zero")
    return dt


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _env_ids(env_ids, num_envs: int, device: torch.device) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(num_envs, device=device, dtype=torch.long)
    return torch.as_tensor(env_ids, device=device, dtype=torch.long).flatten()


def _batch_tensor(value, num_envs: int, device, dtype, name: str) -> torch.Tensor:
    value = torch.as_tensor(value, device=device, dtype=dtype)
    if value.numel() == 1:
        return value.reshape(1).expand(num_envs)
    if value.numel() != num_envs:
        raise ValueError(f"{name} must contain 1 or {num_envs} values")
    return value.reshape(num_envs)


class RhythmHopf:
    """批量生成全局节律，并让各环境独立返回自己的 reset 相位。"""

    def __init__(self, num_envs: int, device, dtype=torch.float32):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.dtype = dtype
        self.mu = 1.0
        self.gamma = 20.0
        self.frequency_slope = 0.48749031
        self.frequency_intercept = 0.60252246

        self.x = torch.empty(self.num_envs, device=self.device, dtype=dtype)
        self.y = torch.empty_like(self.x)
        self.omega = torch.zeros_like(self.x)
        self.reset_phase = torch.empty_like(self.x)
        self.is_running = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.stop_requested = torch.zeros_like(self.is_running)
        self.reset()

    @property
    def phase(self) -> torch.Tensor:
        return torch.remainder(torch.atan2(self.y, self.x), TWO_PI)

    @property
    def frequency(self) -> torch.Tensor:
        return self.omega / TWO_PI

    def velocity_to_frequency(self, velocity: torch.Tensor) -> torch.Tensor:
        velocity = velocity.clamp_min(0.0)
        frequency = self.frequency_slope * velocity + self.frequency_intercept
        return torch.where(
            velocity > 0.0,
            frequency,
            torch.zeros_like(frequency),
        )

    @torch.no_grad()
    def reset(self, env_ids=None) -> None:
        ids = _env_ids(env_ids, self.num_envs, self.device)
        if ids.numel() == 0:
            return
        choice = torch.randint(0, 2, (ids.numel(),), device=self.device)
        self.reset_phase[ids] = choice.to(self.dtype) * math.pi
        radius = math.sqrt(self.mu)
        self.x[ids] = radius * torch.cos(self.reset_phase[ids])
        self.y[ids] = radius * torch.sin(self.reset_phase[ids])
        self.omega[ids] = 0.0
        self.is_running[ids] = False
        self.stop_requested[ids] = False

    @torch.no_grad()
    def step(self, velocity: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
        dt = _check_dt(dt)
        velocity = _batch_tensor(
            velocity, self.num_envs, self.device, self.dtype, "velocity"
        )
        moving = velocity > 0.0

        commanded_omega = TWO_PI * self.velocity_to_frequency(velocity)
        self.omega.copy_(torch.where(moving, commanded_omega, self.omega))
        self.stop_requested.copy_(
            torch.where(moving, False, self.stop_requested | self.is_running)
        )
        self.is_running.logical_or_(moving)

        radius_squared = self.x.square() + self.y.square()
        radial = self.gamma * (self.mu - radius_squared)
        next_x = self.x + (radial * self.x - self.omega * self.y) * dt
        next_y = self.y + (radial * self.y + self.omega * self.x) * dt

        # 未启动的环境保持 reset 状态；停车请求则等待指定 reset 相位。
        next_x = torch.where(self.is_running, next_x, self.x)
        next_y = torch.where(self.is_running, next_y, self.y)
        reset_direction = torch.cos(self.reset_phase)
        at_reset_phase = torch.isclose(self.y, torch.zeros_like(self.y), atol=1e-12)
        at_reset_phase &= self.x * reset_direction > 0.0
        crosses_reset_phase = (self.y * next_y <= 0.0) & (
            next_x * reset_direction > 0.0
        )
        should_stop = self.stop_requested & (at_reset_phase | crosses_reset_phase)

        self.x.copy_(
            torch.where(should_stop, math.sqrt(self.mu) * reset_direction, next_x)
        )
        self.y.copy_(torch.where(should_stop, torch.zeros_like(next_y), next_y))
        self.omega.masked_fill_(should_stop, 0.0)
        self.is_running.masked_fill_(should_stop, False)
        self.stop_requested.masked_fill_(should_stop, False)
        return self.x, self.y


class JointOscillator:
    """用一个 ``[num_envs, 4]`` Hopf 状态批量软跟踪四关节相位。"""

    def __init__(self, num_envs: int, device, dtype=torch.float32):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.dtype = dtype
        self.mu = 1.0
        self.gamma = 20.0
        self.phase_gain = 3.5
        self.frequency_response = 14.0

        shape = (self.num_envs, len(JOINT_NAMES))
        self.x = torch.empty(shape, device=self.device, dtype=dtype)
        self.y = torch.empty_like(self.x)
        self.omega = torch.empty_like(self.x)
        self.phase_offset = torch.empty_like(self.x)
        self.phase_error = torch.empty_like(self.x)
        self.initialized = torch.empty(shape, device=self.device, dtype=torch.bool)
        self.reset()

    @property
    def phase(self) -> torch.Tensor:
        return torch.remainder(torch.atan2(self.y, self.x), TWO_PI)

    @property
    def frequency(self) -> torch.Tensor:
        return self.omega / TWO_PI

    @torch.no_grad()
    def reset(self, env_ids=None) -> None:
        ids = _env_ids(env_ids, self.num_envs, self.device)
        if ids.numel() == 0:
            return
        self.x[ids] = math.sqrt(self.mu)
        self.y[ids] = 0.0
        self.omega[ids] = 0.0
        self.phase_error[ids] = 0.0
        self.initialized[ids] = False

    @torch.no_grad()
    def step(
        self,
        rhythm_phase: torch.Tensor,
        rhythm_frequency: torch.Tensor,
        phase_offset: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dt = _check_dt(dt)
        rhythm_phase = _batch_tensor(
            rhythm_phase, self.num_envs, self.device, self.dtype, "rhythm_phase"
        )
        rhythm_frequency = _batch_tensor(
            rhythm_frequency,
            self.num_envs,
            self.device,
            self.dtype,
            "rhythm_frequency",
        )
        if phase_offset.shape != self.phase_offset.shape:
            raise ValueError(f"phase_offset must have shape {self.phase_offset.shape}")
        self.phase_offset.copy_(phase_offset)

        target_phase = rhythm_phase[:, None] + self.phase_offset
        initialize = ~self.initialized
        self.x.copy_(torch.where(initialize, math.sqrt(self.mu) * torch.cos(target_phase), self.x))
        self.y.copy_(torch.where(initialize, math.sqrt(self.mu) * torch.sin(target_phase), self.y))
        self.initialized.fill_(True)

        self.phase_error.copy_(_wrap_to_pi(target_phase - self.phase))
        target_omega = (
            TWO_PI * rhythm_frequency[:, None]
            + self.phase_gain * torch.sin(self.phase_error)
        )
        target_omega = torch.where(
            rhythm_frequency[:, None] > 0.0,
            target_omega.clamp_min(0.0),
            target_omega,
        )

        decay = math.exp(-self.frequency_response * dt)
        self.omega.copy_(target_omega + (self.omega - target_omega) * decay)
        radius_squared = self.x.square() + self.y.square()
        radial = self.gamma * (self.mu - radius_squared)
        next_x = self.x + (radial * self.x - self.omega * self.y) * dt
        next_y = self.y + (radial * self.y + self.omega * self.x) * dt
        self.x.copy_(next_x)
        self.y.copy_(next_y)

        # Rhythm 停止后，各关节独立收敛到自己的目标相位。
        remaining_error = _wrap_to_pi(target_phase - self.phase)
        settled = (
            (rhythm_frequency[:, None] == 0.0)
            & (remaining_error.abs() < 1e-4)
            & (self.omega.abs() < 1e-3)
        )
        self.x.copy_(torch.where(settled, math.sqrt(self.mu) * torch.cos(target_phase), self.x))
        self.y.copy_(torch.where(settled, math.sqrt(self.mu) * torch.sin(target_phase), self.y))
        self.omega.masked_fill_(settled, 0.0)
        self.phase_error.masked_fill_(settled, 0.0)
        return self.x, self.y


class JointShaper:
    """将四关节相位批量映射为三阶傅里叶关节角。"""

    def __init__(self, device, dtype=torch.float32):
        hip = (0.0078388447, 0.0000546934, 0.2613439175, -0.0482368723,
               -0.0312843541, 0.0047676238, -0.0142935173)
        knee = (0.2367006798, -0.0001668887, 0.3473199593, -0.0100145053,
                0.2530838817, 0.0287142304, 0.0462009950)
        self.coefficients = torch.tensor(
            (hip, hip, knee, knee), device=device, dtype=dtype
        )

    def step(self, phase: torch.Tensor) -> torch.Tensor:
        angle = self.coefficients[:, 0].unsqueeze(0).expand_as(phase).clone()
        for order in range(1, 4):
            angle.add_(self.coefficients[:, 2 * order - 1] * torch.sin(order * phase))
            angle.add_(self.coefficients[:, 2 * order] * torch.cos(order * phase))
        return angle


class LowLimbCPG:
    """速度驱动的四关节批量 CPG。

    Args:
        num_envs: 并行环境数量。
        device: Isaac Lab 环境使用的设备，例如 ``"cuda:0"``。
        dtype: CPG 状态的数据类型，训练中默认使用 ``torch.float32``。
    """

    joint_names = JOINT_NAMES

    def __init__(self, num_envs: int, device, dtype=torch.float32):
        if num_envs <= 0:
            raise ValueError("num_envs must be greater than zero")
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.dtype = dtype
        self.phase_speed_slope = -0.3307487167
        self.phase_speed_intercept = 5.2494736659
        self.max_integration_dt = 0.005

        self._velocity = torch.zeros(num_envs, device=self.device, dtype=dtype)
        self.rhythm = RhythmHopf(num_envs, self.device, dtype)
        self.joints = JointOscillator(num_envs, self.device, dtype)
        self.shaper = JointShaper(self.device, dtype)
        self.hip_knee_phase_difference = torch.full(
            (num_envs,), self.phase_speed_intercept, device=self.device, dtype=dtype
        )
        self.phase_offsets = torch.empty(
            num_envs, len(JOINT_NAMES), device=self.device, dtype=dtype
        )
        self.angles = torch.zeros_like(self.phase_offsets)
        self._update_phase_offsets()

    @property
    def velocity(self) -> torch.Tensor:
        return self._velocity

    @property
    def state(self) -> torch.Tensor:
        """返回适合写入 OBS 的状态：[rhythm xy, joint xy, joint frequency]。"""
        return torch.cat(
            (
                self.rhythm.x[:, None],
                self.rhythm.y[:, None],
                self.joints.x,
                self.joints.y,
                self.joints.frequency,
            ),
            dim=1,
        )

    def _update_phase_offsets(self, env_ids=None) -> None:
        ids = _env_ids(env_ids, self.num_envs, self.device)
        if ids.numel() == 0:
            return
        phase_difference = (
            self.phase_speed_slope * self.velocity[ids] + self.phase_speed_intercept
        )
        self.hip_knee_phase_difference[ids] = phase_difference
        self.phase_offsets[ids, 0] = 0.0
        self.phase_offsets[ids, 1] = math.pi
        self.phase_offsets[ids, 2] = -phase_difference
        self.phase_offsets[ids, 3] = -phase_difference + math.pi

    @torch.no_grad()
    def reset(self, env_ids=None) -> None:
        """只重置指定环境；传入 ``None`` 时重置全部环境。"""
        ids = _env_ids(env_ids, self.num_envs, self.device)
        if ids.numel() == 0:
            return
        self._velocity[ids] = 0.0
        self.rhythm.reset(ids)
        self.joints.reset(ids)
        self._update_phase_offsets(ids)
        self.angles[ids] = 0.0

    def _step_once(self, target_velocity: torch.Tensor, dt: float) -> None:
        self._velocity.copy_(target_velocity)
        self.rhythm.step(target_velocity, dt)
        self._update_phase_offsets()
        self.joints.step(
            self.rhythm.phase,
            self.rhythm.frequency,
            self.phase_offsets,
            dt,
        )
        self.angles.copy_(self.shaper.step(self.joints.phase))

    @torch.no_grad()
    def step(self, target_velocity: torch.Tensor, dt: float) -> torch.Tensor:
        """推进全部环境并返回 ``[num_envs, 4]`` 关节参考角。

        输入可以是标量、``[num_envs]`` 或 ``[num_envs, 1]``。负值会被截断为
        零，因为当前速度—步频模型仅针对向前步速建立。
        """
        dt = _check_dt(dt)
        target_velocity = _batch_tensor(
            target_velocity, self.num_envs, self.device, self.dtype, "target_velocity"
        ).clamp_min(0.0)
        substeps = max(1, math.ceil(dt / self.max_integration_dt))
        substep_dt = dt / substeps
        for _ in range(substeps):
            self._step_once(target_velocity, substep_dt)
        return self.angles


__all__ = [
    "JOINT_NAMES",
    "RhythmHopf",
    "JointOscillator",
    "JointShaper",
    "LowLimbCPG",
]
