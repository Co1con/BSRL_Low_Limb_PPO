from __future__ import annotations

import math

import torch


class MasterHopf:
    """并行环境下的主 Hopf 振荡器，负责统一步态相位。"""

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device,
        dtype: torch.dtype = torch.float32,
        mu: float = 1.0,
        gamma: float = 20.0,
        stop_eps: float = 1e-6,
        phase_eps: float = 2e-2,
        return_omega_min: float = 2.0 * math.pi * 0.0,
        return_omega_max: float = 2.0 * math.pi * 5.0,
    ):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.dtype = dtype
        self.mu = mu
        self.gamma = gamma
        self.stop_eps = stop_eps
        self.phase_eps = phase_eps
        self.return_omega_min = return_omega_min
        self.return_omega_max = return_omega_max

        self.x = torch.empty(num_envs, device=self.device, dtype=self.dtype)
        self.y = torch.empty(num_envs, device=self.device, dtype=self.dtype)
        self.omega = torch.empty(num_envs, device=self.device, dtype=self.dtype)
        self.last_omega = torch.empty(num_envs, device=self.device, dtype=self.dtype)
        self.reset()

    def reset(self, env_ids: torch.Tensor | slice | None = None, phase: float | torch.Tensor = 0.0) -> None:
        if env_ids is None:
            env_ids = slice(None)

        phase = torch.as_tensor(phase, device=self.device, dtype=self.dtype)
        r = math.sqrt(self.mu)
        self.x[env_ids] = r * torch.cos(phase)
        self.y[env_ids] = r * torch.sin(phase)
        self.omega[env_ids] = 0.0
        self.last_omega[env_ids] = self.return_omega_max

    @property
    def phase(self) -> torch.Tensor:
        return torch.remainder(torch.atan2(self.y, self.x), 2.0 * math.pi)

    @property
    def state(self) -> torch.Tensor:
        return torch.stack([self.x, self.y, self.omega, self.phase], dim=-1)

    def step(self, omega_cmd: torch.Tensor, dt: float) -> None:
        omega_cmd = omega_cmd.to(device=self.device, dtype=self.dtype).view(self.num_envs)
        phase = self.phase

        is_walking = torch.abs(omega_cmd) > self.stop_eps
        is_near_start = (phase <= self.phase_eps) | (phase >= 2.0 * math.pi - self.phase_eps)
        should_reset = (~is_walking) & is_near_start
        should_return = (~is_walking) & (~is_near_start)

        self.omega = torch.where(is_walking, omega_cmd, self.omega)
        self.last_omega = torch.where(is_walking, torch.abs(omega_cmd), self.last_omega)

        return_omega = torch.clamp(self.last_omega, self.return_omega_min, self.return_omega_max)
        self.omega = torch.where(should_return, return_omega, self.omega)

        # 停止命令到来时，先让相位回到起点，再重置为静止。
        if torch.any(should_reset):
            self.reset(torch.nonzero(should_reset, as_tuple=False).flatten())

        active = ~should_reset
        if not torch.any(active):
            return

        r2 = self.x * self.x + self.y * self.y
        dx = self.gamma * (self.mu - r2) * self.x - self.omega * self.y
        dy = self.gamma * (self.mu - r2) * self.y + self.omega * self.x
        self.x[active] = self.x[active] + dx[active] * dt
        self.y[active] = self.y[active] + dy[active] * dt

        phase = self.phase
        should_reset = (~is_walking) & ((phase <= self.phase_eps) | (phase >= 2.0 * math.pi - self.phase_eps))
        if torch.any(should_reset):
            self.reset(torch.nonzero(should_reset, as_tuple=False).flatten())


class JointHopf:
    """左右腿关节 Hopf 振荡器，跟随主振荡器并输出 hip/knee 参考角。"""

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device,
        dtype: torch.dtype = torch.float32,
        offset: float = 0.0,
        mu: float = 1.0,
        gamma: float = 20.0,
        coupling: float = 5.0,
    ):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.dtype = dtype
        self.mu = mu
        self.gamma = gamma
        self.coupling = coupling
        self.offset = offset
        self.cos_offset = math.cos(offset)
        self.sin_offset = math.sin(offset)

        coeffs = torch.tensor(
            [
                [-0.000117729978, 0.259456627],
                [0.230165019, -0.0755455646],
                [-0.104658017, -0.336285480],
                [-0.0321349430, -0.147464670],
                [-0.00716299027, 0.0856785938],
                [0.00119805748, 0.00444044707],
                [0.00979784744, 0.0246951121],
            ],
            device=self.device,
            dtype=self.dtype,
        )
        self.coeffs = coeffs.to(device=self.device, dtype=self.dtype)

        self.x = torch.empty(num_envs, device=self.device, dtype=self.dtype)
        self.y = torch.empty(num_envs, device=self.device, dtype=self.dtype)
        self.omega = torch.empty(num_envs, device=self.device, dtype=self.dtype)
        self.reset()

    def reset(self, env_ids: torch.Tensor | slice | None = None, phase: float | torch.Tensor = 0.0) -> None:
        if env_ids is None:
            env_ids = slice(None)

        phase = torch.as_tensor(phase, device=self.device, dtype=self.dtype) + self.offset
        r = math.sqrt(self.mu)
        self.x[env_ids] = r * torch.cos(phase)
        self.y[env_ids] = r * torch.sin(phase)
        self.omega[env_ids] = 0.0

    @property
    def phase(self) -> torch.Tensor:
        return torch.remainder(torch.atan2(self.y, self.x), 2.0 * math.pi)

    @property
    def state(self) -> torch.Tensor:
        return torch.stack([self.x, self.y, self.omega, self.phase], dim=-1)

    @staticmethod
    def basis(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        phase = torch.atan2(y, x)
        return torch.stack(
            [
                torch.ones_like(phase),
                torch.cos(phase),
                torch.sin(phase),
                torch.cos(2.0 * phase),
                torch.sin(2.0 * phase),
                torch.cos(3.0 * phase),
                torch.sin(3.0 * phase),
            ],
            dim=-1,
        )

    @property
    def q_ref(self) -> torch.Tensor:
        return self.basis(self.x, self.y) @ self.coeffs

    def step(self, master_x: torch.Tensor, master_y: torch.Tensor,
             master_omega: torch.Tensor, dt: float) -> torch.Tensor:
        master_x = master_x.to(device=self.device, dtype=self.dtype).view(self.num_envs)
        master_y = master_y.to(device=self.device, dtype=self.dtype).view(self.num_envs)
        master_omega = master_omega.to(device=self.device, dtype=self.dtype).view(self.num_envs)

        target_x = master_x * self.cos_offset - master_y * self.sin_offset
        target_y = master_x * self.sin_offset + master_y * self.cos_offset

        r = torch.sqrt(self.x * self.x + self.y * self.y)
        rt = torch.sqrt(target_x * target_x + target_y * target_y)
        phase_error = (self.x * target_y - self.y * target_x) / torch.clamp(r * rt, min=1e-8)
        self.omega = master_omega + self.coupling * phase_error

        r2 = self.x * self.x + self.y * self.y
        dx = self.gamma * (self.mu - r2) * self.x - self.omega * self.y
        dy = self.gamma * (self.mu - r2) * self.y + self.omega * self.x
        self.x = self.x + dx * dt
        self.y = self.y + dy * dt

        return self.q_ref * torch.sign(master_omega).unsqueeze(-1)


class LowLimbGenerator:
    """低肢 Hopf 轨迹生成器，输入前向速度，输出左右 hip/knee 参考角。"""

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device,
        dtype: torch.dtype = torch.float32,
        velocity_freq_slope: float = 0.45,
        velocity_freq_intercept: float = 0.60,
    ):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.dtype = dtype
        self.velocity_freq_slope = velocity_freq_slope
        self.velocity_freq_intercept = velocity_freq_intercept

        self.master = MasterHopf(num_envs=num_envs, device=self.device, dtype=self.dtype)
        self.left = JointHopf(num_envs=num_envs, device=self.device, dtype=self.dtype, offset=0.0)
        self.right = JointHopf(num_envs=num_envs, device=self.device, dtype=self.dtype, offset=math.pi)

    def reset(self, env_ids: torch.Tensor | slice | None = None, phase: float | torch.Tensor = 0.0) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.master.reset(env_ids, phase)
        self.left.reset(env_ids, self.master.phase[env_ids])
        self.right.reset(env_ids, self.master.phase[env_ids])

    @property
    def state(self) -> torch.Tensor:
        return torch.cat([self.master.state, self.left.state, self.right.state], dim=-1)

    def velocity_to_frequency(self, vx: torch.Tensor) -> torch.Tensor:
        vx = vx.to(device=self.device, dtype=self.dtype).view(self.num_envs)
        abs_vx = torch.abs(vx)

        freq = -0.3407 * abs_vx * abs_vx + 1.1721 * abs_vx + 0.1003
        freq = torch.clamp(freq, min=0.0, max=1.5)

        return torch.where(abs_vx < 1e-6, torch.zeros_like(freq), freq)

    def step(self, freq_hz: torch.Tensor, dt: float) -> torch.Tensor:
        freq_hz = freq_hz.to(device=self.device, dtype=self.dtype).view(self.num_envs)
        self.master.step(2.0 * math.pi * freq_hz, dt)
        left = self.left.step(self.master.x, self.master.y, self.master.omega, dt)
        right = self.right.step(self.master.x, self.master.y, self.master.omega, dt)
        return torch.stack([-left[:, 0], left[:, 1], -right[:, 0], right[:, 1]], dim=-1)

    def step_velocity(self, vx: torch.Tensor, dt: float) -> torch.Tensor:
        return self.step(self.velocity_to_frequency(vx), dt)
