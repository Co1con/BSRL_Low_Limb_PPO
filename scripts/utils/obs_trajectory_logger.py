from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import torch


class ObsTrajectoryLogger:
    """Record selected observation dimensions from one environment during play."""

    def __init__(
        self,
        path: str | Path,
        obs_start: int,
        obs_dim: int = 4,
        env_id: int = 0,
        labels: Sequence[str] | None = None,
    ):
        self.path = Path(path)
        self.obs_start = obs_start
        self.obs_dim = obs_dim
        self.env_id = env_id
        self.labels = list(labels) if labels is not None else [f"obs_{i}" for i in range(obs_dim)]
        if len(self.labels) != obs_dim:
            raise ValueError(f"labels length must match obs_dim: {len(self.labels)} != {obs_dim}")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["step", "time_s", "env_id", *self.labels])

    def close(self) -> None:
        self.file.close()

    def log(self, obs, step: int, time_s: float) -> None:
        if isinstance(obs, tuple):
            obs = obs[0]

        if isinstance(obs, dict):
            if "policy" in obs:
                obs = obs["policy"]
            elif "obs" in obs:
                obs = obs["obs"]
            else:
                raise KeyError(f"Cannot find policy/obs key in observation dict: {list(obs.keys())}")

        if isinstance(obs, (list, tuple)):
            obs = obs[0]

        if obs.ndim == 1:
            obs = obs.unsqueeze(0)

        if obs.ndim != 2:
            raise ValueError(f"Expected obs to have shape [num_envs, obs_dim], got {tuple(obs.shape)}")

        if self.env_id >= obs.shape[0]:
            raise IndexError(f"env_id {self.env_id} is out of bounds for obs with {obs.shape[0]} envs")

        obs_end = self.obs_start + self.obs_dim
        if obs_end > obs.shape[-1]:
            raise IndexError(f"obs slice [{self.obs_start}:{obs_end}] exceeds obs dim {obs.shape[-1]}")

        values = obs[self.env_id, self.obs_start:obs_end].detach().cpu().tolist()
        self.writer.writerow([step, time_s, self.env_id, *values])
        self.file.flush()

    def log_values(self, values: torch.Tensor, step: int, time_s: float) -> None:
        if values.ndim == 1:
            values = values.unsqueeze(0)

        if self.env_id >= values.shape[0]:
            raise IndexError(f"env_id {self.env_id} is out of bounds for values with {values.shape[0]} envs")

        if values.shape[-1] != self.obs_dim:
            raise ValueError(f"Expected values dim {self.obs_dim}, got {values.shape[-1]}")

        row = values[self.env_id].detach().cpu().tolist()
        self.writer.writerow([step, time_s, self.env_id, *row])
        self.file.flush()
