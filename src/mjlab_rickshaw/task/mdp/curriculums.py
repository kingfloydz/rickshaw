"""Curriculum for enabling force economy after tracking is learned."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class ForcePenaltyCurriculum:
  """Increase force penalties smoothly after both tracking terms reach 0.8."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    del cfg
    del env
    self.rho = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice | None,
    forward_reward_name: str = "track_forward_velocity",
    yaw_reward_name: str = "track_yaw_velocity",
    ramp_fraction: float = 0.1,
  ) -> dict[str, float]:
    ids = slice(None) if env_ids is None else env_ids
    if env.common_step_counter > 0:
      episode_steps = env.episode_length_buf[ids].float()
      forward_cfg = env.reward_manager.get_term_cfg(forward_reward_name)
      yaw_cfg = env.reward_manager.get_term_cfg(yaw_reward_name)
      dt = env.step_dt
      forward_mean = env.reward_manager._episode_sums[forward_reward_name][ids] / (
        forward_cfg.weight * dt * episode_steps
      )
      yaw_mean = env.reward_manager._episode_sums[yaw_reward_name][ids] / (
        yaw_cfg.weight * dt * episode_steps
      )
      if torch.all((forward_mean >= 0.8) & (yaw_mean >= 0.8)):
        self.rho += (1.0 - self.rho) * ramp_fraction
        self.rho = min(self.rho, 1.0)

    for name, base_weight in (
      ("peak_force", -3),
      ("force_continuity", -0.5),
      ("force_second_difference", -0.1),
    ):
      env.reward_manager.get_term_cfg(name).weight = base_weight * self.rho
    return {"rho": self.rho}
