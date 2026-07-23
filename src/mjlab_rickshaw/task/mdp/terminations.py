"""Termination terms for the rickshaw task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def bad_orientation(
  env: ManagerBasedRlEnv,
  limit_angle: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  gravity_b = asset.data.projected_gravity_b
  return torch.acos(torch.clamp(-gravity_b[:, 2], -1.0, 1.0)) > limit_angle


def wheel_air(
  env: ManagerBasedRlEnv,
  sensor_name: str = "wheel_contact",
  duration: float = 0.25,
) -> torch.Tensor:
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, ContactSensor)
  air_time = sensor.data.current_air_time
  assert air_time is not None
  return torch.any(air_time > duration, dim=1)
