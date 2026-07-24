"""Actor and privileged critic observations for the rickshaw task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

from ..terrain import TERRAIN_SLOPES
from .actions import TowForceAction
from .frames import ground_normal, terrain_slope

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _asset(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> Entity:
  return env.scene[asset_cfg.name]


def command_forward_yaw(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return command[:, (0, 2)]


def body_linear_velocity(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  return _asset(env, asset_cfg).data.root_link_lin_vel_b


def body_angular_velocity(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  return _asset(env, asset_cfg).data.root_link_ang_vel_b


def projected_gravity(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  return _asset(env, asset_cfg).data.projected_gravity_b


def traction_point_height(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  slope_values: tuple[float, ...] = TERRAIN_SLOPES,
) -> torch.Tensor:
  asset = _asset(env, asset_cfg)
  normal = ground_normal(env, asset.data.site_pos_w.dtype, slope_values)
  point_w = asset.data.site_pos_w[:, asset_cfg.site_ids]
  origin_w = env.scene.env_origins[:, None, :]
  return torch.sum((point_w - origin_w) * normal[:, None, :], dim=-1)


def previous_tow_force(
  env: ManagerBasedRlEnv, action_name: str = "tow_force"
) -> torch.Tensor:
  action = env.action_manager.get_term(action_name)
  assert isinstance(action, TowForceAction)
  return action.current_force_b.reshape(env.num_envs, 6)


def previous_previous_tow_force(
  env: ManagerBasedRlEnv, action_name: str = "tow_force"
) -> torch.Tensor:
  action = env.action_manager.get_term(action_name)
  assert isinstance(action, TowForceAction)
  return action.previous_force_b.reshape(env.num_envs, 6)


def wheel_angular_velocity(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset = _asset(env, asset_cfg)
  return asset.data.joint_vel[:, asset_cfg.joint_ids]


def ground_slope(
  env: ManagerBasedRlEnv,
  slope_values: tuple[float, ...] = TERRAIN_SLOPES,
) -> torch.Tensor:
  return terrain_slope(env, slope_values)


def wheel_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str = "wheel_contact",
  force_threshold: float = 1.0,
) -> torch.Tensor:
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, ContactSensor)
  history = sensor.data.force_history
  assert history is not None
  return (
    (torch.linalg.vector_norm(history, dim=-1) > force_threshold).any(dim=-1).float()
  )
