"""Reward terms for velocity tracking and force economy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .actions import TowForceAction
from .frames import terrain_frame
from .observations import traction_point_height, wheel_contact

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _asset(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> Entity:
  return env.scene[asset_cfg.name]


def _force_action(env: ManagerBasedRlEnv, action_name: str) -> TowForceAction:
  action = env.action_manager.get_term(action_name)
  assert isinstance(action, TowForceAction)
  return action


def track_forward_velocity(
  env: ManagerBasedRlEnv,
  command_name: str,
  sigma: float = 0.4,
) -> torch.Tensor:
  asset = _asset(env, _DEFAULT_ASSET_CFG)
  command = env.command_manager.get_command(command_name)
  assert command is not None
  forward, _, _ = terrain_frame(env, asset)
  velocity = torch.sum(asset.data.root_link_lin_vel_w * forward, dim=-1)
  value = torch.exp(
    -torch.square(velocity - command[:, 0]) / sigma**2
  )
  return value


def track_yaw_velocity(
  env: ManagerBasedRlEnv,
  command_name: str,
  sigma: float = 0.3,
) -> torch.Tensor:
  asset = _asset(env, _DEFAULT_ASSET_CFG)
  command = env.command_manager.get_command(command_name)
  assert command is not None
  _, _, normal = terrain_frame(env, asset)
  yaw_velocity = torch.sum(asset.data.root_link_ang_vel_w * normal, dim=-1)
  value = torch.exp(
    -torch.square(yaw_velocity - command[:, 2]) / sigma**2
  )
  return value


def forward_velocity_error(
  env: ManagerBasedRlEnv,
  command_name: str,
  scale: float = 0.4,
) -> torch.Tensor:
  """Quadratic integrand of the forward velocity tracking error."""
  asset = _asset(env, _DEFAULT_ASSET_CFG)
  command = env.command_manager.get_command(command_name)
  assert command is not None
  forward, _, _ = terrain_frame(env, asset)
  velocity = torch.sum(asset.data.root_link_lin_vel_w * forward, dim=-1)
  return torch.square((velocity - command[:, 0]) / scale)


def yaw_velocity_error(
  env: ManagerBasedRlEnv,
  command_name: str,
  scale: float = 0.3,
) -> torch.Tensor:
  """Quadratic integrand of the yaw velocity tracking error."""
  asset = _asset(env, _DEFAULT_ASSET_CFG)
  command = env.command_manager.get_command(command_name)
  assert command is not None
  _, _, normal = terrain_frame(env, asset)
  yaw_velocity = torch.sum(asset.data.root_link_ang_vel_w * normal, dim=-1)
  return torch.square((yaw_velocity - command[:, 2]) / scale)


def traction_height(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  target_height: float = 0.75,
  sigma: float = 0.05,
  slope_values: tuple[float, ...] = (0.0,),
) -> torch.Tensor:
  height = traction_point_height(env, asset_cfg, slope_values=slope_values)
  return torch.exp(-torch.sum(torch.square(height - target_height), dim=1) / sigma**2)


def undesired_linear_velocity(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset = _asset(env, asset_cfg)
  _, lateral, normal = terrain_frame(env, asset)
  velocity_w = asset.data.root_link_lin_vel_w
  lateral_velocity = torch.sum(velocity_w * lateral, dim=-1)
  normal_velocity = torch.sum(velocity_w * normal, dim=-1)
  return torch.square(lateral_velocity / 0.5) + torch.square(normal_velocity / 0.3)


def roll_pitch_rate(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset = _asset(env, asset_cfg)
  forward, lateral, _ = terrain_frame(env, asset)
  rate_w = asset.data.root_link_ang_vel_w
  return torch.square(torch.sum(rate_w * forward, dim=-1)) + torch.square(
    torch.sum(rate_w * lateral, dim=-1)
  )


def wheel_slip(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  wheel_radius: float = 0.3,
) -> torch.Tensor:
  asset = _asset(env, asset_cfg)
  forward, lateral, _ = terrain_frame(env, asset)
  wheel_vel_w = asset.data.body_link_vel_w[:, asset_cfg.body_ids, :3]
  forward_velocity = torch.sum(wheel_vel_w * forward[:, None, :], dim=-1)
  lateral_velocity = torch.sum(wheel_vel_w * lateral[:, None, :], dim=-1)
  roll_error = (
    forward_velocity - wheel_radius * asset.data.joint_vel[:, asset_cfg.joint_ids]
  )
  return torch.sum(torch.square(roll_error) + torch.square(lateral_velocity), dim=1)


def wheel_lift(
  env: ManagerBasedRlEnv,
  sensor_name: str = "wheel_contact",
  force_threshold: float = 1.0,
) -> torch.Tensor:
  return torch.sum(1.0 - wheel_contact(env, sensor_name, force_threshold), dim=1)


def peak_force(
  env: ManagerBasedRlEnv,
  action_name: str = "tow_force",
  soft_limit: float = 10.0,
  hard_limit: float = 50.0,
) -> torch.Tensor:
  action = _force_action(env, action_name)
  force_peak = torch.linalg.vector_norm(action.current_force_b, dim=-1).amax(dim=1)
  return torch.pow(torch.relu(force_peak - soft_limit) / (hard_limit - soft_limit),3)


def force_continuity(
  env: ManagerBasedRlEnv,
  action_name: str = "tow_force",
  hard_limit: float = 50.0,
) -> torch.Tensor:
  action = _force_action(env, action_name)
  delta = action.current_force_b - action.previous_force_b
  return torch.sum(torch.square(delta), dim=(1, 2)) / (2.0 * hard_limit**2)


def force_second_difference(
  env: ManagerBasedRlEnv,
  action_name: str = "tow_force",
  hard_limit: float = 50.0,
) -> torch.Tensor:
  action = _force_action(env, action_name)
  second_difference = (
    action.current_force_b
    - 2.0 * action.previous_force_b
    + action.previous_previous_force_b
  )
  return torch.sum(torch.square(second_difference), dim=(1, 2)) / (2.0 * hard_limit**2)


def force_difference(
  env: ManagerBasedRlEnv,
  action_name: str = "tow_force",
  hard_limit: float = 50.0,
  forward_weight: float = 0.2,
) -> torch.Tensor:
  action = _force_action(env, action_name)
  forward, lateral, normal = terrain_frame(env, action._entity)
  difference_w = action.current_force_w[:, 0] - action.current_force_w[:, 1]
  difference = torch.stack(
    (
      forward_weight * torch.sum(difference_w * forward, dim=-1),
      torch.sum(difference_w * lateral, dim=-1),
      torch.sum(difference_w * normal, dim=-1),
    ),
    dim=-1,
  )
  return torch.linalg.vector_norm(difference, dim=-1) / (2.0 * hard_limit)


def traction_power(
  env: ManagerBasedRlEnv,
  action_name: str = "tow_force",
  force_limit: float = 50.0,
  velocity_scale: float = 2.0,
) -> torch.Tensor:
  action = _force_action(env, action_name)
  power = torch.abs(
    torch.sum(action.current_force_w * action.hitch_lin_vel_w, dim=-1)
  )
  return torch.sum(power, dim=1) / (2.0 * force_limit * velocity_scale)


def termination(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.termination_manager.terminated.float()
