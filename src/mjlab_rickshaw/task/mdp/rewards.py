"""Reward terms for velocity tracking and force economy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from .actions import TowForceAction
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
  value = torch.exp(
    -torch.square(asset.data.root_link_lin_vel_b[:, 0] - command[:, 0]) / sigma**2
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
  value = torch.exp(
    -torch.square(asset.data.root_link_ang_vel_b[:, 2] - command[:, 2]) / sigma**2
  )
  return value


def traction_height(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  target_height: float = 0.75,
  sigma: float = 0.05,
) -> torch.Tensor:
  height = traction_point_height(env, asset_cfg)
  return torch.exp(-torch.sum(torch.square(height - target_height), dim=1) / sigma**2)


def undesired_linear_velocity(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  vel = _asset(env, asset_cfg).data.root_link_lin_vel_b
  return torch.square(vel[:, 1] / 0.5) + torch.square(vel[:, 2] / 0.3)


def roll_pitch_rate(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  rate = _asset(env, asset_cfg).data.root_link_ang_vel_b
  return torch.square(rate[:, 0]) + torch.square(rate[:, 1])


def wheel_slip(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  wheel_radius: float = 0.3,
) -> torch.Tensor:
  asset = _asset(env, asset_cfg)
  wheel_vel_w = asset.data.body_link_vel_w[:, asset_cfg.body_ids, :3]
  wheel_vel_b = quat_apply_inverse(asset.data.root_link_quat_w[:, None, :], wheel_vel_w)
  roll_error = (
    wheel_vel_b[..., 0] - wheel_radius * asset.data.joint_vel[:, asset_cfg.joint_ids]
  )
  lateral_error = wheel_vel_b[..., 1]
  return torch.sum(torch.square(roll_error) + torch.square(lateral_error), dim=1)


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
  return torch.square(torch.relu(force_peak - soft_limit) / (hard_limit - soft_limit))


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


def termination(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.termination_manager.terminated.float()
