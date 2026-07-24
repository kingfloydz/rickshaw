"""Ground-aligned frame shared by rickshaw rewards and visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.utils.lab_api.math import quat_apply

from ..terrain import TERRAIN_SLOPES

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def terrain_slope(
  env: ManagerBasedRlEnv,
  slope_values: tuple[float, ...] = TERRAIN_SLOPES,
) -> torch.Tensor:
  slopes = torch.as_tensor(slope_values, device=env.device)
  slope_x = slopes[env.scene["terrain"].terrain_types]
  return torch.stack((slope_x, torch.zeros_like(slope_x)), dim=-1)


def ground_normal(
  env: ManagerBasedRlEnv,
  dtype: torch.dtype,
  slope_values: tuple[float, ...] = TERRAIN_SLOPES,
) -> torch.Tensor:
  slope = terrain_slope(env, slope_values).to(dtype=dtype)
  normal = torch.cat((-torch.tan(slope), torch.ones_like(slope[:, :1])), dim=-1)
  return normal / torch.linalg.vector_norm(normal, dim=-1, keepdim=True)


def terrain_frame(
  env: ManagerBasedRlEnv,
  asset: Entity,
  slope_values: tuple[float, ...] = TERRAIN_SLOPES,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Return ground-tangent forward, lateral, and normal unit vectors in world frame."""
  normal = ground_normal(env, asset.data.root_link_quat_w.dtype, slope_values)
  axle_b = torch.zeros_like(normal)
  axle_b[:, 1] = 1.0
  axle_w = quat_apply(asset.data.root_link_quat_w, axle_b)
  forward = torch.cross(axle_w, normal, dim=-1)
  forward = forward / torch.linalg.vector_norm(forward, dim=-1, keepdim=True)
  lateral = torch.cross(normal, forward, dim=-1)
  return forward, lateral, normal
