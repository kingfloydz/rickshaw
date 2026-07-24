"""Force action term for the rickshaw towing points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class TowForceActionCfg(ActionTermCfg):
  """Six-dimensional left/right towing force action."""

  site_names: tuple[str, str] = ("left_tow_hitch", "right_tow_hitch")
  max_force: float = 50.0

  def build(self, env: ManagerBasedRlEnv) -> TowForceAction:
    return TowForceAction(self, env)


class TowForceAction(ActionTerm):
  """Apply body-frame forces at the two hitch sites.

  MuJoCo's ``xfrc_applied`` stores a wrench at a body's center of mass.  The
  two site forces are therefore summed into a base-body force and the exact
  moment about the base COM is added to the applied torque.
  """

  cfg: TowForceActionCfg

  def __init__(self, cfg: TowForceActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    entity = env.scene[cfg.entity_name]
    site_ids, _ = entity.find_sites(cfg.site_names, preserve_order=True)
    body_ids, _ = entity.find_bodies(("base_link",), preserve_order=True)
    self._entity = entity
    self._site_ids = torch.tensor(site_ids, device=self.device, dtype=torch.long)
    self._body_id = body_ids[0]
    self._max_force = cfg.max_force
    self._raw_actions = torch.zeros(self.num_envs, 6, device=self.device)
    self._current_force_b = torch.zeros(self.num_envs, 2, 3, device=self.device)
    self._previous_force_b = torch.zeros_like(self._current_force_b)
    self._previous_previous_force_b = torch.zeros_like(self._current_force_b)

  @property
  def action_dim(self) -> int:
    return 6

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  @property
  def current_force_b(self) -> torch.Tensor:
    return self._current_force_b

  @property
  def previous_force_b(self) -> torch.Tensor:
    return self._previous_force_b

  @property
  def previous_previous_force_b(self) -> torch.Tensor:
    return self._previous_previous_force_b

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_actions[:] = actions
    self._previous_previous_force_b[:] = self._previous_force_b
    self._previous_force_b[:] = self._current_force_b

    force = actions.reshape(self.num_envs, 2, 3) * self._max_force
    norm = torch.linalg.vector_norm(force, dim=-1, keepdim=True)
    scale = self._max_force / torch.clamp_min(norm, self._max_force)
    self._current_force_b[:] = force * scale

  def apply_actions(self) -> None:
    quat_w = self._entity.data.root_link_quat_w
    force_b = self._current_force_b
    quat_sites = quat_w[:, None, :].expand(-1, force_b.shape[1], -1)
    force_w = quat_apply(quat_sites.reshape(-1, 4), force_b.reshape(-1, 3)).view_as(
      force_b
    )
    hitch_pos_w = self._entity.data.site_pos_w[:, self._site_ids]
    root_com_w = self._entity.data.root_com_pos_w[:, None, :]
    torque_w = torch.cross(hitch_pos_w - root_com_w, force_w, dim=-1).sum(dim=1)
    force_sum_w = force_w.sum(dim=1)
    self._entity.data.write_external_wrench(
      force=force_sum_w[:, None, :],
      torque=torque_w[:, None, :],
      body_ids=[self._body_id],
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._raw_actions[env_ids] = 0.0
    self._current_force_b[env_ids] = 0.0
    self._previous_force_b[env_ids] = 0.0
    self._previous_previous_force_b[env_ids] = 0.0


class TowForceVisualization(ManagerTermBase):
  """Draw the two applied towing forces as 3D arrows in the viewer."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    self._action = env.action_manager.get_term("tow_force")
    self._debug_vis_enabled = True

  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device)

  def debug_vis(self, visualizer) -> None:
    if not self._debug_vis_enabled:
      return
    force_b = self._action.current_force_b
    quat_sites = self._action._entity.data.root_link_quat_w[:, None, :].expand(-1, 2, -1)
    force_w = quat_apply(quat_sites.reshape(-1, 4), force_b.reshape(-1, 3)).view_as(force_b)
    hitch_pos_w = self._action._entity.data.site_pos_w[:, self._action._site_ids]

    for env_idx in visualizer.get_env_indices(self.num_envs):
      for hitch_idx, color in enumerate(((0.95, 0.15, 0.1, 1.0), (0.1, 0.35, 0.95, 1.0))):
        start = hitch_pos_w[env_idx, hitch_idx]
        end = start + 0.02 * force_w[env_idx, hitch_idx]
        visualizer.add_arrow(start, end, color=color, width=0.025)
