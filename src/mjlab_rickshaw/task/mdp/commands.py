"""Velocity command with rickshaw telemetry panels for Viser."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg

from .actions import TowForceAction
from .frames import terrain_frame

if TYPE_CHECKING:
  import viser
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class RickshawVelocityCommandCfg(UniformVelocityCommandCfg):
  """Build the velocity command with Viser telemetry."""

  def build(self, env: ManagerBasedRlEnv) -> RickshawVelocityCommand:
    return RickshawVelocityCommand(self, env)


class RickshawVelocityCommand(UniformVelocityCommand):
  """Uniform velocity command with five live plots for the selected environment."""

  _HISTORY_LENGTH = 200

  def __init__(self, cfg: RickshawVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._viewer_enabled = None
    self._viewer_sliders: dict[int, Any] = {}
    self._viewer_get_env_idx = None
    self._plot_env_idx = -1
    self._plot_handles: list[Any] = []
    self._plot_histories = [
      (deque(maxlen=self._HISTORY_LENGTH), deque(maxlen=self._HISTORY_LENGTH))
      for _ in range(5)
    ]

  def _update_metrics(self) -> None:
    forward, lateral, normal = terrain_frame(self._env, self.robot)
    velocity_w = self.robot.data.root_link_lin_vel_w
    actual_linear = torch.stack(
      (
        torch.sum(velocity_w * forward, dim=-1),
        torch.sum(velocity_w * lateral, dim=-1),
      ),
      dim=-1,
    )
    actual_yaw = torch.sum(self.robot.data.root_link_ang_vel_w * normal, dim=-1)
    max_command_step = self.cfg.resampling_time_range[1] / self._env.step_dt
    self.metrics["error_vel_xy"] += (
      torch.linalg.vector_norm(self.vel_command_b[:, :2] - actual_linear, dim=-1)
      / max_command_step
    )
    self.metrics["error_vel_yaw"] += (
      torch.abs(self.vel_command_b[:, 2] - actual_yaw) / max_command_step
    )

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx,
    on_change=None,
    request_action=None,
  ) -> None:
    """Create command controls and five task-specific telemetry panels."""
    del request_action
    import viser.uplot
    from viser import Icon

    with server.gui.add_folder(name.capitalize()):
      self._viewer_enabled = server.gui.add_checkbox("Enable", initial_value=False)
      for axis, label, max_value in (
        (0, "lin_vel_x", self.cfg.ranges.lin_vel_x[1]),
        (2, "ang_vel_z", self.cfg.ranges.ang_vel_z[1]),
      ):
        limit = server.gui.add_slider(
          f"Max {label}",
          initial_value=max_value,
          step=0.1,
          min=0.1,
          max=10.0,
        )
        slider = server.gui.add_slider(
          label,
          min=-max_value,
          max=max_value,
          step=0.05,
          initial_value=0.0,
        )

        @limit.on_update
        def _(_event, _slider=slider, _limit=limit) -> None:
          _slider.min = -_limit.value
          _slider.max = _limit.value

        if on_change is not None:

          @slider.on_update
          def _(_event) -> None:
            on_change()

        self._viewer_sliders[axis] = slider

      zero_button = server.gui.add_button("Zero", icon=Icon.SQUARE_X)

      @zero_button.on_click
      def _(_event) -> None:
        for slider in self._viewer_sliders.values():
          slider.value = 0.0

    self._viewer_get_env_idx = get_env_idx
    with server.gui.add_folder("Rickshaw telemetry", expand_by_default=True):
      for title, labels, colors in (
        ("Forward velocity (m/s)", ("Target", "Actual"), ("#d62728", "#1f77b4")),
        ("Yaw velocity (rad/s)", ("Target", "Actual"), ("#d62728", "#1f77b4")),
        ("Tow force forward (N)", ("Left", "Right"), ("#d62728", "#1f77b4")),
        ("Tow force lateral (N)", ("Left", "Right"), ("#d62728", "#1f77b4")),
        ("Tow force normal (N)", ("Left", "Right"), ("#d62728", "#1f77b4")),
      ):
        self._plot_handles.append(
          server.gui.add_uplot(
            data=(np.array([]), np.array([]), np.array([])),
            series=(
              viser.uplot.Series(label="Steps"),
              viser.uplot.Series(label=labels[0], stroke=colors[0], width=2),
              viser.uplot.Series(label=labels[1], stroke=colors[1], width=2),
            ),
            scales={
              "x": viser.uplot.Scale(
                time=False, auto=False, range=(-self._HISTORY_LENGTH, 0)
              ),
              "y": viser.uplot.Scale(auto=True),
            },
            legend=viser.uplot.Legend(show=True),
            title=title,
            aspect=2.0,
          )
        )

  def compute(self, dt: float) -> None:
    super().compute(dt)
    if self._viewer_enabled is not None and self._viewer_enabled.value:
      env_idx = self._viewer_get_env_idx()
      for axis, slider in self._viewer_sliders.items():
        self.vel_command_b[env_idx, axis] = slider.value
    self._update_plots()

  def _debug_vis_impl(self, visualizer) -> None:
    action = cast(TowForceAction, self._env.action_manager.get_term("tow_force"))
    force_w = action.current_force_w
    hitch_pos_w = action.hitch_pos_w

    colors = ((0.95, 0.15, 0.1, 1.0), (0.1, 0.35, 0.95, 1.0))
    for env_idx in visualizer.get_env_indices(self.num_envs):
      for hitch_idx, color in enumerate(colors):
        start = hitch_pos_w[env_idx, hitch_idx]
        end = start + 0.02 * force_w[env_idx, hitch_idx]
        visualizer.add_arrow(start, end, color=color, width=0.025)

  def _update_plots(self) -> None:
    if not self._plot_handles:
      return
    env_idx = self._viewer_get_env_idx()
    if env_idx != self._plot_env_idx:
      for histories in self._plot_histories:
        for history in histories:
          history.clear()
      self._plot_env_idx = env_idx

    action = cast(TowForceAction, self._env.action_manager.get_term("tow_force"))
    forward, lateral, normal = terrain_frame(self._env, self.robot)
    basis_w = torch.stack((forward[env_idx], lateral[env_idx], normal[env_idx]))
    force = action.current_force_w[env_idx] @ basis_w.T
    forward_velocity = torch.sum(
      self.robot.data.root_link_lin_vel_w[env_idx] * forward[env_idx]
    )
    yaw_velocity = torch.sum(
      self.robot.data.root_link_ang_vel_w[env_idx] * normal[env_idx]
    )
    values = (
      torch.stack(
        [
          torch.stack(
            [
              self.vel_command_b[env_idx, 0],
              forward_velocity,
            ]
          ),
          torch.stack(
            [
              self.vel_command_b[env_idx, 2],
              yaw_velocity,
            ]
          ),
          force[:, 0],
          force[:, 1],
          force[:, 2],
        ]
      )
      .cpu()
      .numpy()
    )
    for handle, histories, pair in zip(
      self._plot_handles, self._plot_histories, values, strict=True
    ):
      histories[0].append(pair[0])
      histories[1].append(pair[1])
      length = len(histories[0])
      x = np.arange(-length + 1, 1, dtype=np.float64)
      handle.data = (
        x,
        np.fromiter(histories[0], dtype=np.float64, count=length),
        np.fromiter(histories[1], dtype=np.float64, count=length),
      )
