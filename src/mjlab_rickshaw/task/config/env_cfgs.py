"""Environment configuration for the flat-ground rickshaw task."""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from mjlab_rickshaw.asset_zoo.rickshaw import get_rickshaw_cfg
from mjlab_rickshaw.task import mdp
from mjlab_rickshaw.task.terrain import TERRAIN_SLOPES, make_sloped_terrain_cfg


def make_rickshaw_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create the base rickshaw force-tracking configuration."""
  hitch_cfg = SceneEntityCfg("robot", site_names=("left_tow_hitch", "right_tow_hitch"))
  wheel_cfg = SceneEntityCfg(
    "robot",
    body_names=("left_wheel_link", "right_wheel_link"),
    joint_names=("left_wheel_joint", "right_wheel_joint"),
  )

  wheel_contact = ContactSensorCfg(
    name="wheel_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=("left_wheel_collision", "right_wheel_collision"),
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
    history_length=3,
  )

  actor_terms = {
    "command": ObservationTermCfg(
      func=mdp.command_forward_yaw,
      params={"command_name": "twist"},
    ),
    "body_linear_velocity": ObservationTermCfg(func=mdp.body_linear_velocity),
    "body_angular_velocity": ObservationTermCfg(func=mdp.body_angular_velocity),
    "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
    "traction_point_height": ObservationTermCfg(
      func=mdp.traction_point_height,
      params={"asset_cfg": hitch_cfg, "slope_values": TERRAIN_SLOPES},
    ),
    "previous_force": ObservationTermCfg(func=mdp.previous_tow_force),
    "previous_previous_force": ObservationTermCfg(func=mdp.previous_previous_tow_force),
  }
  critic_terms = {
    **actor_terms,
    "wheel_angular_velocity": ObservationTermCfg(
      func=mdp.wheel_angular_velocity,
      params={"asset_cfg": wheel_cfg},
    ),
    "ground_slope": ObservationTermCfg(
      func=mdp.ground_slope,
      params={"slope_values": TERRAIN_SLOPES},
    ),
    "wheel_contact": ObservationTermCfg(
      func=mdp.wheel_contact,
      params={"sensor_name": wheel_contact.name},
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  commands: dict[str, CommandTermCfg] = {
    "twist": mdp.RickshawVelocityCommandCfg(
      entity_name="robot",
      debug_vis=True,
      resampling_time_range=(3.0, 6.0),
      rel_standing_envs=0.1,
      rel_forward_envs=0.15,
      heading_command=False,
      ranges=mdp.RickshawVelocityCommandCfg.Ranges(
        lin_vel_x=(0.0, 2.0),
        lin_vel_y=(0.0, 0.0),
        ang_vel_z=(-0.8, 0.8),
      ),
    )
  }

  rewards = {
    "track_forward_velocity": RewardTermCfg(
      func=mdp.track_forward_velocity,
      weight=2.0,
      params={"command_name": "twist", "sigma": 0.4},
    ),
    "track_yaw_velocity": RewardTermCfg(
      func=mdp.track_yaw_velocity,
      weight=0.75,
      params={"command_name": "twist", "sigma": 0.3},
    ),
    "traction_point_height": RewardTermCfg(
      func=mdp.traction_height,
      weight=0.8,
      params={
        "asset_cfg": hitch_cfg,
        "target_height": 0.75,
        "sigma": 0.05,
        "slope_values": TERRAIN_SLOPES,
      },
    ),
    "undesired_linear_velocity": RewardTermCfg(
      func=mdp.undesired_linear_velocity,
      weight=-0.20,
    ),
    "roll_pitch_rate": RewardTermCfg(func=mdp.roll_pitch_rate, weight=-0.10),
    "wheel_slip": RewardTermCfg(
      func=mdp.wheel_slip,
      weight=-0.15,
      params={"asset_cfg": wheel_cfg, "wheel_radius": 0.3},
    ),
    "wheel_lift": RewardTermCfg(
      func=mdp.wheel_lift,
      weight=-0.50,
      params={"sensor_name": wheel_contact.name},
    ),
    "peak_force": RewardTermCfg(
      func=mdp.peak_force,
      weight=-3.0,
      params={"soft_limit": 10.0, "hard_limit": 50.0},
    ),
    "opposing_force": RewardTermCfg(
      func=mdp.opposing_force,
      weight=-1,
      params={"hard_limit": 50.0},
    ),
    "force_continuity": RewardTermCfg(
      func=mdp.force_continuity,
      weight=-1,
      params={"hard_limit": 50.0},
    ),
    "force_second_difference": RewardTermCfg(
      func=mdp.force_second_difference,
      weight=-0.3,
      params={"hard_limit": 50.0},
    ),
    "termination": RewardTermCfg(func=mdp.termination, weight=-200.0),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=make_sloped_terrain_cfg(),
      ),
      entities={"robot": get_rickshaw_cfg()},
      sensors=(wheel_contact,),
      num_envs=4096,
      env_spacing=4.0,
      extent=3.0,
    ),
    observations=observations,
    actions={
      "tow_force": mdp.TowForceActionCfg(
        entity_name="robot",
        site_names=("left_tow_hitch", "right_tow_hitch"),
        max_force=50.0,
      )
    },
    commands=commands,
    events={
      "reset_rickshaw": EventTermCfg(
        func=envs_mdp.reset_scene_to_default,
        mode="reset",
      )
    },
    rewards=rewards,
    terminations={
      "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
      "nan_detection": TerminationTermCfg(func=envs_mdp.nan_detection),
      "bad_orientation": TerminationTermCfg(
        func=mdp.bad_orientation,
        params={"limit_angle": math.radians(45.0)},
      ),
      "wheel_air": TerminationTermCfg(
        func=mdp.wheel_air,
        params={"sensor_name": wheel_contact.name, "duration": 0.25},
      ),
    },
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="base_link",
      distance=4.0,
      elevation=-12.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      njmax=256,
      nconmax=64,
      contact_sensor_maxmatch=16,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=20.0,
  )


def rickshaw_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Return training or interactive play configuration."""
  cfg = make_rickshaw_env_cfg()
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
  return cfg
