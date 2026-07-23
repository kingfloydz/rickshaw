"""Registered rickshaw task configurations."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import rickshaw_flat_env_cfg
from .rl_cfg import rickshaw_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Rickshaw-Force-Flat",
  env_cfg=rickshaw_flat_env_cfg(),
  play_env_cfg=rickshaw_flat_env_cfg(play=True),
  rl_cfg=rickshaw_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

__all__ = ["rickshaw_flat_env_cfg", "rickshaw_ppo_runner_cfg"]
