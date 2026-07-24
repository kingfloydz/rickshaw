"""Per-environment inclined terrain configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np
from mjlab.terrains import TerrainGeneratorCfg
from mjlab.terrains.terrain_generator import (
  SubTerrainCfg,
  TerrainGeometry,
  TerrainOutput,
)

TERRAIN_SLOPES = tuple(i / 100.0 for i in range(-8, 11))


@dataclass(kw_only=True)
class InclinedPlaneCfg(SubTerrainCfg):
  """Finite plane with a fixed x-axis inclination in radians."""

  angle: float
  thickness: float = 0.2

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    del difficulty, rng
    body = spec.body("terrain")
    half_thickness = self.thickness / 2.0
    geom = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(self.size[0] / 2.0, self.size[1] / 2.0, half_thickness),
      pos=(self.size[0] / 2.0, self.size[1] / 2.0, -half_thickness * math.cos(self.angle)),
      quat=(math.cos(-self.angle / 2.0), 0.0, math.sin(-self.angle / 2.0), 0.0),
      rgba=(0.35, 0.38, 0.42, 1.0),
    )
    return TerrainOutput(
      origin=np.array([self.size[0] / 2.0, self.size[1] / 2.0, 0.0]),
      geometries=[TerrainGeometry(geom=geom)],
    )


def make_sloped_terrain_cfg() -> TerrainGeneratorCfg:
  """Create 19 fixed slopes with slightly lower weight at larger inclinations."""
  sub_terrains = {
    f"slope_{slope:+.2f}": InclinedPlaneCfg(
      angle=slope,
      proportion=1.0 if abs(slope) <= 0.05 else 0.7,
    )
    for slope in TERRAIN_SLOPES
  }
  return TerrainGeneratorCfg(
    curriculum=True,
    size=(40.0, 40.0),
    num_rows=1,
    sub_terrains=sub_terrains,
    color_scheme="none",
    add_lights=True,
  )
