"""Per-environment inclined terrain configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

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
class InclinedPlanesCfg(SubTerrainCfg):
  """Sequence of finite planes with fixed x-axis inclinations."""

  angles: tuple[float, ...]
  thickness: float = 0.2
  _index: int = field(init=False, default=0)

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    del difficulty, rng
    angle = self.angles[self._index]
    self._index += 1
    body = spec.body("terrain")
    half_thickness = self.thickness / 2.0
    geom = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(self.size[0] / 2.0, self.size[1] / 2.0, half_thickness),
      pos=(self.size[0] / 2.0, self.size[1] / 2.0, -half_thickness * math.cos(angle)),
      quat=(math.cos(-angle / 2.0), 0.0, math.sin(-angle / 2.0), 0.0),
      rgba=(0.35, 0.38, 0.42, 1.0),
    )
    return TerrainOutput(
      origin=np.array([self.size[0] / 2.0, self.size[1] / 2.0, 0.0]),
      geometries=[TerrainGeometry(geom=geom)],
    )


def make_sloped_terrain_cfg() -> TerrainGeneratorCfg:
  """Create one fixed 100 m by 100 m patch for each slope."""
  return TerrainGeneratorCfg(
    size=(100.0, 100.0),
    num_rows=1,
    num_cols=len(TERRAIN_SLOPES),
    sub_terrains={"slopes": InclinedPlanesCfg(angles=TERRAIN_SLOPES)},
    color_scheme="none",
    add_lights=True,
  )
