"""MuJoCo entity configuration for the rickshaw model."""

from pathlib import Path

import mujoco
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

RICKSHAW_XML = Path(__file__).parent / "xmls" / "rickshaw.xml"
RICKSHAW_ASSET_DIR = RICKSHAW_XML.parent / "assets"


def get_spec() -> mujoco.MjSpec:
  """Load the package-local MJCF asset."""
  xml = RICKSHAW_XML.read_text(encoding="utf-8")
  assets = {
    f"assets/{path.name}": path.read_bytes()
    for path in RICKSHAW_ASSET_DIR.glob("*.stl")
  }
  return mujoco.MjSpec.from_string(xml, assets=assets)


# The URDF wheel axes are body-Y axes.  The asset has no driven joints; the
# policy applies external forces at the two hitch sites.
RICKSHAW_ARTICULATION = EntityArticulationInfoCfg()

# The pitch is the static geometry solution that keeps the wheel bottoms at
# z=0 while placing both hitch sites at z=0.75 m.
RICKSHAW_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.014208598811700812),
  rot=(0.9880885597862666, 0.0, -0.15388631524440816, 0.0),
  lin_vel=(0.0, 0.0, 0.0),
  ang_vel=(0.0, 0.0, 0.0),
  joint_pos={".*": 0.0},
  joint_vel={".*": 0.0},
)


def get_rickshaw_cfg() -> EntityCfg:
  """Return a fresh entity config for each environment instance."""
  return EntityCfg(
    init_state=RICKSHAW_INIT_STATE,
    spec_fn=get_spec,
    articulation=RICKSHAW_ARTICULATION,
  )
