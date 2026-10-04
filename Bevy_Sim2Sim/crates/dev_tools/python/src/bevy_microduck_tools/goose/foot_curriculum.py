"""Native mjlab foot sensing and the first Goose swing curriculum.

Sites are massless sensing frames at the frozen sole patch centers. They add
no collision shapes or actuators. The 65-value Actor and original drive stay
unchanged; upstream sensors and rewards own the contact and swing histories.
This curriculum covers the admitted flat-floor development domain only.
"""
import json
from pathlib import Path

import mujoco
import numpy as np
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef
from mjlab.sensor import RingPatternCfg, TerrainHeightSensorCfg
from mjlab.tasks.velocity.mdp import rewards

from .source_training import make_sole_bounded_forward_cfg

REVISION = "goose_flat_native_foot_course_v1"
CONTACT_SENSOR = "goose_feet_ground"
HEIGHT_SENSOR = "goose_feet_height"
SITE_NAMES = ("right_sole_center", "left_sole_center")


def add_foot_sensors(cfg, contract_path: Path):
    """Attach public native sensors without changing the received physics.

    ``found`` counts native constraint contacts, including predictive contacts;
    it is deliberately conservative air time, not a load-bearing test. Actual
    force and whole-foot clearance remain separate evaluation measurements.
    Native step sensors/rewards retain upstream's one-step derived-data lag.
    """
    contract = json.loads(contract_path.read_text())
    patches = contract["contact_mapping"]["ground_contact_quadrature"]
    if [p["geom"] for p in patches] != [
            "right_flexible_sole", "left_flexible_sole"]:
        raise ValueError("Native foot course requires the frozen right/left sole order")
    if any(s.name in (CONTACT_SENSOR, HEIGHT_SENSOR) for s in cfg.scene.sensors):
        raise ValueError("Goose foot sensing is already attached")
    entity = cfg.scene.entities["robot"]
    original_spec = entity.spec_fn

    def sensed_spec():
        spec = original_spec()
        for name, patch in zip(SITE_NAMES, patches, strict=True):
            center = np.asarray(patch["bottom_corners_body_m"], dtype=float).mean(axis=0)
            spec.body(patch["body"]).add_site(name=name, pos=center,
                type=mujoco.mjtGeom.mjGEOM_SPHERE, size=(.001, .001, .001),
                rgba=(0., 1., 0., 0.), group=5)
        return spec

    entity.spec_fn = sensed_spec
    cfg.scene.sensors = (*cfg.scene.sensors,
        ContactSensorCfg(name=CONTACT_SENSOR,
            primary=ContactMatch(mode="geom", entity="robot",
                pattern=tuple(p["geom"] for p in patches)),
            secondary=ContactMatch(mode="geom", pattern="ground"),
            fields=("found", "force"), reduce="netforce", num_slots=1,
            track_air_time=True, global_frame=True),
        TerrainHeightSensorCfg(name=HEIGHT_SENSOR,
            frame=tuple(ObjRef(type="site", name=name, entity="robot")
                        for name in SITE_NAMES),
            pattern=RingPatternCfg.single_ring(radius=.01, num_samples=6),
            ray_alignment="world", max_distance=1., exclude_parent_body=True,
            include_geom_groups=(0,), debug_vis=False))
    return cfg


def make_foot_course_cfg(model_path: Path, contract_path: Path, **kwargs):
    """Add mature velocity-task swing terms to the sole-bounded forward task.

    The original small command range needs a .01m/s activation threshold.
    Zero commands receive no swing reward. The bounded native air-time term
    stops rewarding a permanently raised foot. Qualification independently
    requires actual alternating steps and useful travel, beyond these rewards.
    """
    cfg = add_foot_sensors(make_sole_bounded_forward_cfg(
        model_path, contract_path, **kwargs), contract_path)
    cfg.rewards["feet_air_time"] = RewardTermCfg(func=rewards.feet_air_time,
        weight=1., params={"sensor_name": CONTACT_SENSOR, "threshold_min": .05,
            "threshold_max": .5, "command_name": "velocity", "command_threshold": .01})
    cfg.rewards["feet_clearance"] = RewardTermCfg(func=rewards.feet_clearance,
        weight=-2., params={"target_height": .02, "height_sensor_name": HEIGHT_SENSOR,
            "command_name": "velocity", "command_threshold": .01,
            "asset_cfg": SceneEntityCfg("robot", site_names=SITE_NAMES, preserve_order=True)})
    return cfg
