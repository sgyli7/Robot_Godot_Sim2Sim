"""Opt-in privileged terrain observations from the pinned mjlab workflow.

The public Actor, physical model, action units, rewards and existing flat
profiles stay unchanged. A new Critic requires a fresh native optimizer;
callers must declare an Actor-only transfer rather than a full PPO resume.
"""
from __future__ import annotations

import copy

from mjlab.sensor import RayCastSensorCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


REVISION = "goose_move_mature_terrain_critic_v1"


def with_mature_move_terrain_critic(cfg):
    """Reuse upstream terrain scan and foot truth for the Critic only.

    Terrain rays retain the upstream yaw frame, grid, range, geometry group
    and scaling. Existing Goose ground/height sensors supply native foot
    observations. No terrain truth is added to the 65-value Actor.
    """
    if cfg.scene.terrain is None or "critic" not in cfg.observations:
        raise ValueError("Terrain Critic requires an existing terrain profile")
    names = {sensor.name for sensor in cfg.scene.sensors}
    required = {"goose_feet_ground", "goose_feet_height"}
    if not required.issubset(names) or "terrain_scan" in names:
        raise ValueError("Terrain Critic requires declared Goose foot sensors")
    stock = make_velocity_env_cfg()
    scan = copy.deepcopy(next(sensor for sensor in stock.scene.sensors
        if sensor.name == "terrain_scan"))
    if not isinstance(scan, RayCastSensorCfg) or scan.include_geom_groups != (0,):
        raise ValueError("Pinned terrain scan must read terrain geometry only")
    scan.frame.name = "torso"
    scan.debug_vis = False
    result = copy.deepcopy(cfg)
    result.scene.sensors = (*result.scene.sensors, scan)
    for name in ("height_scan", "foot_height", "foot_air_time",
                 "foot_contact", "foot_contact_forces"):
        if name in result.observations["critic"].terms:
            raise ValueError("Terrain Critic observations must not be duplicated")
        term = copy.deepcopy(stock.observations["critic"].terms[name])
        if term.params.get("sensor_name") == "foot_height_scan":
            term.params["sensor_name"] = "goose_feet_height"
        elif term.params.get("sensor_name") == "feet_ground_contact":
            term.params["sensor_name"] = "goose_feet_ground"
        result.observations["critic"].terms[name] = term
    if result.observations["actor"] != cfg.observations["actor"]:
        raise ValueError("Terrain Critic must preserve the public Actor profile")
    return result
