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

from .source_training import make_sole_bounded_forward_cfg, make_smooth_forward_cfg

REVISION = "goose_flat_native_foot_course_v1"
RIGID_POSTURE_REVISION = "goose_rigid_native_posture_course_v1"
RIGID_MOTION_REVISION = "goose_rigid_native_motion_course_v1"
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
    patches = (contract["rigid_native"]["foot_sensor_sites"] if "rigid_native" in contract
        else contract["contact_mapping"]["ground_contact_quadrature"])
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
            center = (np.asarray(patch["center_body_m"], dtype=float)
                if "center_body_m" in patch
                else np.asarray(patch["bottom_corners_body_m"], dtype=float).mean(axis=0))
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
    contract = json.loads(contract_path.read_text())
    rigid = "rigid_native" in contract
    base = make_smooth_forward_cfg if rigid else make_sole_bounded_forward_cfg
    cfg = add_foot_sensors(base(
        model_path, contract_path, **kwargs), contract_path)
    if rigid:
        from mjlab.managers.termination_manager import TerminationTermCfg
        cfg.terminations["rigid_contact_limit"] = TerminationTermCfg(
            func=rigid_contact_limit_exceeded, params={"max_depth_m": .005})
    cfg.rewards["feet_air_time"] = RewardTermCfg(func=rewards.feet_air_time,
        weight=1., params={"sensor_name": CONTACT_SENSOR, "threshold_min": .05,
            "threshold_max": .5, "command_name": "velocity", "command_threshold": .01})
    cfg.rewards["feet_clearance"] = RewardTermCfg(func=rewards.feet_clearance,
        weight=-2., params={"target_height": .02, "height_sensor_name": HEIGHT_SENSOR,
            "command_name": "velocity", "command_threshold": .01,
            "asset_cfg": SceneEntityCfg("robot", site_names=SITE_NAMES, preserve_order=True)})
    return cfg


def rigid_collision_depth(env):
    """Current rigid depth without changing the live derived-data snapshot.

    Terminations run before upstream refreshes derived data for observations.
    Running FK on live data here would mix current poses with previous-step
    velocities in rewards. Use separate FK-only data; no extra integration,
    contact solve, or change to the real world's allocation occurs.
    """
    import torch
    import warp as wp
    import mujoco_warp as mjw

    if not hasattr(env, "_goose_rigid_support"):
        from .upstream_discrete_mjlab import CANDIDATE as DISCRETE_CANDIDATE
        contract = json.loads(env.cfg.actions["goose"].contract_path.read_text())
        if contract["candidate"] == DISCRETE_CANDIDATE:
            from .native_geometry315 import collision_geom_vertices
        else:
            from .native_geometry import collision_geom_vertices
        model = env.sim.mj_model
        env._goose_rigid_support = [(g, torch.as_tensor(
            collision_geom_vertices(model, g),
            dtype=env.sim.data.geom_xpos.dtype, device=env.device))
            for g in range(model.ngeom) if int(model.geom_type[g]) in
                (int(mujoco.mjtGeom.mjGEOM_MESH), int(mujoco.mjtGeom.mjGEOM_BOX))]
        if len(env._goose_rigid_support) != 11 or env.contact_adapter is not None:
            raise ValueError("Rigid course requires11 original convex leaves,native contact")
    with wp.ScopedDevice(env.sim.wp_device):
        if not hasattr(env, "_goose_rigid_depth_data"):
            env._goose_rigid_depth_data = mjw.make_data(
                env.sim.mj_model, nworld=env.num_envs, nconmax=0,
                nccdmax=0, njmax=0, njmax_nnz=0)
        data = env._goose_rigid_depth_data
        wp.copy(data.qpos, env.sim.wp_data.qpos)
        if env.sim.mj_model.nmocap:
            wp.copy(data.mocap_pos, env.sim.wp_data.mocap_pos)
            wp.copy(data.mocap_quat, env.sim.wp_data.mocap_quat)
        mjw.kinematics(env.sim.wp_model, data)
    positions = wp.to_torch(data.geom_xpos)
    matrices = wp.to_torch(data.geom_xmat)
    depths = []
    for geom, vertices in env._goose_rigid_support:
        rotation = matrices[:, geom].reshape(env.num_envs, 3, 3)
        z = vertices@rotation[:, 2, :].T+positions[:, geom, 2]
        depths.append(-z.min(dim=0).values)
    return torch.stack(depths).max(dim=0).values.clamp_min(0)


def rigid_contact_limit_exceeded(env, max_depth_m=.005):
    return rigid_collision_depth(env) > max_depth_m


def make_rigid_posture_course_cfg(model_path: Path, contract_path: Path, **kwargs):
    """Bounded follow-up to the recorded short-episode rigid course failure.

    Use upstream posture and termination rewards, leaving the plant, action
    limits, Actor, command sampling and failure thresholds unchanged. This is
    a separate task, not a silent revision of existing training receipts.
    """
    from mjlab.envs.mdp import is_terminated
    from .artifacts import JOINT_ORDER

    if "rigid_native" not in json.loads(contract_path.read_text()):
        raise ValueError("Rigid posture course requires computer-only rigid feet")
    cfg = make_foot_course_cfg(model_path, contract_path, **kwargs)
    cfg.rewards["pose"] = RewardTermCfg(func=rewards.variable_posture,
        weight=1., params={"asset_cfg": SceneEntityCfg("robot",
            joint_names=JOINT_ORDER, preserve_order=True), "command_name": "velocity",
            "std_standing": {".*": .15}, "std_walking": {".*": .35},
            "std_running": {".*": .35}, "walking_threshold": .01,
            "running_threshold": 1.5})
    # RewardManager multiplies by .02s: each failed episode receives -4,
    # whereas a regular timeout remains eligible for value bootstrapping.
    cfg.rewards["failed_episode"] = RewardTermCfg(func=is_terminated, weight=-200.)
    return cfg


def make_rigid_motion_course_cfg(model_path: Path, contract_path: Path, **kwargs):
    """Learn through finite contact transients; keep depth as an acceptance metric.

    Recorded legal-action cases exceed 5mm yet remain upright and settle with
    zero commands. End learning episodes on actual falls or timeout, rather
    than equating that transient with falling. Independent depth acceptance
    remains 5mm; this task does not confer physics or movement qualification.
    """
    from mjlab.managers.metrics_manager import MetricsTermCfg

    cfg = make_rigid_posture_course_cfg(model_path, contract_path, **kwargs)
    del cfg.terminations["rigid_contact_limit"]
    cfg.metrics["rigid_contact_depth_m"] = MetricsTermCfg(func=rigid_collision_depth)
    return cfg
