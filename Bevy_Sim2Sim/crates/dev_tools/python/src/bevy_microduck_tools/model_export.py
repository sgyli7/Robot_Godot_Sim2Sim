"""Export actual BAM-edited MuJoCo compilation, not raw XML PD guesses."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

from .serialization import identity, sha256_file, snapshot, write_json
from .source_adapter import JOINT_ORDER

FIELDS = (
    "body_parentid", "body_pos", "body_quat", "body_mass", "body_inertia", "body_ipos", "body_iquat",
    "jnt_type", "jnt_bodyid", "jnt_qposadr", "jnt_dofadr", "jnt_pos", "jnt_axis", "jnt_range", "jnt_limited",
    "jnt_solref", "jnt_solimp", "jnt_margin", "jnt_stiffness",
    "dof_bodyid", "dof_jntid", "dof_armature", "dof_damping", "dof_frictionloss", "dof_invweight0", "dof_solref", "dof_solimp",
    "geom_type", "geom_bodyid", "geom_pos", "geom_quat", "geom_size", "geom_dataid",
    "geom_contype", "geom_conaffinity", "geom_condim", "geom_friction", "geom_solref", "geom_solimp", "geom_priority",
    "mesh_vertadr", "mesh_vertnum", "mesh_vert", "mesh_faceadr", "mesh_facenum", "mesh_face",
    "mesh_pos", "mesh_quat", "mesh_scale", "mesh_graphadr", "mesh_graph",
    "site_bodyid", "site_pos", "site_quat", "sensor_type", "sensor_objtype", "sensor_objid", "sensor_adr", "sensor_dim",
    "actuator_trntype", "actuator_trnid", "actuator_dyntype", "actuator_gaintype", "actuator_biastype",
    "actuator_gear", "actuator_gainprm", "actuator_biasprm", "actuator_ctrlrange", "actuator_forcerange",
    "actuator_ctrllimited", "actuator_forcelimited", "key_time", "key_qpos", "key_qvel", "key_ctrl", "qpos0", "qpos_spring", "exclude_signature",
)


def export_compiled(cfg, output: Path, *, family: str, timing_plan: dict) -> dict:
    import mujoco
    entity = cfg.scene.entities["robot"].build()
    entity.spec.option.timestep = cfg.sim.mujoco.timestep
    model = entity.spec.compile()
    object_kinds = {"jnt": "JOINT"}
    count_names = {"actuator": "nu"}
    names = {kind: [mujoco.mj_id2name(model, getattr(mujoco.mjtObj, f"mjOBJ_{object_kinds.get(kind, kind.upper())}"), i)
                    for i in range(getattr(model, count_names.get(kind, f"n{kind}")))]
             for kind in ("body", "jnt", "geom", "site", "sensor", "actuator", "mesh", "key")}
    # Enum uses JOINT whereas size attribute is njnt.
    fields = {name: getattr(model, name).tolist() for name in FIELDS if hasattr(model, name)}
    actuator_names = names["actuator"]
    if tuple(actuator_names) != JOINT_ORDER or model.nu != 14:
        raise ValueError(f"Actuator mapping mismatch: {actuator_names}")
    servo_joint_ids = [names["jnt"].index(name) for name in JOINT_ORDER]
    home = [float(model.key_qpos[-1, model.jnt_qposadr[joint_id]]) for joint_id in servo_joint_ids]
    bam = []
    for instance in entity.actuators:
        parameter_path = Path(instance.cfg._resolved_json_path)
        actual = instance._bam_model.actuator
        bam.append({"parameters": json.loads(parameter_path.read_text()), "parameter_path": str(parameter_path),
                    "parameter_sha256": sha256_file(parameter_path), "cfg": snapshot(instance.cfg),
                    "firmware": {name: getattr(actual, name) for name in
                                 ("kp", "vin", "error_gain", "max_pwm", "max_current")},
                    "external_load": "-qfrc_bias + qfrc_constraint - qfrc_friction",
                    "friction_motor_history": "previous qfrc_actuator applied by solve",
                    "voltage_sag_history": "previous computed motor_torque before solve/clipping"})
    output.mkdir(parents=True, exist_ok=True)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.nkey - 1)
    mujoco.mj_forward(model, data)
    write_json(output / "home_pose_oracle.json", {"schema": "mujoco_home_pose_oracle_v1",
               "method": "actual compiled key_qpos -> mj_resetDataKeyframe -> mj_forward; no integration",
               "names": names, "qpos": data.qpos.tolist(), "qvel": data.qvel.tolist(),
               "body_positions": data.xpos.tolist(), "body_quaternions": data.xquat.tolist(),
               "inertial_positions": data.xipos.tolist(), "inertial_matrices": data.ximat.tolist(),
               "geom_positions": data.geom_xpos.tolist(), "geom_matrices": data.geom_xmat.tolist()})
    assets = {}
    xml = entity.spec.to_xml()
    vfs = {name: bytes(content) for name, content in entity.spec.assets.items()}
    from mjlab_microduck.robot.microduck_constants import MICRODUCK_ALLCOLLISIONS_XML, MICRODUCK_ALLCOLLISIONS_ROLLERS_XML
    source_xml = MICRODUCK_ALLCOLLISIONS_ROLLERS_XML if family == "roller_allcollisions" else MICRODUCK_ALLCOLLISIONS_XML
    xml_tree = ET.fromstring(xml)
    compiler = xml_tree.find("compiler")
    for kind in ("mesh", "texture"):
        directory = compiler.get(f"{kind}dir", "") if compiler is not None else ""
        for element in xml_tree.findall(f"asset/{kind}"):
            filename = element.get("file")
            if filename:
                key = str(Path(directory) / filename)
                if key not in vfs:
                    vfs[key] = (source_xml.parent / key).read_bytes()
    for index, (name, content) in enumerate(vfs.items()):
        path = output / "effective_assets" / f"asset_{index}.bin"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(bytes(content))
        assets[name] = {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": len(content)}
    # Prove VFS reconstruction, rather than assuming the XML alone is sufficient.
    reconstructed = mujoco.MjModel.from_xml_string(xml, vfs)
    if reconstructed.nq != model.nq or reconstructed.ngeom != model.ngeom:
        raise ValueError("XML plus captured VFS did not reproduce robot structure")
    write_json(output / "effective_assets.json", {"schema": "mujoco_vfs_manifest_v1", "assets": assets,
               "xml_independently_loadable": not bool(assets), "recompile_verified": True,
               "recompile_method": "MjModel.from_xml_string(effective_robot.xml, {original_vfs_name: recorded_bytes})"})
    report = {"schema": "microduck_compiled_v1", "family": family, "names": names, "fields": fields,
              "counts": {name: int(getattr(model, name)) for name in ("nq", "nv", "nu", "nbody", "njnt", "ngeom", "nmesh", "nsensor", "nkey", "neq")},
              "missing_optional_fields": [name for name in FIELDS if not hasattr(model, name)],
              "joint_order": list(JOINT_ORDER), "home": home, "bam": bam, "timing_plan": timing_plan,
              "units": "SI; MuJoCo Z-up; quaternions wxyz; principal inertia in inertial frame",
              "per_step_mutable_fields": ["dof_frictionloss", "dof_damping"],
              "note": f"BAM writes per-step friction after initial compilation. XML requires effective_assets.json VFS; recompile verification checks nq/ngeom structure only, not bit-exact numeric reconstruction. Home pose oracle is actual mj_forward, not integration. Actual nexclude={model.nexclude}; exclude_signature exports every explicit source pair exclusion. Actual disableflags={int(model.opt.disableflags)}; parent-filter/weld collision semantics remain MuJoCo source defaults, separate from explicit excludes."}
    report["sha256"] = identity(report)
    write_json(output / "compiled_robot.json", report)
    write_json(output / "bam_params.json", bam)
    (output / "effective_robot.xml").write_text(xml)
    return report


def export_bam_reference(cfg, output: Path, *, cases: int = 512, seed: int = 20260928) -> dict:
    """Evaluate authoritative Torch BAM on reproducible stress/DR inputs."""
    import inspect
    import torch
    from bam.actuator import TorchBackend
    from bam.mjlab import BamActuator
    if cases < 16:
        raise ValueError("At least sixteen electrical/friction boundary cases are required")
    entity = cfg.scene.entities["robot"].build()
    instance = entity.actuators[0]
    act = instance._bam_model.actuator
    act.backend = TorchBackend()
    params = json.loads(Path(instance.cfg._resolved_json_path).read_text())
    generator = torch.Generator().manual_seed(seed)
    def random_tensor(low, high, width=14):
        return low + torch.rand((cases, width), generator=generator) * (high - low)
    inputs = {"target": random_tensor(-2.5, 2.5), "position": random_tensor(-.5, .5),
              "velocity": random_tensor(-30, 30), "previous_motor_computed": random_tensor(-1, 1),
              "previous_actuator_applied": random_tensor(-1, 1)}
    raw_external = {"qfrc_bias": random_tensor(-.5, .5), "qfrc_constraint": random_tensor(-1.5, 1.5),
                    "qfrc_friction": random_tensor(-.1, .1)}
    for tensor in list(inputs.values()) + list(raw_external.values()):
        tensor[0].zero_()
    # Explicit current/PWM saturation, high-speed back-EMF and strict directional ties.
    for index in range(1, 9):
        for tensor in list(inputs.values()) + list(raw_external.values()):
            tensor[index].zero_()
    inputs["target"][1].fill_(20.0)
    inputs["target"][2].fill_(-20.0)
    inputs["velocity"][3].fill_(120.0)
    inputs["velocity"][4].fill_(-120.0)
    for index, value in ((5, .1), (6, -.1)):
        inputs["previous_actuator_applied"][index].fill_(value)
        raw_external["qfrc_constraint"][index].fill_(value)
    inputs["target"][7].fill_(.02)
    inputs["target"][8].fill_(-.02)
    inputs["velocity"][7].fill_(10.0)
    inputs["velocity"][8].fill_(-10.0)
    inputs["external_load"] = -raw_external["qfrc_bias"] + raw_external["qfrc_constraint"] - raw_external["qfrc_friction"]
    environments = {"supply_voltage": random_tensor(6.5, 8.2, 1), "voltage_drop_gain": random_tensor(0, .2, 1),
                    "minimum_voltage": torch.full((cases, 1), 6.0), "kp_scale": random_tensor(.9, 1.1, 1),
                    "kd_scale": random_tensor(.85, 1.15, 1), "friction_scale": random_tensor(.5, 1.5, 1)}
    vin = (environments["supply_voltage"] - environments["voltage_drop_gain"] *
           inputs["previous_motor_computed"].abs().sum(dim=-1, keepdim=True)).clamp_min(6.0)
    act.vin = vin
    act.kp = 200.0 * environments["kp_scale"]
    velocity = inputs["velocity"] * environments["kd_scale"]
    control = act.compute_control(inputs["target"], inputs["position"], velocity, cfg.sim.mujoco.timestep)
    torque = act.compute_torque(control, True, inputs["position"], velocity)
    stribeck = torch.exp(-torch.pow(inputs["velocity"].abs() / params["dtheta_stribeck"], params["alpha"]))
    budget = instance._compute_friction_budget(inputs["previous_actuator_applied"], inputs["external_load"], stribeck)
    budget *= environments["friction_scale"]
    rows = [{"input": {key: tensor[index].tolist() for key, tensor in inputs.items()},
             "environment": {key: float(tensor[index, 0]) for key, tensor in environments.items()},
             "external_load_decomposition": {key: tensor[index].tolist() for key, tensor in raw_external.items()},
             "expected": {"motor_torque": torque[index].tolist(), "friction_budget": budget[index].tolist(),
                          "effective_voltage": float(vin[index, 0]), "viscous_damping": float(torch.tensor(params["friction_viscous"]))}}
            for index in range(cases)]
    parameters = {key: value for key, value in params.items() if key not in {"R", "model", "actuator", "q_offset"}}
    parameters.update(resistance=params["R"], error_gain=act.error_gain, firmware_kp=200.0,
                      max_pwm=act.max_pwm, max_current=act.max_current)
    result = {"schema": "bam_torch_reference_v1", "dtype": "float32", "cases": rows, "seed": seed,
              "parameters": parameters, "kt_squared_f64": params["kt"] ** 2,
              "bam_source_sha256": sha256_file(Path(inspect.getfile(BamActuator))),
              "parameter_sha256": sha256_file(Path(instance.cfg._resolved_json_path)),
              "limits": "torque/friction budget oracle; static friction clipping still belongs to physics solver"}
    result["sha256"] = identity(result)
    write_json(output, result)
    return result


def export_observation_action_reference(env, output: Path) -> dict:
    """Authoritative upstream boundary with deterministic sensor DR disabled locally.

    This is a mathematical boundary oracle, not a rollout or trained-policy score.
    Native states retain the actual reset body pose. Delay/noise/normalization are
    distinct stages; the policy normalizer remains embedded in the exported model.
    """
    import torch
    robot = env.scene["robot"].data
    action = env.action_manager.get_term("joint_pos")
    ids = action.target_ids
    original_bias = robot.encoder_bias.clone()
    original_imu = getattr(env, "_imu_misalign_quat", None)
    previous_raw = env.action_manager.action.clone()
    previous_targets = robot.joint_pos_target.clone()
    actual_scale = torch.broadcast_to(torch.as_tensor(action.scale, device=env.device), (env.num_envs, 14)).detach().cpu().tolist()[0]
    actual_home = robot.default_joint_pos[:, ids].detach().cpu().tolist()[0]
    actual_offset = torch.broadcast_to(torch.as_tensor(action.offset, device=env.device), (env.num_envs, 14)).detach().cpu().tolist()[0]
    robot.encoder_bias.zero_()
    env._imu_misalign_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device)
    rows = []
    terms = env.cfg.observations["actor"].terms
    def vector(value):
        return value.detach().cpu().tolist()[0]
    try:
        for case in range(16):
            raw = torch.linspace(-.15, .15, 14, device=env.device).unsqueeze(0) * (case / 15)
            if case % 2:
                raw = -raw
            env.action_manager.process_action(raw)
            values = []
            term_lengths = {}
            for name, term in terms.items():
                value = term.func(env, **term.params)
                if term.scale is not None:
                    value = value * torch.as_tensor(term.scale, device=env.device)
                values.append(value)
                term_lengths[name] = value.shape[-1]
            expected_observation = torch.cat(values, dim=-1)
            action.apply_actions()
            rows.append({"state": {"gyro": vector(robot.root_link_ang_vel_b),
                          "gravity": vector(robot.projected_gravity_b),
                          "q": vector(robot.joint_pos[:, ids]), "qd": vector(robot.joint_vel[:, ids])},
                         "last_raw_action": vector(raw),
                         "commands": {"locomotion": vector(env.command_manager.get_command("twist")),
                                      "head": vector(values[-2]), "body": vector(values[-1])},
                         "home": vector(robot.default_joint_pos[:, ids]), "raw_action": vector(raw),
                         "expected_observation": vector(expected_observation),
                         "expected_position_target": vector(robot.joint_pos_target[:, ids])})
    finally:
        robot.encoder_bias.copy_(original_bias)
        if original_imu is None:
            del env._imu_misalign_quat
        else:
            env._imu_misalign_quat = original_imu
        env.action_manager.process_action(previous_raw)
        robot.joint_pos_target.copy_(previous_targets)
    report = {"schema": "source_observation_action_reference_v1", "dtype": "float32",
              "term_order": list(terms), "term_lengths": term_lengths, "actuator_order": list(JOINT_ORDER),
              "action_scale": actual_scale, "home": actual_home,
              "action_offset": actual_offset, "observation_reference": actual_home,
              "pre_oracle_reset_joint_positions": robot.joint_pos[:, ids].detach().cpu().tolist()[0],
              "pre_oracle_reset_joint_targets": previous_targets[:, ids].detach().cpu().tolist()[0],
              "pre_oracle_encoder_bias": original_bias[:, ids].detach().cpu().tolist()[0],
              "profile": "native boundary; noise/encoder bias/mounting misalignment disabled for oracle only; no delays",
              "normalization": "inside ONNX actor, not this 61D boundary", "cases": rows}
    report["sha256"] = identity(report)
    write_json(output, report)
    return report
