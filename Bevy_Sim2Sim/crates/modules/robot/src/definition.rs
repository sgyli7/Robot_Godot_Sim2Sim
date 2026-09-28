//! Immutable, hash-bound compilation of the actual BAM-edited MuJoCo model.
//!
//! JSON is a development interchange format. Runtime configuration loads actual
//! RON bytes through the same model validation; neither replaces compiling MJCF.

use crate::{ACTION_DIMENSION, RobotError, contract::ACTUATOR_ORDER};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashSet},
    fs,
    path::Path,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelCounts {
    pub nq: usize,
    pub nv: usize,
    pub nu: usize,
    pub nbody: usize,
    pub njnt: usize,
    pub ngeom: usize,
    pub nmesh: usize,
    pub nsensor: usize,
    pub nkey: usize,
    pub neq: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelNames {
    pub body: Vec<Option<String>>,
    pub jnt: Vec<Option<String>>,
    pub geom: Vec<Option<String>>,
    pub site: Vec<Option<String>>,
    pub sensor: Vec<Option<String>>,
    pub actuator: Vec<Option<String>>,
    pub mesh: Vec<Option<String>>,
    pub key: Vec<Option<String>>,
}

/// MuJoCo `mjNREF`: solver reference width of `jnt_solref`.
const SOURCE_JNT_SOLREF_WIDTH: usize = 2;
/// MuJoCo `mjNIMP`: solver impedance width of `jnt_solimp`.
const SOURCE_JNT_SOLIMP_WIDTH: usize = 5;

/// Preserve source doubles until the explicit backend boundary. Principal
/// inertia axes are body_iquat, never the body frame's XYZ diagonal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelFields {
    pub body_parentid: Vec<usize>,
    pub body_pos: Vec<[f64; 3]>,
    pub body_quat: Vec<[f64; 4]>,
    pub body_mass: Vec<f64>,
    pub body_inertia: Vec<[f64; 3]>,
    pub body_ipos: Vec<[f64; 3]>,
    pub body_iquat: Vec<[f64; 4]>,
    pub jnt_type: Vec<u32>,
    pub jnt_bodyid: Vec<usize>,
    pub jnt_qposadr: Vec<usize>,
    pub jnt_dofadr: Vec<usize>,
    pub jnt_pos: Vec<[f64; 3]>,
    pub jnt_axis: Vec<[f64; 3]>,
    pub jnt_range: Vec<[f64; 2]>,
    pub jnt_limited: Vec<bool>,
    /// Source joint-limit solver reference, one `mjNREF` row per joint.
    /// Legacy frozen exports omit it. Stored values are not a qualified limit law.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub jnt_solref: Option<Vec<[f64; SOURCE_JNT_SOLREF_WIDTH]>>,
    /// Source joint-limit solver impedance, one `mjNIMP` row per joint.
    /// Legacy frozen exports omit it. Stored values are not a qualified limit law.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub jnt_solimp: Option<Vec<[f64; SOURCE_JNT_SOLIMP_WIDTH]>>,
    /// Source joint-limit margin, one value per joint.
    /// Legacy frozen exports omit it. Stored values are not a qualified limit law.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub jnt_margin: Option<Vec<f64>>,
    pub dof_bodyid: Vec<usize>,
    pub dof_jntid: Vec<usize>,
    pub dof_armature: Vec<f64>,
    pub dof_damping: Vec<f64>,
    pub dof_frictionloss: Vec<f64>,
    /// Source compilation-time diagonal inverse weight used by joint-limit
    /// regularization. It is not the pose-dependent effective inverse mass.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dof_invweight0: Option<Vec<f64>>,
    pub geom_type: Vec<u32>,
    pub geom_bodyid: Vec<usize>,
    pub geom_pos: Vec<[f64; 3]>,
    pub geom_quat: Vec<[f64; 4]>,
    pub geom_size: Vec<[f64; 3]>,
    pub geom_dataid: Vec<i32>,
    pub geom_contype: Vec<u32>,
    pub geom_conaffinity: Vec<u32>,
    pub geom_condim: Vec<u32>,
    pub geom_friction: Vec<[f64; 3]>,
    pub mesh_vertadr: Vec<usize>,
    pub mesh_vertnum: Vec<usize>,
    pub mesh_faceadr: Vec<usize>,
    pub mesh_facenum: Vec<usize>,
    pub mesh_vert: Vec<[f64; 3]>,
    pub mesh_face: Vec<[usize; 3]>,
    pub mesh_pos: Vec<[f64; 3]>,
    pub mesh_quat: Vec<[f64; 4]>,
    pub mesh_scale: Vec<[f64; 3]>,
    /// Legacy development exports omit native collision hulls. An importer
    /// must reject absent hulls rather than reconstructing an approximate one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mesh_graphadr: Option<Vec<i32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mesh_graph: Option<Vec<i32>>,
    pub site_bodyid: Vec<usize>,
    pub site_pos: Vec<[f64; 3]>,
    pub site_quat: Vec<[f64; 4]>,
    pub actuator_trntype: Vec<u32>,
    pub actuator_trnid: Vec<[i32; 2]>,
    pub actuator_gear: Vec<[f64; 6]>,
    pub actuator_forcerange: Vec<[f64; 2]>,
    pub actuator_forcelimited: Vec<bool>,
    pub key_qpos: Vec<Vec<f64>>,
    pub qpos0: Vec<f64>,
    /// Preserve all remaining solver, sensor and actuator parameters verbatim.
    #[serde(flatten)]
    pub other: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompiledModel {
    pub schema: String,
    pub family: String,
    pub counts: ModelCounts,
    pub names: ModelNames,
    pub fields: ModelFields,
    pub home: [f64; ACTION_DIMENSION],
    pub joint_order: [String; ACTION_DIMENSION],
    pub bam: Vec<serde_json::Value>,
    pub timing_plan: serde_json::Value,
    pub missing_optional_fields: Vec<String>,
    pub per_step_mutable_fields: Vec<String>,
    pub note: String,
    pub units: String,
    pub sha256: String,
}

/// Validated data remains immutable for the episode's lifetime.
pub struct RobotDefinition {
    model: CompiledModel,
    file_sha256: String,
    actuator_joint_ids: [usize; ACTION_DIMENSION],
}

impl RobotDefinition {
    pub fn load_json(path: &Path, expected_file_sha256: &str) -> Result<Self, RobotError> {
        let (bytes, file_sha256) = read_bound_bytes(path, expected_file_sha256)?;
        let model: CompiledModel =
            serde_json::from_slice(&bytes).map_err(|error| invalid(error.to_string()))?;
        Self::from_model(model, file_sha256)
    }

    pub fn load_ron(path: &Path, expected_file_sha256: &str) -> Result<Self, RobotError> {
        let (bytes, file_sha256) = read_bound_bytes(path, expected_file_sha256)?;
        let model: CompiledModel =
            ron::de::from_bytes(&bytes).map_err(|error| invalid(error.to_string()))?;
        Self::from_model(model, file_sha256)
    }

    /// Development exporters may write this string. Loading never mutates or
    /// reloads the validated configuration during an episode.
    pub fn to_ron(&self) -> Result<String, RobotError> {
        ron::ser::to_string_pretty(&self.model, ron::ser::PrettyConfig::default())
            .map_err(|error| invalid(error.to_string()))
    }

    fn from_model(model: CompiledModel, file_sha256: String) -> Result<Self, RobotError> {
        let actuator_joint_ids = validate(&model)?;
        Ok(Self {
            model,
            file_sha256,
            actuator_joint_ids,
        })
    }

    pub fn model(&self) -> &CompiledModel {
        &self.model
    }
    pub fn file_sha256(&self) -> &str {
        &self.file_sha256
    }
    pub fn actuator_joint_ids(&self) -> &[usize; ACTION_DIMENSION] {
        &self.actuator_joint_ids
    }
}

fn read_bound_bytes(
    path: &Path,
    expected_file_sha256: &str,
) -> Result<(Vec<u8>, String), RobotError> {
    let bytes = fs::read(path).map_err(|error| invalid(error.to_string()))?;
    let file_sha256 = format!("{:x}", Sha256::digest(&bytes));
    if file_sha256 != expected_file_sha256 {
        return Err(invalid("compiled model file SHA256 mismatch"));
    }
    Ok((bytes, file_sha256))
}

fn invalid(message: impl Into<String>) -> RobotError {
    RobotError::Contract(message.into())
}

fn validate(model: &CompiledModel) -> Result<[usize; ACTION_DIMENSION], RobotError> {
    let c = &model.counts;
    let f = &model.fields;
    if model.schema != "microduck_compiled_v1"
        || c.nu != ACTION_DIMENSION
        || c.neq != 0
        || !["leg_allcollisions", "roller_allcollisions"].contains(&model.family.as_str())
        || model
            .joint_order
            .iter()
            .map(String::as_str)
            .ne(ACTUATOR_ORDER)
        || model.units != "SI; MuJoCo Z-up; quaternions wxyz; principal inertia in inertial frame"
    {
        return Err(invalid(
            "unsupported compiled model identity, mapping or units",
        ));
    }
    if !model.missing_optional_fields.is_empty() {
        return Err(invalid("compiled model has missing fields"));
    }
    let expected_joints = if model.family == "leg_allcollisions" {
        15
    } else {
        19
    };
    if c.njnt != expected_joints
        || c.nbody != c.njnt + 1
        || c.nv != c.njnt + 5
        || c.nq != c.nv + 1
        || c.nkey != 1
        || model.sha256.len() != 64
    {
        return Err(invalid("unexpected model topology or keyframe count"));
    }
    macro_rules! lengths {
        ($count:expr; $($field:expr),+ $(,)?) => {
            if [$($field.len()),+].iter().any(|length| *length != $count) {
                return Err(invalid("compiled model array length mismatch"));
            }
        };
    }
    lengths!(c.nbody; model.names.body, f.body_parentid, f.body_pos, f.body_quat,
        f.body_mass, f.body_inertia, f.body_ipos, f.body_iquat);
    lengths!(c.njnt; model.names.jnt, f.jnt_type, f.jnt_bodyid, f.jnt_qposadr,
        f.jnt_dofadr, f.jnt_pos, f.jnt_axis, f.jnt_range, f.jnt_limited);
    lengths!(c.nv; f.dof_bodyid, f.dof_jntid, f.dof_armature, f.dof_damping, f.dof_frictionloss);
    lengths!(c.ngeom; model.names.geom, f.geom_type, f.geom_bodyid, f.geom_pos,
        f.geom_quat, f.geom_size, f.geom_dataid, f.geom_contype, f.geom_conaffinity,
        f.geom_condim, f.geom_friction);
    lengths!(c.nmesh; model.names.mesh, f.mesh_vertadr, f.mesh_vertnum,
        f.mesh_faceadr, f.mesh_facenum, f.mesh_pos, f.mesh_quat, f.mesh_scale);
    lengths!(c.nu; model.names.actuator, f.actuator_trntype, f.actuator_trnid,
        f.actuator_gear, f.actuator_forcerange, f.actuator_forcelimited);
    lengths!(c.nkey; model.names.key, f.key_qpos);
    lengths!(c.nsensor; model.names.sensor);
    lengths!(model.names.site.len(); f.site_bodyid, f.site_pos, f.site_quat);
    lengths!(c.nq; f.qpos0, f.key_qpos[0]);
    for names in [
        &model.names.body,
        &model.names.jnt,
        &model.names.actuator,
        &model.names.mesh,
        &model.names.site,
    ] {
        let unique: HashSet<_> = names.iter().filter_map(Option::as_deref).collect();
        if unique.len() != names.len() {
            return Err(invalid("missing or duplicate model name"));
        }
    }
    for body in 0..c.nbody {
        if (body == 0 && f.body_parentid[body] != 0) || (body > 0 && f.body_parentid[body] >= body)
        {
            return Err(invalid("non-topological body parent mapping"));
        }
        vector(&f.body_pos[body])?;
        vector(&f.body_ipos[body])?;
        quaternion(&f.body_quat[body])?;
        quaternion(&f.body_iquat[body])?;
        let mass = f.body_mass[body];
        let inertia = f.body_inertia[body];
        if !mass.is_finite()
            || (body > 0 && mass <= 0.0)
            || mass < 0.0
            || inertia
                .iter()
                .any(|x| !x.is_finite() || (body > 0 && *x <= 0.0) || *x < 0.0)
            || (0..3).any(|i| inertia[i] > inertia[(i + 1) % 3] + inertia[(i + 2) % 3] + 1e-12)
        {
            return Err(invalid("invalid physical mass or principal inertia"));
        }
    }
    let mut body_with_joint = HashSet::new();
    for joint in 0..c.njnt {
        let body = f.jnt_bodyid[joint];
        if body == 0
            || body >= c.nbody
            || !body_with_joint.insert(body)
            || (joint == 0 && (f.jnt_type[joint] != 0 || body != 1))
            || (joint > 0 && f.jnt_type[joint] != 3)
        {
            return Err(invalid(
                "expected one free root and one hinge per remaining body",
            ));
        }
        let width = if joint == 0 { 6 } else { 1 };
        let qwidth = if joint == 0 { 7 } else { 1 };
        if f.jnt_dofadr[joint]
            .checked_add(width)
            .is_none_or(|end| end > c.nv)
            || f.jnt_qposadr[joint]
                .checked_add(qwidth)
                .is_none_or(|end| end > c.nq)
        {
            return Err(invalid("joint address exceeds compiled state"));
        }
        for dof in f.jnt_dofadr[joint]..f.jnt_dofadr[joint] + width {
            if f.dof_jntid[dof] != joint || f.dof_bodyid[dof] != body {
                return Err(invalid("DOF map disagrees with joint map"));
            }
        }
        vector(&f.jnt_pos[joint])?;
        vector(&f.jnt_axis[joint])?;
        if joint > 0 && (f.jnt_axis[joint].iter().map(|x| x * x).sum::<f64>() - 1.0).abs() > 1e-8 {
            return Err(invalid("hinge axis must have unit length"));
        }
        vector(&f.jnt_range[joint])?;
        if f.jnt_limited[joint] && f.jnt_range[joint][0] >= f.jnt_range[joint][1] {
            return Err(invalid("invalid joint limits"));
        }
    }
    validate_source_joint_limit_parameters(f, c.njnt)?;
    for values in [&f.dof_armature, &f.dof_damping, &f.dof_frictionloss] {
        if values.iter().any(|x| !x.is_finite() || *x < 0.0) {
            return Err(invalid("negative or non-finite DOF parameters"));
        }
    }
    let mut actuator_joints = [0; ACTION_DIMENSION];
    for (index, expected_name) in ACTUATOR_ORDER.iter().enumerate() {
        let joint = model
            .names
            .jnt
            .iter()
            .position(|name| name.as_deref() == Some(expected_name))
            .ok_or_else(|| invalid("missing driven joint"))?;
        actuator_joints[index] = joint;
        if model.names.actuator[index].as_deref() != Some(expected_name)
            || f.actuator_trntype[index] != 0
            || f.actuator_trnid[index][0] != joint as i32
            || f.actuator_gear[index] != [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            || !f.actuator_forcelimited[index]
        {
            return Err(invalid(
                "non-unit or misordered joint actuator transmission",
            ));
        }
        vector(&f.actuator_forcerange[index])?;
        if f.actuator_forcerange[index][0] >= 0.0
            || f.actuator_forcerange[index][1] <= 0.0
            || model.home[index] != f.key_qpos[0][f.jnt_qposadr[joint]]
        {
            return Err(invalid(
                "home pose or force range disagrees with compilation",
            ));
        }
    }
    for geom in 0..c.ngeom {
        if f.geom_bodyid[geom] >= c.nbody || f.geom_type[geom] != 7 || f.geom_condim[geom] != 3 {
            return Err(invalid(
                "unsupported geom mapping, shape or contact dimension",
            ));
        }
        vector(&f.geom_pos[geom])?;
        quaternion(&f.geom_quat[geom])?;
        vector(&f.geom_size[geom])?;
        vector(&f.geom_friction[geom])?;
        if f.geom_friction[geom].iter().any(|x| *x < 0.0)
            || f.geom_dataid[geom] < 0
            || f.geom_dataid[geom] as usize >= c.nmesh
        {
            return Err(invalid("invalid mesh geom data or friction"));
        }
    }
    for mesh in 0..c.nmesh {
        let vertices = span(&f.mesh_vert, f.mesh_vertadr[mesh], f.mesh_vertnum[mesh])?;
        let faces = span(&f.mesh_face, f.mesh_faceadr[mesh], f.mesh_facenum[mesh])?;
        if vertices.len() < 4
            || faces.is_empty()
            || faces.iter().flatten().any(|v| *v >= vertices.len())
        {
            return Err(invalid("invalid compiled mesh topology"));
        }
        for vertex in vertices {
            vector(vertex)?;
        }
        vector(&f.mesh_pos[mesh])?;
        quaternion(&f.mesh_quat[mesh])?;
        vector(&f.mesh_scale[mesh])?;
        if f.mesh_scale[mesh].iter().any(|x| *x <= 0.0) {
            return Err(invalid("invalid mesh scale"));
        }
    }
    for site in 0..model.names.site.len() {
        if f.site_bodyid[site] >= c.nbody {
            return Err(invalid("invalid site body"));
        }
        vector(&f.site_pos[site])?;
        quaternion(&f.site_quat[site])?;
    }
    vector(&f.qpos0)?;
    vector(&f.key_qpos[0])?;
    Ok(actuator_joints)
}

fn validate_source_joint_limit_parameters(
    fields: &ModelFields,
    njnt: usize,
) -> Result<(), RobotError> {
    let presence = [
        fields.jnt_solref.is_some(),
        fields.jnt_solimp.is_some(),
        fields.jnt_margin.is_some(),
        fields.dof_invweight0.is_some(),
    ];
    if presence.iter().any(|present| *present) && presence.iter().any(|present| !*present) {
        return Err(invalid("source joint-limit parameter set is incomplete"));
    }
    if let Some(solref) = &fields.jnt_solref {
        if solref.len() != njnt {
            return Err(invalid("jnt_solref length must equal njnt"));
        }
        if solref.iter().flatten().any(|value| !value.is_finite()) {
            return Err(invalid("jnt_solref values must be finite"));
        }
    }
    if let Some(solimp) = &fields.jnt_solimp {
        if solimp.len() != njnt {
            return Err(invalid("jnt_solimp length must equal njnt"));
        }
        if solimp.iter().flatten().any(|value| !value.is_finite()) {
            return Err(invalid("jnt_solimp values must be finite"));
        }
    }
    if let Some(margin) = &fields.jnt_margin {
        if margin.len() != njnt {
            return Err(invalid("jnt_margin length must equal njnt"));
        }
        if margin.iter().any(|value| !value.is_finite()) {
            return Err(invalid("jnt_margin values must be finite"));
        }
    }
    if let Some(inverse_weight) = &fields.dof_invweight0 {
        if inverse_weight.len() != fields.dof_bodyid.len() {
            return Err(invalid("dof_invweight0 length must equal nv"));
        }
        if inverse_weight
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(invalid("dof_invweight0 values must be positive and finite"));
        }
    }
    Ok(())
}

fn vector(values: &[f64]) -> Result<(), RobotError> {
    if values
        .iter()
        .all(|v| v.is_finite() && v.abs() <= f64::from(f32::MAX))
    {
        Ok(())
    } else {
        Err(invalid("model value cannot be represented at f32 boundary"))
    }
}

fn quaternion(values: &[f64; 4]) -> Result<(), RobotError> {
    vector(values)?;
    if (values.iter().map(|v| v * v).sum::<f64>() - 1.0).abs() > 1e-8 {
        return Err(invalid("non-unit model quaternion"));
    }
    Ok(())
}

fn span<T>(values: &[T], start: usize, count: usize) -> Result<&[T], RobotError> {
    let end = start
        .checked_add(count)
        .ok_or_else(|| invalid("mesh span overflow"))?;
    values
        .get(start..end)
        .ok_or_else(|| invalid("mesh span out of bounds"))
}

#[cfg(test)]
#[path = "definition_tests.rs"]
mod tests;
