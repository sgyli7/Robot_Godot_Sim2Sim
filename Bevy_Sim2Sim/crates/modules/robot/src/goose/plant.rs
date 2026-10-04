//! SI mechanical data exported from a frozen Goose candidate.

use std::{collections::HashSet, fs, path::Path};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    RobotError,
    goose::contract::{GOOSE_JOINT_ORDER, GooseControlContract},
};

/// A source-derived mechanical tree, including collision and linkage provenance.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct GoosePlant {
    pub schema: String,
    pub candidate_id: String,
    pub neutral_contract_sha256: String,
    pub derived_contract_sha256: String,
    pub model_sha256: String,
    pub robot_mass_kg: f64,
    pub physics_hz: u32,
    pub joint_order: Vec<String>,
    pub bodies: Vec<GooseBody>,
    pub joints: Vec<GooseJoint>,
    pub colliders: Vec<GooseCollider>,
    pub exclusions: Vec<[String; 2]>,
    pub jaw_loop: GooseJawLoop,
    #[serde(default)]
    pub numerical_experiment: Option<GooseNumericalExperiment>,
}

/// Explicit experimental numerical selection; never mixed into physical inertia.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct GooseNumericalExperiment {
    pub source_method: String,
    pub predictive_scalar_stops: bool,
    pub jaw_pin_basis: String,
}

/// Complete inertia is expressed at COM in the body frame, never just diagonal.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct GooseBody {
    pub name: String,
    pub parent: Option<String>,
    pub translation_world_m: [f64; 3],
    pub rotation_world_wxyz: [f64; 4],
    pub mass_kg: f64,
    pub com_local_m: [f64; 3],
    pub inertia_at_com_body_kg_m2: [[f64; 3]; 3],
}

/// Hinge/slide values retain their SI meaning and distinct drive/feedback axes.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct GooseJoint {
    pub name: String,
    pub parent: String,
    pub child: String,
    pub kind: String,
    pub origin_world_m: [f64; 3],
    pub axis_world: [f64; 3],
    pub range: [f64; 2],
    pub stiffness_n_m: f64,
    pub damping: f64,
    pub armature: f64,
    pub frictionloss: f64,
    pub active_axis: Option<usize>,
    pub feedback_axis: Option<usize>,
    pub driven: bool,
}

/// Collision geometry has zero additional mass; masks use MuJoCo's OR rule.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct GooseCollider {
    pub name: String,
    pub body: String,
    pub kind: String,
    pub local_position_m: [f64; 3],
    pub local_rotation_wxyz: [f64; 4],
    pub vertices_local_m: Option<Vec<[f64; 3]>>,
    pub half_extents_m: Option<[f64; 3]>,
    pub friction: f64,
    pub contact_patch: Option<String>,
    pub contype: u32,
    pub conaffinity: u32,
}

/// Actual coupler-to-jaw pin, at the closed branch of the four-bar mechanism.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct GooseJawLoop {
    pub parent_body: String,
    pub rotor_body: String,
    pub coupler_body: String,
    pub jaw_body: String,
    pub output_pin_world_m: [f64; 3],
    pub rotation_axis_world: [f64; 3],
}

impl GoosePlant {
    /// Bind the exact control bytes and source self-contact exceptions.
    pub fn bound_control_contract(&self, bytes: &[u8]) -> Result<GooseControlContract, RobotError> {
        self.validate()?;
        if format!("{:x}", Sha256::digest(bytes)) != self.derived_contract_sha256 {
            return Err(invalid("Goose control bytes do not match the plant hash"));
        }
        let contract: GooseControlContract = serde_json::from_slice(bytes)
            .map_err(|error| invalid(format!("Goose control JSON: {error}")))?;
        contract.validate()?;
        if contract.candidate != self.candidate_id || contract.model_sha256 != self.model_sha256 {
            return Err(invalid(
                "Goose control/model identity differs from its plant",
            ));
        }
        if self.is_task_proxy() {
            let value: serde_json::Value = serde_json::from_slice(bytes)
                .map_err(|error| invalid(format!("Goose collision contract JSON: {error}")))?;
            let rows = value["collision_exclusions"]
                .as_array()
                .ok_or_else(|| invalid("Source collision exclusions absent"))?;
            let mut expected = HashSet::new();
            for row in rows {
                let mut pair = [
                    row["body1"]
                        .as_str()
                        .ok_or_else(|| invalid("Invalid source exclusion"))?
                        .to_owned(),
                    row["body2"]
                        .as_str()
                        .ok_or_else(|| invalid("Invalid source exclusion"))?
                        .to_owned(),
                ];
                pair.sort();
                expected.insert(pair);
            }
            let observed: HashSet<_> = self
                .exclusions
                .iter()
                .cloned()
                .map(|mut pair| {
                    pair.sort();
                    pair
                })
                .collect();
            if rows.len() != expected.len()
                || self.exclusions.len() != observed.len()
                || expected != observed
            {
                return Err(invalid(
                    "Native collision exclusions differ from source contract",
                ));
            }
        }
        Ok(contract)
    }

    /// Return the parsed contract and the hash of the exact bytes read.
    pub fn read(path: &Path) -> Result<(Self, String), RobotError> {
        let bytes = fs::read(path).map_err(|error| invalid(error.to_string()))?;
        let plant: Self = serde_json::from_slice(&bytes)
            .map_err(|error| invalid(format!("Goose plant JSON: {error}")))?;
        plant.validate()?;
        Ok((plant, format!("{:x}", Sha256::digest(bytes))))
    }

    /// Validate identity, physical tensors, the tree and the actual motor mapping.
    pub fn validate(&self) -> Result<(), RobotError> {
        let identity_valid = matches!(
            (self.schema.as_str(), self.candidate_id.as_str()),
            (
                "goose_plant50_v1",
                "goose_460_full50_v1"
                    | "goose_460_condensed50_v1"
                    | "goose_task_collision_v1_full50"
                    | "goose_task_collision_v1_condensed50"
            ) | ("goose_plant50_experimental_v2", "goose_460_full50_be_v2")
                | ("goose_task_proxy_plant_v1", "goose_task_proxy_11_v1")
        );
        if !identity_valid
            || self.physics_hz != 50
            || self
                .joint_order
                .iter()
                .map(String::as_str)
                .ne(GOOSE_JOINT_ORDER)
        {
            return Err(invalid(
                "Goose mechanical candidate identity/50 Hz mismatch",
            ));
        }
        let experimental = self.candidate_id == "goose_460_full50_be_v2";
        match (&self.numerical_experiment, experimental) {
            (None, false) => {}
            (Some(config), true)
                if config.source_method == "goose_joint_backward_euler_predictive_v1"
                    && config.predictive_scalar_stops
                    && config.jaw_pin_basis == "coupler_axis" => {}
            _ => {
                return Err(invalid(
                    "Goose numerical experiment must match its explicit candidate version",
                ));
            }
        }
        for hash in [
            &self.neutral_contract_sha256,
            &self.derived_contract_sha256,
            &self.model_sha256,
        ] {
            if hash.len() != 64
                || !hash
                    .bytes()
                    .all(|value| value.is_ascii_digit() || (b'a'..=b'f').contains(&value))
            {
                return Err(invalid("Goose mechanical identity needs lowercase SHA256"));
            }
        }
        let expected_bodies = if self.is_condensed() || self.is_task_proxy() {
            21
        } else {
            33
        };
        if self.bodies.len() != expected_bodies || self.joints.len() + 1 != self.bodies.len() {
            return Err(invalid("Goose candidate body/coordinate counts mismatch"));
        }
        if self.is_task_proxy()
            && (self.colliders.len() != 11
                || self
                    .colliders
                    .iter()
                    .any(|collider| collider.kind != "convex_mesh"))
        {
            return Err(invalid(
                "Goose task proxy requires its eleven actual convex leaves",
            ));
        }
        let names: HashSet<_> = self.bodies.iter().map(|body| body.name.as_str()).collect();
        if names.len() != self.bodies.len() || !names.contains("torso") {
            return Err(invalid("Goose body names are duplicate or root is absent"));
        }
        let mut assembled = HashSet::new();
        let mut mass = 0.0;
        for body in &self.bodies {
            if body.name == "torso" {
                if body.parent.is_some() {
                    return Err(invalid("Goose torso must be a free root"));
                }
            } else if !body
                .parent
                .as_deref()
                .is_some_and(|parent| assembled.contains(parent))
            {
                return Err(invalid("Goose bodies must form a parent-first tree"));
            }
            finite(&body.translation_world_m)?;
            finite(&body.com_local_m)?;
            quaternion(body.rotation_world_wxyz)?;
            if !body.mass_kg.is_finite() || body.mass_kg <= 0.0 {
                return Err(invalid("Invalid Goose body mass"));
            }
            inertia(body.inertia_at_com_body_kg_m2)?;
            mass += body.mass_kg;
            assembled.insert(body.name.as_str());
        }
        if !self.robot_mass_kg.is_finite() || (mass - self.robot_mass_kg).abs() > 1e-9 {
            return Err(invalid("Goose total mass does not match its physical tree"));
        }
        let mut joint_names = HashSet::new();
        let mut children = HashSet::new();
        let mut drives = [false; 18];
        let mut feedback = [false; 18];
        for joint in &self.joints {
            if !joint_names.insert(&joint.name)
                || !children.insert(&joint.child)
                || !names.contains(joint.parent.as_str())
                || !names.contains(joint.child.as_str())
                || self
                    .bodies
                    .iter()
                    .find(|body| body.name == joint.child)
                    .and_then(|body| body.parent.as_ref())
                    != Some(&joint.parent)
                || !matches!(joint.kind.as_str(), "hinge" | "slide")
            {
                return Err(invalid("Invalid Goose joint tree"));
            }
            finite(&joint.origin_world_m)?;
            unit_axis(joint.axis_world)?;
            finite(&joint.range)?;
            if joint.range[0] >= joint.range[1] {
                return Err(invalid("Invalid Goose joint range"));
            }
            for coefficient in [
                joint.stiffness_n_m,
                joint.damping,
                joint.armature,
                joint.frictionloss,
            ] {
                if !coefficient.is_finite() || coefficient < 0.0 {
                    return Err(invalid("Invalid Goose passive coefficient"));
                }
            }
            if joint.driven != joint.active_axis.is_some() {
                return Err(invalid("Goose drive flag/axis mismatch"));
            }
            if let Some(axis) = joint.active_axis {
                if axis >= 18
                    || drives[axis]
                    || (axis == 5 && joint.name != "beak_input_rotor")
                    || (axis != 5 && joint.name != GOOSE_JOINT_ORDER[axis])
                {
                    return Err(invalid("Goose action must drive its declared motor once"));
                }
                drives[axis] = true;
            }
            if let Some(axis) = joint.feedback_axis {
                if axis >= 18 || feedback[axis] || joint.name != GOOSE_JOINT_ORDER[axis] {
                    return Err(invalid(
                        "Goose feedback must retain the original 18-axis order",
                    ));
                }
                feedback[axis] = true;
            }
        }
        if !drives.into_iter().all(|value| value) || !feedback.into_iter().all(|value| value) {
            return Err(invalid("Goose drive/feedback mapping is incomplete"));
        }
        let mut collider_names = HashSet::new();
        for collider in &self.colliders {
            if !collider_names.insert(&collider.name) || !names.contains(collider.body.as_str()) {
                return Err(invalid("Invalid Goose collider owner or duplicate name"));
            }
            finite(&collider.local_position_m)?;
            quaternion(collider.local_rotation_wxyz)?;
            if !collider.friction.is_finite() || collider.friction < 0.0 {
                return Err(invalid("Invalid Goose friction"));
            }
            match collider.kind.as_str() {
                "convex_mesh" => {
                    let vertices = collider
                        .vertices_local_m
                        .as_ref()
                        .ok_or_else(|| invalid("Goose convex vertices absent"))?;
                    if vertices.len() < 4 {
                        return Err(invalid("Goose convex shape has too few vertices"));
                    }
                    for vertex in vertices {
                        finite(vertex)?;
                    }
                }
                "box" => {
                    let extents = collider
                        .half_extents_m
                        .ok_or_else(|| invalid("Goose box extents absent"))?;
                    if extents
                        .into_iter()
                        .any(|extent| !extent.is_finite() || extent <= 0.0)
                    {
                        return Err(invalid("Invalid Goose box extent"));
                    }
                }
                _ => return Err(invalid("Unsupported Goose collision shape")),
            }
        }
        for pair in &self.exclusions {
            if pair[0] == pair[1] || !pair.iter().all(|name| names.contains(name.as_str())) {
                return Err(invalid("Invalid Goose collision exclusion"));
            }
        }
        for name in [
            &self.jaw_loop.parent_body,
            &self.jaw_loop.rotor_body,
            &self.jaw_loop.coupler_body,
            &self.jaw_loop.jaw_body,
        ] {
            if !names.contains(name.as_str()) {
                return Err(invalid("Goose four-bar body absent"));
            }
        }
        finite(&self.jaw_loop.output_pin_world_m)?;
        unit_axis(self.jaw_loop.rotation_axis_world)?;
        Ok(())
    }

    /// This version uses the explicitly declared task-proxy contact runtime.
    pub fn is_task_proxy(&self) -> bool {
        self.schema == "goose_task_proxy_plant_v1" && self.candidate_id == "goose_task_proxy_11_v1"
    }

    /// Pad masses have been merged; target construction still needs a contact law.
    pub fn is_condensed(&self) -> bool {
        matches!(
            self.candidate_id.as_str(),
            "goose_460_condensed50_v1" | "goose_task_collision_v1_condensed50"
        )
    }
}

fn invalid(message: impl Into<String>) -> RobotError {
    RobotError::Contract(message.into())
}
fn finite(values: &[f64]) -> Result<(), RobotError> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(invalid("Nonfinite Goose mechanical data"))
    }
}
fn unit_axis(axis: [f64; 3]) -> Result<(), RobotError> {
    finite(&axis)?;
    if (axis.iter().map(|value| value * value).sum::<f64>() - 1.0).abs() > 1e-8 {
        Err(invalid("Nonunit Goose joint axis"))
    } else {
        Ok(())
    }
}
fn quaternion(value: [f64; 4]) -> Result<(), RobotError> {
    finite(&value)?;
    if (value.iter().map(|part| part * part).sum::<f64>() - 1.0).abs() > 1e-8 {
        Err(invalid("Nonunit Goose mechanical quaternion"))
    } else {
        Ok(())
    }
}
fn inertia(tensor: [[f64; 3]; 3]) -> Result<(), RobotError> {
    for row in &tensor {
        finite(row)?;
    }
    for row in 0..3 {
        for column in 0..3 {
            if (tensor[row][column] - tensor[column][row]).abs() > 1e-12 {
                return Err(invalid("Asymmetric Goose inertia"));
            }
        }
    }
    let minor = tensor[0][0] * tensor[1][1] - tensor[0][1] * tensor[1][0];
    let determinant = tensor[0][0] * (tensor[1][1] * tensor[2][2] - tensor[1][2] * tensor[2][1])
        - tensor[0][1] * (tensor[1][0] * tensor[2][2] - tensor[1][2] * tensor[2][0])
        + tensor[0][2] * (tensor[1][0] * tensor[2][1] - tensor[1][1] * tensor[2][0]);
    if tensor[0][0] <= 0.0 || minor <= 0.0 || determinant <= 0.0 {
        Err(invalid("Nonpositive Goose inertia"))
    } else {
        Ok(())
    }
}
