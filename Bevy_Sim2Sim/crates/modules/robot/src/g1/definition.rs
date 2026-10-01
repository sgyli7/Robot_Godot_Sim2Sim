//! Byte-bound source USD geometry and immutable completed-frame interchange.

use super::{contract::JOINT_NAMES, policy::bound_bytes};
use crate::RobotError;
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, path::Path};

pub const USD_SHA256: &str = "a7a2bab76981d19a1d76adecdfffec9b52afa34df9ba8e288ccedf410d3ce6bd";
pub const ARENA_COMMIT: &str = "7d75c95934c51a0318c957a8831e862ca43c53b5";

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourcePose {
    pub position: [f64; 3],
    pub rotation_wxyz: [f64; 4],
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Body {
    pub name: String,
    pub path: String,
    pub rest_pose: SourcePose,
    pub mass: f64,
    pub center_of_mass: [f64; 3],
    pub principal_inertia: [f64; 3],
    pub principal_axes_wxyz: [f64; 4],
    pub diagnostic_mass_defaults: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum JointKind {
    Fixed,
    Revolute,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Joint {
    pub name: String,
    pub parent: usize,
    pub child: usize,
    pub frame_parent: SourcePose,
    pub frame_child: SourcePose,
    pub kind: JointKind,
    pub axis: String,
    pub limits: [f64; 2],
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "shape", rename_all = "snake_case", deny_unknown_fields)]
pub enum Shape {
    ConvexHull {
        points: Vec<[f64; 3]>,
    },
    Sphere {
        radius: f64,
        local_pose: SourcePose,
    },
    Capsule {
        radius: f64,
        half_height: f64,
        axis: String,
        local_pose: SourcePose,
    },
    Box {
        half_extents: [f64; 3],
        local_pose: SourcePose,
    },
}

#[derive(Clone, Debug, Deserialize)]
pub struct Collision {
    pub body: usize,
    pub path: String,
    #[serde(flatten)]
    pub shape: Shape,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MassDefaultNote {
    pub body: String,
    pub interpretation: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceModel {
    pub schema: String,
    pub usd_sha256: String,
    pub arena_commit: String,
    pub exporter_sha256: String,
    pub openusd_version: Vec<u32>,
    pub units: String,
    pub bodies: Vec<Body>,
    pub joints: Vec<Joint>,
    pub collisions: Vec<Collision>,
    pub diagnostic_notes: Vec<MassDefaultNote>,
}

pub struct G1Definition {
    model: SourceModel,
    file_sha256: String,
    driven: [usize; 43],
}

impl G1Definition {
    pub fn load(path: &Path, expected_sha256: &str) -> Result<Self, RobotError> {
        let model: SourceModel = serde_json::from_slice(&bound_bytes(path, expected_sha256)?)
            .map_err(|e| invalid(e.to_string()))?;
        if model.schema != "arena_g1_usd_v1"
            || model.usd_sha256 != USD_SHA256
            || model.arena_commit != ARENA_COMMIT
            || model.units != "metres_kilograms_radians_z_up"
            || model.bodies.len() != 53
            || model.joints.len() != 52
            || model.collisions.len() != 52
            || model.bodies[0].name != "pelvis"
        {
            return Err(invalid("unexpected G1 source identity, topology or units"));
        }
        let mut names = HashSet::new();
        for body in &model.bodies {
            if !names.insert(&body.name)
                || !body.mass.is_finite()
                || body.mass <= 0.
                || body
                    .principal_inertia
                    .iter()
                    .any(|v| !v.is_finite() || *v <= 0.)
            {
                return Err(invalid("invalid G1 body name or mass properties"));
            }
            finite(&body.center_of_mass)?;
            rotation(&body.principal_axes_wxyz)?;
            body.rest_pose.validate()?;
        }
        let mut parent = vec![None; 53];
        let mut driven = [usize::MAX; 43];
        names.clear();
        for (id, joint) in model.joints.iter().enumerate() {
            if !names.insert(&joint.name)
                || joint.parent >= 53
                || joint.child == 0
                || joint.child >= 53
                || joint.child == joint.parent
                || parent[joint.child].replace(joint.parent).is_some()
            {
                return Err(invalid("invalid G1 joint tree"));
            }
            joint.frame_parent.validate()?;
            joint.frame_child.validate()?;
            finite(&joint.limits)?;
            if !matches!(joint.axis.as_str(), "X" | "Y" | "Z") {
                return Err(invalid("unknown USD joint axis"));
            }
            if joint.kind == JointKind::Revolute {
                let slot = JOINT_NAMES
                    .iter()
                    .position(|name| *name == joint.name)
                    .ok_or_else(|| invalid("USD hinge absent from WBC order"))?;
                if joint.limits[0] >= joint.limits[1] || driven[slot] != usize::MAX {
                    return Err(invalid("duplicate WBC joint or invalid limits"));
                }
                driven[slot] = id;
            }
        }
        if driven.contains(&usize::MAX) {
            return Err(invalid("missing WBC hinge"));
        }
        for body in 1..53 {
            let mut cursor = body;
            let mut seen = HashSet::new();
            while cursor != 0 {
                if !seen.insert(cursor) {
                    return Err(invalid("cyclic G1 articulation"));
                }
                cursor = parent[cursor].ok_or_else(|| invalid("disconnected G1 body"))?;
            }
        }
        for collider in &model.collisions {
            if collider.body >= 53 {
                return Err(invalid("missing collider body"));
            }
            match &collider.shape {
                Shape::ConvexHull { points } => {
                    if points.len() < 4 {
                        return Err(invalid("degenerate hull input"));
                    }
                    for p in points {
                        finite(p)?;
                    }
                }
                Shape::Sphere { radius, local_pose } => {
                    positive(&[*radius])?;
                    local_pose.validate()?;
                }
                Shape::Capsule {
                    radius,
                    half_height,
                    axis,
                    local_pose,
                } => {
                    positive(&[*radius, *half_height])?;
                    local_pose.validate()?;
                    if !matches!(axis.as_str(), "X" | "Y" | "Z") {
                        return Err(invalid("invalid capsule axis"));
                    }
                }
                Shape::Box {
                    half_extents,
                    local_pose,
                } => {
                    positive(half_extents)?;
                    local_pose.validate()?;
                }
            }
        }
        for (i, body) in model
            .bodies
            .iter()
            .enumerate()
            .filter(|(_, b)| b.diagnostic_mass_defaults)
        {
            if !matches!(
                body.name.as_str(),
                "imu_in_pelvis" | "d435_link" | "imu_in_torso" | "mid360_link"
            ) || model.collisions.iter().any(|c| c.body == i)
                || !model
                    .joints
                    .iter()
                    .any(|j| j.child == i && j.kind == JointKind::Fixed)
            {
                return Err(invalid("mass fallback outside shape-less fixed sensor"));
            }
        }
        Ok(Self {
            model,
            file_sha256: expected_sha256.into(),
            driven,
        })
    }
    pub fn model(&self) -> &SourceModel {
        &self.model
    }
    pub fn file_sha256(&self) -> &str {
        &self.file_sha256
    }
    pub fn driven_joints(&self) -> &[usize; 43] {
        &self.driven
    }
}

impl SourcePose {
    fn validate(&self) -> Result<(), RobotError> {
        finite(&self.position)?;
        rotation(&self.rotation_wxyz)
    }
}

/// Engine Y-up poses from one completed native step; never renderer interpolation.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct G1BodyPose {
    pub body: usize,
    pub translation: [f32; 3],
    pub rotation_xyzw: [f32; 4],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct G1BodyFrame {
    pub usd_sha256: String,
    pub episode_id: u64,
    pub source_tick: u64,
    pub sim_time: f64,
    pub bodies: Vec<G1BodyPose>,
}

impl G1BodyFrame {
    pub fn validate(&self) -> Result<(), RobotError> {
        if self.usd_sha256 != USD_SHA256
            || self.bodies.len() != 53
            || !self.sim_time.is_finite()
            || self.sim_time < 0.
        {
            return Err(invalid("invalid G1 frame identity or size"));
        }
        let mut seen = [false; 53];
        for body in &self.bodies {
            if body.body >= 53 || std::mem::replace(&mut seen[body.body], true) {
                return Err(invalid("duplicate or invalid G1 frame body"));
            }
            finite(&body.translation.map(f64::from))?;
            rotation(&body.rotation_xyzw.map(f64::from)).map_err(|cause| {
                invalid(format!("G1 completed frame body {}: {cause}", body.body))
            })?;
        }
        Ok(())
    }
}

fn finite(values: &[f64]) -> Result<(), RobotError> {
    if values
        .iter()
        .all(|v| v.is_finite() && (*v as f32).is_finite())
    {
        Ok(())
    } else {
        Err(invalid("G1 source cannot be represented in f32"))
    }
}
fn positive(values: &[f64]) -> Result<(), RobotError> {
    finite(values)?;
    if values.iter().all(|v| *v > 0.) {
        Ok(())
    } else {
        Err(invalid("nonpositive G1 shape"))
    }
}
fn rotation(values: &[f64; 4]) -> Result<(), RobotError> {
    finite(values)?;
    if (values.iter().map(|v| v * v).sum::<f64>() - 1.).abs() <= 2e-6 {
        Ok(())
    } else {
        Err(invalid("G1 source quaternion is not normalized"))
    }
}
fn invalid(message: impl Into<String>) -> RobotError {
    RobotError::Contract(message.into())
}
