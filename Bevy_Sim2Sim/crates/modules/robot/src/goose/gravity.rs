//! Nominal neck gravity feedforward.
//!
//! The arithmetic follows `NominalNeckGravity.__call__` and `quat_matrix` in
//! `stage_one_gravity.py`, source SHA-256
//! `da0e918deaacc822289a4ce1d1013a22ae9ee89c26929640960af2d3ce34e949`.
//! This is the nominal encoder and IMU neck feedforward. Simulator truth,
//! policy inference, runtime root pose, randomized mass, contact, and velocity
//! stay outside its inputs.
//!
//! [`GooseNominalGravity::torque_nm`] takes the first six absolute encoder
//! coordinates in `GOOSE_JOINT_ORDER`. Those coordinates stay in the original
//! absolute encoder frame. It also takes an IMU quaternion in `wxyz` order and
//! divides that quaternion by its Euclidean norm, matching the source. The five
//! results are neck moments in newton-metres. The original upper-triangular
//! descendant mask over the first six bodies is preserved, and both passive
//! mimic links contribute to those five moments.

use std::collections::BTreeMap;

use serde::Deserialize;

use super::contract::GOOSE_JOINT_ORDER;
use crate::RobotError;

const SCHEMA: &str = "goose_task_proxy_si_v1";
const CANDIDATE: &str = "goose_task_proxy_11_rigid_braking_v1";
const GRAVITY_M_S2: f64 = 9.81;
const DRIVEN_JOINTS: usize = 6;
const TORQUE_JOINTS: usize = 5;

/// Validated nominal neck mass, center of mass, pivots, axes, and passive mimic law.
#[derive(Clone, Debug)]
pub struct GooseNominalGravity {
    pivots: Vec<[f64; 3]>,
    links: Vec<NominalLink>,
}

/// One moving body in parent-first order. Frame zero is the torso pivot.
#[derive(Clone, Debug)]
struct NominalLink {
    parent: usize,
    axis: [f64; 3],
    skew: [[f64; 3]; 3],
    mass_kg: f64,
    com_local_m: [f64; 3],
    angle: LinkAngle,
}

#[derive(Clone, Debug)]
enum LinkAngle {
    Encoder(usize),
    Mimic {
        encoder: usize,
        multiplier: f64,
        offset_rad: f64,
    },
}

#[derive(Deserialize)]
struct NominalContract {
    schema: String,
    candidate: String,
    root_origin_at_zero_m: [f64; 3],
    joints: Vec<NominalJoint>,
    /// Absent passive links match the source `dict.get` default.
    #[serde(default)]
    passive_linkage_joints: Vec<PassiveJoint>,
    bodies: Vec<NominalBody>,
}

#[derive(Deserialize)]
struct NominalJoint {
    name: String,
    parent: String,
    pivot_world_at_zero_m: [f64; 3],
    axis_parent: [f64; 3],
}

#[derive(Deserialize)]
struct PassiveJoint {
    name: String,
    parent: String,
    pivot_world_at_zero_m: [f64; 3],
    axis_parent: [f64; 3],
    mimic_joint: String,
    mimic_multiplier: f64,
    mimic_offset_rad: f64,
}

#[derive(Deserialize)]
struct NominalBody {
    name: String,
    mass_kg: f64,
    com_local_m: [f64; 3],
}

#[derive(Clone, Copy)]
struct BodyMass {
    mass_kg: f64,
    com_local_m: [f64; 3],
}

impl GooseNominalGravity {
    /// Loads the nominal subset from contract JSON.
    ///
    /// The bytes may be the full frozen contract or a smaller document with
    /// the fields this feedforward reads. Schema and candidate must be
    /// `goose_task_proxy_si_v1` and `goose_task_proxy_11_rigid_braking_v1`.
    /// The first six joints must match `GOOSE_JOINT_ORDER` and form a
    /// parent-first chain from `torso`, as must any passive links. Used masses
    /// must be finite and positive, axes must be unit length, and every used
    /// coordinate must be finite. This is the nominal subset check. Full SI,
    /// hash, and control binding stay with the caller.
    pub fn from_contract_bytes(bytes: &[u8]) -> Result<Self, RobotError> {
        let parsed: NominalContract = serde_json::from_slice(bytes)
            .map_err(|error| invalid(format!("malformed Goose nominal gravity subset: {error}")))?;
        if parsed.schema != SCHEMA || parsed.candidate != CANDIDATE {
            return Err(invalid(
                "Goose nominal gravity schema or candidate mismatch",
            ));
        }
        require_finite_point(
            parsed.root_origin_at_zero_m,
            "nonfinite Goose nominal root pivot",
        )?;
        let neck_names_match = parsed
            .joints
            .iter()
            .take(DRIVEN_JOINTS)
            .map(|joint| joint.name.as_str())
            .eq(GOOSE_JOINT_ORDER.into_iter().take(DRIVEN_JOINTS));
        if !neck_names_match {
            return Err(invalid(
                "Goose nominal neck chain must be the first six Goose joints",
            ));
        }

        let mut bodies = BTreeMap::new();
        for body in &parsed.bodies {
            if body.name.is_empty() || bodies.contains_key(&body.name) {
                return Err(invalid("duplicate or empty Goose nominal body"));
            }
            bodies.insert(
                body.name.clone(),
                BodyMass {
                    mass_kg: body.mass_kg,
                    com_local_m: body.com_local_m,
                },
            );
        }

        let mut names = vec!["torso".to_owned()];
        let mut pivots = vec![parsed.root_origin_at_zero_m];
        let mut links = Vec::with_capacity(DRIVEN_JOINTS + parsed.passive_linkage_joints.len());
        for (index, joint) in parsed.joints.iter().take(DRIVEN_JOINTS).enumerate() {
            push_link(
                &mut names,
                &mut pivots,
                &mut links,
                &bodies,
                &joint.name,
                &joint.parent,
                joint.pivot_world_at_zero_m,
                joint.axis_parent,
                LinkAngle::Encoder(index),
            )?;
        }
        for passive in &parsed.passive_linkage_joints {
            if !passive.mimic_multiplier.is_finite() || !passive.mimic_offset_rad.is_finite() {
                return Err(invalid("nonfinite Goose nominal mimic"));
            }
            let encoder = parsed
                .joints
                .iter()
                .take(DRIVEN_JOINTS)
                .position(|joint| joint.name == passive.mimic_joint)
                .ok_or_else(|| invalid("Goose nominal mimic joint is outside the first six"))?;
            push_link(
                &mut names,
                &mut pivots,
                &mut links,
                &bodies,
                &passive.name,
                &passive.parent,
                passive.pivot_world_at_zero_m,
                passive.axis_parent,
                LinkAngle::Mimic {
                    encoder,
                    multiplier: passive.mimic_multiplier,
                    offset_rad: passive.mimic_offset_rad,
                },
            )?;
        }
        Ok(Self { pivots, links })
    }

    /// First five neck gravity moments, in newton-metres.
    ///
    /// `neck_and_jaw_q` holds raw absolute encoder radians for `neck_yaw`,
    /// `neck_pitch`, `neck_mid_pitch`, `head_pitch`, `head_roll`, and
    /// `beak_hinge`. `imu_wxyz` is normalized by its Euclidean norm when that
    /// norm is finite and nonzero.
    pub fn torque_nm(
        &self,
        neck_and_jaw_q: [f64; 6],
        imu_wxyz: [f64; 4],
    ) -> Result<[f64; 5], RobotError> {
        if neck_and_jaw_q.iter().any(|value| !value.is_finite()) {
            return Err(RobotError::NonFinite("Goose nominal encoder"));
        }
        let mut positions = vec![[0.0; 3]; self.links.len() + 1];
        let mut rotations =
            vec![[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]; self.links.len() + 1];
        // The source places the torso origin at zero. The contract root is only the torso pivot.
        positions[0] = [0.0, 0.0, 0.0];
        rotations[0] = quat_matrix(imu_wxyz)?;

        let mut world_axes = [[0.0; 3]; DRIVEN_JOINTS];
        let mut centers = vec![[0.0; 3]; self.links.len()];
        let mut forces = vec![[0.0; 3]; self.links.len()];
        for (link_index, link) in self.links.iter().enumerate() {
            let frame = link_index + 1;
            let parent = link.parent;
            let parent_rotation = rotations[parent];
            let offset = sub(self.pivots[frame], self.pivots[parent]);
            positions[frame] = add(positions[parent], matvec(parent_rotation, offset));
            if let LinkAngle::Encoder(index) = link.angle {
                world_axes[index] = matvec(parent_rotation, link.axis);
            }
            let local = rodrigues(link.skew, link.angle.radians(neck_and_jaw_q));
            rotations[frame] = matmul(parent_rotation, local);
            centers[link_index] = add(positions[frame], matvec(rotations[frame], link.com_local_m));
            forces[link_index] = [0.0, 0.0, -GRAVITY_M_S2 * link.mass_kg];
        }

        let mut torque = [0.0; TORQUE_JOINTS];
        for joint in 0..TORQUE_JOINTS {
            let mut moment = [0.0; 3];
            for body in 0..self.links.len() {
                if !contributes(joint, body) {
                    continue;
                }
                let arm = sub(centers[body], positions[joint + 1]);
                moment = add(moment, cross(arm, forces[body]));
            }
            torque[joint] = -dot(moment, world_axes[joint]);
        }
        if torque.iter().any(|value| !value.is_finite()) {
            return Err(RobotError::NonFinite("Goose nominal gravity"));
        }
        Ok(torque)
    }
}

impl LinkAngle {
    fn radians(&self, q: [f64; 6]) -> f64 {
        match self {
            LinkAngle::Encoder(index) => q[*index],
            LinkAngle::Mimic {
                encoder,
                multiplier,
                offset_rad,
            } => *multiplier * q[*encoder] + *offset_rad,
        }
    }
}

fn push_link(
    names: &mut Vec<String>,
    pivots: &mut Vec<[f64; 3]>,
    links: &mut Vec<NominalLink>,
    bodies: &BTreeMap<String, BodyMass>,
    name: &str,
    parent: &str,
    pivot: [f64; 3],
    axis: [f64; 3],
    angle: LinkAngle,
) -> Result<(), RobotError> {
    if name.is_empty() || names.iter().any(|existing| existing == name) {
        return Err(invalid("duplicate or empty Goose nominal link"));
    }
    let parent_index = names
        .iter()
        .position(|existing| existing == parent)
        .ok_or_else(|| invalid("Goose nominal linkage is not parent-first"))?;
    require_finite_point(pivot, "nonfinite Goose nominal pivot")?;
    require_unit_axis(axis)?;
    let body = require_body(bodies, name)?;
    links.push(NominalLink {
        parent: parent_index,
        axis,
        skew: skew(axis),
        mass_kg: body.mass_kg,
        com_local_m: body.com_local_m,
        angle,
    });
    names.push(name.to_owned());
    pivots.push(pivot);
    Ok(())
}

/// Source mask: upper triangle over the first six bodies, then ones for each passive body.
fn contributes(joint: usize, body: usize) -> bool {
    if body < DRIVEN_JOINTS {
        body >= joint
    } else {
        true
    }
}

fn require_body<'a>(
    bodies: &'a BTreeMap<String, BodyMass>,
    name: &str,
) -> Result<&'a BodyMass, RobotError> {
    let body = bodies
        .get(name)
        .ok_or_else(|| invalid("missing Goose nominal body"))?;
    if !body.mass_kg.is_finite() || body.mass_kg <= 0.0 {
        return Err(invalid("Goose nominal mass must be finite and positive"));
    }
    if body.com_local_m.iter().any(|value| !value.is_finite()) {
        return Err(invalid("nonfinite Goose nominal center of mass"));
    }
    Ok(body)
}

fn require_finite_point(point: [f64; 3], message: &str) -> Result<(), RobotError> {
    if point.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(invalid(message))
    }
}

fn require_unit_axis(axis: [f64; 3]) -> Result<(), RobotError> {
    require_finite_point(axis, "nonfinite Goose nominal axis")?;
    let norm_sq = axis.iter().map(|value| value * value).sum::<f64>();
    if (norm_sq - 1.0).abs() > 1e-8 {
        Err(invalid("Goose nominal axis must be a unit vector"))
    } else {
        Ok(())
    }
}

fn quat_matrix(wxyz: [f64; 4]) -> Result<[[f64; 3]; 3], RobotError> {
    if wxyz.iter().any(|value| !value.is_finite()) {
        return Err(RobotError::NonFinite("Goose nominal IMU"));
    }
    let norm_sq = wxyz.iter().map(|value| value * value).sum::<f64>();
    if !norm_sq.is_finite() {
        return Err(RobotError::NonFinite("Goose nominal IMU"));
    }
    if norm_sq == 0.0 {
        return Err(invalid("zero Goose nominal IMU quaternion"));
    }
    let scale = norm_sq.sqrt().recip();
    let w = wxyz[0] * scale;
    let x = wxyz[1] * scale;
    let y = wxyz[2] * scale;
    let z = wxyz[3] * scale;
    Ok([
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ])
}

fn skew(axis: [f64; 3]) -> [[f64; 3]; 3] {
    let [x, y, z] = axis;
    [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]]
}

fn rodrigues(skew_matrix: [[f64; 3]; 3], angle: f64) -> [[f64; 3]; 3] {
    let sine = angle.sin();
    let cosine = angle.cos();
    let squared = matmul(skew_matrix, skew_matrix);
    std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            let identity = if row == column { 1.0 } else { 0.0 };
            identity + sine * skew_matrix[row][column] + (1.0 - cosine) * squared[row][column]
        })
    })
}

fn matmul(left: [[f64; 3]; 3], right: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            left[row][0] * right[0][column]
                + left[row][1] * right[1][column]
                + left[row][2] * right[2][column]
        })
    })
}

fn matvec(matrix: [[f64; 3]; 3], vector: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|row| {
        matrix[row][0] * vector[0] + matrix[row][1] * vector[1] + matrix[row][2] * vector[2]
    })
}

fn add(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|axis| left[axis] + right[axis])
}

fn sub(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|axis| left[axis] - right[axis])
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn invalid(message: impl Into<String>) -> RobotError {
    RobotError::Contract(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    const ABSOLUTE_ERROR_NM: f64 = 1e-10;

    /// Nominal subset copied from the independent gravity fixture. Expected
    /// torques below are those fixture outputs, not values produced by this port.
    const NOMINAL_SUBSET: &str = r#"{
  "schema": "goose_task_proxy_si_v1",
  "candidate": "goose_task_proxy_11_rigid_braking_v1",
  "root_origin_at_zero_m": [
    0.0,
    0.0,
    0.0
  ],
  "joints": [
    {
      "name": "neck_yaw",
      "parent": "torso",
      "pivot_world_at_zero_m": [
        0.043,
        0.0,
        0.3457
      ],
      "axis_parent": [
        0.0,
        0.0,
        1.0
      ]
    },
    {
      "name": "neck_pitch",
      "parent": "neck_yaw",
      "pivot_world_at_zero_m": [
        0.043,
        0.0,
        0.41169999999999995
      ],
      "axis_parent": [
        0.0,
        1.0,
        0.0
      ]
    },
    {
      "name": "neck_mid_pitch",
      "parent": "neck_pitch",
      "pivot_world_at_zero_m": [
        -0.03,
        0.0,
        0.5157
      ],
      "axis_parent": [
        0.0,
        1.0,
        0.0
      ]
    },
    {
      "name": "head_pitch",
      "parent": "neck_mid_pitch",
      "pivot_world_at_zero_m": [
        0.042,
        0.0,
        0.6087
      ],
      "axis_parent": [
        0.0,
        1.0,
        0.0
      ]
    },
    {
      "name": "head_roll",
      "parent": "head_pitch",
      "pivot_world_at_zero_m": [
        0.081,
        0.0,
        0.5892000000000001
      ],
      "axis_parent": [
        1.0,
        0.0,
        0.0
      ]
    },
    {
      "name": "beak_hinge",
      "parent": "head_roll",
      "pivot_world_at_zero_m": [
        0.16,
        0.0,
        0.5797
      ],
      "axis_parent": [
        0.0,
        1.0,
        0.0
      ]
    }
  ],
  "passive_linkage_joints": [
    {
      "name": "beak_input_rotor",
      "parent": "head_roll",
      "pivot_world_at_zero_m": [
        0.129,
        0.0,
        0.6047
      ],
      "axis_parent": [
        0.0,
        1.0,
        0.0
      ],
      "mimic_joint": "beak_hinge",
      "mimic_multiplier": 1.0,
      "mimic_offset_rad": 0.0
    },
    {
      "name": "beak_coupler_link",
      "parent": "beak_input_rotor",
      "pivot_world_at_zero_m": [
        0.123,
        0.0,
        0.5943076951545867
      ],
      "axis_parent": [
        0.0,
        1.0,
        0.0
      ],
      "mimic_joint": "beak_hinge",
      "mimic_multiplier": -1.0,
      "mimic_offset_rad": 0.0
    }
  ],
  "bodies": [
    {
      "name": "neck_yaw",
      "mass_kg": 0.45294349124138206,
      "com_local_m": [
        -1.7268310993634461e-07,
        -0.0007048891988300066,
        0.060960584342048596
      ]
    },
    {
      "name": "neck_pitch",
      "mass_kg": 0.6107350780492311,
      "com_local_m": [
        -0.05799548597643267,
        -0.0009078086584944603,
        0.08185788015302148
      ]
    },
    {
      "name": "neck_mid_pitch",
      "mass_kg": 0.5145364404531154,
      "com_local_m": [
        0.052393753795951604,
        0.002063425478576288,
        0.06752041621990124
      ]
    },
    {
      "name": "head_pitch",
      "mass_kg": 0.05813070923092986,
      "com_local_m": [
        0.020548682972421815,
        0.00557068165139365,
        -0.021424731068606317
      ]
    },
    {
      "name": "head_roll",
      "mass_kg": 0.6593799033333092,
      "com_local_m": [
        0.06081670316810249,
        -0.0013338700177895505,
        0.00943233500530316
      ]
    },
    {
      "name": "beak_hinge",
      "mass_kg": 0.12444184333803009,
      "com_local_m": [
        0.022105349738602786,
        0.0021783860398677996,
        -0.0158287818399776
      ]
    },
    {
      "name": "beak_input_rotor",
      "mass_kg": 0.01713015122738488,
      "com_local_m": [
        -0.000722912037014467,
        0.030037179254881457,
        -0.0012521203775123224
      ]
    },
    {
      "name": "beak_coupler_link",
      "mass_kg": 0.005177307249452544,
      "com_local_m": [
        0.015499999888487326,
        0.034964001297447034,
        -0.012500000368660724
      ]
    }
  ]
}
"#;

    const GOLDEN: [([f64; 6], [f64; 4], [f64; 5]); 12] = [
        (
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [
                -0.0,
                -0.3878614344564497,
                -1.722727867831325,
                -0.8478209634658435,
                0.0008546088564479566,
            ],
        ),
        (
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.955336489125606, 0.0, 0.29552020666133955, 0.0],
            [
                0.0033176462397869117,
                -1.9531357628325132,
                -1.9836431897083628,
                -0.623896964433475,
                0.0007053391260437313,
            ],
        ),
        (
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.9800665778412416, 0.19866933079506122, 0.0, 0.0],
            [
                0.1510400580523785,
                -0.35724403835584245,
                -1.586737442341191,
                -0.7808948193463335,
                -0.011717600669139766,
            ],
        ),
        (
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [
                -0.0,
                0.3878614344564497,
                1.722727867831325,
                0.8478209634658435,
                -0.0008546088564479566,
            ],
        ),
        (
            [0.15, 0.2, -0.15, 0.2, 0.1, 0.2],
            [1.0, 0.0, 0.0, 0.0],
            [
                -0.0,
                -0.7815411959596245,
                -1.7128957938665565,
                -0.782640496182299,
                -0.0018105037115877104,
            ],
        ),
        (
            [0.15, 0.2, -0.15, 0.2, 0.1, 0.2],
            [0.955336489125606, 0.0, 0.29552020666133955, 0.0],
            [
                0.0677057138441757,
                -2.244645928460154,
                -1.8272021022900784,
                -0.45413594704923615,
                -0.003529929902268969,
            ],
        ),
        (
            [0.15, 0.2, -0.15, 0.2, 0.1, 0.2],
            [0.9800665778412416, 0.19866933079506122, 0.0, 0.0],
            [
                0.30074524481424136,
                -0.5531134713398884,
                -1.5345820596281645,
                -0.7408522091741773,
                -0.012161550648014233,
            ],
        ),
        (
            [0.15, 0.2, -0.15, 0.2, 0.1, 0.2],
            [0.0, 1.0, 0.0, 0.0],
            [
                -0.0,
                0.7815411959596245,
                1.7128957938665565,
                0.782640496182299,
                0.0018105037115877104,
            ],
        ),
        (
            [-0.2, -0.1, 0.15, -0.1, -0.2, 0.5],
            [1.0, 0.0, 0.0, 0.0],
            [
                -0.0,
                -0.25382469153453624,
                -1.7714202430515484,
                -0.8411649453672903,
                0.005168845932287183,
            ],
        ),
        (
            [-0.2, -0.1, 0.15, -0.1, -0.2, 0.5],
            [0.955336489125606, 0.0, 0.29552020666133955, 0.0],
            [
                -0.02283070543683178,
                -1.7231593007710335,
                -2.004824003075967,
                -0.6370864582088769,
                0.006790574600443869,
            ],
        ),
        (
            [-0.2, -0.1, 0.15, -0.1, -0.2, 0.5],
            [0.9800665778412416, 0.19866933079506122, 0.0, 0.0],
            [
                0.0976622453250777,
                -0.44540424286871017,
                -1.7074725298885813,
                -0.7667734807352228,
                -0.003321324085813468,
            ],
        ),
        (
            [-0.2, -0.1, 0.15, -0.1, -0.2, 0.5],
            [0.0, 1.0, 0.0, 0.0],
            [
                -0.0,
                0.25382469153453624,
                1.7714202430515484,
                0.8411649453672903,
                -0.005168845932287183,
            ],
        ),
    ];

    fn assert_nm(got: [f64; 5], expected: [f64; 5], label: &str) {
        for axis in 0..5 {
            let error = (got[axis] - expected[axis]).abs();
            assert!(
                error <= ABSOLUTE_ERROR_NM,
                "{label} axis {axis} absolute error {error} Nm"
            );
        }
    }

    #[test]
    fn matches_twelve_independent_golden_torques() {
        assert_eq!(GOLDEN.len(), 12);
        let model = GooseNominalGravity::from_contract_bytes(NOMINAL_SUBSET.as_bytes())
            .expect("nominal subset");
        for (index, (q, imu, expected)) in GOLDEN.iter().enumerate() {
            let torque = model.torque_nm(*q, *imu).expect("golden encoder and IMU");
            assert_nm(torque, *expected, &format!("case {index}"));
        }
    }

    #[test]
    fn normalizes_imu_quaternion_like_the_source() {
        let model = GooseNominalGravity::from_contract_bytes(NOMINAL_SUBSET.as_bytes())
            .expect("nominal subset");
        let (q, imu, expected) = GOLDEN[1];
        let scaled = imu.map(|value| value * 2.0);
        let torque = model.torque_nm(q, scaled).expect("scaled IMU");
        assert_nm(torque, expected, "scaled IMU");
    }

    #[test]
    fn rejects_invalid_quaternion_and_nonfinite_encoder() {
        let model = GooseNominalGravity::from_contract_bytes(NOMINAL_SUBSET.as_bytes())
            .expect("nominal subset");
        let q = [0.0; 6];
        assert!(matches!(
            model.torque_nm(q, [0.0, 0.0, 0.0, 0.0]),
            Err(RobotError::Contract(_))
        ));
        assert!(matches!(
            model.torque_nm(q, [f64::NAN, 0.0, 0.0, 0.0]),
            Err(RobotError::NonFinite(_))
        ));
        assert!(matches!(
            model.torque_nm(q, [f64::INFINITY, 0.0, 0.0, 0.0]),
            Err(RobotError::NonFinite(_))
        ));
        assert!(matches!(
            model.torque_nm([f64::NAN, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
            Err(RobotError::NonFinite(_))
        ));
        assert!(matches!(
            model.torque_nm(
                [f64::INFINITY, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0]
            ),
            Err(RobotError::NonFinite(_))
        ));
    }

    #[test]
    fn rejects_malformed_nominal_subset() {
        let wrong_schema =
            NOMINAL_SUBSET.replace("goose_task_proxy_si_v1", "goose_task_proxy_si_v0");
        assert!(GooseNominalGravity::from_contract_bytes(wrong_schema.as_bytes()).is_err());
        let negative_mass = NOMINAL_SUBSET.replace("0.45294349124138206", "-0.45294349124138206");
        assert!(GooseNominalGravity::from_contract_bytes(negative_mass.as_bytes()).is_err());
        let forward_parent =
            NOMINAL_SUBSET.replace("\"parent\": \"neck_yaw\"", "\"parent\": \"head_roll\"");
        assert!(GooseNominalGravity::from_contract_bytes(forward_parent.as_bytes()).is_err());
        assert!(GooseNominalGravity::from_contract_bytes(b"{").is_err());
    }
}
