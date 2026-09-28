//! Source-basis forward kinematics for checking the imported MJCF geometry.
//! This is a geometric validator; runtime body poses come from the physics world.

use crate::{RobotError, definition::RobotDefinition};
use serde::Serialize;

#[derive(Debug, Clone, Copy, Serialize)]
pub struct SourceBodyPose {
    pub position: [f64; 3],
    pub rotation_wxyz: [f64; 4],
}

pub fn source_body_poses(
    definition: &RobotDefinition,
    qpos: &[f64],
) -> Result<Vec<SourceBodyPose>, RobotError> {
    let model = definition.model();
    let f = &model.fields;
    if qpos.len() != model.counts.nq || qpos.iter().any(|x| !x.is_finite()) {
        return Err(RobotError::Contract(
            "invalid source kinematic state".into(),
        ));
    }
    let mut result = vec![
        SourceBodyPose {
            position: [0.0; 3],
            rotation_wxyz: [1.0, 0.0, 0.0, 0.0]
        };
        model.counts.nbody
    ];
    for body in 1..model.counts.nbody {
        let joint = f
            .jnt_bodyid
            .iter()
            .position(|id| *id == body)
            .ok_or_else(|| RobotError::Contract("body has no source joint".into()))?;
        let address = f.jnt_qposadr[joint];
        if f.jnt_type[joint] == 0 {
            let rotation = [
                qpos[address + 3],
                qpos[address + 4],
                qpos[address + 5],
                qpos[address + 6],
            ];
            if (rotation.iter().map(|x| x * x).sum::<f64>() - 1.0).abs() > 2e-6 {
                return Err(RobotError::Contract(
                    "free-root quaternion is not normalized".into(),
                ));
            }
            result[body] = SourceBodyPose {
                position: [qpos[address], qpos[address + 1], qpos[address + 2]],
                rotation_wxyz: rotation,
            };
            continue;
        }
        let angle = qpos[address] - f.qpos0[address];
        let half_sine = (angle * 0.5).sin();
        let axis = f.jnt_axis[joint];
        let hinge = [
            (angle * 0.5).cos(),
            axis[0] * half_sine,
            axis[1] * half_sine,
            axis[2] * half_sine,
        ];
        let local_rotation = multiply(f.body_quat[body], hinge);
        // T_body · T_pivot · R_hinge · T_-pivot, before the parent transform.
        let rotated_pivot = rotate(hinge, f.jnt_pos[joint]);
        let pivot_shift = std::array::from_fn(|i| f.jnt_pos[joint][i] - rotated_pivot[i]);
        let body_shift = rotate(f.body_quat[body], pivot_shift);
        let local_position = std::array::from_fn(|i| f.body_pos[body][i] + body_shift[i]);
        let parent = result[f.body_parentid[body]];
        let shift = rotate(parent.rotation_wxyz, local_position);
        result[body] = SourceBodyPose {
            position: std::array::from_fn(|i| parent.position[i] + shift[i]),
            rotation_wxyz: multiply(parent.rotation_wxyz, local_rotation),
        };
    }
    Ok(result)
}

fn multiply([aw, ax, ay, az]: [f64; 4], [bw, bx, by, bz]: [f64; 4]) -> [f64; 4] {
    [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]
}

fn rotate(q: [f64; 4], v: [f64; 3]) -> [f64; 3] {
    let value = multiply(
        multiply(q, [0.0, v[0], v[1], v[2]]),
        [q[0], -q[1], -q[2], -q[3]],
    );
    [value[1], value[2], value[3]]
}
