//! Numerical port fixture with mocked forces; no physics or policy qualification.

use robot_minigame::goose::{
    contract::{GooseControlContract, GooseNativeState},
    gravity::GooseNominalGravity,
    native_drive::GooseNativeDrive,
};
use serde::Deserialize;
use serde_json::json;
use std::{env, fs, path::PathBuf};

#[derive(Deserialize)]
struct Row {
    q: [f64; 18],
    qd: [f64; 18],
    motor_velocity: [f64; 18],
    quaternion: [f64; 4],
    action: [f64; 18],
    lower: [f64; 18],
    upper: [f64; 18],
    target: [f64; 18],
    ctrl: [f64; 18],
    feed: [f64; 5],
    mock_actual_force: [f64; 18],
    next_heat: [f64; 18],
    next_phase: f64,
    next_action: [f64; 18],
}
#[derive(Deserialize)]
struct Case {
    delayed: bool,
    strength: f64,
    rows: Vec<Row>,
}
#[derive(Deserialize)]
struct Fixture {
    schema: String,
    cases: Vec<Case>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let p: Vec<PathBuf> = env::args_os().skip(1).map(PathBuf::from).collect();
    if p.len() != 3 || p[2].exists() {
        return Err("usage: goose_native_drive_parity_probe CONTRACT FIXTURE NEW_OUTPUT".into());
    }
    let bytes = fs::read(&p[0])?;
    let contract: GooseControlContract = serde_json::from_slice(&bytes)?;
    contract.validate()?;
    let fixture: Fixture = serde_json::from_slice(&fs::read(&p[1])?)?;
    if fixture.schema != "source_driver_mock_force_fixture_v1" {
        return Err("mock fixture schema mismatch".into());
    }
    let mut count = 0;
    let mut max_error = 0.0_f64;
    let mut check = |a: f64, b: f64| -> Result<(), Box<dyn std::error::Error>> {
        max_error = max_error.max((a - b).abs());
        if !a.is_finite() || !b.is_finite() || (a - b).abs() > 1e-10 {
            return Err(format!("source driver port mismatch {a} vs {b}").into());
        }
        Ok(())
    };
    for case in fixture.cases {
        let case_count = case.rows.len();
        let gravity = GooseNominalGravity::from_contract_bytes(&bytes)?;
        let mut drive =
            GooseNativeDrive::new(contract.clone(), gravity, case.strength, case.delayed)?;
        for row in case.rows {
            let state = GooseNativeState {
                gyro_body_rad_s: [0.0; 3],
                projected_gravity: [0.0, 0.0, -1.0],
                joint_position_rad: row.q,
                joint_velocity_rad_s: row.qd,
                root_position_world_m: [0.0; 3],
                root_rotation_world_wxyz: row.quaternion,
            };
            let packet = drive.prepare(&state, row.motor_velocity, row.action)?;
            // A second prepare cannot silently replace the outstanding update.
            if drive
                .prepare(&state, row.motor_velocity, row.action)
                .is_ok()
            {
                return Err("duplicate prepare accepted".into());
            }
            for i in 0..18 {
                check(packet.target_rad[i], row.target[i])?;
                check(packet.lower_limit_nm[i], row.lower[i])?;
                check(packet.upper_limit_nm[i], row.upper[i])?;
                if i == 5 {
                    check(packet.external_effort_nm[i], row.ctrl[i])?;
                } else {
                    let bias = (row.lower[i] + row.upper[i]) * 0.5;
                    check(packet.external_effort_nm[i], bias)?;
                    check(
                        packet.symmetric_limit_nm[i],
                        (row.upper[i] - row.lower[i]) * 0.5,
                    )?;
                    check(
                        packet.motor_target_rad[i],
                        row.ctrl[i] - bias / contract.joints[i].kp_nm_rad,
                    )?;
                    if i < 5 {
                        check(
                            (row.ctrl[i] - row.target[i]) * contract.joints[i].kp_nm_rad,
                            row.feed[i],
                        )?;
                    }
                }
            }
            drive.finish(&packet, row.mock_actual_force)?;
            for i in 0..18 {
                check(drive.squared_torque_ewma()[i], row.next_heat[i])?;
                check(drive.previous_action()[i], row.next_action[i])?;
            }
            check(drive.phase(), row.next_phase)?;
            if drive.finish(&packet, row.mock_actual_force).is_ok() {
                return Err("duplicate finish accepted".into());
            }
            count += 1;
        }
        if drive.control_count() as usize != case_count {
            return Err("mock update count mismatch".into());
        }
    }
    fs::write(
        &p[2],
        serde_json::to_vec_pretty(&json!({"schema":"goose_native_drive_mock_parity_v1",
        "mock_updates":count,"maximum_absolute_error":max_error,"actual_integrals":0,"inferences":0,"optimizer_updates":0,
        "scope":"pure source driver port with mocked final forces; no simulation or motor qualification"}))?,
    )?;
    println!(
        "STATUS: success; mock_updates={count}; actual_integrals=0; maximum_error={max_error}"
    );
    Ok(())
}
