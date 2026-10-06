//! Read-only geometry and actual-body measurements for rigid Goose probes.

use rapier3d::{
    geometry::{ContactData, ContactManifold, ContactManifoldData},
    parry::query::{DefaultQueryDispatcher, PersistentQueryDispatcher},
    prelude::*,
};
use robot_minigame::goose::plant::GoosePlant;
use serde_json::{Value, json};
use simulation_minigame::{SimulationWorld, goose::builder::GooseAssembly};

pub fn support_vertices(
    _simulation: &SimulationWorld,
    assembly: &GooseAssembly,
    plant: &GoosePlant,
) -> Result<Vec<(RigidBodyHandle, Vec<Vector>)>, Box<dyn std::error::Error>> {
    let mut support = Vec::new();
    for geom in &plant.colliders {
        let local = simulation_minigame::goose::builder::native_source_pose(
            geom.local_position_m,
            geom.local_rotation_wxyz,
        )?;
        let vertices = if let Some(points) = &geom.vertices_local_m {
            points.clone()
        } else {
            let s = geom.half_extents_m.ok_or("Collider support absent")?;
            [-1.0, 1.0]
                .into_iter()
                .flat_map(|x| {
                    [-1.0, 1.0].into_iter().flat_map(move |y| {
                        [-1.0, 1.0]
                            .into_iter()
                            .map(move |z| [x * s[0], y * s[1], z * s[2]])
                    })
                })
                .collect()
        };
        support.push((
            assembly.body_handles[&geom.body],
            vertices
                .into_iter()
                .map(|v| {
                    local
                        * Vector::from_array(robot_minigame::basis::source_to_engine_vector(
                            v.map(|v| v as f32),
                        ))
                })
                .collect::<Vec<_>>(),
        ));
    }
    Ok(support)
}

pub fn boundary(
    simulation: &SimulationWorld,
    assembly: &GooseAssembly,
    plant: &GoosePlant,
    contract: &robot_minigame::goose::contract::GooseControlContract,
    support: &[(RigidBodyHandle, Vec<Vector>)],
) -> Result<Value, Box<dyn std::error::Error>> {
    let state = assembly.state(simulation)?;
    let observation = contract
        .observation(&state, [0.0; 3], [0.0; 18], 0.0)?
        .to_vec();
    let mut mass = 0.0_f64;
    let mut com = [0.0_f64; 3];
    let mut norm_error = 0.0_f32;
    let mut min_floor_y = f32::INFINITY;
    for body in &plant.bodies {
        let native = &simulation.world.bodies[assembly.body_handles[&body.name]];
        let center = native.position() * native.mass_properties().local_mprops.local_com;
        for i in 0..3 {
            com[i] += body.mass_kg * center[i] as f64;
        }
        mass += body.mass_kg;
        norm_error = norm_error.max((native.rotation().length_squared() - 1.0).abs());
    }
    for (handle, vertices) in support {
        let body = &simulation.world.bodies[*handle];
        for v in vertices {
            min_floor_y = min_floor_y.min((body.position() * *v).y);
        }
    }
    let mut self_depth = 0.0_f32;
    let mut fresh_pairs = 0_usize;
    let native: Vec<_> = simulation.world.colliders.iter().collect();
    for (a_index, (_, a)) in native.iter().enumerate() {
        let Some(first) = a.parent() else {
            continue;
        };
        let Some(first_name) = assembly
            .body_handles
            .iter()
            .find_map(|(n, h)| (*h == first).then_some(n))
        else {
            continue;
        };
        for (_, b) in native.iter().skip(a_index + 1) {
            let Some(second) = b.parent() else {
                continue;
            };
            let Some(second_name) = assembly
                .body_handles
                .iter()
                .find_map(|(n, h)| (*h == second).then_some(n))
            else {
                continue;
            };
            if first == second
                || plant.joints.iter().any(|j| {
                    (&j.parent == first_name && &j.child == second_name)
                        || (&j.parent == second_name && &j.child == first_name)
                })
                || plant.exclusions.iter().any(|p| {
                    (&p[0] == first_name && &p[1] == second_name)
                        || (&p[0] == second_name && &p[1] == first_name)
                })
                || !a.collision_groups().test(b.collision_groups())
            {
                continue;
            }
            let mut manifolds = Vec::<ContactManifold>::new();
            let mut workspace = None;
            let actual_a = simulation.world.bodies[first].position()
                * *a.position_wrt_parent()
                    .ok_or("Collider parent pose absent")?;
            let actual_b = simulation.world.bodies[second].position()
                * *b.position_wrt_parent()
                    .ok_or("Collider parent pose absent")?;
            <DefaultQueryDispatcher as PersistentQueryDispatcher<
                ContactManifoldData,
                ContactData,
            >>::contact_manifolds(
                &DefaultQueryDispatcher,
                &(actual_a.inverse() * actual_b),
                a.shape(),
                b.shape(),
                0.0,
                &mut manifolds,
                &mut workspace,
            )?;
            fresh_pairs += 1;
            for point in manifolds.iter().flat_map(|m| &m.points) {
                self_depth = self_depth.max(-point.dist);
            }
        }
    }
    let mut solve_point_depth = 0.0_f32;
    let mut impulse_points = 0_usize;
    for pair in simulation.live_contact_pairs() {
        for m in pair.solver_manifolds() {
            for p in &m.points {
                solve_point_depth = solve_point_depth.max(-p.dist);
                impulse_points += usize::from(p.data.impulse != 0.0);
            }
        }
    }
    Ok(
        json!({"joint_position_rad":state.joint_position_rad.to_vec(), "joint_velocity_rad_s":state.joint_velocity_rad_s.to_vec(),
        "root_position_source_m":state.root_position_world_m, "root_rotation_source_wxyz":state.root_rotation_world_wxyz,
        "zero_command_observation_65":observation, "com_engine_m":com.map(|v|v/mass),
        "upright":-state.projected_gravity[2], "floor_support_depth_m":(-min_floor_y).max(0.0),
        "fresh_self_depth_m":self_depth, "fresh_self_queries":fresh_pairs,
        "native_solve_point_depth_m":solve_point_depth, "native_nonzero_normal_impulse_points":impulse_points,
        "max_quaternion_norm_squared_error":norm_error, "jaw_pin_error_m":assembly.jaw_pin_error_m(simulation)}),
    )
}
