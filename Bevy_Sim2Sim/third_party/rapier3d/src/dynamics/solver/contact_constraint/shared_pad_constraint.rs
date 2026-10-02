//! Shared physical K/C coordinates at the existing native algebraic barriers.

use super::GenericContactConstraint;
use crate::alloc_prelude::*;
use crate::dynamics::{ExperimentalSharedPadResult, Multibody};
use crate::math::{DIM, DVector, Real};

struct Row {
    contact: Option<(usize, usize)>,
    j: [f64; 6],
    wj: [f64; 6],
    virtual_j: Vec<f64>,
    gap: f64,
    old_impulse: f64,
}

fn solve_unilateral(a: &na::DMatrix<f64>, b: &na::DVector<f64>) -> (na::DVector<f64>, f64, usize) {
    let n = b.len();
    let mut lambda = na::DVector::<f64>::zeros(n);
    let mut active = Vec::<usize>::new();
    let mut iterations = 0;
    loop {
        let residual = a * &lambda + b;
        let next = (0..n)
            .filter(|i| !active.contains(i) && residual[*i] < -1e-9)
            .min_by(|i, k| residual[*i].total_cmp(&residual[*k]));
        let Some(next) = next else {
            break;
        };
        active.push(next);
        loop {
            iterations += 1;
            assert!(
                iterations <= 16 * n + 32,
                "shared-pad algebraic bound exceeded"
            );
            let reduced =
                na::DMatrix::from_fn(active.len(), active.len(), |i, k| a[(active[i], active[k])]);
            let rhs = na::DVector::from_fn(active.len(), |i, _| -b[active[i]]);
            let trial = reduced
                .lu()
                .solve(&rhs)
                .expect("singular shared-pad active set");
            assert!(
                trial.iter().all(|v| v.is_finite()),
                "nonfinite shared-pad impulses"
            );
            if trial.iter().all(|v| *v > 0.0) {
                lambda.fill(0.0);
                for (i, id) in active.iter().enumerate() {
                    lambda[*id] = trial[i];
                }
                break;
            }
            let step = active
                .iter()
                .enumerate()
                .filter(|(i, _)| trial[*i] <= 0.0)
                .map(|(i, id)| lambda[*id] / (lambda[*id] - trial[i]))
                .fold(1.0, f64::min);
            let mut leaving = Vec::new();
            for (i, id) in active.iter().enumerate() {
                let old = lambda[*id];
                if trial[i] <= 0.0 && old / (old - trial[i]) <= step + 1e-14 {
                    leaving.push(*id);
                }
                lambda[*id] += step * (trial[i] - old);
            }
            for id in &leaving {
                lambda[*id] = 0.0;
            }
            active.retain(|id| !leaving.contains(id));
        }
    }
    let residual = a * &lambda + b;
    let error = (0..n)
        .map(|i| {
            if lambda[i] > 1e-12 {
                residual[i].abs()
            } else {
                (-residual[i]).max(0.0)
            }
        })
        .fold(0.0, f64::max);
    assert!(
        lambda.iter().all(|v| v.is_finite() && *v >= 0.0) && error <= 2e-8,
        "shared-pad complementarity failed"
    );
    (lambda, error, iterations)
}

/// Couple actual signed native J/WJ and original shared compression/stops.
/// No pose write, independent rigid mass, extra physical mass or integration.
pub(crate) fn solve_shared_pad_block(
    constraints: &mut [GenericContactConstraint],
    jacobians: &DVector,
    solver_vels: &mut DVector,
    multibody: &mut Multibody,
    dt: Real,
) {
    assert_eq!(
        multibody.ndofs(),
        6,
        "isolated six-DoF shared-pad owner required"
    );
    let slot = multibody.solver_id as usize;
    let state = multibody
        .shared_pad_state
        .as_mut()
        .expect("shared-pad mechanics not configured");
    let h = dt as f64;
    let p = state.mechanics.len();
    assert_eq!(p, 6, "six original regions required");
    let diagonal: Vec<f64> = state
        .mechanics
        .iter()
        .map(|m| h * m.damping_n_s_m as f64 + h * h * m.stiffness_n_m as f64)
        .collect();
    let virtual_free: Vec<f64> = state
        .mechanics
        .iter()
        .zip(&state.compression_m)
        .map(|(m, q)| {
            -(m.stiffness_n_m as f64) * q / (m.damping_n_s_m as f64 + h * m.stiffness_n_m as f64)
        })
        .collect();
    let mut rows = Vec::new();
    for (ci, c) in constraints.iter().enumerate() {
        assert!(
            c.ndofs1 + c.ndofs2 == 6 && c.limit == 0.0 && !c.physical_normal,
            "shared-pad fixture rejects other topology or point springs/friction"
        );
        let (owner, fixed, offset) = if c.ndofs1 == 6 {
            (c.solver_vel1, c.solver_vel2, 0)
        } else {
            (c.solver_vel2, c.solver_vel1, 2 * c.ndofs1)
        };
        assert!(
            owner as usize == slot && fixed == u32::MAX,
            "mixed or moving shared-pad owners"
        );
        let binding = c.shared_pad.expect("shared-pad region binding absent");
        assert!(
            binding.patch_index < p && binding.axis_cosine > 0.5,
            "invalid shared-pad binding"
        );
        for pi in 0..c.num_contacts as usize {
            let jid = c.j_id + pi * 12 * DIM + offset;
            let j = core::array::from_fn(|i| jacobians[jid + i] as f64);
            let wj = core::array::from_fn(|i| jacobians[jid + 6 + i] as f64);
            let mut virtual_j = vec![0.0; p];
            virtual_j[binding.patch_index] = binding.axis_cosine as f64;
            rows.push(Row {
                contact: Some((ci, pi)),
                j,
                wj,
                virtual_j,
                gap: c.shared_nominal_gaps[pi] as f64
                    + binding.axis_cosine as f64 * state.compression_m[binding.patch_index],
                old_impulse: c.normal_part[pi].impulse as f64,
            });
        }
    }
    let contacts = rows.len();
    assert!(contacts <= 96, "shared-pad contact capacity exceeded");
    for upper in [false, true] {
        for i in 0..p {
            let mut virtual_j = vec![0.0; p];
            virtual_j[i] = if upper { -1.0 } else { 1.0 };
            rows.push(Row {
                contact: None,
                j: [0.0; 6],
                wj: [0.0; 6],
                virtual_j,
                gap: if upper {
                    state.mechanics[i].travel_m as f64 - state.compression_m[i]
                } else {
                    state.compression_m[i]
                },
                old_impulse: 0.0,
            });
        }
    }
    let mut free: [f64; 6] = core::array::from_fn(|i| solver_vels[slot + i] as f64);
    for row in &rows[..contacts] {
        for i in 0..6 {
            free[i] -= row.wj[i] * row.old_impulse;
        }
    }
    let n = rows.len();
    let response = na::DMatrix::<f64>::from_fn(n, n, |i, k| {
        (0..6).map(|d| rows[i].j[d] * rows[k].wj[d]).sum::<f64>()
            + (0..p)
                .map(|d| rows[i].virtual_j[d] * rows[k].virtual_j[d] / diagonal[d])
                .sum::<f64>()
    });
    // Keep the measured native response; do not replace it with a rigid-body diagonal.
    let mut a = response;
    for i in 0..n {
        a[(i, i)] += (1e-7 * a[(i, i)]).max(1e-12);
    }
    let b = na::DVector::<f64>::from_fn(n, |i, _| {
        (0..6).map(|d| rows[i].j[d] * free[d]).sum::<f64>()
            + (0..p)
                .map(|d| rows[i].virtual_j[d] * virtual_free[d])
                .sum::<f64>()
            + rows[i].gap / h
    });
    let (lambda, residual, iterations) = solve_unilateral(&a, &b);
    let mut compression_velocity = virtual_free;
    for (i, row) in rows.iter().enumerate() {
        if let Some((ci, pi)) = row.contact {
            constraints[ci].normal_part[pi].impulse = lambda[i] as Real;
            for d in 0..6 {
                free[d] += row.wj[d] * lambda[i];
            }
        }
        for d in 0..p {
            compression_velocity[d] += row.virtual_j[d] * lambda[i] / diagonal[d];
        }
    }
    let compression: Vec<f64> = state
        .compression_m
        .iter()
        .zip(&compression_velocity)
        .map(|(q, v)| q + h * v)
        .collect();
    let violation = compression
        .iter()
        .zip(&state.mechanics)
        .map(|(q, m)| (-q).max(q - m.travel_m as f64).max(0.0))
        .fold(0.0, f64::max);
    assert!(
        violation <= 2e-8 && compression.iter().chain(&free).all(|v| v.is_finite()),
        "shared-pad stroke or velocity failed"
    );
    let (calls, biased, difference) = if let Some(previous) = &state.pending {
        (
            previous.solve_calls + 1,
            previous.biased_compression_m.clone(),
            compression
                .iter()
                .zip(&previous.biased_compression_m)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max),
        )
    } else {
        (1, compression.clone(), 0.0)
    };
    state.pending = Some(ExperimentalSharedPadResult {
        compression_m: compression,
        compression_velocity_m_s: compression_velocity,
        lower_stop_impulses_n_s: lambda.rows(contacts, p).iter().copied().collect(),
        upper_stop_impulses_n_s: lambda.rows(contacts + p, p).iter().copied().collect(),
        complementarity_residual_m_s: residual,
        max_travel_violation_m: violation,
        algebraic_iterations: iterations,
        solve_calls: calls,
        biased_compression_m: biased,
        barrier_compression_difference_m: difference,
    });
    for d in 0..6 {
        solver_vels[slot + d] = free[d] as Real;
    }
}
