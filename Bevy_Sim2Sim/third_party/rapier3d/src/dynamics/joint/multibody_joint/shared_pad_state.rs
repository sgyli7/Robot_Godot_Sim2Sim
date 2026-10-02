//! Explicit CPU shared-pad experiment; history commits once after a successful native Tick.

use super::Multibody;
use crate::alloc_prelude::*;
use crate::math::Real;

/// Original per-region physical mechanics. Virtual mass is zero: physical pad
/// mass and its complete inertia have already been condensed into the foot.
#[derive(Copy, Clone, Debug)]
pub struct ExperimentalSharedPadMechanics {
    /// Original patch stiffness in N/m, before any point-count allocation.
    pub stiffness_n_m: Real,
    /// Original slide damping in N s/m.
    pub damping_n_s_m: Real,
    /// Original positive compression stroke in metres.
    pub travel_m: Real,
}

/// Candidate history produced at the two existing native algebraic barriers.
#[derive(Clone, Debug)]
pub struct ExperimentalSharedPadResult {
    /// Final algebraic candidate; committed history uses the integration barrier.
    pub compression_m: Vec<f64>,
    /// Final virtual slide velocity, in m/s.
    pub compression_velocity_m_s: Vec<f64>,
    /// Lower-bound reaction impulses, in N s.
    pub lower_stop_impulses_n_s: Vec<f64>,
    /// Upper-bound reaction impulses, in N s.
    pub upper_stop_impulses_n_s: Vec<f64>,
    /// Maximum unilateral velocity-equation residual, in m/s.
    pub complementarity_residual_m_s: f64,
    /// Maximum original stroke violation, in metres.
    pub max_travel_violation_m: f64,
    /// Bounded active-set work; not temporal integration steps.
    pub algebraic_iterations: usize,
    /// Two existing biased/relaxation barrier solves per native Tick.
    pub solve_calls: usize,
    /// Compression associated with the velocity used to integrate positions.
    pub biased_compression_m: Vec<f64>,
    /// Difference between the two algebraic candidates, in metres.
    pub barrier_compression_difference_m: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct SharedPadState {
    pub mechanics: Vec<ExperimentalSharedPadMechanics>,
    pub compression_m: Vec<f64>,
    pub pending: Option<ExperimentalSharedPadResult>,
    pub last_committed_epoch: Option<u64>,
    pub pending_committed: bool,
}

impl Multibody {
    /// Configure the immutable original mechanics once, starting uncompressed.
    /// This diagnostic does not qualify articulated owners or frictional contact.
    pub fn sim2sim_configure_shared_pads(
        &mut self,
        mechanics: Vec<ExperimentalSharedPadMechanics>,
    ) {
        assert!(
            self.ndofs() == 6
                && self.num_links() == 2
                && mechanics.len() == 6
                && self.shared_pad_state.is_none(),
            "one isolated six-DoF owner and six original patches required"
        );
        assert!(
            mechanics.iter().all(|p| p.stiffness_n_m.is_finite()
                && p.stiffness_n_m > 0.0
                && p.damping_n_s_m.is_finite()
                && p.damping_n_s_m >= 0.0
                && p.travel_m.is_finite()
                && p.travel_m > 0.0),
            "invalid original shared-pad mechanics"
        );
        self.shared_pad_state = Some(SharedPadState {
            compression_m: vec![0.0; mechanics.len()],
            mechanics,
            pending: None,
            last_committed_epoch: None,
            pending_committed: true,
        });
    }

    /// Committed physical compression history; never a fitted action target.
    pub fn sim2sim_shared_pad_compressions(&self) -> Option<&[f64]> {
        self.shared_pad_state
            .as_ref()
            .map(|s| s.compression_m.as_slice())
    }

    /// Read the current Tick's candidate and barrier measurements.
    pub fn sim2sim_shared_pad_result(&self) -> Option<&ExperimentalSharedPadResult> {
        self.shared_pad_state
            .as_ref()
            .and_then(|s| s.pending.as_ref())
    }

    /// Commit exactly once after native full-step observations confirm completion.
    /// A failed Tick leaves prior history intact and cannot be silently restarted.
    pub fn sim2sim_commit_shared_pad_tick(&mut self) -> Result<(), &'static str> {
        let observation = &self.sim2sim_observation;
        if !observation.valid || observation.full_step_dt().to_bits() != (0.02 as Real).to_bits() {
            return Err("shared-pad commit requires one completed native 20ms Tick");
        }
        let state = self
            .shared_pad_state
            .as_mut()
            .ok_or("shared-pad mechanics absent")?;
        if state.last_committed_epoch == Some(observation.epoch) {
            return Err("shared-pad history already committed this Tick");
        }
        let pending = state.pending.as_ref().ok_or("shared-pad solve absent")?;
        if pending.solve_calls != 2
            || pending.barrier_compression_difference_m > 2e-8
            || pending.max_travel_violation_m > 2e-8
        {
            return Err("shared-pad solve timing, barrier consistency or stroke failed");
        }
        state
            .compression_m
            .clone_from(&pending.biased_compression_m);
        state.last_committed_epoch = Some(observation.epoch);
        state.pending_committed = true;
        Ok(())
    }

    pub(crate) fn begin_shared_pad_tick(&mut self, dt: Real) {
        if let Some(state) = &mut self.shared_pad_state {
            assert_eq!(
                dt.to_bits(),
                (0.02 as Real).to_bits(),
                "shared-pad experiment requires 50Hz"
            );
            assert!(
                state.pending.is_none() || state.pending_committed,
                "previous shared-pad Tick was not successfully committed"
            );
            state.pending = None;
            state.pending_committed = false;
        }
    }
}
