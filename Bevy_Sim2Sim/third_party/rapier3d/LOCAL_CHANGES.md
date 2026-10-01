# Local patches

The published rapier3d 0.35.3 crate is vendored unchanged except for files carrying a local modification notice. UPSTREAM_SOURCE.json records the published archive and all 218 original file hashes. The upstream crate reports a dirty VCS state; the published crate bytes are the baseline. The default-disabled sim2sim-observation feature adds read-only diagnostics without changing solver formulas, constants, thresholds, physical update order, mass computation, or temporal subdivisions.

Stage A records actual pre-inverse-mass inertial/gravity projections, literal guard flags, and original generic-joint impulses with a separate articulation dry-friction subset. Its partial getter semantics are retained. Stage B independently audits the native cached contact selection before integration and reconciles every final generated original normal/two-tangent row side after restitution. The separate contact-complete getter requires both partial validity and true contact coverage. Neither getter qualifies complete BAM external load or MuJoCo bias equivalence.

The same read-only stage also records `Jᵀ` of the exact user force/torque queued on each body. The paired-force diagnostic uses this to distinguish 14-channel actuator mapping from moving-state integration differences; it is not an actuator history API or a BAM admission signal.

Observation publication supports only the serial, single-temporal-step, single-CCD-step backend. Native parallel physics remains available but both observation getters return None. Same-owner contact sides retain both already-signed original J contributions. Contact impulses use native final total_impulse, including this step's applied warmstart, never old seeds added again. No rolling/torsional contact compatibility is asserted. Internal test-only malformed diagnostics do not qualify physical behavior.

The separate default-disabled `sim2sim-limit-row-trace` feature depends on `sim2sim-observation`. On the supported serial backend, it copies each internal one-sided limit row's coordinate, generalized velocity, RHS, and accumulated impulse at three existing stage barriers: after the biased solve, after position integration, and after the unbiased solve. It is read-only and has no production BAM consumer. This trace establishes why the final generic-row impulse can be zero after a nonzero position-correction impulse; it does not establish MuJoCo limit-force equivalence.

The separate default-disabled `sim2sim-plain-mass-probe` feature is a causal diagnostic, not a production solver option. Its per-multibody switch defaults false. Only when the paired-force probe explicitly enables it, the free-acceleration solve uses the already assembled plain `inv_augmented_mass` rather than `acc_inv_augmented_mass`, and skips the implicit-Coriolis energy guard for that solve. The observation records the selection and actual guard flags; the paired probe reports the matrix used. The feature-off binary retains the original solver path; the feature-on binary with its switch off reproduces the prior paired-force reports exactly after removing the extra status fields. See `docs/paired_mass_causal_probe.md` for the frozen robot-only comparison.

## Fixed-scene CCD cache maintenance

The runtime pipeline preserves the upstream fixed-target cache across force, torque, and wake-only changes. Collider modification/removal or a rigid-body position, collider membership, type, or enabled-state change invalidates the cache, including steps with no active CCD sweep. This avoids repeated fixed-geometry collection and prevents a later sweep from reusing stale geometry. The sweep algorithm, solver equations, temporal subdivisions, and collider insertion order are unchanged.

`src/dynamics/ccd/ccd_solver.rs` adds an internal invalidation method and test-only build counter; `src/pipeline/physics_pipeline/substep.rs` classifies scene changes and invalidates before the CCD branch. Regression tests cover torque-only reuse, inactive-CCD collection avoidance, and geometry invalidation before a later sweep. These runtime changes apply independently of the observation features.

The empty local `[workspace]` in `Cargo.toml` permits the vendored backend's regression tests to run independently from the parent workspace. It does not change its dependency or feature defaults.

## Explicit predictive multibody stops

`MultibodyJoint::set_predictive_limits_enabled` defaults off on every joint.
When explicitly selected, each limited scalar coordinate emits two native
unilateral rows enforcing `min <= q + dt*v <= max` at the original bounds.
They use the existing augmented mass, drive, damping, solver and one integration;
no coordinate overwrite, extra time step or stop-position margin is introduced.
Relative CFM `1e-7` (absolute floor `1e-12`) regularizes redundant rigid stops.
The rows keep their velocity-cap reference in the final stabilization pass.
The existing single-row path is retained when this option is off. The source-limit
diagnostic and predictive mode reject simultaneous selection on one joint.

The bounded Goose development probe selects this mode with `--limit-mode predictive`.
No production robot assembly enables it yet. Primitive regressions in the
simulation crate check unchanged default crossing, one-step stopping, unchanged
interior motion, reachable endpoints, and unforced kinetic-energy dissipation.

With the existing default-disabled `sim2sim-limit-row-trace` feature, the same
three staged barriers also copy original impulse-joint rows with a multibody
second block and an empty/fixed first block. For same-owner loops the copied
block is the native relative Jacobian. Raw lambda and signed generalized
impulse are observed before position integration and after relaxation.
Internal limit trace now recognizes both signed unit Jacobians, including
predictive lower-stop rows. This instrumentation only reads solver state.
