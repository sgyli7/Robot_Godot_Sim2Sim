# T1 AGILE native runner candidate

`simulation_minigame::g1::agile_runner::AgileRunner` owns one free-root G1 and
its floor in one Rapier world. Its separate config requires definition/ORT byte
identities, the AGILE model path, episode, source Z-up root pose, both contact
frictions and explicit finite `default_positions` in the 43-joint WBC order.
Those same frozen task defaults initialize physical q and policy-relative q.
There is no implicit Homie model, history, pose or lower-body gain substitution.

The only backend is the unqualified native ForceBased candidate. All 43
parameters come from `robot_minigame::g1::agile::parameters`, including source
armatures. A non-integrating initial motor installation is separate from runtime
counts. Every accepted boundary runs one real recurrent inference, one complete
43-motor target update and one existing `step_with_torques(&[])` at 20 ms. It
adds no external PD, hidden substeps or frequency change. Motor-update and empty
external-torque counts are not actual motor effort measurements.

The native generalized q/dq, root quaternion and current root motion are read
through the shared assembly seam. Angular velocity is expressed in actor/root
link axes after the proper source/engine basis conversion. COM identifies the
velocity's origin; it does not name the principal-inertia axes. Angular velocity
is the same at every point of a rigid body, so no principal-axis rotation is
applied. Projected gravity is the inverse root-quaternion rotation of source
`[0,0,-1]`. Input quaternion/state snapshots are never normalized or repaired.
The shared measurement's linear `root_velocity_source` keeps its existing free
root-frame-origin meaning; it is not renamed or fabricated as a COM linear sensor.
The 80 observation values, 12 raw action feedback, h/c shapes, 43 target ordering
and pinned model bytes retain the already-audited AGILE contract.

Public frame/state/measurement/count methods provide read-only evidence.
Snapshot or operational errors latch terminal failure. Caller guards run before
and after inference. A rejected second guard can leave recurrence advanced with
zero integration; this owner must be cold rebuilt and has no resume/reset API.
A failed backend integration without a validated completed boundary is not
relabeled as a fresh completed frame.

The ignored stand diagnostic requires a frozen JSON matching
`AgileRunnerConfig`, including static-task shoulder defaults: indices 16/17 are
`0.25/0.5`, and 30/31 are `-0.25/-0.5`. It uses those explicit upper defaults,
zero navigation and the source task's 0.75 m height command for at most 150 ticks.
It stops on any runtime error, measured source position limits beyond 0.001 rad,
pelvis height below 0.35 m, upright cosine below 0.5 or external torque leakage.

After the main agent integrates the two parameterized assembly seams and compiles
from the fixed approved source/target, invoke a bounded owner-controlled run:

```bash
G1_AGILE_CONFIG=/absolute/frozen_agile_config.json \
G1_AGILE_OUTPUT=/absolute/new_agile_stand_receipt.json \
G1_CODE_COMMIT=FULL_COMPILED_SOURCE_COMMIT \
timeout 60 cargo test --locked --offline -p simulation_minigame --lib \
  g1::agile_runner::tests::real_agile_static_open_arm_stand_diagnostic -- \
  --ignored --exact --test-threads=1 --nocapture
```

The output parent must exist; `create_new` refuses replacement. Optional
`G1_AGILE_CONFIG_SHA256` verifies the JSON bytes too. The execution wrapper must
bind the compiled source and actual test executable hash. Receipts retain the
original configuration, source/model identities, initial frame/measurement,
all completed steps, actual inference/motor/integration counts and failures.
Failure evidence is saved before the test returns nonzero. The shared USD/body
export identity and the T1 AGILE policy revision are recorded separately.

The owner compiled and its three pure guards passed. The first actual run at
`2381619` stopped after 119 integrations (2.38 s): left ankle roll reached
0.307715 rad against its 0.261800 rad upper limit. Horizontal drift was 0.742738 m
and pelvis height had fallen to 0.444884 m. This is a failed standing candidate;
the limit is a late symptom, and has not been widened. Replaying all 119 actual
inputs through the original upstream recurrent implementation gave a maximum
target difference of 2.03e-6 rad, including retained quaternion rounding.

The ignored diagnostic also accepts `G1_AGILE_DIAGNOSTIC_PGS_PASSES=4` for a
single bounded comparison. It changes only Rapier's internal constraint sweeps
before the same integration; the ordinary owner still selects one pass. Every
receipt records the actual value. `num_solver_iterations`, which splits the
timestep in this Rapier version, stays one, as do CCD substeps. This comparison
does not change gains, armatures, defaults, frequency, limits or policy history.

Even a completed 150-tick budget remains
`qualified=false`: standing duration, physical migration, materials, real-time
pacing, grasping and visual task success require separate actual evidence.
