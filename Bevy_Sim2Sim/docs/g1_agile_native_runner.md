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
zero navigation and the source task's 0.75 m height command for 150 ticks by default.
`G1_AGILE_DIAGNOSTIC_TICKS` permits an explicit finite budget from 1 to 1500.
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

The bounded four-pass comparison completed 150 ticks, with maximum drift
0.059584 m, final pelvis height 0.754199 m and upright cosine 0.997673.
Both initial states, all actuator parameters and the first real inference were
identical to the one-pass failure. This supports insufficient constraint
convergence as a contributor to the failure at fixed 20 ms; it does not yet
prove source physics parity.

This task owner now selects four Rapier internal constraint sweeps before the
same integration. The shared world and Homie defaults are unchanged. The
ignored diagnostic retains `G1_AGILE_DIAGNOSTIC_PGS_PASSES=1` as a control. Every
receipt records the actual value. `num_solver_iterations`, which splits the
timestep in this Rapier version, stays one, as do CCD substeps. This comparison
does not change gains, armatures, defaults, frequency, limits or policy history.

The subsequent actual offline 1500-tick run at `793d4a9` completed 30 seconds
without a fall or source-limit violation. Its first 150 steps exactly matched
the preceding four-pass run. Maximum drift was 0.080320 m, minimum pelvis
height 0.745768 m and minimum upright cosine 0.996036. The owner now enforces
these original joint position bounds before inference and after integration;
an error latches failure without relabeling the last validated frame.

`AgileWorker` uses the existing bounded physics scheduling through a distinct
`AgileCommand` / `AgileStep` type. It retains AGILE defaults, gains and recurrence;
sharing the clock does not convert commands through Homie. The original worker's
21 deterministic lifecycle guards and the new typed command guard passed.

The actual background-owner diagnostic at `aa297a3` completed 1500 real model
inferences, motor updates and integrations. Active simulation/wall time was
30 / 30.001458 seconds (ratio 0.999951), with zero missed control deadlines or
pending ticks. All 1500 completed boundaries were observed; one initial display
publication was replaced. The final explicit pause closed the owner. The same
0.080320 m drift agrees with the offline run. Invoke the ignored test
`g1::worker::tests::real_agile_owner_standing_clock_diagnostic` with the frozen
`G1_AGILE_CONFIG`, `G1_AGILE_CONFIG_SHA256`, new `G1_AGILE_WORKER_OUTPUT` and
compiled `G1_CODE_COMMIT` in a finite process wrapper.

Immutable evidence is under
`/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/g1_agile_convergence_001/manifest.json`
and `g1_agile_owner_clock_001/manifest.json`. The original failed one-pass run and
an invalid first replay comparison remain preserved with their explicit review.
These flat-floor standing diagnostics remain `qualified=false`: original task
materials/layout, full source physics parity, walking, real grasp/release and
visual task success still require their own actual evidence.


The development-only `StaticStartupWorker` also has a separate typed
`StaticMemoryPlaceGoal`, admitted only after the two unchanged N1.7 chunks
and all 250 traditional static transfer ticks (the actual 390-Tick boundary).
This goal explicitly preserves the 140-Tick RGB origin and 390-Tick observation,
rigid grasp/static target assumptions and public geometry identity. It does not
claim current object detections. Mixed/reset episodes, changed goals, invalid
rigid transforms, old timestamps and foreign geometry are rejected.

The bounded placement uses published collision vertices and self IMU to derive
an initial lowering displacement; it changes only the left arm and left hand.
It lowers for 100 ticks, opens the original seven hand targets toward zero for
50 ticks, retracts 4 cm over 50 ticks, and physically settles for 125 ticks.
AGILE, original force/gain/material settings and one 20 ms integration per tick
remain unchanged. Whole-path geometry preparation belongs outside the physical
Tick; the owner checks one increment per update. No contact or world truth drives
these commands. The goal permits at most 8 seconds of disclosed RGB memory age;
this finite diagnostic is not an unbounded perception fallback.

The ignored development test
`g1_station_environment::static_transfer_diagnostic::saved_memory_static_place_in_original_world`
accepts a hash-bound `G1_STATIC_TRANSFER_FIXTURE` and
`G1_STATIC_TRANSFER_FIXTURE_SHA256`. Its saved camera/action wall timestamps
are explicitly refreshed only for offline mechanical testing, preserving the
origin/current memory age. It executes 715 real native/WBC ticks and records
all steps for the independent active-contact placement audit. Passing the Rust
execution test alone is not proof of release, support, a two-second placement
window, fresh model execution, Qwen task execution, or formal task qualification.


The first single-case offline mechanical placement at `230de1b` completed all
715 actual native/WBC ticks. Its first 390 body records exactly matched the
preceding live image transfer. The unchanged independent v4 audit found 4.0
continuous seconds with every apple collision vertex in the plate footprint,
positive current plate support, no actual robot contact, low linear/angular
velocity and continuous standing. All 125 settling samples passed those rules;
final speeds were 1.84e-6 m/s and 5.41e-5 rad/s. This is one saved-original
P1/seed42 mechanical diagnostic, with zero new VLA/Qwen requests or images;
it is not a formal autonomous task trial or an 8/10 result. Raw steps and the
independent result are retained in scratch case 0363 for immutable sealing.


The separate ignored test `saved_memory_static_place_with_native_four_passes`
uses `spawn_static_native_four_passes_comparison` only for one fixed-input
mechanical comparison. It selects the existing native four PGS passes before
clock startup and retains the same existing predictive bounds. Normal/static
sixteen-pass diagnostic constructors remain unchanged; there is no runtime
parameter selection or fallback. Reset recreates the selected comparison.
The hash-bound saved 16-pass reference images/actions/memory are explicitly
reused, with offline-only wall age refresh. Their geometry is a fixed input,
not newly measured four-pass RGB. Cadence, forces/materials and model bytes are
unchanged; the comparison must not count as autonomous or formal qualification.
Failed owner snapshots preserve actual integration/inference counters if a
post-integration guard refuses the last completed record. No parameter grid is
performed. The frozen trajectory and independent physical audit, not the test
exit status alone, determine what this comparison shows.
