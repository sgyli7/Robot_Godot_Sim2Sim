# Original G1 task objects in the native physics owner

T1 AGILE and T2 Homie accept an optional `task_objects` startup configuration.
Each owner creates its matching dynamic objects before its first integration:
apple/plate for T1, brown box/blue bin for T2. The same Rapier world contains the
free robot, floor and objects. Only immutable completed object frames cross the
worker display slot. These frames are render/independent-acceptance truth and do
not enter Qwen, VLA or body-controller self-state observations. Reset recreates
the scene and clears the old object frame with the old robot frame.

`unitree_g1_task_asset_query.py` references the original default prims, overrides
the original task scales, and queries PhysX cooked convex pieces and mass while
the timeline remains stopped. It performs zero integrations. The isolated
installed source runtime is 6.1.0-rc.26; this does **not** reproduce T1's matching
6.0.0-dev2 runtime or prove either full source task. Bootstrap uses the existing
isolated ARM64 libgomp preload/EULA environment. Kit normal cleanup aborted after
a successful saved query; the final invocation uses the installed documented
`close(skip_cleanup=True, exit_code=...)`, preserving failure/success status.
The initial bootstrap failures and cleanup abort remain in the evidence bundle.

`unitree_g1_task_asset_export.py` is a CPU OpenUSD conversion. It preserves every
cooked convex part, original polygon winding, task scale, queried COM/principal
inertia, material, and authored CCD setting. Original asset hashes are checked
before and after conversion. The box retains its analytic cube, and the bin
retains its opening. There is no whole-object convex replacement, artificial
attachment, fixed destination body, sleeping or added damping.

| Object | Native mass (kg) | Convex pieces | Friction | Restitution | CCD |
|---|---:|---:|---:|---:|---|
| Apple | 0.09897653 | 256 | 0.8 | 0 | false |
| Plate | 0.5 | 256 | 2 | 0.1 | true |
| Brown box | 0.1000000015 | 1 | 5 | 0 | false |
| Blue bin | 421.3587952 | 16 | 1 | 0 | false |

The heavy bin mass comes from the actual original source density and scale;
it is not replaced with a guessed light mass or an anchor.

## Frozen native contact diagnosis

The original thin plate failed with Rapier's default contact clustering. A
one-variable comparison held geometry, mass, materials, 50 Hz dt, one solver
integration and four non-integrating PGS sweeps constant. More PGS sweeps,
reconstructed per-part hull boundaries, restoring the authored CCD, and disabling
contact recycling did not solve the failure. Disabling only contact clustering
did. The failed default merged 256 raw plate manifolds into one cluster with four
solver points; the successful case retained per-part solver contacts. This
isolates clustering in this native setup, not a general claim about all Rapier
contact configurations or the internal reducer's exact failure mechanism.

The public G1 task scene constructor now disables clustering **only in its owner
world**. Shared physics, MicroDuck, and G1 configurations without task objects
keep their existing settings. Contact recycling stays enabled. No formal
frequency, temporal subdivision, source gains or physics acceptance threshold
was changed.

At source `8c90e604993a02cc0ea565e6775ab234c4a746a6`:

- The 500-Tick default control still failed. The production task constructor
  passed both a 5 mm support-gap start and a 0.3 m drop. All four dynamic objects
  remained continuously in supporting contact below 0.02 m/s and 0.1 rad/s for
  the last 485 and 455 Ticks, respectively. Bin-opening and box-solid probes passed.
- The actual T1 background owner ran the free 53-link G1 plus apple and plate
  for 1500 native inferences/integrations at 50 Hz: 30.0 simulated seconds,
  30.002067 wall seconds, ratio 0.9999311, zero missed control deadlines or pending
  Ticks. Maximum root drift was 0.0840065 m with no fall/source-limit failure.
  Both objects met the support/static thresholds for the last 1482 observed
  consecutive Ticks. Median/P95/max boundary duration: 4.004/4.110/7.700 ms.
- The owner contained 56 bodies, 55 colliders and 52 multibody joint handles.
  Robot and object snapshots had episode 21 and matching completed Tick identity.
  The owner closed after its explicit 1500-Tick command expired.
- Workspace library regression passed 155 tests, with 27 explicit diagnostics
  ignored. Model-dependent tests above were invoked separately.

Evidence, including all failing controls, source query, derived asset bytes,
exact command/binary identities, and complete trajectories:
`/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/g1_task_objects_contact_001/manifest.json`.
The same-world standing placement is deliberately separated on a flat diagnostic
floor. No robot/object grasp or full task was executed. This is not T1/T2
qualification, matched full source rollout, source background parity, or finished
science-station integration.

## Reproduction

Use frozen source query SHA
`9e1f6b0d9d45cf34c84f71a8ccf4bf71689263b31b22547f67cd616956920d55`
and native object definition v2 SHA
`c4ed318c8a8242a2ff7c69bec816f9ca1b870af34be578618407132dc8d26988`.
Model and original USD bytes remain in `/home/ethan/models/unitree_g1`, outside
Git. The frozen command receipts include exact executable hashes and environment
inputs; use new output paths when repeating a diagnostic.

```bash
flock /tmp/sai-g1-cargo.lock env \
  CARGO_TARGET_DIR=/home/ethan/Projects/Sai_Lab/Bevy_Sim2Sim/target \
  cargo test --locked --offline -p simulation_minigame --lib --no-run -j 2
```

Invoke `g1::task_objects::tests::real_native_task_object_contact_diagnostic`
with `--exact --ignored --nocapture`. Required variables:
`G1_TASK_OBJECTS_DEFINITION`, `G1_TASK_OBJECTS_SHA256`,
`G1_TASK_OBJECTS_OUTPUT`, `G1_CODE_COMMIT`. Production constructor selection:
`G1_TASK_OBJECTS_DIAGNOSTIC_CONTACT_MODE=task_owner`.
Bounded starts: `near_support_5mm` or `drop_0_3m`; Tick budgets: 150 or 500.
`task_owner` requires at least 100 consecutive supported/static Ticks.

For the real clock use
`g1::worker::tests::real_agile_owner_standing_clock_diagnostic`, frozen
`task_owner_world_20261001_2200/config.json` and its SHA
`4fbbcb8d3ab34b43e53fa62d8d6b3805c77a6f12aaeeda4893c045e3d3888e99`.
Set `G1_AGILE_CONFIG`, `G1_AGILE_CONFIG_SHA256`, a new
`G1_AGILE_WORKER_OUTPUT`, and `G1_CODE_COMMIT`.

Original object textures and native visual registration are documented in
`g1_task_visual.md`. Full background MDL and texture dependency closure,
artifact-specific license mapping, and full source/native task rollouts remain
unfinished.

## Original T1 shelf migration

The T1 task constructor can now add the original invisible shelf support in the
same owner world. The source Arena revision is
`8b4a3a47fc53de23e8205089d71109a2e2348acd`. A uniform source-world translation of
`[0,0,0.795]` puts the original ground at native Z=0, robot root at
`[0.25,0.08,0.795]`, and shelf center at `[0.62,0,0.745]`. Shelf size remains
`[0.8,1.5,0.04]`, with the source 5 mm contact offset recorded. The source shelf
has no authored physics material; native friction 0.5 is an explicit candidate
choice. The 16 original hand/finger collision bodies receive dynamic friction
5 with the max combine rule. Rapier's single friction coefficient does not
represent the source static/dynamic pair 6/5 exactly.

The original startup geometry overlaps the shelf by about 42 mm at
`pelvis_contour_link` and 6 mm at each hip-pitch link. These are measured initial
overlaps, rather than a reason to disable real collisions. The original-source
layout exposes a native standing failure that the separated flat-floor test did
not cover. The background owner stops on left ankle roll crossing the unchanged
source limit at Tick 1099 (21.98 simulated seconds). There was no observed fall
before that guard. All 1098 valid background frames and measurements exactly
match the corresponding offline native replay.

Bounded one-variable controls retain 50 Hz, one integration per Tick, four
non-integrating PGS sweeps, original gains and model bytes:

| Native shelf case | Result |
|---|---|
| Original contacts; repeated cold runs | Limit guard at Tick 1099 |
| Only floor friction 1.0 instead of 0.5 | Limit guard at Tick 1175 |
| Only contact recycling disabled | Limit guard at Tick 1125 |
| Original cooked pelvis convex topology | Limit guard at Tick 1200 |
| Original cooked topology for all 38 mesh colliders | Limit guard at Tick 1200 |
| Robot/shelf pairs excluded, diagnostic only | Completed 1500 Ticks |

Collision masking is not a production fix or task success. The copied cooking
topology comes from a zero-integration 6.1 query; its runtime mismatch with T1
6.0 remains explicit. Source evidence and complete failed trajectories are
frozen at
`/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/g1_t1_shelf_diagnosis_001/manifest.json`.

The matching T1 reference uses the original ARM64 image manifest
`sha256:9bc527c60c1cdde532b05859b955e866e6115ac5bf2fe22eaa24daa6ea8783e6`
for `6.0.0-dev2`, whose actual build is
`6.0.0-rc.22+release.33481.407f3ea1.gl`, and Lab revision
`e57379c634b42db5a0fe9f754341be6e2a7c7c43`. Dependencies are installed only in
the isolated owned container. Its original Torch 2.10.0+cu128 GPU JIT fusion
fails on GB10 with an NVRTC architecture error. The reduced reference harness
disables only this process-local GPU fusion optimization, retaining the original
operators. The original quaternion transform's 64-sample CPU/GPU comparison
passed with maximum component error 2.384e-7. This checks that transform, not
every computation in the complete policy pipeline.

Both fresh source PhysX standing processes completed 1500 control Ticks and
6000 physics steps at the original 200/50 Hz. No-shelf maximum root drift was
0.100234 m; original-shelf maximum drift was 0.024930 m. The source shelf case's
left ankle roll stayed in `[-0.016866,0.002709]` rad, inside its original limits.
The source/native initial 43 joint angles and first cold inference's 43 targets
are identical. Initial body position error across all 53 bodies is at most
2.682e-7 m; mass error is at most 2.384e-7 kg, and COM positions match exactly.
These comparisons narrow the failing candidate to migrated dynamics. They do
not identify a particular solver mechanism by themselves.

The original SDK reports static joint friction 0.03 on 35 joints and zero on
the eight hip/knee joints; dynamic and viscous joint friction are zero. A
source-only diagnostic setting all three SDK friction properties to zero still
completed 1500 standing Ticks with the shelf (maximum drift 0.024919 m). That
omission alone therefore does not reproduce the failure in the reference
200 Hz setup. The original reference configuration and native production
configuration remain unchanged.

A separate native oracle diagnostic executes the byte-bound original 1500-Tick
target sequence through real native motors and contacts, with zero policy
inferences. It preserves the original native shelf layout and apple/plate.
At source `bbe85238f35f5084e95370be73b4732698139395`, it stopped after 974
motor updates and integrations: upright cosine 0.483894 crossed the unchanged
fall guard, root height was 0.513734 m, and maximum horizontal drift was
0.469336 m. The final state is explicitly invalid for runtime admission.
Failure can occur without recurrent policy feedback. This experiment is not
a production controller, and does not prove identical source/native scenes:
the reduced reference omitted props while the native owner retained them.

Matching source setup, startup failures, both original standing traces, SDK
property comparisons, friction control, and failed native oracle replay are
frozen separately at
`/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/g1_t1_source_physx_001/manifest.json`.

The harness `unitree_g1_t1_source_shelf.py` is a reduced ground/robot/shelf
standing comparison. It intentionally records `qualified=false` and
`source_task_rollout_verified=false`. Original background, props, cameras and
VLA are absent. Initialization can advance the source simulator; recorded
physics-step counts start after reset. These source diagnostics do not change
the native formal 50 Hz frequency or prove full T0, T1 grasp/release, or T2
walking while carrying.

## Matched T1 collision query

The isolated original 6.0.0-dev2 image now completed the same stopped-timeline,
zero-integration asset query in its actual build
`6.0.0-rc.22+release.33481.407f3ea1.gl` / OpenUSD 25.11. All query outcomes are
checked separately from process exit status. The standalone image has no pip
`isaacsim` distribution metadata and its public `SimulationApp.close` lacks the
6.1 `exit_code` parameter; both are handled explicitly. The first incompatible
query and its error receipt remain preserved.

This matching T1 build produces 253 apple convex pieces and mass
0.09702564776 kg; the earlier 6.1 data had 256 pieces and mass 0.09897653013 kg.
Apple extrema and COM also differ. The plate still has 256 pieces / 0.5 kg.
The new alternate diagnostic definition is
`native_task_objects_t1_60_diagnostic_v2.json`, SHA-256
`19eb60783008e3f08d82a1cf402c590395df1e98c4c089fb1247f8ed7d9a88a0`.
Original USD/weights and the prior object definition are unchanged. Only T1
objects are instantiated with this alternate definition; T2 is not switched to
the 6.0 query's box/bin data and its matching runtime remains unverified.

With unchanged source action chunks, predictive limits and sixteen sweeps, the
matched T1 objects completed all 160 native integrations. The apple was lifted
and released onto the plate, but its final linear/angular speed was
0.07430 m/s / 2.92203 rad/s. Matching the source cooking therefore does not by
itself establish released-object stability or autonomous grasping.

Independent contact samples also identify `other_task_kind` from actual
owner-local body handles. This read-only annotation distinguishes the plate
from the shelf without assuming handle indices and never enters model input.
The ignored `real_static_source_action_release_window_diagnostic` runs the
unchanged 160 saved action frames, then a separately labelled, finite 200-Tick
body-target hold in that same physical world. It performs no new VLA inference,
coordinate write, expired-chunk extension or observation restamp. It reports
release/support/velocity windows but does not verify whole-object containment
or grant task/runtime safe-hold qualification.

The actual matched-object release diagnostic completed 360 native body
inferences and integrations, with zero VLA calls and zero missed deadlines.
The same last source command was explicitly held for 200 Ticks after its 160
saved action frames. Released, plate-supported, slow motion and standing
conditions held for the final 3.3 continuous seconds. The first 160 physical
state/action rows match the original background replay at every original f32
bit pattern, with f64 simulation times exactly equal. The new read-only task
contact identity does not change that trajectory.

`unitree_g1_static_placement_audit.py` independently evaluates every collision
vertex against the actual moving plate's original convex XY outer footprint,
with a disclosed vertical-prism diagnostic target. Positive plate support,
no robot contact, speed limits, standing and 101 consecutive 50 Hz samples are
required separately. This trace passed for 3.3 seconds with a minimum geometric
margin of 0.01393 m. In-memory controls that move the apple one metre outside or
remove a physical Tick fail the same audit. The target rule is explicitly
not yet frozen for the formal ten-episode suite, and source-action execution
plus a diagnostic hold is not autonomous task success. Native RGB grasping,
continuous live waiting/holding, T1/T2 scores and station integration remain open.
