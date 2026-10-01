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

Original object textures are cached with byte receipts. Full background MDL and
texture dependency closure, artifact-specific license mapping, task visual
registration, and full source/native task rollouts remain unfinished.
