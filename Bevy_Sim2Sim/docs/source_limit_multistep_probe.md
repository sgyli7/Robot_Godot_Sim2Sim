# Frozen head_roll limit: same-world eight-step diagnostic

Date: 2026-09-28. This experiment advances **one** native `SimulationWorld` and **one** frozen MuJoCo `MjData` for eight consecutive `1/60 s` steps. Both start with `head_roll` at its upper limit +`0.02 rad`, all generalized velocities zero, and actuator torque zero throughout. The two robot forms are evaluated separately. No world is rebuilt between steps. This is a diagnostic of the default-off source-style limit row, not a production selector or a complete Sim2Sim acceptance.

The native probe is `source_limit_continuous_probe`, compiled only with `sim2sim_source_limit_probe` and requiring the explicit `--source-limit-continuous-diagnostic` switch. It accepts only the SHA-bound scratch definitions and frozen qpos bytes; the qvel input must have source `nv` entries, all zero. The native report records every completed integration/observation epoch, head-roll pre/post q and qdot, final row impulse and equivalent force, and contact row and pair impulses. The feature-off production path is unchanged. The companion `crates/dev_tools/python/scripts/source_limit_continuous_compare.py` checks each frozen `.mjb` SHA, MuJoCo 3.10.0, timestep, Euler/Newton settings, creates one `MjData` per form, and calls `mj_step` eight times without resetting. It invokes the native binary once per form and compares every step.

| Input | Leg SHA256 | Roller SHA256 |
| --- | --- | --- |
| Frozen source `.mjb` | `832e1f08a1e328d8498be565874691348b89bc13041ff0d9eebf3f06c6bdc47c` | `ea54c7c2b0aa8fbc3e9a0e3a431214650cbd4bdbd934d6707bb0aa52ac475191` |
| Native enriched-with-hulls scratch definition | `e91d67ba25efe3b61b65e77c754a4278f24f37253a86acdf81543a7de51f67bb` | `6f88aa286e487c031250b427595a640b9f597604c9a20dcbca6fd69c49109c40` |
| Initial source qpos | `e16f8e7ddf1d1cf2061b4c760faa1cbf78536d1cd8e6b992ca415eef10ae21bf` | `df50ef2c31243ef01c6419f18238008c9a2d8d3e444073b82c61cb27b3fb7773` |
| Explicit zero qvel file | `b65778c25e5ddc78b31aadcff3227f93156ce0cb396a6f5c2ebd8615987b9d2a` | `b071c6ce6c4da113902b1fb6822864744f59fe3291e92ff167e285f41f29e66a` |

The raw native reports are `.scratch/source_limit_continuous_v1/{leg,roller}_native_8step.json`, SHA256 `65273b4fa756e6659e5a2200e4966a21657a627be60765ed16a16d00dc37f988` and `bb7a21447b0b9f02f7b4e651c0de1b2acf7de689b73a665c09aa1dee33b73d7b`. The per-step source/native comparison is `.scratch/source_limit_continuous_v1/continuous_comparison.json`, SHA256 `edd57e61f1b452da68ef305b350d75a9f461739d9023d07747f2305732715971`. Two runs with the final binary and script produced byte-identical native and comparison reports. Both forms show exactly one assembly build, integration/observation epochs `1..8`, and constant native topology epochs.

The replay script fails on any activation disagreement, nonzero source or target contact impulse, limit-force error at or above `1e−6 N·m`, step-end angle error at or above `1e−6 rad`, or step-end angular-velocity error at or above `1e−5 rad/s`. The native probe also checks that its final limit-row impulse equals the generalized joint observation for this frozen single-row case. These gates were rerun after the independent review without changing the report hashes.

| Form | Max step-end q error | Max step-end qdot error | Max signed limit-force error | Limit activation disagreements | Steps with nonzero native contact impulse |
| --- | ---: | ---: | ---: | ---: | ---: |
| Leg | `2.32555e−8 rad` | `1.84497e−7 rad/s` | `1.11359e−8 N·m` | `0 / 8` | `0 / 8` |
| Roller | `1.82652e−8 rad` | `1.43633e−7 rad/s` | `1.64251e−8 N·m` | `0 / 8` | `0 / 8` |

Both source and native have an active upper-limit row on steps 1–5 and no active row on steps 6–8. Only step 1 has nonzero unilateral force (leg source `−0.0351088073 N·m`, native `−0.0351088185 N·m`; roller source `−0.0351108102 N·m`, native `−0.0351108266 N·m`). On steps 2–5 the joint remains beyond its limit but moves inward: the row exists and its unilateral force is zero on both sides. By step 8, leg qdot is source `−0.2999925259` versus native `−0.2999927104 rad/s`; roller is `−0.2999929633` versus `−0.2999930978 rad/s`.

**Contact topology is not equivalent.** The source reports `ncon=0` on all eight steps. Rapier reports five active robot self-collider pairs each step, with nine solver contact points in the first step of each form; the report identifies and deterministically sorts those pairs. Every pair's recorded normal impulse and every articulation DOF's observed normal/tangent impulse are zero throughout. No scene object was included. Thus the measured limit trajectory is a zero-contact-**impulse** case, not proof of collision/contact equivalence.

From the `Bevy_Sim2Sim` directory, with the frozen scratch inputs available:

```bash
cargo build -p dev_tools_minigame --bin source_limit_continuous_probe --features sim2sim_source_limit_probe
/home/ethan/Projects/Sai_Lab/upstream/pollen-robotics-microduck_rl/.venv/bin/python crates/dev_tools/python/scripts/source_limit_continuous_compare.py
```

Formatting and the feature-gated build passed. The binary also refuses a missing explicit switch or a mismatched qpos SHA without writing an output report. No `SourceCollisionWorld` step was used. Eight zero-torque steps at one isolated limit do not establish longer stability, simultaneous limits, nonzero actuation, meaningful contacts, nine ONNX behavior, or BAM `previous_solve_load` availability.
