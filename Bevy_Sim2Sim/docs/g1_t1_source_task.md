# Original T1 scene and task reproduction

`unitree_g1_t1_source_task.py` calls the original
`GalileoG1StaticPickAndPlaceEnvironment.get_env` with `g1_wbc_agile_joint`.
It preserves all four scene assets, original positions/scales, background
deactivation event, finger material, head camera and six-second episode.
Only byte-bound asset storage locations and the process-local GPU TorchScript
fusion optimization are redirected as described in `g1_task_objects.md`.

The source runtime is the retained ARM64 `6.0.0-dev2` container,
`6.0.0-rc.22+release.33481.407f3ea1.gl`, with Arena
`8b4a3a47fc53de23e8205089d71109a2e2348acd` and Lab
`e57379c634b42db5a0fe9f754341be6e2a7c7c43`. Its original 200 Hz PhysX / 50 Hz
control configuration is a reference reproduction. Formal native Rapier remains
50 Hz with one integration per tick.

The full scene/camera probe at `ec88bf5` completed 40 control ticks, with all
original assets present and actual 640×480 RGB/self-state captures. The initial
full ONNX rollout at `f47d652` made eight real five-graph CUDA calls and executed
the returned 40-frame chunks. It stayed upright, but did not lift the apple and
ended at the original 300-tick timeout. Original success and object-dropped terms
were both false. This is a failed source task, not native task qualification.

## Static hand input correction

The original export `graph.yaml` incorrectly labels `preprocess_state` hand
inputs as thumb-first. The original Arena joint groups and numerical ONNX
normalization use index(2), middle(2), thumb(3). This is measured rather than
inferred from a successful import or a zero-hand smoke input:

- Thirty-three deterministic midpoint/random percentile samples were evaluated
  through the original CPU `preprocess_state.onnx` and frozen training statistics.
- Original training order maximum normalization error was `3.03094024e-7`.
- Following the declared hand labels produced maximum error `1.79734287` and
  saturated several hand channels even for the training percentile midpoints.
- Zero normalized actions decoded to the original action percentile midpoints
  with maximum error `2.23517418e-8`, including the original base commands.

`unitree_g1_static_contract_probe.py` reproduces this comparison using the
original source joint YAML. The observation/client/server use schema
`unitree_g1_static_observation_v2`; both hand arrays are canonical index/middle/thumb.
The old v1 schema is rejected. Weight bytes, ONNX graphs, normalization constants
and published metadata are preserved. Mobile GN1.6 input semantics are unchanged.
This correction alone does not establish why the first grasp failed.

The local reference files are bound by these hashes:

| File | SHA-256 |
| --- | --- |
| Source joint groups | `28226939039bd29fce7bf93d7f3fccce66bd3aa74a4c49e89d8229cd946357ae` |
| Training statistics | `cd012265f59ea0b82571aa888356ddf87b797ab91c3c8f74675678e1779c121a` |
| Export graph.yaml | `21a771433d79195cde9a2fa102671bbe6764cbd231ec8b060cae1775175d86d2` |
| preprocess_state.onnx | `33b5eff99674daac13d65a92b5274efc5efd52b0ad21618ebcbf2327394555fa` |
| decode_action.onnx | `13cef938a2d07b98e79a7666c67f81ea4ea7b97091c75494b4f932c0df62cf5d` |

## Execution and evidence boundaries

The source reproduction bridge accepts bounded RGB/self-state messages through
one private Unix socket and loads the same verified five-graph static model used
by the native localhost service. It accepts at most eight calls per process.
The source simulator performs synchronous inference, then starts frame zero,
matching Arena's original action chunk ordering. Its offline wall time is not a
native 1× result. Simulator truth is recorded separately in the trajectory and
never sent to the model.

`PolicyActionQueue` now separately retains the unchanged observation stamp and
owner-side execution start/end. Observation age is independently bounded at
admission; execution begins at frame zero after admission. An owner stall skips
elapsed execution frames rather than replaying a backlog. Clock rewind, reset,
wrong profile, wrong revision, invalid frame and stale replies retain their guards.
Admission and emission metadata expose actual observation ages.

`simulation::g1::task_policy` maps decoded groups to the distinct AGILE/Homie-v2
commands in canonical physical upper-body order. Both original WBC configs give
the waist to the lower-body policy. Decoded waist joint angles are retained as
evidence and never converted to torso RPY; the original GR00T torso command is
zero. `ArenaTaskWorker` now admits these chunks in the same background owner
that runs WBC, motors and Rapier. The live development camera entry uses it;
production game/UI task registration and task qualification remain pending.

`unitree_g1_source_material_cache.py` caches only original relative USD/MDL file
references, with bounded downloads and SHA-256 receipts. The closed file graph
contains 174 files / 1,262,303,096 bytes. Commented MDL example texture paths do
not count as dependencies. Built-in MDL modules remain resolved by the original
runtime. File-reference closure does not by itself qualify complete rendered
material compatibility, licenses, physics or tasks.

Use fresh output/episode identities for each invocation. Container outputs are
under `/tmp` and copied into an exclusive `.scratch/g1` case directory by the
host diagnostic wrapper. Initial and every 40th actual camera frame, requests,
replies, full measured trace, termination causes, reset metrics, model stage
timings and artifact identities are retained. Arena automatically resets on a
terminal step; the log explicitly identifies that terminal step and does not
mislabel the reset state as the terminal physical pose.

No source task outcome qualifies the required native ≥8/10 scores or the
two-second released-object acceptance test. These remain independent gates.

## Reset camera diagnosis and source task reproduction

The original Lab 6 RTX update cache is keyed by simulator identity and physics
step count. The reset changes robot state at the same count; the first RGB could
therefore describe the pre-reset hand pose while its measured joints describe
the reset pose. A render-only reset pump plus the public camera cache reset now
refreshes that image before any policy request. Both SDK counters and all
robot/object physical state are asserted unchanged; no integration or pose write
is used. This refresh is enabled for every source harness run.

With the same matched ONNX model, original scene and seed 42, the refreshed
source runs triggered Arena's success at Tick 132 and Tick 140, each with four
real complete VLA calls. Actual SDK integration counts were 528 and 560. The
second run also verified a normal bridge shutdown and model close after early
termination. The former timeout runs and failed first cleanup remain evidence.

Original dataset action replay independently triggered the source success at
Tick 120 (480 actual integrations), with the robot upright. Dataset revision is
`37ba80a99486a4c308477854f54b4e82e77777dc`; it is an open-loop diagnostic,
not autonomous perception. It uses the original Arena action mapping and never
writes body poses. Upstream success ends/reset the episode immediately and
does not prove the required two-second released-object stability criterion.

The frozen N1.7 release image helper always letterboxes its RGB, including when
its saved configuration says false. Its published ONNX black padding is
consistent with that release helper. Current-main preprocessing must not be
substituted for the matched release. The optional root PyTorch checkpoint has
been hash-verified in the shared cache; its gated base processor is not needed
to run the already validated complete ONNX source path.
