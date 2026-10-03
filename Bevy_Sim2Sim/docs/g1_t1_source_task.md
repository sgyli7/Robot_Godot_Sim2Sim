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

## Native hand-limit diagnosis

The same four unchanged chunks from the successful source run were executed by
one native owner. The original four PGS sweeps and an isolated sixteen-sweep
control both stopped at Tick 157: the left index proximal joint crossed its
original zero upper bound by 0.00554 and 0.00497 radians respectively. The
0.001-radian protection threshold was retained. Source traces and the 154-frame
published episode have no measured limit excursions above 0.0001 radians.

The bundled reduced-coordinate limit row activates only after the coordinate
has crossed the bound. Increasing convergence iterations does not prevent the
first 20 ms crossing. An explicit development feature compares two unilateral
velocity constraints that enforce `lower <= q + dt*v <= upper` before the same
integration. It changes no coordinates, source ranges, gains, action data or
root constraints. The ordinary constructor still uses its original limits;
this comparison is not silently enabled in normal builds.

The predictive comparison completed all 160 original frames, WBC calls, torque
updates and integrations with four PGS sweeps, one 20 ms integration per Tick,
and a free six-DoF root. The apple physically rose about 9 cm and moved toward
the plate, but the final release drifted outside the target and did not meet
the velocity/stability gate. Native live RGB inference also completed four
actual VLA calls and 160 physical Ticks without the former limit failure; its
grasp remained unsuccessful. Neither result qualifies T1.

`TaskObjectFrame` now includes aggregated last-solve contact counterparts,
distances and normal impulses, annotated in the frozen 53-body order. These are
independent evidence only and remain absent from the model wire. A second
160-Tick run produced identical mechanical state/action rows after excluding
the new evidence fields and naturally different wall-clock observation ages.
Speculative positive-distance pairs do not by themselves prove a grasp.

## Joint mapping and contact convergence controls

`unitree_g1_t1_source_task.py --joint-kinematics-audit` records the original SDK's
world Jacobian, all named joint coordinates and body-link poses at existing
40-Tick boundaries. It performs no additional integration or coordinate write;
SDK counters are checked before/after each read. The original expert-action
diagnostic was bounded at 80 controls / 320 source integrations for this audit,
with no task-policy inference. Source physics remains the reference 200/50 Hz.

`unitree_g1_joint_kinematics_audit.py` independently compares all 43 immediate
child angular Jacobian columns and 53 links against a zero-Tick native assembly
initialized at exactly the measured q/root. This initialization is explicitly
an oracle mapping test with zero native integration/inference, never task
performance or a controller. Both startup and the bent left-hand grasp posture
passed: maximum native position error was 3.080e-7 m, rotation-matrix component
error 1.284e-6 and joint-axis vector error 9.544e-7. A deliberately reversed
finger axis failed the same comparison. These controls exclude the tested
axis/sign/FK mismatch; they do not prove migrated contact dynamics.

An isolated contact-convergence comparison now combines the explicit predictive
limit rows with 16 nonintegrating PGS sweeps. Use the same original chunk/config
bytes and append `--predictive-limits --constraint-sweeps 16` to
`g1_task_chunk_probe ... --offline-diagnostic`. It retains the free root,
original ranges/gains, 50 Hz, one 20 ms integration per Tick and actual owner
execution. The normal constructor remains unchanged.

The four- and sixteen-sweep runs both completed all 160 source action frames.
Maximum apple height was 0.87048 / 0.87738 m. With sixteen sweeps the last state
had no robot-object contact and a positive normal impulse from the plate;
apple linear speed was 0.00439 m/s and angular speed was 0.10756 rad/s. The
four-sweep final state drifted outside the plate at 0.11897 m/s. Whole-object
containment and two continuous released/stable seconds were not established,
and neither offline action replay is autonomous task success. The larger sweep
count also does not eliminate the measured transient contact penetrations.

## Causal renderer comparison

`--renderer-socket` is a finite diagnostic with the original source physics,
AGILE controller, self-state, action timing and ONNX policy unchanged. At each
existing 40-Tick policy boundary a separate
`unitree_g1_source_render_bridge.py` initializes a fresh native **zero-Tick**
assembly from the measured SDK robot/prop poses and returns actual Bevy RGB.
The original SDK image is retained alongside the image used by the policy.
All 53 links, named joint coordinates and both prop frames are independently
checked; source q/dq/root/prop state and integration counters must remain
unchanged across the blocked render request. Simulator truth reaches only the
renderer and evidence files. The policy wire remains RGB plus named self-state.

This is explicitly external-pose render initialization, not native task
execution or a controller. The native capture receipt must report zero WBC,
torque, task-policy calls and integrations. The source camera stamp records
the actual native image acquisition wall time and the stationary measured
source boundary; both renderer identities remain in separate evidence. Neither
side's frequency changes, and this paused comparison cannot qualify 1× runtime.
The comparison freezes the current original-background/EV11.7/no-directional-
shadow renderer candidate and matched 6.0 prop definition. Its outcome is not
known until the original source task has executed these observations.

The frozen original-background candidate failed this causal test: eight actual
policy calls executed 300 source controls / 1200 original PhysX integrations,
then the original six-second timeout fired without success or apple lift.
Across all eight boundaries the largest native body-position error was
5.36e-7 m and joint-axis error 1.43e-6; source physical-state change while
rendering was exactly zero. The earlier original-RTX camera reference succeeded
with four calls at 140 controls. This one comparison establishes that the
current Bevy observations can break the original-source task independently of
the migrated physics. It does not exclude additional native dynamic issues or
establish success rates. Inspection then found three omitted visible boxes:
they have no bound material but do have explicitly authored black displayColor.
The original-RTX camera control was also rerun with the current harness: it
again succeeded with four actual policy calls, at 146 controls / 584 source
integrations. This supports the image-domain comparison against a current
reference, while retaining the original auto-reset/termination limitations.
Restoring the black-box geometry/displayColor alone also failed the bounded
source task: eight calls, 300 controls / 1200 integrations, with an initial
apple height of -0.00790 m and maximum 0.00906 m. The model contacted/moved the
object but the original success term remained false. All same-pose and blocked-
state checks passed. This repair is retained as source-asset fidelity work;
it is not reported as task success.

After mapping the authored background albedo offsets, a finite two-patch
background-only diffuse calibration was frozen before task evaluation. The
source-physics/native-image comparison then lifted the apple by 0.10657 m and
transferred it above the plate, but still timed out at 300 controls without
release. No further lighting search was selected. The identical frozen camera
candidate in the actual native owner completed eight image-policy calls and
320 native 50 Hz steps; the apple ended released, supported by the plate and
nearly stationary, with the robot standing. Full per-Tick containment/release
verification is recorded separately. These paused camera/inference diagnostics
are not continuous 1× operation or the required ten-episode acceptance suite.

## Finite grasp wait controls

The ignored tests `real_static_grasp_target_hold_diagnostic` and
`real_static_measured_arm_grasp_hold_diagnostic` use the same byte-bound
`G1_STATIC_RELEASE_CONFIG`, `G1_STATIC_RELEASE_ACTIONS`, corresponding SHA256
variables, `G1_STATIC_RELEASE_OUTPUT` and `G1_CODE_COMMIT` as the existing release
diagnostic. They execute exactly 100 unchanged saved source frames, then 25 real
50 Hz WBC/motor/integration steps in the same free-root world, with no new VLA
calls. This is a separately labelled explicit test-private wait, never a chunk
extension, action restamp or installed runtime fallback.

The frozen grasp-wait rule requires all 25 samples to retain robot support
impulse above 1e-6 N·s and standing, apple height at least 0.8371 m (initial
height plus 0.05 m), and maximum displacement 0.025 m. Retaining the last full
target kept support/standing throughout but displaced the apple 0.03351 m,
so the rule failed. Replacing only arm targets by measured self-state, with
zero navigation and unchanged finger preload, displaced it 0.05526 m and
lowered it to 0.81035 m; the same rule failed. No object truth selected either
controller target. These outcomes reject both candidates for precise grasp
waiting; success after empty-hand release cannot establish grasp-wait safety.
Further wait-target searching is not selected. The next timing comparison will
prefetch inference during an existing unexpired chunk while preserving every
original action frame and explicitly measuring observation age; a missed
replacement deadline must pause rather than retain expired targets.

## Original-source image-age comparison

The optional `--policy-prefetch-after-ticks 10` is a finite timing diagnostic
with the original RTX observations, AGILE controller and 200/50 Hz source
physics/control. It observes at controls 10/50/90/130/170 and installs each
replacement at 40/80/120/160/200, preserving all forty predecessor frames and
all forty replacement frames. Thus the replacement image is thirty controls
(0.6 s) old in simulation time. **Inference still pauses source physics** at
capture: this comparison does not establish asynchronous or continuous 1×
operation. The native frequency remains 50 Hz with one integration per Tick.

The mode requires `--policy-socket`, original RTX rendering, 80–240 controls in
whole forty-frame chunks, and no expert or external-renderer mode. A single
pending chunk retains its acquisition Tick and sequence; the boundary guard
rejects a missing/partial replacement, incomplete predecessor, early/late
application or rebasing. Traces record the actually applied sequence and frame
SHA256 independently of a queued future result.

The frozen seed-42 comparison used the same harness and original model/export.
Immediate observations triggered Arena success at control 137, with four
actual VLA calls and 548 source integrations. The delayed observation run
completed 240 controls, six actual calls and 960 integrations without a
success trigger. All 240 applied delayed-run frames were independently
matched to the original replies; the robot's minimum upright value was
0.99658. The apple's final horizontal distance from the plate was 0.17516 m,
versus 0.03179 m just before baseline termination. The two initial robot states
were identical, but RTX images and the first predicted chunks were not byte
identical. This is evidence against the present 0.6 s scheduling candidate,
not an isolated proof of the only failure cause. No offset sweep or hidden
frame skipping was selected.

The baseline's original success trigger is not the independent two-second
placement rule: its terminal step automatically resets the source environment.
Neither run qualifies native T1 or the ten-episode scores. The different
baseline completion Tick also failed an exploratory exact-repeat check against
an earlier control-146 run; that failed check is retained separately.


## Native scientific-station preparation

An explicit optional `station` capture configuration pins model, manifest and
layout SHA256. Static station visual surfaces and collision shapes are derived
from the same checked `StationScene` read, in engine Y-up metres. Immutable
Rapier shapes and mass properties are prepared before the owner clock starts.
T1 AGILE installs these static colliders into its existing robot/task world and
removes the original40×40m floor before any integration. The source shelf,
apple, plate, recurrent controller, actuator parameters and task profile are
unchanged. No extra world, runtime pose adjustment, hidden floor or substep is
added. Unrelated movable station props are omitted from both display and physics.
The G1 scene retains1080p/MSAA8 and owns its main camera; station materials and
static geometry use their existing rendering path.

This initial station seam admits the independent T1 AGILE owner only. T2's full
warehouse background is rejected because overlaying it would create unrelated
or invisible contacts. Existing explicit T1 predictive-limit and16nonintegrating
PGS diagnostics remain available; the ordinary AGILE default stays4. The earlier
320Tick visual release evidence used the explicit16PGS factory, not that default.
Every station owner step records public environment identity, collider coverage,
zero preparation integrations and original-floor removal. Integration, rendering
and strict task acceptance must be tested separately; this preparation alone is
not station task success or frozen8of10 qualification.

The first native-station live pilot (`20261003_0298`, clean0848e5c) used8fresh
original N1.7 calls and320single50Hz integrations. All camera pixels and original
stamps match the captured model inputs. The robot remains standing but the apple
stays on the shelf: strict placement0seconds. Camera/model pauses and18missed
control deadlines also prevent real-time qualification.

A bounded offline mechanical comparison (`0299`) preserves all320original
successful0071actions and their simulation/frame identity. Only offline wall
admission times are refreshed. The original-floor reference reproduces all320
robot body frames exactly in float32 and passes2.76seconds of strict placement;
the station world with identical actions fails. Public-geometry vertical rays
confirm groundY=0, upward normals and no duplicate ground under either foot.
The imported tessellation has separate equal-position vertices. G1 static meshes
now weld those vertices and apply Rapier/Parry `FIX_INTERNAL_EDGES` contact-normal
handling without changing surface coordinates or coverage. The third320Tick
comparison still fails placement, so internal edges alone do not explain the
failure. These960CPUbody-controller integrations are mechanical diagnostics,
with0fresh images/VLA calls, and are not autonomous task qualification.

The40Tick-per-world contact audit (`0301`,80real body steps) distinguishes cached
empty manifolds from solver-active contact points. Both original-floor and
station active foot normals are vertical. The oblique empty cached normals do
not establish a contact defect. Larger initial impulse differences involve the
original robot contour and original task table. No mass, gain, pose or support
geometry is changed to remove those contacts.

One finite initialization comparison (`0302`) executes60original zero-navigation,
0.75metre-height/default-upper AGILE commands in each world,120real body steps.
Last20Tick maximum self speed is0.0040m/s on the original floor and0.0043m/s in
the station. These are finite startup checks; original robot/table contacts
remain, and they do not establish feet-only support or task success.

The development `static_startup` owner makes that initialization explicit in
the same task world and clock. Loading performs0integrations. After exactly60
real20ms steps, at least20consecutive self-velocity/IMU checks must satisfy
speed<=0.03m/s and upright cosine>0.95. Only then may unchanged original N1.7
chunks enter, using an actual image from simulation time>=1.2seconds. Startup
records are separate from VLA execution records; all380integrations of an
eight-chunk pilot remain in the evidence. Reset reconstructs the world and
startup gate; stale episode/image results cannot continue it. This is an
explicit bounded comparison, not an automatic fallback or qualified executor.

The real owner/reset test (`0303`) passes with121real body integrations and0fresh
VLA calls: an old pre-startup image is rejected at60without an extra integration,
a reset repeats the60real startup ticks, and one saved original action frame is
admitted at the new boundary. This offline fixture explicitly refreshes its
image admission stamp and does not claim fresh vision. Its first failed test
attempt executed60real startup ticks before a test-side consumed-snapshot
assertion failed; that failure and its log are retained separately.

The first same-world startup/vision pilot (`0305`, clean5f4d5fb) performs all380
real integrations and8fresh original N1.7 calls. All8RGB inputs match captured
pixels and retain Tick60/100/.../340stamps. The robot stands throughout, but the
apple has54positive robot-contact ticks and0ticks without shelf contact. Strict
placement remains0seconds: startup readiness has not solved grasping. Active
sim/wall ratio0.993with2misses is separate from camera/model-paused continuous
ratio0.640and does not qualify continuous1x operation.

The station entry recorded requested450ambient/15000directional/shadow-on
lighting rather than the successful source-near0071baseline's fixed
2328.263982ambient/981.363192directional/shadow-off settings at the sameEV100=11.7.
This is a disclosed rendering/profile difference in addition to the mechanical
floor comparison. The initial explicit lighting comparison (`0306`) is rejected
before world creation because the prior diffuse-light guard allowed only a
source background. It has0physics integrations and0VLA calls. Native station
captures now accept explicitly supplied finite bounded light energies as well;
normal defaults and geometry remain unchanged. The next comparison restores
those three known baseline fields, with no lighting or physical parameter sweep.

The0308run reveals that those requested receipt fields did not describe effective
station lights: native station setup skipped the capture-light branch. All8images
and original action blocks are byte-identical to0305, despite different requested
light parameters. Both runs execute380real steps and fail placement. Effective
station defaults are450ambient with RGB(0.78,0.76,0.85),10000directional and shadows
enabled. Older raw receipts and sealed evidence remain unchanged; this explicit
correction supersedes their station-light interpretation.

Native station capture now applies explicit lighting fields after the station
has spawned its actual light entities, and records ambient color/energy plus
each directional light's effective color/energy/rotation/shadow state. Existing
station defaults are preserved when no override is requested. No additional
light or floor is added. The ECS integration test verifies both the unchanged
default entities and effective overrides; all43development tests pass. The next
fixed known-baseline comparison must additionally verify actual effective
lighting and changed pixels before it is considered a valid visual comparison.

The first effective-lighting station pilot (`0310`, clean25bb013) passes the
independent placement rule for2.98seconds, with minimum full-apple footprint
margin0.009105m, positive plate support, actual robot release and standing.
It performs380real single50Hz integrations and8fresh original N1.7 forwards.
All8RGB inputs, original action arrays and image stamps are independently
matched to native execution. Its first60startup body steps exactly match the
failed0305run aside from episode identity; only after actual lighting/pixels
change does the first original task chunk change. This demonstrates a concrete
perception/configuration fix, not that all mechanical migration risks vanished.
Active ratio0.98897,1control miss and0pending debt are recorded separately from
continuous boundary ratio0.67613with explicit camera/model pauses. This one case
does not establish8of10, Qwen task selection or continuous real-time performance.

The static server now accepts a fixed `--seed-offset` (unsigned32-bit, default0)
added to each request sequence for the original graph's initial noise. Default
sampling is unchanged. Both health and capture receipts identify the offset and
actual sampling seed; no model/action/preprocessing contract field changes.
The14CPU protocol tests include real HTTP calls proving changed sampling input,
unchanged actual RGB/stamps/action transport and exclusive evidence writes.

`unitree_g1_static_suite.py` consumes a byte-frozen manifest of5positions x2seed
offsets(0,42). It checks code/tool/model/build identity before starting, executes
one fresh physical owner per case, and serially shares the original five-graph
service within each seed group. The paired read-only capture auditor validates
all380owner records, original model arrays, images, sampling identity and actual
station light entities. Placement thresholds and75second application timeout
are frozen before any case runs. This intermediate profile suite explicitly
retains camera/model pauses and cannot qualify Qwen or continuous execution;
its task score is reported separately. Failed/partial cases and owned process
cleanup remain in the output rather than being silently retried or dropped.

The first frozen suite launch (`0313`) ends before model readiness with0real
integrations and0VLA calls. Its launcher incorrectly resolved the virtualenv
`bin/python` symlink to the base uv interpreter, losing NumPy/site-packages.
The failed lifecycle and logs remain. The launcher now preserves that path and
checks the actual virtualenv prefix, NumPy, ONNX Runtime and CUDA provider before
loading any weights. The corrected manifest keeps all ten case configurations
and thresholds byte-equivalent, updates only tool/source identities, and names
the retained zero-Tick failed batch. No task case or score is silently retried.

The corrected run (`0316`) completes all five seed0 cases:2pass,3fail,
1900actual integrations and40fresh N1.7 forwards, with standing throughout.
Its second model group never starts because a port probe without SO_REUSEADDR
rejects the previous group's closed connections. The probe now matches the
HTTP server's reuse policy and still rejects an active listener. A CPU socket
test exercises both outcomes. Explicit continuation pins the old summary and
manifest, checks unchanged cases/thresholds/model/binary, re-audits every retained
case and checks the closed owner's40actual calls before starting only seed42.
The combined report retains separate source identities and new/retained counts;
it never rewrites or reruns the first five cases. Even five further successes
would give at most7of10, so this batch cannot meet the frozen8of10 threshold.

The completed continuation (`0319`) gives2of10, all10standing,3800actual native
integrations and80fresh original forwards. The first five were independently
re-audited rather than repeated; the last five use0fdab26with the same Rust binary.
This is a failed intermediate paused profile score, not formal task acceptance.
An offline source-FK audit (`0320`) matches native fingertip-link origins within
submicrometre error in nine cases; one case reaches1.9mm. Tracking-error maxima
also occur in successful cases and do not establish an actuator root cause.
Link origins are only geometric proxies; contact truth remains auditor-only.

The source-bound material comparison (`0321`) uses all49bound original USD
materials with exactly the same mesh data and USD identity. Startup's60full body
steps are unchanged, actual pixels change, and the failed position still does
not lift the apple. It is retained as a negative comparison, not promoted as a
grasp fix. No physical parameter or graph contract changes.

The ignored `finite_static_grasp_alignment_comparison` is a bounded mechanical
fixture: two fresh native station owners, each60startup +80saved commands
+20explicit last-target hold ticks. Only the candidate's left-arm targets change,
using one fixed oracle-derived initial-position translation and source FK/IK;
fingers, body commands, gains, physics and all original saved bytes stay intact.
It creates no images or task-policy calls and cannot earn an autonomous score.
The diagnostic API rejects VLA admission, navigation magnitude above0.01and
steps outside60..160; tiny original decoded navigation residuals stay unchanged.
normal workers/queues do not call it or acquire a new fallback behavior.

The first alignment attempt (`0322`) stops after62actual ticks because its
initial zero-navigation guard rejects an original -0.000629rad/s yaw output.
No candidate world has run. Its partial trace stays retained; only that guard
is corrected, with unchanged saved commands, oracle delta and grasp thresholds.

The corrected320Tick comparison (`0323`) reproduces all140reference body steps
exactly. Oracle translation prolongs hand-only contact, but both20Tick waits
lose the apple: reference retains support1/20ticks, candidate9/20; neither meets
the frozen height/displacement/standing/contact hold rule. It is not a fix.
An optional test-only20Tick finger-preload candidate then uses only original
joint encoders: flexion closes at1rad/s within source joint limits, and stops
incrementing a target at0.08rad directional position error. The4Nm/rad source
stiffness and5Nm motor caps do not change. Error is a load proxy, not proof of
object contact. The candidate has no runtime/UI path and no autonomous score.
Its paired worlds start with the same prior oracle-aligned prefix, allowing the
independent auditor to isolate this feedback change and retain any failure.

The paired self-encoder comparison (`0324`) also fails: support improves from
9/20 to13/20wait ticks, but both apples fall. All320integrations and original
body updates are retained; the reference exactly reproduces0323and both140Tick
prefixes match. No preload threshold or gain search follows. An offline audit
(`0325`) separates approximately26mm of palm joint motion from approximately2mm
of root motion. At the hold boundary the elbow still moves at -1.37rad/s with
a -0.152rad last-target error; this is an unsettled arm, not a proven finger
force problem. Original robot masses/COM and measured self-state predict the
gravity-support target offset using the unchanged source stiffness.

The same finite test can now preload both reference and candidate during their
20Tick hold. It starts that feedback from command80(the first explicit hold
command), allowing one separately declared gravity-compensated arm-hold
comparison. This test seam still has no worker/UI path, object-truth controller
input or autonomous score. Previously frozen fixtures retain their behavior:
their commands79and80are identical and the new reference flag defaults false.

The gravity-compensated320Tick comparison (`0326`) is also negative. Its
reference exactly reproduces all160steps of0324's preload candidate, and both
140Tick prefixes match. Candidate palm displacement drops from29.2mm to13.4mm,
but hand-only support drops from13/20to11/20ticks and the apple still falls.
This rejects unsettled-arm correction as a sufficient grasp fix; no gravity,
preload, friction or gain sweep follows.

`unitree_g1_t1_source_grasp.py` provides one finite original200/50Hz source
comparison with the same100saved original commands after60startup updates,
original G1/finger friction, source shelf and original apple/plate USD at the
matched initial poses. Its reduced scene omits background/cameras and has no
VLA or expert claim. Original action groups are mapped by source joint names.
Each control update must advance exactly4original integrations, at most640.
Object poses/velocities are recorded only for the independent auditor. This
entry does not change native50Hz or establish contact retention, visual task
success or formal task acceptance by itself.

The initial source launch (`0327`) has0integrations: the SDK container's UID1234
could not write its newly created output directory. Only that run directory's
ACL is corrected for the retained retry (`0328`); no global permissions or
foreign process changes occur. The retry completes160original controls and
640original integrations with the same saved commands. Initial robot joints
and apple/plate poses match, but source startup moves the robot and apple by
approximately20mm while native startup does not. The source apple stays above
the shelf during the20Tick wait, but moves36.9mm and fails the frozen25mm wait
component. Source contact support is not measured by this pose-only probe;
neither case is a qualified grasp or autonomous task score. The different
post-startup geometry prevents attributing the outcome to one contact setting.

A separate `static_marker_assets` camera configuration now admits only the
explicit original AGILE station0/60Tick no-policy diagnostic with16nonintegrating
solver iterations and predictive limits. It loads the published
`assets/g1_fiducials/static_apple_plate.json` labels: IDs31/32, black square
sizes20/60mm, overall white-margin sizes25/75mm, fixed object-local centers
[.002,0,.046] and[0,0,.0045] metres, both facing source+Z. They are declared
render-only planar labels; physics geometry/materials/mass do not change.
Existing mobile labels and their activation gates retain their original values.
Original unmarked VLA entries reject this new static calibration mode.

`unitree_g1_static_vision.py` reads actual640x480PNG and a whitelisted
`g1_static_marker_observation_v1` of original pinhole calibration,43joint
encoders and IMU/self velocities. It rejects object poses, world-root position,
contacts, saved actions and unknown fields. Public label PnP plus original
self-FK yields root-relative apple/plate transforms; duplicate IDs, short edges
(<8px), bad reprojection(>1px), wrong calibration and nonfinite data are rejected.
This classical localization tool neither proposes actuation nor earns a task
success. An actual native view must verify detections and pose error before
any live task controller may consume them.

A preparation-time inspection catches a marker-entry mismatch before any
launch: the ordinary standalone AGILE worker cannot select the required
predictive16solver owner. The unused0329config/build are retained with0real
steps. The marker entry therefore uses the existing typed static task owner
with no policy, and60Tick views reuse its exact explicit startup command and
self-state gate. The startup helper is shared with the original live profile;
no new physics method or altered startup command is introduced.0Tick views
initialize that same owner and perform no integration. Any policy combined
with static labels, other profile or different Tick budget remains rejected.

The first actual static marker view (`0330`,4cc72ec) completes60original owner
integrations/WBC calls with0VLA/Qwen. All60body records exactly match the
unmarked P1reference; camera, self-state and task display use the same Tick60.
Apple ID31is detected with18.43px minimum edge,0.456px reprojection RMS and
2.14mm independent position error. Plate ID32is hidden inside its original
mesh, so this is only partial perception proof and cannot admit a task.

Calibration v2moves only the plate label above the original cooked surface:
the source geometry's maximum localZ is0.02346462m; labelZ is0.0255m. The
apple label stays unchanged. Version2and both fixed public mount centers are
now explicit byte-bound document fields checked by renderer and localizer;
stale/changed mounts are rejected rather than applying new parameters to old
pictures. The original0330document, localizer, raw RGB, partial result and
independent audit remain retained. No physics or learned-policy field changes.

The v2 native P1 view (`0331`,9421b78) passes the pre-frozen 10mm position,
8px edge and 1px reprojection gate for both targets: apple2.137mm and
plate2.913mm. Its60body records remain identical to the unmarked reference.
The subsequent unchanged five-position check (`0332`) retains this P1 result
and adds four60Tick views, for240new integrations and300unique integrations
overall, with0VLA/Qwen. Only3/5perception gates pass. P0apple error is19.522mm
despite0.172px reprojection RMS; P3apple is visible in RGB but not decoded.
Plate errors in all five views remain below4mm. These are perception failures,
not task successes or permission to actuate. The original 10mm gate is retained;
native frozen T1 remains2/10 and full-task qualification remains false.

The retained RGB corner audit (`0333`) finds0.8-1.3px contour errors on the
16-19px apple label; the P3quadrilateral is found but fails dictionary decoding.
A full public-pattern registration prototype (`0334`) reduces errors for the
four decoded labels but incorrectly selects a nearly uniform rejected contour
for P3. That negative result is retained. Explicit positive contrast, exact
36cell pattern agreement and per-cell confidence reject this false candidate
(`0335`); offline apple errors then range0.355-6.494mm on the same five images.
No native Tick, model call, mount, camera, physics setting or gate changes.

`unitree_g1_static_label_fit.py` implements this declared classical correction
as a separate bounded RGB module. It fits the full disclosed apple pattern,
models raster samples at half-pixel centers and returns continuous projection
corners using the original camera calibration. Fits move observed corners at
most3px, have at most100optimizer evaluations, and inspect at most16rejected
quadrilaterals. Positive contrast>=0.5, normalized residual<=0.15, exact payload
and black-border bits, and cell confidence>=0.15 are required. Uniform/wrong
patterns, clipped/small contours, duplicate admitted labels and excessive
candidate counts are rejected. The existing PnP edge/reprojection and external
10mm position gates remain unchanged. The plate path is unchanged; self sensor
whitelisting, public asset hashes and no-actuation/task-unqualified outputs are
retained. This tool needs the already-used OpenCV/NumPy/SciPy runtime and does
not modify or feed marked images into the original learned grasp profile.

The committed full-pattern module (`0336`,6c9ab24) reprocesses the same five
byte-bound native RGB/self-state pairs without additional simulation or camera
captures. All five unchanged external perception gates pass; apple errors are
0.486-6.494mm and plate errors are unchanged. NineCPUadmission/isolation tests
pass. Candidate scanning takes2.16s P50/3.16s P95on this machine, so this result
does not qualify a live moving-object controller or replace task acceptance.

Static startup can now select the existing explicit prefetch diagnostic. Its
first actual observation and action slot start at Tick60rather than Tick0;
future40frame chunks retain original image timestamps and fixed starts. The
same typed startup owner admits future chunks through its existing original
VLA command wrapper, with no second world or controller. Requests still begin
10Ticks into the active chunk, matching the existing static prefetch window;
the resulting observation age is measured, not erased. Late replies or missed
replacement slots pause explicitly; no chunk is rebased. This changes the
observation/execution schedule and must be separately tested for task behavior;
the paused2/10profile is neither overwritten nor promoted.

The actual startup-prefetch pilot (`0337`,262a41c) completes380single50Hz
integrations/WBC calls and8fresh N1.7inferences. Every original frame executes
once, at fixed starts60,100,...,340; subsequent RGB comes from70,110,...,310
without restamping. The320Tick task section has6.379914s between first and last
completed boundaries, a1.000013sim/wall ratio and0observed pending ticks.
Four individual control deadlines are missed. InferenceP50 is0.317s/P95 0.680s
(including the first cold request). The robot stands throughout, but placement
fails: the apple has48hand-only support samples before slipping during transfer.
Owned app/model processes are reaped. This proves one uninterrupted task clock,
not successful continuous task execution or an8/10qualification.

Original source rollouts0034/0036apply the first original policy block at reset
Tick0, whereas0337adds60standing initialization ticks. The source's ordinary
success term also immediately auto-resets and therefore does not establish
the required2second stability. A separate bounded no-extra-startup comparison
uses the existing ordinary task owner and320original frames; no frequency,
actuator, scene, weight, age threshold or physics setting changes. Its audit
keeps0/60startup identities explicit rather than mixing their counts or scores.

The no-extra-startup pilot (`0338`,11a1fd9) completes320native/WBC ticks and
8fresh unchanged original model calls. All frames and actual inputs pass the
independent prefetch audit; task clock ratio1.000344 and0pending ticks again
pass, with6individual control misses. The robot stands but placement still
fails. Therefore extra standing initialization is not established as the main
transfer failure; this negative result is retained without further startup
duration searches or any upgrade to the2/10frozen profile.

The retained original SDK run0036 stopped at control140/560physical steps when
the original success term triggered auto-reset. It has4fresh original VLA
calls, but no complete two-second release/support window; that weak source
termination never qualifies the native task. An explicit source-only
`--placement-continuation` audit retains300controls, original200/50Hz physics,
drop/timeout terms, camera, actions and dynamics. It records the original success
term while suppressing only its early reset, queries existing apple/plate/all53
robot contact reporting without adding schemas, and traces body poses only to
the independent auditor. All2517apple collision vertices must fit the same
moving plate footprint; plate upward force, original speed/standing bounds and
two continuous seconds remain required. Release additionally needs strictly
positive same-Tick exported-collider AABB separation from every robot shape;
overlap remains inconclusive, even with zero reported force. Source results
remain separate from native/formal acceptance. Contact tensors use the
[documented impulse-to-force timestep](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.0/extensions/runtime/source/omni.physics.tensors/docs/api/python.html),
here the actual original last-solve0.005seconds; no extra step is performed.
