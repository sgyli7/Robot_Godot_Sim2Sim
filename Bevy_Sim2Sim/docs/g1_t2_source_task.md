# T2 original source diagnostic

`crates/dev_tools/python/scripts/unitree_g1_t2_source_task.py` builds the frozen
Arena brown-box-to-blue-bin task with its original Homie v2 controller. It uses
the T2 Arena/Isaac Lab revisions and Isaac 6.1 assets checked by
`unitree_g1_source_stand.py`; it checks T2's own policy joint YAML. This path
does not load T1's AGILE controller, N1.7 export or forty-frame contract.

The default invocation verifies source and asset identities without starting a
GPU. `--run-source --ticks 0` captures the original reset camera and named
self-state without a physical integration. Motion requires the actual N1.6
loopback service and accepts only its original fifty-frame, 20 ms replies.
Foreign episode/sequence/profile, changed timing/horizon, malformed/nonfinite
actions and extra object-truth groups are rejected. Only RGB and measured named
joints enter the policy request. The separate trace's `acceptance_truth_only`
object poses and velocities never enter the request.

Two source stacks are separately pinned by `--source-profile`. The existing
`development_0_3` path uses Arena `7d75c959`, Lab `ae37b028` and SDK 6.1.0.0.
The selected published-stack path, `release_0_2_1`, uses Arena `8b4a3a47`,
Lab `e57379c6` and the original Docker SDK build
`6.0.0-rc.22+release.33481.407f3ea1.gl`. Both retain T2 Homie v2 and the same
N1.6 checkpoint, fifty-frame decoder, joint groups and instruction. Loading
that SDK does not load T1's AGILE/N1.7 policy. The released task's original
success term is a proximity check; it cannot establish formal placement.

The source diagnostic keeps 200 Hz physics, 50 Hz control, the original
thirty-second episode, task placement/randomization and scale. Inference pauses
source physics. It does not change native Rapier's required 50 Hz single
integration timing, qualify continuous 1× operation, or prove a native task.
An original terminal step automatically resets the environment; a source
success trigger cannot replace the independent released/stable two-second rule.
The trace records actual source integration counts and applied-frame identities.

## Preparation and entry points

The isolated `arena_spark_cu13` environment originally supported the standing
reference but lacked the full task registry's RSL-RL imports. The original
Isaac Lab `pyproject.toml` requires `rsl-rl-lib==5.4.1`; its frozen `uv.lock`
also pins `tensordict==0.13.0`, `onnxscript==0.7.2.dev20260725`,
`onnx-ir==0.2.1`, `orjson==3.11.9`, `pyvers==0.2.3` and
`importlib-metadata==9.0.0`. Those packages were added only to this isolated
environment, using `uv pip install --python PATH --no-deps` with the explicit
pins. Original Torch/CUDA packages were retained. CPU import of the original
registry runner symbols passed; no training was run. Installed versions and
the upstream lock hash are recorded in the evidence.

Run preflight from `Bevy_Sim2Sim`, using a fresh output filename:

```bash
/home/ethan/models/unitree_g1/envs/arena_spark_cu13/bin/python \
  crates/dev_tools/python/scripts/unitree_g1_t2_source_task.py \
  --arena-source /home/ethan/Projects/Sai_Lab/upstream/unitree_g1/isaaclab_arena \
  --lab-source /home/ethan/Projects/Sai_Lab/upstream/unitree_g1/isaaclab \
  --homie-assets /home/ethan/models/unitree_g1/homie_v2 \
  --task-assets /home/ethan/models/unitree_g1/task_assets/20261001_frozen/Assets/Isaac/6.1/Isaac/IsaacLab \
  --output .scratch/NEW_CASE/source.json --episode-id 1
```

For the zero-Tick camera diagnostic add `--run-source --ticks 0`. Use the
same process environment as the proven source standing reference:
`LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1`,
`OMNI_KIT_ACCEPT_EULA=YES`, `ACCEPT_EULA=Y`,
`ISAACLAB_ARENA_FORCE_EXIT_ON_COMPLETE=1`, `PYTHONUNBUFFERED=1`.
Use a bounded parent timeout; `--startup-trace` records one 45-second startup
stack and enables SDK startup logs without changing dynamics. A cold camera
startup timeout and the missing-dependency failures remain separate evidence.

For the published source stack, select `--source-profile release_0_2_1` and
the corresponding frozen source roots and 6.0 asset paths. In a source container
whose editable packages are installed from `/workspace/Arena` and
`/workspace/IsaacLab`, use those actual import roots. Their Git worktree metadata
may refer to host-only paths. `--source-tree-receipt PATH --source-tree-sha256 SHA`
then verifies a host-frozen manifest against every listed source byte, its
profile and both revision identities. The current manifest covers 3205 files.
Host preparation must check both original Git revisions and a clean tracked
tree before producing it; freeze its hash outside the container command. Wrong
revisions, changed bytes, omitted roots, duplicates and escaping paths fail.

For actual motion, first start one original model owner in the matched isolated
N1.6 environment:

```bash
/home/ethan/models/unitree_g1/envs/policy_gr00t_n16_cu13/bin/python -u \
  crates/dev_tools/python/scripts/unitree_g1_mobile_server.py \
  --gr00t-source /home/ethan/Projects/Sai_Lab/upstream/unitree_g1/isaac_gr00t_n16 \
  --model-root /home/ethan/models/unitree_g1/mobile_box/dfe74af855007f26093f362cd2d7a2f404b64b93 \
  --receipt /home/ethan/Projects/Sai_Lab/.scratch/unitree_g1/policy/mobile_box_files.json \
  --runtime-env /home/ethan/models/unitree_g1/envs/policy_gr00t_n16_cu13 \
  --port 5558 --seed 42 --capture-dir .scratch/NEW_CASE/policy_captures
```

After its `/health` reports the expected revision and zero prior calls, add
`--run-source --policy-port 5558 --ticks 1500 --seed 42` to the source command.
Stop this owned foreground model process with Ctrl-C after the finite source
run. The initial model instruction remains the published fixed brown-box task;
this is not arbitrary-target language generalization. Every run needs a fresh
episode and output directory, so source reset faults cannot contaminate a
second episode in the same SDK process.

For a bridge-network source container, retain the HTTP owner at
`127.0.0.1:5558` and use `unitree_g1_source_http_bridge.py` in the host:

```bash
python3 crates/dev_tools/python/scripts/unitree_g1_source_http_bridge.py \
  --socket /absolute/private/mount/pEPISODE.sock --port 5558 \
  --max-calls 30 --output .scratch/NEW_CASE/transport.json
```

The source command receives `--policy-socket /evidence/pEPISODE.sock` instead
of `--policy-port`. This finite transport loads no model, creates no network
listener and forwards request/reply bytes unchanged, with bounded sizes and
timeouts. It records byte hashes separately from actual model inference counts.
Its socket must live in the private evidence mount shared with that container;
host and original image UIDs differ, so only that socket uses mode 0666. After
an early source termination send the framed `{"stop":true}` control request, or
close the owned transport process. Close the model owner separately. Never
reuse an earlier episode's socket, output or model counters.

## Reset-camera result

The first complete task construction captured the actual original 640×480
head-camera RGB and named self-state, with zero task integrations and zero VLA
calls. The brown box and the two hands are visible in the original shelf scene.
The render-only reset refresh changed neither physical state nor SDK counters;
all fifty original URDF mesh files were verified against their parent-bound
receipt. The SDK process completed and released its GPU resources. This proves
the original task/camera preparation path, not grasping or moving-box success.

Earlier evidence retains a 180-second cold camera-startup timeout, missing
`rsl_rl`/`tensordict` registry imports, and the original SDK's newly required
individual STL retrieval failure. The local path adapter now redirects only
verified mesh receipt URLs to their matching cached files, preserving all
original URDF contents and physical parameters.

## First complete policy run and initial-contact isolation

The source run completed thirty actual original N1.6 calls, 1500 controls and
6000 PhysX integrations. All 1499 available post-control trace frames were
matched to the corresponding decoded original actions; the terminal frame
auto-resets and supplies no pre-reset physical snapshot. The original task
reported timeout, with success false. G1 remained upright (minimum upright
0.98715) and moved 1.23515 m, while the box stayed behind. This is a failed
source task, with no carrying or placement claim. Both owned processes closed
and released GPU resources.

The box already acquired 4.70143 m/s upward velocity in the first 20 ms. The
finite `--reset-contact-audit --ticks 10` control uses only the reset measured
upper-body pose, zero navigation and the original Homie controller at .75 m
base height; it calls no VLA. Its box trajectory for the first ten controls
matches the policy-run trajectory exactly. Thus the VLA is not necessary for
this initial launch. The next investigation concerns original scene/reset
contacts, before altering policy parameters or navigation. This explicit
ten-control fixture is not a validated waiting controller or installed native
fallback.

The same fixture also subscribes to the existing box's original PhysX contact
reports, using the [official reporting API][contact_reports], without creating
or modifying report schemas. The observation-only run preserved the complete
ten-control trajectory byte-for-byte but received zero callback events. This
cannot prove absence of contact or identify a colliding actor; the source
callback route is uninformative in this run. Contact geometry/cooking still
needs an independent inspection. No timestep, asset geometry, gains, masses,
initial poses, success thresholds or model weights were changed.

## Actual geometry and source-stack comparison

`--initial-collision-audit` reads the actual SDK stage at reset, with measured
poses for all 53 robot bodies, the box/bin and the original background's dynamic
tabletop and drill. It uses the [official synchronous cooking API][cooking]
without enabling query support, writing USD schemas or taking a physics step.
The initial query covered 305 enabled colliders and exported 171 convex hulls.
Independent convex/box SAT found only six rack hull overlaps, approximately
0.044 mm deep. Robot convex hulls were disjoint; analytic robot shapes were
excluded by disjoint bounds. The ground plane and triangle surfaces remain
explicit limitations, and geometric overlap is not live contact attribution.
The entire ten-control trajectory remained identical to the previous fixture.

The separate `--scene-overlap-audit --reset-contact-audit --ticks 10` explicitly
enables only source scene-query support and reads an inset .0999 m half-extent
box at reset. It hits only the box itself and preserves the same ten-control
trajectory. This excludes a large initial overlap through that query, not all
possible later contacts. Existing sensor forces at control boundaries are zero
and do not exclude a transient substep impulse.

The published-stack ten-control fixture used exactly the same measured initial
robot joints, root, box and bin state. It retained zero VLA calls and 40 original
source integrations. The box moved at most 0.00175927 m, with first-control
vertical velocity -0.0000832623 m/s. The development stack's corresponding
values were 0.550429 m and +4.70143 m/s. Thus the initial launch is sensitive to
the composite source stack. SDK, Lab, Arena, Torch and Warp versions differ;
this does not identify one SDK bug or isolate a single causal change. The
selected next task run uses the published stack rather than tuning around that
launch. No controller, task geometry, reset pose, frequency or model was tuned.

The full published-stack run subsequently triggered its original proximity
success after 19 actual N1.6 inferences, 942 controls and 3768 integrations.
Independent checks matched all 941 available post-control action hashes and
all 19 unchanged transport request/reply byte pairs. Model counters reported
19 successful and zero failed inferences; no prop truth entered requests.
The robot moved 1.18305 m and remained upright (minimum upright 0.992552).
The box moved 1.85548 m toward the bin; actual camera frames at controls 300
and 600 show it between the hands during the move. The owned model and
transport processes closed afterward.

This remains source evidence with inference pauses. The terminal control 942
auto-resets, so the last available physical sample is control 941. Its box
speed is still 1.54792 m/s and its horizontal distance to the bin center is
0.114308 m. The proximity trigger cannot prove a supported, released and
stable two-second placement, native Rapier transfer, continuous 1× execution,
or the required ten-episode score. Those gates remain open. The source-stack
failure is preserved, while the next work is T2 native body/action/geometry
migration using this released reference.

Before that transfer, the original 6.0 Homie stand/walk URLs were independently
hashed and matched both cached binaries (1,886,682 bytes each). A source AST
comparison also found identical `G1_CFG` assignments and Homie reset,
observation, goal and action functions between the two pinned Arena revisions;
the Homie YAML, helper and canonical joint-order files are byte-identical.
The constructor's download-cache flag differs. These checks bind the selected
body contract without claiming identical SDK actuation or native stability.

The audit also retains two failed geometry-reader assumptions (unmeasured
dynamic background and lowercase robot path) and a failed released-stack import
root check. The latter was resolved by verifying the actual installed copied
source roots against all 3205 frozen source files, not by relaxing byte checks.

`--source-profile release_0_2_1 --reset-contact-audit --body-contract-audit
--ticks 10 --run-source` records actual original Homie buffers in
`source.captures/body_contract.json`. This separate diagnostic uses five zero
navigation commands followed by five [.1,0,0] commands, fixed reset upper
targets and the original .75 m height. It reads the pre-control 43 named joints,
516-value history, original network output and all 43 targets actually supplied
to the articulation. It neither reconstructs source observations nor makes a
second inference. Physical state and SDK counters are unchanged by reading.
It cannot be combined with collision/query diagnostics or a VLA service.

The first capture completed 10 controls / 40 original integrations / zero VLA
calls. Its byte identity is
`ea0148e895e4cf7ee662e14eea69f277c36312e7d821fcf198831a1edc0d00c6`.
The Rust ignored test
`g1::policy::tests::released_scene_observation_and_real_onnx_parity` binds this
receipt, both released source revisions, the source-tree manifest, SDK build,
joint names and both original weights. It runs five stand and five walk ORT
inferences with independently accumulated native history. Maximum absolute
errors are 7.1525574e-7 for input and action, and 1.7881393e-7 for all 43
articulation targets, below the fixed 1e-5 tolerance. This proves numerical
contract parity on these actual source frames, not native dynamic stability.
Set `G1_ORACLE`, `G1_ORACLE_SHA256`, `G1_MODEL_DIR`, `G1_ORT`,
`G1_ORT_SHA256` and a fresh `G1_PARITY_OUTPUT`, then run:

```bash
cargo test --locked --offline -p robot_minigame --lib \
  g1::policy::tests::released_scene_observation_and_real_onnx_parity -- \
  --ignored --exact --test-threads=1 --nocapture
```

The published-stack zero-control `--initial-collision-audit` also completed with
zero VLA calls, zero integrations and zero physical-state change. It recorded
305 enabled colliders, 177 cooked convex parts and all 53 measured robot poses.
The corresponding development stack had 171 parts. Specifically, the original
shelf frame cooks to 34 rather than 31 hulls and the shelf surfaces to 93 rather
than 90; all other cooked hull counts agree. This is an observed cooking
difference between the composite stacks, not proof that these six additional
parts alone cause or solve the earlier launch. Do not use the development
cooking as the released reference. The ground plane remains an explicitly
unsupported analytic shape in this geometry reader; the two dynamic background
bodies (tabletop and drill) retain measured poses and require actual mass,
material and owner mapping before native insertion.

The existing all-four-object query/export at SDK
`6.0.0-rc.22+release.33481.407f3ea1.gl` already includes both original T2 assets.
The frozen export identity is
`19eb60783008e3f08d82a1cf402c590395df1e98c4c089fb1247f8ed7d9a88a0`,
bound to the host-path-remapped query
`70ca6c1513c29b0e942535190c11f5e73c742dd281fe41543be51f573ced81bb`.
The .1 kg box is one original cube; the scaled bin is 422.090668 kg and 16
original convex parts at [4,2,1] scale. The independent original 6.0 downloads
match both USD identities. This binds existing geometry to the selected T2
runtime without recooking or modifying its old receipt/qualification fields;
it does not establish native scene contact or placement success.

[contact_reports]: https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/physics/rb_physics.html
[cooking]: https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/extensions/runtime/source/omni.physx/docs/api/python.html


## Released background in the native owner (2026-10-02, cases 0119–0126)

`task_background.rs` now prepares the full released background before any
mutation, then adds 246 static colliders on one fixed group and four colliders
on the actual dynamic drill/table owners. It reuses the existing owner floor;
with G1, brown box and blue bin the same Rapier world has 59 bodies and 305
colliders. The only scene offset is `[0,0,0.795]`, including props, robot, floor
and dynamic owner poses. No runtime pose repair, anchor, grasp constraint or
object truth input was introduced.

The SDK6.0 query completed at zero controls/integrations, with zero original USD
writes and zero dynamic-state change. Its 251 collision nodes comprise 130
fitted boxes, 106 authored triangle meshes, 13 convex/decomposition meshes,
one analytic cube and the existing plane. The SDK cooked 294 convex parts.
The static property query initially failed because it requires a rigid body;
its convex-representation interface also explicitly rejects fitted boxes.
Both failures remain in cases 0119/0120. The successful query uses an isolated,
unattached and never-integrated stage containing exact mesh/approximation/
world-transform copies solely to read SDK fitted-box properties. It does not
add rigid-body APIs to the live source background. Copy cooking is not a claim
of full source-world physics parity.

The table's authored `[1,1,0.7]` scale is baked into its collision and visual
vertices, while measured rigid poses contain no scale. Actual reset table/drill
poses and velocities, including their source startup settling, are retained.
Measured table/drill masses are 220.4559021 and 0.89999998 kg. Actual COM-frame
principal axes diagonalize the measured link-frame inertia; the exporter
checks that basis rather than treating the nine inertia components as a diagonal.
Native materials preserve the original bindings/combine rules. T2 does **not**
receive T1's prestartup finger friction override: the released source call is
confined to the static-task environment. Source G1 IdealPD gains/effort limits
still agree with `g1/actuator.rs`; no gain or friction sweep was performed.

Frozen external files (outside ordinary Git):

- Background query SHA256: `5e08ee1046f1b81df0d2e914973be246fdb995de7a4cb376545da530cc7fd323`.
- `native_t2_background_sdk60_v1.json` SHA256:
  `0ecc6d502967d7cca82bd208b2338dedf86505c781bf5d47551b186fa73b86e6`.
- `native_t2_background_visual_v1.json` SHA256:
  `e77c76cf3b78146da33e845f746218b009121f7aba6d1ca0b78b32892a7bb9b6`.

The physics exporter preserves SDK fitted-box pose/volume without inflating
boxes. Maximum authored-vertex enclosure difference is 0.095701 mm; maximum
relative box-volume difference is 1.042e-6. Static meshes use 113097 authored
polygon-fan triangles, **not an export of the SDK's cooked triangle topology**.
That parity remains unverified. Native contact qualification is false. Visuals
include 122 meshes/17 textures, bind the frozen physics background and update
the table/drill from immutable actual owner frames. Their disclosed mobile
region is `[-1.5,-3.5,0] .. [2.5,1.5,2.5]`; 46 unmapped surfaces remain omitted,
and MDL/lighting/renderer parity is false. The earlier static visual profile
and its three deactivations remain distinct.

Case 0123 exercised exactly 150 real Homie inferences, motor updates and native
20ms integrations with all background contacts. It survived three simulated
seconds (minimum upright cosine 0.9990059), with actual box/shelf support impulse
about 0.01962 N·s and settled bin/table support. It was headless and faster than
wall time; it does not qualify T0, carrying or 1× time. A negative measured mass
file was rejected before adding any partial background to that real world.

The optional mobile diagnostic factory selects the existing **4 nonintegrating
PGS sweep candidate** only for a fresh Homie_v2/N1.6 owner with complete released
background and native force motors. Public loading stays at one sweep. The
render diagnostic uses `diagnostic_constraint_sweeps: 4`, without T1 predictive
limits. There is no 8/16-sweep mobile path or frequency change.

Case 0124 saved actual 640×480 ego RGB and a 1920×1080/MSAA8 window with zero
inferences/integrations. Case 0125 then made two genuine local N1.6 calls and
100 native physics boundaries; case 0126 made eight genuine calls and 400
boundaries. Every saved policy PNG matches the model's RGB byte input exactly,
and captured model inputs contain only ego RGB and the five named joint groups.
Body-policy counts, motor updates and integrations agree at every Tick, with
one 20ms integration and CCD maximum one. Both runs pause between chunks for
camera/model inference, and do **not** claim 1× continuous timing.

Neither run picked up the box. Both had twelve solver-active robot/box pair Ticks,
but final box displacement was only about 0.086 mm and it remained supported by
the original shelf. The eight-call run stayed upright (minimum cosine 0.9937149)
but moved 0.459677 m from its reset XY as the policy began turning toward the bin.
Grasp/transport/stable release and formal T2 8/10 remain unpassed. The owned
N1.6 services were closed after each finite run; failures are not relabeled as
safe-stop task success. No longer repetition is justified before the identical-
action source/native and contact/actuator comparisons resolve the missed grasp.

Reproduction examples (run from `Bevy_Sim2Sim` with frozen files installed):

```bash
flock /tmp/sai-g1-cargo.lock env CARGO_BUILD_JOBS=2 CARGO_TARGET_DIR=/home/ethan/Projects/Sai_Lab/Bevy_Sim2Sim/target cargo build -p bevy_sim2sim --features dev_tools,dev_tools_minigame/g1_constraint_diagnostic
/home/ethan/Projects/Sai_Lab/Bevy_Sim2Sim/target/debug/bevy_sim2sim --scene g1_camera_diagnostic --robot g1 --g1-config .scratch/g1/t2_native_live_rgb_20261002_0125/config.json --g1-ticks 100 --output .scratch/g1/NEW_T2_LIVE_OUTPUT
```

Start the one original local N1.6 owner first using the exact `model_command`
in case 0125/0126 `execution.json` with a fresh capture directory; do not reuse a
foreign process at port5558. Close only that owned process after the finite test.
The preserved drivers contain the exact preparation, source query/export,
start/stop, binary/config/source identities, budgets and result locations.


Case 0127 isolates the action stream from visual decisions: exactly the two
actual native N1.6 output chunks from case 0125 were executed for 100 released
source controls / 400 original200Hz reference integrations, with **zero new
model calls**. This is explicitly a causal action replay, not autonomy, a VLA
rollout, task recovery or qualified success. Both streams retain their original
50-frame/20ms action identity. The original source stayed upright (minimum
0.9962804), but the source box moved 13.7639 mm while the native box moved only
0.0859 mm. Source final box height after the disclosed offset was 0.8794577 m,
versus native 0.8657259 m. Final root positions differed by about
`[-.011707,-.010039,.004160]` m, and canonical joint state differences reached
0.220898 rad (left-arm deltas up to about0.0233rad). No grasp qualification is
inferred from that source movement. Contact outcome differs under identical
commands; actual shape rest/contact offsets and contact-force/pose mapping must
be checked before selecting a corrective change. This evidence does not justify
changing gains, formal frequency, replaying a success trajectory or repeating
longer native runs.


Cases 0128/0129 queried the original SDK contact/material and explicit actuator
inputs at reset, with zero integrations, model calls, schema writes or physical
state change. The first query failed to encode the schema's automatic `-inf`
offset sentinel; the corrected query preserves it as a tagged value, never a
finite physical offset. All 52 robot shapes use friction0.5/restitution0;
box friction5, bin friction1; all effective rest offsets are0. Source effective
contact offsets range0.0004905–0.00285431m for the robot,0.004m for the box and
0.001060592m for the bin. These are **not** Rapier contact skins or its0.02m
prediction distance. Source gain/effort/velocity values cover all43 joints and
agree with the frozen Homie parameter table; equal inputs do not establish
implicit50Hz/native versus explicit200Hz/source actuator equivalence.

A closer independent read of cases0125/0126 shows their twelve reported
robot/box pair Ticks were **speculative**, with positive gaps about15–25mm and
exactly zero normal impulse. They do not demonstrate touch or gripping. In the
identical-action source replay the box already rises from the first control,
reaching8.762mm above reset at control50; the13.764mm final movement must not be
credited to grasping. Cases0130/0131 retain actual SDK link poses under the same
100 commands: palm origins differ from native by up to37.7mm, distal finger
origins by up to45.8mm. No model/truth contract was changed or new VLA call made.

Original contact-report APIs were already present; observing them made zero
schema writes but returned no callback events. A second read-only contact tensor
view bound the actual box and all16 hand bodies, with zero state/counter change.
Its **support-force positive control failed**: net box force was also zero while
the box was physically supported. Therefore neither callback absence nor zero
hand force is treated as proof of no source contact. This instrumentation remains
explicit acceptance-only evidence and cannot qualify grasp or policy recovery.

The actual reset RGB comparison then exposed a deterministic visual exporter
error: all49 source body visual meshes have bound `UsdPreviewSurface` materials
(48white and onegray), but the old exporter read only absent `displayColor`,
substituting gray-purple. This source-authored material omission is the next
bounded correction. Tracking/cooked-shape and timing gaps remain independently
open; no friction/gain/PGS/frequency or longer-run sweep follows this observation.


Source body material repair (cases0132–0138) preserves all49 mesh point/index
arrays and the53-body order exactly. A fresh unused physical export differs
only in exporter identity; the installed physics cache is untouched. The new
external visual file is `homie_v2/g1_visuals_bound_material_v1.json`, SHA256
`298d427d9bf3078a212f256faee95ff7ba23216a26032ac32acd7101b4d70220`.
The exporter now reads the actual bound shader's `diffuse_color_constant`:
48white meshes and onegray mesh. Old visual files keep their original behavior.

The color-only native preview0133 still applied station cool bands/hatching.
Bound-source materials therefore now use standard PBR without the extra ink
shell. Roughness0.6/metallic0, illumination/exposure/camera, object/background
materials, geometry and every physical parameter are unchanged. RTX/MDL parity
and preservation of the body's authored face-varying normals remain unproven.
This fixes an authored material/shader mismatch, without adjusting sensor pixels,
lighting values, model preprocessing, source images or action semantics.

PBR preview0134 first failed with0integrations because the shared readiness
observer awaited unused enamel/ink pipelines. The source-only diagnostic now
awaits the actual StandardMaterial fragment shader, while the existing station
entry still requires both enamel and ink. Actual GPU preview0136 passed with
49bound-material meshes and zero model/control/physical updates; legacy GPU
preview0138 also passed with the original visual cache. The current binary SHA
is `234188af7b49b0505a94ee2c89f1c891ff34816f9ca3605b09ebb49ca044e314`.

Case0135 used two genuine N1.6 calls/100nativeTicks under the repaired visuals.
Unlike the old gray fallback, it applied nonzero hand/box impulses at five Ticks
and moved the box about10.09mm, but did not lift it off the shelf. The original
source live task first makes a substantial lift around controls150–200; a
100Tick prefix alone is not a valid grasp-failure benchmark. Case0137 therefore
covers exactly four calls/200Ticks, without a longer or parameter sweep. Its
first100joint states are byte-for-byte identical to0135. It stays upright,
applies positive hand/box impulses at17Ticks and raises the box at most29.70mm,
but the shelf remains a support at **every** Tick. At200 it rests on the shelf;
held lift/carry/release remain unpassed. The corresponding original live source
box height is1.006593m after the disclosed offset. This outcome motivates one
explicit diagnostic transplant of the original four source command chunks;
replay results cannot count as autonomous task or benchmark success.

Both finite services were closed after use. Actual model NPZ input RGB matches
the saved native PNG bytes; only the named joint self state is included alongside
RGB. One integration/20msTick and distinct VLA/body-policy counts are verified.
Final source-material/PBR Rust library checks passed172tests/34ignored, the app
CLI check passed1test, and the source/contact Python checks passed26tests.

Case0139 is the finite mechanical control prompted by0137: the first four
**original** source replies at0/50/100/150 were copied with SHA256 identities,
validated through the matched task action contract and executed in a fresh native
T2 world. Their action values, source episode and simulation timeline are
unchanged. A test-only copy explicitly replaces the old wall admission stamp;
this is disclosed in every chunk receipt and never enters runtime execution.
The fixture SHA256 is
`61868bb49efb02c0e297ae15d04e298bafeb5649656428a4f8f0e349eecd9b88`.

This mechanical control **passes** the frozen held-lift rule. All200 actual
Homie calls/control updates/integrations complete at the unchanged50Hz/one-step,
4PGS diagnostic settings. There are0 new VLA calls. Box hand impulses occur at
85Ticks; positive shelf support ends at177. At182–200 (19 consecutive samples)
the box is at least50mm above reset height, has real hand impulse, has no positive
shelf support impulse and the robot remains upright. Maximum rise is80.35mm,
minimum upright cosine0.99524; the final box is still held above the shelf.
The test is free-running (0.715s wall time for4s simulated time), so it proves
neither autonomous execution, continuous1×, stable final placement nor carrying.
It cannot count toward either formal10-episode task acceptance.

Reproduce this exact mechanical comparison using fresh output paths and the
frozen runner/action files:

```bash
G1_CODE_COMMIT=$(git rev-parse HEAD) \
G1_MOBILE_REPLAY_CONFIG=.scratch/g1/t2_native_original_source_grasp_20261002_0139/runner.json \
G1_MOBILE_REPLAY_CONFIG_SHA256=08bc9d2932dc50186cdc65ec41e7214c249f7a183f61c299aed4a4e22b4fdfe7 \
G1_MOBILE_REPLAY_ACTIONS=.scratch/g1/t2_native_original_source_grasp_20261002_0139/actions.json \
G1_MOBILE_REPLAY_ACTIONS_SHA256=61868bb49efb02c0e297ae15d04e298bafeb5649656428a4f8f0e349eecd9b88 \
G1_MOBILE_REPLAY_OUTPUT=.scratch/g1/NEW_MOBILE_MECHANICAL_RECEIPT.json \
CARGO_BUILD_JOBS=2 \
CARGO_TARGET_DIR=/home/ethan/Projects/Sai_Lab/Bevy_Sim2Sim/target \
flock /tmp/sai-g1-cargo.lock cargo test -p simulation_minigame \
  --features g1_constraint_diagnostic --lib \
  real_mobile_source_grasp_window_diagnostic -- --ignored --nocapture --test-threads=1
```

The reset input comparison independently finds all five source/native self-state
arrays **exactly equal** (all zero), with the same shapes/dtypes and fixed
instruction. The RGB differs, and the two first decoded arm chunks differ by
up to0.2243rad/0.1973rad. This prioritizes a finite paired real-model image
comparison with identical random-generator state, rather than another physical
parameter or longer-rollout sweep. Successful source actions establish that
this native grasp window is mechanically possible; they do not establish the
whole2m carry, or rule out tracking/contact limitations later in that task.

Case0140 performs three actual N1.6 calls in one freshly loaded frozen owner:
source reset RGB, native reset RGB, then source RGB again. Each call restores
the exact saved CPU/CUDA random state from immediately after model loading;
all five measured state groups and the fixed instruction are identical. No
pixels, camera poses, preprocessing, weights or physical state are changed.
Both original first replies are reproduced **bit-for-bit**, and the repeated
source reply is also identical. Replacing only the actual input image exactly
reproduces the0.22428rad/0.19726rad arm-command difference. This establishes a
causal image effect on the initial command; it does not claim a particular
material, light or later feedback difference is the sole cause of task failure.
There are3 actual model calls,0 physics steps, no HTTP service and no autonomous
rollout. The owned model is closed after the24.62s bounded comparison. The next
render repair must use source-authored data and a new finite live grasp test;
substituting source screenshots is not a task capability.

Cases0141–0144 locate and repair one concrete background material import bug.
Read-only camera rays identify the gray left-hand foreground as the original
`hesai_box_06`. Its shader is **MI_LampCeilingA**, SHA256
`2e2aef644544b619f5df253eacbaed941be89bc1018fb58cbb47d0e8ac1d4192`,
with UE4 `AlbedoTexture/MainNormalInput/MergeMapInput`, linear tint,
desaturation and a green-channel roughness remap. The prior generic OmniPBR
reader ignored these differently named inputs and silently produced gray.
Only this inspected graph is explicitly mapped; unknown MDL parity is unproven.

The new external T2 visual file is
`native_t2_background_visual_warehouse_v2.json`, SHA256
`1243478cc15a87229a7f348699c7e0b9fbe605a4643a83931c79079db6279c66`.
All122 geometry/normal/UV/triangle/owner arrays, all other119 materials and the
46 disclosed omissions are identical to v1. Exactly three original cardboard
materials now retain their three byte-bound2048x2048 textures, authored tint,
linear desaturation and roughness remap. Their albedo alpha is255 everywhere;
the source graph does not use the merge texture's red channel as occlusion.
No lighting, exposure, camera, model or physical parameter is changed.

Three actual-source USD regression tests pass, including malformed graph entry
and nonfinite desaturation rejection. Rendering checks pass31tests/2ignored;
the development app build and actual zero-Tick GPU capture0143 pass, with0model
or physical updates. The new binary SHA256 is
`bd6cc5ee48ce181c68768cbe43962463b58daa0e77567d89f1111bc209175d91`.
Old files keep the original fragment behavior through a zero/default extension.

The same frozen200Tick/four-real-call visual grasp budget0144 **fails**. All200
real body calls/updates/integrations complete; minimum upright cosine0.993606,
maximum robot XY drift0.174337m. The box remains shelf-supported at everyTick,
with0positive hand impulses and at most0.038mm rise. Relative to the source first
chunk, arm maxima improve from0.22428/0.19726rad to0.12878/0.11666rad; this
numerical proximity is not a task success measure. It does not prove the
remaining lighting, normals or feedback gap is harmless. The service is closed
and the full actual input/response/trajectory records are retained. No longer
rollout or parameter sweep follows this negative result; source actual render
settings and source-authored normal correspondence are the next bounded audit.
