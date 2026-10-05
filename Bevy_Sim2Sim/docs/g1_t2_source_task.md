# T2 original source diagnostic

The opt-in native `current_regrasp_stability_probe` adds exactly two100-Tick
stationary holds after the separately journaled regrasp lift. It requires
`current_regrasp_lift_probe`; each hold preserves the completed lift's original
joint targets and needs a fresh actual paired RGB at850,950and1050. The20mm
bound applies both to the850postload reference and each consecutive interval.
The earlier650-to850 failed pickup verdict remains unchanged. This diagnostic
admits no carry/release and is not a task success or a continuous1× result.

The separate `verified_regrasp_pickup` station option permits the existing
transport/placement route only after both postload image checks complete. Its
fixed transport reference is the actual1050Tick image, with the original20mm
cumulative slip bound. The original rejected650-to850 entry stays in the log.
Transport uses the previously declared bin-placement primary and pregrasp
secondary cameras, with their original extrinsics and80mm vertical baseline;
neither learned-policy camera changes. Scan/carry stage expiry is relative to
the actual new entry boundary; the global3300Tick budget remains unchanged.

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
G1_MOBILE_REPLAY_CONFIG="$PWD/.scratch/g1/t2_native_original_source_grasp_20261002_0139/runner.json" \
G1_MOBILE_REPLAY_CONFIG_SHA256=08bc9d2932dc50186cdc65ec41e7214c249f7a183f61c299aed4a4e22b4fdfe7 \
G1_MOBILE_REPLAY_ACTIONS="$PWD/.scratch/g1/t2_native_original_source_grasp_20261002_0139/actions.json" \
G1_MOBILE_REPLAY_ACTIONS_SHA256=61868bb49efb02c0e297ae15d04e298bafeb5649656428a4f8f0e349eecd9b88 \
G1_MOBILE_REPLAY_OUTPUT="$PWD/.scratch/g1/NEW_MOBILE_MECHANICAL_RECEIPT.json" \
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

Cases0145–0146 add an exclusive released zero-step render audit, rejecting model,
motion or other-query combinations before source access. The actual SDK read
records eight rectangular-light transforms and modern-schema default inputs, the existing scene DB
ambient settings, postprocessing inputs and87robot meshes including invisible
collision/proxy meshes. It preserves the complete physical state and bothSDK
counters exactly at0, with0USD/render-setting writes and0model calls. No dome or
native directional-light equivalence is inferred. Queried global tone-mapping
settings include operator6/whiteScale40.2, histogram disabled and sceneDB ambient
color(0.1,0.1,0.1)/intensity1; null settings remain explicitly unavailable. These
are measured inputs, not proof that RTX/Bevy photometric units or sensor pipelines
are equivalent. The owned source process closes after16.45s.

Cases0147–0150 establish a separate deterministic normal-generation bug.
Bevy's default angle weighting rejects squared edge-length products below
`f32::EPSILON`, discarding millimeter-size source triangles expressed in meters.
In the six actual wrist/palm/elbow meshes, it produces zero normals at94–99% of
face corners (bothpalms above99%). Their original authored normal vectors are
finite and use right-handed winding. A real small-triangle regression first
fails under the old method, then passes under the new source-only angle weighting.
The correction evaluates angles in f64 and discards truly degenerate edges,
without modifying a point, index, transform, body or physical parameter.
Legacy station-style G1 files retain the existing normal behavior.

The real Rust comparison against the six hash-bound source meshes passes at all
307056face corners with0zero normals; mean source-direction cosine exceeds
0.999999996 in everymesh. The 40.03MB fixture SHA256 is
`7a75aabecf545e66d84137af9aa1c80396f030825bd2776de5ed24b9015d2a5d`.
The first ignored-test invocation used an incorrect Cargo-relative fixture path;
that harness failure remains recorded, followed by the successful absolute-path
invocation. Rendering tests pass32/3ignored and the app rebuild passes. Actual
zero-Tick source-normal GPU capture0149 passes, with binary SHA256
`f21f835bc73eefada2c98b6f861483607f16782ddc09212fc78009b7b1bfcb7a`.
The target-box pixels remain exactly unchanged from0143; hand-region pixel
changes are small, so corrected normals do not establish brightness parity.

The fixed four-call200Tick realRGB loop0150 again fails held lift:0hand impulses,
200shelf-support samples, at most0.038mm rise, minimum upright0.993468 and maximum
XY drift0.180417m. All200bodycalls/updates/integrations and actual model pixel
bytes are verified. The owned model closes. This rejects normals as a sufficient
explanation for the present grasp failure, while retaining a validated rendering
repair. Further longer/physical-parameter sweeps remain at0; the next comparison
must be justified by actual source photometric/postprocessing data.

Cases0151–0155 resolve two source-data interpretation gaps without changing
formal physics. The actual RobotHeadCam and its640x480 RenderProduct use
`acesApproximation`, exposure:fStop5, ISO100, exposure time0.02s,
responsivity1.102670908 and disabled autoexposure. These camera attributes agree
with the measured global settings; they supersede global exposure controls as
described in [NVIDIA's camera documentation](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/cameras.html).
The packaged original RTX UI maps operator6 to ACES and exposes whiteScale
only for Hable(operator5). Thus40.2 is not an ACES exposure multiplier.

The native diagnostic explicitly used `Tonemapping::None` and no HDR
intermediate. An opt-in `diagnostic_aces_fitted` comparison now enables Bevy's
HDR intermediate and fitted ACES in the matched mobile-background diagnostic,
leaving all public scene defaults unchanged. The GPU receipt reads the actual
ego-camera components (`ego_camera_hdr=true`, `ego_camera_tonemapping=aces_fitted`).
It does not claim that fitted ACES equals the closed RTX approximation.
The zero-step0153 GPU capture passes. The one four-call200Tick0154 run still
fails held lift:0positive hand impulses,200shelf-supported ticks, at most0.038mm
rise, upright>=0.993600, drift<=0.178126m. All actual model pixels and200single
20ms integrations are independently verified; the owned model closes. Exposure,
lighting and every physical parameter remain fixed in this isolated comparison.

The initial light audit missed the old USD attributes, reading only modern
`inputs:*` schema defaults. The complete authored-source query now records the
legacy `intensity=200000`, `width=200`, `height=28` and warm color on all eight
original ceiling RectLights. Their world scale produces a1.96x0.084m panel,
rather than the modern un-authored1x1 fallback. Case0155 performs one bounded
source-only light positive control: baseline, explicit modern intensity0,
legacy intensity0, and restored original values, with six render pumps each.
Both zero conditions reduce mean RGB by about44/255; restored light values
recover the bright box/hands. The restored image differs from baseline by
6.397/255, so temporal rendering has not reproduced identical pixels and this
is not an exact attribute-precedence/shader-equivalence proof. It demonstrates
that these eight source lights materially affect the original camera and
cannot be replaced by an unrelated directional light without validation.
Only temporary light intensity overrides were made in the ephemeral source
session layer, then cleared; original attribute values are restored. Actual
robot/prop state and both physics counters remain exactly unchanged at0.
Parser isolation checks pass10/10. No native lighting sweep, training, model
change, autonomous carry, placement or1x qualification is claimed.

Cases0156–0160 add a strictly opt-in source-panel renderer comparison. It loads
only the frozen actual SDK light-query bytes(SHA256
`d23f8c54bcbdfcc394efb0a23b0273c3c7379152b66361dc773e4733f33a482b`,
bounded512KiB), verifies the runtime, eight paths, legacy inputs and transforms,
then creates four fixed spatial samples per panel. There are no physical
substeps or object/body edits. The on-axis patch intensity is L*A/4 candela;
Bevy's point power parameter is4pi times this value,103446.37 per sample.
This uses [the USD light luminance convention](https://openusd.org/dev/api/usd_lux_page_front.html)
and the installed Bevy point-light conversion. No brightness is fitted to a
task result. All source panels, including those outside the selected visual
mesh region, are retained.

The initial hemisphere spot approximation produces severe grain/shadow
suppression in zero-step0157 and is not sent to a model. Its almost180degree
perspective shadow projection is ill-conditioned (about10000m half-width at1m
distance); this is a geometric explanation, not a separate shadow-off causal
proof. The final implementation uses32point emitters, ordinary cube shadows
and one generated64pixel-per-face linear Lambertian angular mask. The mask's
UV/face directions follow the actual Bevy shader; its forward/black-backlight
regression passes. This remains finite spatial quadrature with20m light range,
raster shadows and noRTXGI, not exact RTX area-light parity. The existing
ambient450 and EV1009.7 are held fixed for this isolated light comparison.
The native directional light is omitted only in this explicit profile.

Bevy's light-texture shader path needs an explicit build feature. Zero-step0158
rejects the incomplete earlier build before any physical/model call; its
"GPU lacks" error was actually caused by the missing compile feature. A new
development-only `g1_source_lighting` feature enables `bevy/pbr_light_textures`;
the configuration now rejects builds lacking that feature before loading a
worker. It also requires actual GPU binding-array support and refuses an
isotropic fallback. Source light clustering reserves the measured131072index
capacity before extraction. Default game features/scenes remain unchanged.

```bash
cargo build -p bevy_sim2sim --features dev_tools,dev_tools_minigame/g1_constraint_diagnostic,dev_tools_minigame/g1_source_lighting
```

Supply `diagnostic_aces_fitted=true` and a `diagnostic_source_rect_lighting`
object containing the absolute actual-query `path` and SHA256 above, alongside
the original mobile background configuration. The zero-step0159 GPU capture
passes at1920x1080/MSAA8 with a real640x480 ego image; actual camera HDR/ACES
and GPU mask support are recorded. Binary SHA256:
`824d059652afa72b8b399a08c01a28ae80f03a9a50dddef4ea0d8357bd8a7d09`.

The first realRGB four-call200Tick comparison0160 **passes the bounded held-lift
stage**: maximum box rise0.115835m,84positive robot-contact samples,
15consecutive held-lift ticks186–200, and no background support in those ticks.
Minimum upright is0.994435 and maximum XY drift0.136773m. The exact same held
rule used in source-command diagnostic0139 is retained (>=50mm rise, positive
robot support, no background support, upright>.95). All200actual body policy
calls/control updates/single20ms integrations and all four real model calls
are verified. Captured PNG/model NPZ pixels are identical; only RGB and named
self state enter N1.6. There is no source-action replay. The owned service
closes after54.06s. Native step trace SHA256:
`dc414ab60a0ed20cd71d89bd3bfbd31a724fb03ade0a4e8f1b1a6ea8bec4f769`.
This enables a next finite carry/release-stage test. It does not qualify the
complete T2 skill, science station,1x timing, strict placement or8/10 benchmark.

The expanded-feature regression0161 passes174workspace library checks with
37explicitly ignored resource/experiment cases, plus the app CLI check and the
actual frozen-panel energy/basis comparison. Zero-step0162 also passes the
legacy station shader configuration at1920x1080: actual HDR=false,
tonemapping=none, original450ambient/15000directional, and no source-panel
profile. No physics or model calls occur in that legacy GPU check.

The independent contact-ID audit0160 maps every positive supporting body in
ticks186–200 to original left/right palm or finger links; both hands support
the object throughout those15ticks, with no torso/root/leg support. It is
acceptance data only and is not sent to the policy.

Case0163 records an admission failure: the old CLI rejects1000Ticks before
renderer/physics/model inference (the owned service had been loaded, then
closed with0successful calls). Case0164 instead uses the existing400Tick/eight
call budget without parameter changes. It preserves a214Tick consecutive
held-lift interval through tick400, with minimum upright0.994079. The box moves
0.598388m mainly while the body turns; root XY displacement remains below
0.137949m. Walk-body policy is selected at236ticks, including200held ticks.
This proves sustained grip through turning/control transitions, not a2m carry
or release. All400integrations/control/body calls and eight realRGB model
inputs are verified; the owned service closes after60.22s. Trace SHA256:
`5717e16d93f91a896878f219e9707c3f5021e119b4d2f0a1c054b602da40dc89`.

The initial `g1_mobile_carry_diagnostic` entry admits exactly1000Ticks and
20original50frame chunks, with a finite120s application timeout. It requires
the matched mobile T2 physics/background, the verified source-light profile,
the existing4PGS development candidate and no UI/static/prefetch combination.
The normal camera/static budgets remain400/150; CLI argument ordering cannot
bypass them. Two CLI checks pass, including ordinary-scene rejection of1000
and carry-scene rejection of other budgets. This is a bounded stage entry,
not a registration of qualified user-facing execution.

```bash
cargo run -p bevy_sim2sim --features dev_tools,dev_tools_minigame/g1_source_lighting,dev_tools_minigame/g1_constraint_diagnostic -- --scene g1_mobile_carry_diagnostic --robot g1 --g1-config /absolute/mobile_20_chunk_config.json --g1-ticks 1000 --output /absolute/new_evidence_directory
```

Carry-stage0166 completes20real N1.6 calls and1000actual native single20ms
integrations. The robot remains upright (minimum0.990024), travels1.128330m
from reset XY and carries the box1.796482m from reset. There are883positive
hand-support ticks,848with both original hands; no supporting robot contact
belongs to a torso/leg. The maximum uninterrupted raised-held interval is356
Ticks. The final box is[-0.322805,-1.393144,0.907623]source, still held, moving
0.272038m/s and1.041169rad/s. No bin-support impulse occurs. Actual PNG/model
NPZ inputs remain byte-identical and contain only RGB/named self state. The
owned model closes after83.07s. Trace SHA256:
`81d3e0f9edbc3432c634b2f0b5075dba4595f03c225732c940b761582d579e46`.
This proves native physical carry beyond the original reaching range, but
not the strict2m/release/stability task or1x timing.

Inspection at the fixed endpoint finds a partial approach to the bin, no drop
or fall. Chunk19 lowers the pelvis, chunk20 raises it and commands forward
motion again; the original policy has not completed its placement phase.
The dedicated carry entry consequently retains the1000/20budget and adds one
fixed1500Tick/30chunk release-stage budget, with the same120s timeout. All
scene/model/light/physics/gain/friction/seed parameters stay fixed. Arbitrary
longer budgets remain rejected; this is one justified continuation-stage test,
not an unbounded repeat/parameter sweep.

`unitree_g1_mobile_placement_audit.py` independently uses the upward prism over
convex part2 of the frozen original bin: its flat interior floor, excluding the
rim. It transforms every original box collision vertex into the actual moving
bin frame. Every vertex must fit that footprint and lie above its lowest local
floorZ. All robot links must be physically separated, the bin must supply a
positive upward normal impulse, and strict0.02m/s/0.1rad/s speed limits and
standing must hold for101consecutive50Hz samples spanning2s. Fall history is
sticky. This target is a disclosed diagnostic rule, not the frozen ten-episode
formal suite. Actual old0166geometry fails the rule and records0s placement;
its absent bin contacts require no inference of missing impulse directions.

The physical trace now publishes the last-solve normal impulse acting on each
object in source Z-up coordinates, excluding friction. This is read-only
acceptance data and never enters model observations/control. A100step real
gravity-contact check verifies its upward sign for both collider insertion
orders (Rapier's solver applies `-normal` on collider1). Eight adversarial
Python checks cover moving/rotating targets, rim overhang, side-only support,
unknown/touching robot distance, strict speed thresholds, sticky falls, missing
Ticks/resets and the101sample duration rule. Both CLI checks also pass.

```bash
python crates/dev_tools/python/scripts/unitree_g1_mobile_placement_audit.py --definition /absolute/frozen_task_objects.json --trace /absolute/native/owner_steps.jsonl --output /absolute/new_placement_audit.json
```

The single release-stage trial0168 completes30real RGB N1.6 calls and1500
single20ms integrations but **fails**: first loss of hand support occurs at727;
the box ultimately lies on the floor outside the target. Minimum upright is
0.990278; maximum box displacement1.582313m. Independent original-interior-floor
placement remains0s, including support-direction checks. Owned N1.6 closes after
99.54s. Trace SHA256:
`de83f4fe43e715ea0dd7b2787f73f6105013d4487dc8f55f2a75be7f983a72ed`.
No further longer VLA budget is admitted by this diagnosis.

Actual copied source files0169 match the frozen released source tree0109:
T2 success checks only absolute object/bin XYZ proximity (0.260/0.130/0.150m).
It does not check release, supporting contact or speed/stable duration. This
explains why source proximity success is insufficient evidence; it does not
establish what release examples were used during weight training.

Finite input control0170 runs four actual N1.6 calls and zero physics steps.
Two second-frame inputs from0166/0168 have identical named self state and differ
in only one RGB component by1/255 at[y362,x189,R]. Restoring the same CPU/CUDA
RNG after their identical first call exactly reproduces both original second
outputs; an A-repeat is bit-exact. The image change alone produces maximum
arm-target differences0.003960/0.003431rad. This establishes the initial action
branch's input sensitivity, not the sole cause of the eventual slip. Inputs
are exact captured pixels; no quantization/filtering/normalization workaround
is installed. The model closes after23.40s.

`real_mobile_fixed_grip_body_carry_diagnostic` is an ignored, hash-bound
mechanical comparison. It holds the exact last saved upper targets after a
200Tick grasp fixture, changes only body navigation, retains one native
integration per Tick and never performs pose writes. Original source-grasp0171
keeps grip through fixed turn/walk/stop (final701consecutive held ticks;
minimum upright0.991701), but the350Tick timed walk covers1.105823m rather than
the required2m, so its test correctly fails. Commands0.4rad/s turn/0.3m/s walk
are command values, not claims of achieved speed. No VLA runs in this check.

Matched native-grasp0172 copies the exact first four real replies from failed
0168. Its first200body/joint/action/object/contact samples reproduce0168
exactly. Holding the last upper targets while steering toward+pi/2 using only
robot quaternion and measured velocity still drops the box at473Ticks,
minimum upright0.993887. A single straight-back comparison0173 retains the
same captured heading and commands-0.3m/s with the same self-state feedback;
it drops at1071Ticks after1.262310m root travel, minimum upright0.994449.
Both stop on the independent failure guard, remain unqualified, use0new VLA
calls and retain failed traces. The feedback cases have fixed750/1000/100
maximum turn/walk/stop budgets and abort instead of repeating indefinitely.
Robot/object truth is used only for logging/independent failure abortion;
navigation commands read robot self state, not prop/contact/acceptance data.
No production navigation or grip controller is registered by these tests.

The200Tick boundary comparison provides a next geometric question: original
source-grasp palms are0.243239m apart with box center about5.25mm above their
mean height; native-grasp palms are0.234067m apart with box center about51.35mm
above them. Native total positive robot normal impulse is0.743422N.s versus
source0.562707N.s, so "just insufficient lateral squeeze" is not established.
The next diagnosis examines actual wrist/contact geometry and retention;
there is no new lighting/gain/friction/PGS/long-VLA sweep.

The offline geometry audit0174 evaluates the frozen G1 joint frames and
validates forward kinematics against all53actual native body poses (maximum
position error2.59e-7m). The source-grasp desired palm gap is0.156043m; the
native-grasp desired gap is0.125295m despite an actual gap0.234067m. Thus simply
increasing closing pressure is not justified. A single contact-cache
counterfactual0175 disables recycling only after the identical200Tick grasp:
it drops earlier, at453rather than473Ticks. This negative remains recorded;
normal cache behavior is retained.

A classical grip candidate0176 instead uses the measured source desired gap,
preserves the native desired palm midpoint and both orientations, and solves
only the two seven-joint arms within original joint limits. Lower and finger
targets remain unchanged. Maximum target change is0.042889rad; position and
orientation residuals are below1e-6. The28target file SHA256 is
`8b03a5513d15c941452535f7f38ef9ef759c1e4af0b382cf1e94ec7d6d5be784`.
This is explicitly a traditional controller correction, not original VLA
output. It is not registered in the production task/UI.

The600Tick retention test0176 still **fails** its unchanged400consecutive
held-Tick rule: brief source-scene box contact ends at359, leaving only241
final held Ticks. It does retain the box through the former473Tick drop point.
The longer bounded body test0177 retains it for590final Ticks but fails the
unchanged750Tick turn deadline at949. Its recorded last150Ticks switch between
stand and turn14times at the0.06rad acceptance boundary; only6consecutive
samples meet the20sample requirement. Original Homie selects stand below a
strict0.05 command norm.

A test-only turn controller now enters stand at0.03rad and resumes turning at
0.06rad, keeping the acceptance tolerance0.06rad,20consecutive samples, and
750Tick deadline unchanged. Its regression covers both turn directions and
the observed boundary oscillation. Trial0178 completes turn/walk/stop with
continuous hand contact but correctly fails physical distance: walk1.991714m,
settled1.974342m, although own-velocity odometry reaches2m. The old failed
result is preserved. A5cm control margin, derived from this measured stopping
and odometry discrepancy, changes the command goal to2.05m; independent
physical acceptance stays2m. Attempt0179 fails compilation before any model
or integration (oversized JSON receipt macro); splitting that receipt fixes
compilation without increasing crate recursion limits.

The single corrected trial0180 completes1391actual Homie calls, torque
updates and native20ms integrations,0new VLA calls. Independent pre-walk root
position audit measures2.041199m walking and2.027090m after100stop Ticks.
All694walk/stop samples have positive original palm/finger support, no other
robot-body or scene support, box height at least0.821997m, and minimum upright
0.992599. Trace SHA256:
`139262d1f87dabc0108a318ec14d7626b865a28a34a896eacab04f7c32f361fa`.
This passes only the disclosed saved-grasp/classical-grip/self-state-navigation
mechanical2m check. It does not establish live-RGB autonomous carry, movement
toward the source bin, release, formal8/10, science-station/Qwen execution,
terrain or1x timing. All cases retain one integration/Tick,50Hz, original
masses, force gains, friction, default contact caching and the same4PGS
nonintegrating diagnostic setting. No training or further live VLA is run.

`mobile_grip` now computes this bounded correction directly from the original
G1 joint frames, measured lower-joint positions and an admitted mobile VLA
upper command. It takes no scene/root-position/object/contact input. Original
lower/finger targets, navigation and pelvis semantics remain unchanged;
unreachable, nonfinite, incoherent-clock or out-of-limit inputs fail before
actuation. Cross-language frozen real-grasp check0182 differs from the
independent offline candidate by at most5.96e-8rad, with five negative-input
checks passing and zero inference/integration.

`mobile_navigation` holds an explicit observation-stamped goal and consumes
one current-episode quaternion/velocity measurement per50Hz boundary. It
rejects repeated/skipped/foreign states and has finite750/1000/100Tick phases.
`mobile_assist` provides distinct `OriginalVla` and `ClassicalCarry` commands
and execution receipts. It owns the existing `ArenaTaskRunner`, preserving
the same Rapier world, models, bounds and all original VLA bytes. Handoff must
follow a complete original chunk and a fresh observation of the current
physical boundary. Goal mutation, expiry or failed correction halts; reset
rebuilds all histories. None of these components is included by default or
advertised as a qualified UI skill.

A finite owner now requests pause on its real completed boundary, invalidating
its command before remaining catch-up Ticks. The scheduler regression checks
an80ms debt, one actual transport-fixture boundary, retained debt, no second
integration and no resumed work from that old command. Existing owners retain
the default behavior. The timing ledger can be serialized without altering
clock accounting.

Actual runtime-calculated grip0183 and the typed owner0184 both complete1524
native steps, walking2.037898m and settling2.019864m. All692walk/stop samples
have only positive original hand contacts, no scene support and upright at
least0.993010. The actual background worker0185 reproduces all1524body states
bit-exactly, including the200Tick original native grasp. It records every
completed Tick, drops zero trace records, misses zero control deadlines,
auto-pauses at completion, and adds zero Ticks during the following second.
Active simulation/wall time is30.48/30.499902s (ratio0.999347). Initial load,
chunk-boundary pauses and the final observation second remain disclosed.
This is a scheduler/mechanical result with0new VLA, not a rendered continuous
1x task or autonomous qualification. Its physical trace SHA256 is
`eda1930073a3a12bc91308c3dfc27f1a70ed43b10445083df2be4f714bb22530`.
The first background-test compilation lacked timing serialization; that
zero-step failure log is preserved. Final checks0186 pass62simulation checks
with the explicit diagnostic feature (33ignored real-fixture checks), and
56default checks (25ignored); no default assist or physical setting changes.

The separate `g1_mobile_assist_diagnostic` Bevy entry admits a maximum2050
native Ticks and exactly four **fresh** RGB N1.6 grasp calls, then requests an
actual ego image at the200Tick boundary before admitting one manually specified
2m clear-aisle goal. It has no saved-action/upper-target fixture input. The
same typed worker computes the traditional correction from actual self state,
navigates and auto-pauses; raw VLA replies remain unchanged and classical
commands/receipts are explicitly separated in every owner record. The goal
origin is manual diagnostic configuration, not Qwen selection or model output.
All normal camera/static/carry entries retain their existing budgets. This
entry does not qualify source-bin placement, autonomous goal choice, formal
8/10, continuous rendered1x or UI execution.

Startup validation0187 passes three CLI checks and three actual early-rejection
cases: missing goal, a five-call budget and assist configuration supplied to a
normal camera scene. They reject before world/model/output creation. The
initial CLI check revealed its older per-argument1500ceiling; the failed log is
retained and scene-specific bounds now pass independently of argument order.
The app builds with source-light and constraint diagnostic features; ordinary
`dev_tools` also checks without the assist feature. Legacy render/physics input
and MicroDuck contracts remain unchanged.

```bash
cargo run -p bevy_sim2sim --features dev_tools,dev_tools_minigame/g1_source_lighting,dev_tools_minigame/g1_constraint_diagnostic -- --scene g1_mobile_assist_diagnostic --robot g1 --g1-config /absolute/four_call_mobile_assist_config.json --g1-ticks 2050 --output /absolute/new_evidence_directory
```

The additional `mobile_assist` configuration holds only
`heading_yaw_source_rad` and `relative_distance_m` (exactly2for this finite
entry). Observation/episode/frame/time identities are supplied by the actual
native camera handoff, never fabricated in configuration. The independent
physics ledger records actual completion Ticks, which may be below2050.

Real Bevy trial0188 passes the independently checked mechanical stage with
four fresh native RGB N1.6 grasp inferences and1524actual Homie/torque/physics
updates. Each saved RGB pixel array equals its actual model input; only RGB
and the original named self-state arrays enter inference. No saved grasp or
fixed upper-target fixture is loaded. Actual walking displacement is2.037898m,
settled displacement2.019864m; all692walking/stopping boundaries have positive
hand-only support, no scene support and upright at least0.993010. All1524owner
records are present, with zero control deadlines missed/pending Ticks. Active
simulation/wall ratio is0.999323; inference pauses remain explicit, so this
does not qualify a continuous rendered1x task. The original bin's independent
stable-placement result is zero seconds: this manual clear-aisle goal still
holds the box and is not bin placement. The owned model closes after its four
successful inferences. Formal T2, Qwen goal selection and8/10 remain open.
Raw native trace SHA256:
`35e82f10084d448f77caac9b884a191fde56a954f1284dbcdd6e780f93104858`.

The next separate `g1_mobile_scan_diagnostic` entry is a finite public-map
search, not a walking/placement test. It admits exactly four fresh original
unmarked RGB grasp calls, then only a measured-yaw turn and100standing Ticks,
with1050maximum native Ticks. The same typed native owner computes the known
grip correction; a completed scan auto-pauses and cannot enter walking.
Fresh observation identity/age guards apply to every handoff. The old2m
carry entry and all default physics/contracts retain their bounds.

The initial0190 scene mounts disclosed DICT_4X4_50 printed markers21on the bin
floor (16cm black square) and22on the original box's top (10cm). That top mount
is preserved in the failed trial's code/receipt. The subsequent visibility
entry puts22on the original robot-facing local+X face instead. Fixed white
margins, metric mounts and PNG hashes are
recorded. Marker visibility begins at actual Tick200, so the four original
grasp images remain unmarked. Markers inherit displayed native object poses;
they add no physics, light, constraint or object sensor.

`unitree_g1_mobile_vision.py` accepts actual640x480PNG and a separately written
whitelist of pinhole calibration and original joint/root self sensors. It
rejects extra fields, foreign robot identity and duplicate markers, computes
camera mounting from original FK, and bounds IPPE square reprojection/error
and pixel support. Task/world/camera truth poses and contact evidence are
excluded. Any approach proposal is derived from RGB marker poses and self
orientation, explicitly classical and **not automatically executed**. Target
visibility, original-bin approach and physical release remain separate gates.

Entry checks0189 pass63simulation checks with33ignored real-fixture checks,
four CLI guards and an ordinary-feature compile. Independent camera mounting
from actual native self state agrees with the render camera within9.78e-8m
and1.14e-7rotation-matrix elements; no world poses enter localization. These
are zero-physics/zero-model calibration and entry checks, not a target-visibility
or placement result.

Live trial0190 completes663actual native Ticks with four fresh unmarked RGB
grasp calls and zero walking commands. The final100Ticks retain hand-only
support, minimum scan upright0.994135, no missed deadlines or pending debt;
active sim/wall ratio0.998337. Both marker detections fail. Independent optical
audit shows all bin corners in view but occluded by the carried box; all top
box-marker corners are above the image and its face points away. This is an
actual visibility failure, not missing weights or service failure. The model
closes after its four calls; camera/object truth is auditor-only.

Finite saved-prefix trials0191/0192 preserve all663body states bit-exactly and
call no fresh VLA. Vertical-only lowering stops safely at726completed Ticks
when a single IK target change would exceed0.1rad. Independent FK shows the
right-arm Jacobian minimum singular value falling from0.06338to0.00884 as its
reach extends. Preserving shoulder-to-palm radius reaches790Ticks, then
fails the unchanged solver bound; the actual shoulder-to-wrist radius still
grows0.317114→0.335083m and the minimum singular value falls to0.00236.
Neither failure relaxes joint/solver bounds or modifies physical parameters.

Preserving **shoulder-to-wrist** reach instead passes the zero-world/model
planning envelope0193 and physical trial0194. It lowers both palm targets18cm
over150native Ticks, adding only the shared inward shift calculated from
original self-state FK; rotations, palm gap and all fingers remain unchanged.
The913Tick trial includes100settling Ticks and keeps hand-only support through
all250lower/settle samples, upright at least0.999392. Box height moves
0.938082→0.779938m. The commanded additional inward shift is0.150518m.
Independent optical geometry places both bin and robot-facing box markers
inside the image, with no box occlusion of the bin; this still requires actual
RGB detection. Its trace SHA256 is
`89589917a32ca6cad9054ac916fa5f4990c9e2c9914e494fd1b0fdd0465354cb`.

The separate `g1_mobile_target_view_diagnostic` admits1300maximum Ticks/four
fresh unmarked grasp calls, the public-map scan, then exactly18cm/150Tick
lowering and100settling Ticks. A fresh actual stationary RGB stamp admits that
lowering on the same typed owner. Original scan/carry budgets stay unchanged.
The final actual RGB/self-sensor whitelist feeds classical marker localization;
no target height, object pose, contact truth or saved action enters execution.
This visibility pose is not bin approach, physical release or task acceptance.

Fresh Bevy trial0196 completes913 actual 50Hz Ticks after four unmarked RGB
N1.6 calls; all913physical body states match the offline0194trial except the
new episode identity. All250lower/stop Ticks retain hand-only support. Both
markers decode from the actual final RGB. Independent audit measures horizontal
position errors0.000650m for the box marker and0.003892m for the bin marker,
with reprojection RMS0.342/0.560pixels. The strictly whitelisted perception
input contains only camera calibration and original self sensors. Its visual
navigation proposal is heading−1.614499rad and relative distance1.731257m;
the proposal has **not** been executed. This trial records every physical
boundary, no missed deadlines/pending debt, active sim/wall ratio0.998771,
and closes the owned model after four successful inferences. It does not
qualify original-bin placement, continuous unpaused1x or formal8/10.
Raw trace SHA256:
`ec9c9cd9cdf4a449f97c0cd26d08a14e76ff511f0ff02948fb02ff9f0065c2e9`.


The saved0196 prefix plus its actual RGB marker proposal is mechanically
checked in0197 before admitting online motion. A disclosed0.65m reobservation
margin produces a1.081257m coarse waypoint. The sole native owner completes
1307single50Hz integrations; the first913body states match0196exactly. All394
carry/stop samples retain hand-only support, upright at least0.992602, and
actual root displacement is1.135662m. Independent geometry keeps both markers
visible at the final boundary. This is a saved-fixture control diagnostic with
zero fresh VLA calls; it grants no autonomous or placement qualification.
Trace SHA256:
`157f71d99300bcea80573bc948bff7ec98b2df3ccdccc9bf705795d5fec070ef`.

`g1_mobile_target_approach_diagnostic` separately requires3150maximum Ticks,
four fresh original grasp calls, matched source scene/4PGS/light/fiducials,
and the same18cm visibility lowering. Its `mobile_scan.vision` pins absolute
Python/script/original-definition paths and SHA256 identities. After lowering,
a bounded local worker receives a new actual RGB and whitelisted self-state
capture. It has one reply slot and a3second deadline; timeout/cancellation kills
only its owned subprocess. Reply admission checks episode/frame/time and input
hashes, both distinct marker identities, residual≤1pixel and edge≥8pixels.
The executed goal records both the raw visual proposal and traditional0.65m
reobservation margin. Physics stays paused during this diagnostic localization;
there is no extra integration, truth input, Qwen target selection or release.
Startup0198 rejects missing vision, wrong script/definition hashes, vision in
the old target-view scene and an extra grasp call before output/world/model
creation. Three reply-guard checks and six scene-budget/CLI checks pass.


Live entry0199 stops before any model call or physics integration because the
original17,171,205byte collider definition exceeded the initially allocated4MiB
worker file budget. The owned model is closed and the failed startup is retained.
The pinned definition gets a finite32MiB read ceiling; its SHA256 identity is
unchanged. Actual saved RGB worker roundtrip0200 then loads the original file,
detects21/22 and verifies the bound reply in0.45seconds, with zero model calls
or physics steps. This tests the worker contract only, not fresh execution.


Entry0201 exposes a second pre-integration guard still using the1300Tick
view ceiling for the new3150Tick approach mode; zero model calls or native
steps occur and the owned service closes. Both guards now read one explicit
per-mode budget. A focused regression verifies2050/1050/1300/3150 separately
and confirms ordinary modes have no assisted budget. No physics/action limit
or prior scene budget is increased.


Fresh online approach0203 completes1307single50Hz body updates after four
new actual RGB N1.6 calls. The local worker reads the exact913Tick RGB and
whitelisted self state, produces heading−1.614499rad/distance1.731257m, and
the same owner executes its disclosed1.081257m coarse waypoint. All394
carry/stop samples retain hand-only support; root displacement1.135662m,
upright≥0.992602, active sim/wall0.998895, zero missed/pending Ticks. A new
actual near image detects both markers: independent horizontal errors2.22mm
for box and1.12mm for bin, bin residual0.0448px/minimum edge38.85px. Its
remaining visual distance is0.573745m. The owned model closes after four calls.
This establishes online visual approach, not Qwen choice, release or8/10.
Trace SHA256:
`9a948fd67a266a1cf4fd910cfc967d65410d86e3b5df1e1716bb10e35eebca82`.

Saved-prefix near approach0204 reproduces all1307body states from0203, then
admits a newer camera-bound goal while retaining the lowered upper command.
It fails its finite walk deadline after2326native Ticks, with no new VLA
calls. At1404the box first receives both original table-leg and bin support;
no robot/bin impulse occurs. The robot remains upright. Independent geometry
shows the original table top0.5061–0.5306m and stable bin origin0.5304m.
The carried cube's lowest tilted corner is0.5532m at the near view and0.5431m
at first contact, below the original bin rim≈0.581m. The height/geometry data
are acceptance-only and never become command input. This is a collision
clearance failure, not a newly qualified release or model/frequency issue.

Completed carry handoff now preserves the current upper/finger command and
requires a strictly newer current camera boundary; active-goal replacement
remains rejected. Navigation also checks forward progress from its original
self velocity: after one gait-start window, less than3cm along the requested
heading over50Ticks raises a bounded failure requiring pause/reset. Lateral
drift cannot mask blockage; alternating zero/positive gait velocity is allowed.
Simulation checks0205:66pass/35explicit real-asset trials ignored. The next
stage must restore visual/known-geometry clearance before walking over the
bin; merely extending the collision deadline is not an adopted remedy.

Actual near RGB plus the pinned public collision geometry produces the0206
clearance proposal without world/object poses: raise0.130138m over131Ticks
to retain10cm above the rim. The separate bounded self-FK raising controller
preserves palm orientations/gap/fingers, then stands100Ticks. Saved-fixture
trial0208 raises the box0.703970→0.818480m, but its fine approach is rejected
by the forward-progress guard at1757actual Ticks. Read-only contact replay0209
reproduces all1757body states exactly and identifies first non-floor robot
contact at1674: left hip-yaw link against the original table legs,3.369N·s.
This is distinct from the earlier lowered-box/rim collision. No physical
parameter or formal frequency is changed.

Saved-fixture standoff0210 uses an explicitly manual diagnostic20cm earlier
waypoint; this margin is not presented as an online visual plan. It completes
1761native Ticks with no non-floor robot contact, but the box slips into the
bin before a release command. Current episode/camera boundaries after raising
are synthetic diagnostic fixtures, and all fresh-model counts are zero.

The bounded release controller opens the commanded palm gap to30cm over100
Ticks, retaining the palm midpoint/orientations/fingers and standing125physical
Ticks afterward. Trial0212 reproduces0210's first1761body states exactly and
finishes1986actual50Hz single integrations. The actual palm gap is0.300431m at
the end of opening and0.303367m at the final boundary. Independent original-bin
floor-footprint/support/separation/speed/standing criteria hold continuously
for5.38seconds, but hand contact first disappears at1681, before opening1762.
**This is development placement evidence, not controlled release or formal T2
success.** It cannot count toward8/10. Trace SHA256:
`4beea127e0b6b8a67973f03e9345b9dd4136cb921beb9d344795d165d63e1906`.

Read-only self-FK/contact audit measures the held actual palm gap≈22.8cm
against the unchanged15.6043cm command, with original contact-loaded tracking
error; original FK agrees with actual bodies below0.2µm. The box is already
about7cm below the palm midpoint at the near boundary, and slides farther
during the final walk. No held sample fully fits the bin-floor footprint before
hand loss, so simply opening earlier does not satisfy placement. This rules
out claiming that the existing slipping trajectory is a successful planned
release; fresh raised RGB and controlled retention/placement remain required.
Checks0213:68simulation checks pass/35real-asset trials explicitly ignored;
both development-feature and default entry compile. Contact auditing is
test-only and reads the last solve without refreshing contacts or integrating.

`g1_mobile_target_raise_view_diagnostic` retains the3150Tick/four-original-grasp
budget and matched source profile. At the completed coarse waypoint, a **new
actual near image** feeds the bounded marker worker. Optional
`mobile_scan.vision.task_geometry` pins the public source collision file to
`19eb60783008e3f08d82a1cf402c590395df1e98c4c089fb1247f8ed7d9a88a0`;
the worker reads vertex geometry only, never runtime poses. Original marker
mounts and self quaternion yield gravity-aligned box minimum/bin maximum and
a disclosed10cm clearance. The admitted bounded raise carries that exact
current image stamp, executes on the sole native owner, stands100Ticks and
captures a new actual raised image. Other scenes reject the geometry option.
This mode does not fine-walk, release or advertise qualified execution.

Worker0214 consumes the saved actual near RGB in385ms and exactly reproduces
the independently calculated0206goal0.130138m/131Ticks, with zero physics/model
work. Admission rejects old frames, foreign public geometry, detached raise
distance and truth claims. Checks0215:36development-library checks pass/two
real-asset checks ignored; seven CLI checks pass; default entry compiles.
Fresh raised-view execution is a separate required test.

Fresh raised-view0216 completes1497single50Hz integrations after four new
actual RGB grasp calls. Its fresh near image1314 produces a different bounded
raise,0.082747m/83Ticks, followed by100standing Ticks; all183raise/stand samples
retain hand-only support. First100body states match0203, with the first physical
difference at101; the whole fresh prefix is not claimed bit-exact. Every new
model input still matches its actual captured RGB array exactly. Active sim/wall
ratio0.999003, zero missed/pending Ticks; owned model closes. The final new
raised image1497 detects box22 with independent0.94mm horizontal error, while
box geometry occludes bin21. It is not evidence for final approach or release.
Trace SHA256:
`c1d17a80eaa23d00c55844e4a8b21f0ac748588152678c884babc4d9c91f9a7f`.

`g1_mobile_target_memory_view_diagnostic` adds a current raised RGB localization
without further motion. The admitted actual near image supplies target21's pose,
original self quaternion and image/input hashes. Raising integrates **original
root velocity** at each20ms control boundary; absolute root position and object
state are excluded. A current image/self quaternion propagates that recent
target into the current root frame. The static-target assumption is explicit;
memory expires after8sim seconds/12wall seconds, rejects reset/old frame/unbound
hashes and requires a current actual box22 detection. A newly visible target
more than3cm from the predicted target rejects the static assumption. Remembered
targets are stored separately from current marker detections, and any navigation
proposal remains unexecuted by this view entry. Other entries retain the
two-current-marker rule and their existing budgets.

Saved actual raised RGB worker0217 localizes the occluded target in412ms using
183original velocity samples,3.66sim seconds/4697ms image age. Independent
acceptance-only target error is1.74mm, with no physics/model work. Its proposed
heading−1.592415rad/distance0.525429m is not executed. Seven focused worker checks
cover actual roundtrip, old/reset/expired memory, unbound reply and missing
current-box input. Fresh runtime target-memory admission remains a separate test.

Fresh runtime0219 completes1597single50Hz integrations/four original actual
RGB grasp calls. The current near RGB1360 admits0.136479m/137Tick raising plus
100standing Ticks, all237with hand-only support. At the new actual raised
camera1597, bin21 is occluded and box22 is visible. The bounded worker uses
exactly the owner's original velocity integral and4.74sim seconds/5801ms
target-image age. Its independent target error is7.35mm; navigation proposal
heading−1.566022rad/distance0.653836m remains **unexecuted**. Active sim/wall
0.998837, zero missed/pending Ticks; owned model closes. No saved grasp or
synthetic camera enters this runtime. Checks0218:38development checks/eight
CLI checks/68simulation checks pass; default entry compiles. Trace SHA256:
`763fe92c04bef1f6dc7a97136f12ff0ccd3921ba7be2596be41605ed373b9607`.

The proposed off-center printed target board is **not adopted**. Zero-world
geometric analysis0220 rejects full-board visibility: both the20cm board and a
smaller10cm black tag on the original floor retain occluded corners. No GPU
rollout, integration or model call is spent on that candidate. Rejected source
and projection evidence are retained outside Git, and the four owned candidate
files return to the prior committed center-marker implementation. The next
observation action must use actual robot/camera motion and preserve grip;
marker relocation does not solve the final-view requirement in this scene.

Actual raised RGB/self sensors/public cube geometry predict a bounded standing
view turn0221:−0.33rad exposes the complete original target marker with37.6px
minimum edge and16px image margins under a rigid-grip prediction. This is a
zero-world geometric plan; hand-mesh occlusion still needs new actual RGB.
The explicit `ClassicalReobserve` owner command requires completed raising,
a newer current raised camera stamp and≤0.6rad turn, retains current upper/finger
targets and rejects active-stage replacement. It does not relax the initial
scan entry or enter walking.

Saved-prefix physical trial0222 reproduces all1597body states from fresh0219,
then performs276turn/standing Ticks on the same world. It completes1873single
50Hz integrations, upright≥0.994917 and without any non-floor robot contact.
However, hand-only box support first disappears at1710. This is **not** a safe
grip-preserving observation action and does not authorize the online turn. It
narrows the failure beyond final walking/table contact: raised carrying posture
also loses grip during a standing turn. No new model call occurs. Trace SHA256:
`7abdec20e2075885290c0bc387bc9acf86591fe238ed7b70ed8a993f5e6a30df`.
The next bounded comparison must restore the independently validated original
calibrated transport posture before body motion; friction/gain/force sweeps and
longer failed walk deadlines are not adopted remedies.

Original self FK comparison0223 identifies a common0.159071m palm translation
back to the episode's calibrated transport posture: root-frame
`[0.15251725, 0.000852408, 0.04518119]`. Left/right disagreement is0.624µm;
palm rotations and fingers agree. The pure200IK-update/100standing-command
envelope preserves the0.156043m commanded gap, with maximum joint increment
0.007208rad and13nm rounded palm residual. It performs no integration or model
call. `ClassicalRestore` permits only this≤18cm common rigid-grip translation
at a completed raised carry boundary, followed by100physical standing Ticks;
it reads original episode commands and current self joints, never object truth.

Trial0224 is rejected before restore integration because the original-posture
cache accidentally retained the predecessor VLA's small navigation command.
The cache now explicitly stores stationary posture only; executed VLA/scan
commands are unchanged. Corrected trial0225 reproduces the first1597body states
bit-exact from fresh0219, then restores for200Ticks, stands for100 and repeats
the same bounded view-turn heading for256Ticks. All556post-raise samples have
hand-only box support, upright≥0.994888 and no non-floor robot contact. This
counterfactual supports restoring arm posture before observation motion;
unchanged raised posture in0222 lost support at1710. Total2153single50Hz
integrations, no fresh VLA call. Trace SHA256:
`c89df1860a7e9dd06ed90964ee1c38a2c1926471e2b13eb8103dbb6bf3a40423`.
The turn's future camera identity in this fixture is explicitly synthetic.
Fresh online restore/current-camera/turn observation remains required, and
neither trial counts as formal T2 placement or qualified safe execution.

`g1_mobile_target_restored_view_diagnostic` is a separate3150Tick development
entry. Four fresh unmarked RGB grasp calls and the actual scan/lower/coarse/
near-clearance raise are unchanged. New real images bind the stationary restore
and subsequent view-turn goals to their current native boundaries. The episode
original calibrated upper posture is restored over200Ticks plus100standing
Ticks. One disclosed fixed−0.33rad standing observation turn uses original
self yaw, followed by a new actual RGB localization requiring both current
markers. Target memory and navigation proposals are not executed here.
This grants no formal placement, generalized target ability or task UI execution.


Fresh online0231 completes2112single50Hz integrations/four new actual RGB
grasp calls. All231raise/stand and300restore/stand samples retain hand-only
support; the subsequent standing turn first loses support at1897,59Ticks after
its current1838boundary. The final actual RGB shows the dropped box, and target
localization fails. Upright≥0.994460, zero control misses/pending Ticks, active
sim/wall0.998875; owned model closes. Trace SHA256:
`e583f2f9b448ad46b0767640552c105d1a054b0c371990df889be85bbc93ca99`.
Saved0219/0225retention does not qualify this fresh episode. Earlier startup0229
fails before any inference/integration because the binary lacks the existing
source-light feature; corrected build0230 preserves it and records the exact
binary SHA. This is not a light/physics parameter experiment.

The actual post-restore image1838also exposes an observation gap: fingers
occlude box22, so no current box detection is available before the turn.
The restored-view entry now requires a bound actual box-only image and self FK
proximity before admitting any observation motion. This mode cannot admit a
navigation proposal, clearance or target memory, and cannot substitute for the
existing two-marker target localizer. Missing box22 or distance>0.25m from the
measured palm midpoint leaves the native owner paused. Visibility/proximity is
only a necessary guard, never proof of secure grip. The original GR00T camera,
action contract, formal clock, gains and materials remain unchanged. Further
body-motion trials require usable current grip observation; restoration alone
is insufficient.

The separate1050Tick `g1_mobile_auxiliary_view_diagnostic` entry keeps the
four original unmarked learned grasp images and calibrated standing scan.
It then captures a disclosed passive head sensor: original mount plus0.15m
source Z and15degrees optical upward rotation, with the unchanged640x480
pinhole. The original VLA rejects this sensor profile. Render-only public labels
use bin21 at local[0,0.18,0.60], Rx90degrees, black0.16m; box22 at
[0.1005,0,-0.04], Ry90degrees, black0.06m. PNG identities and200Tick activation
remain unchanged. The perception worker binds the fixed layout document hash
and camera profile to the actual image/self-sensor input. No object truth is
available to it. Geometry control0236 predicts visibility; this prediction is
not actual RGB evidence. This entry executes no lowering, raising, restoration,
post-carry visibility turn, navigation proposal or release. It grants no task
qualification. The formal50Hz clock and all original physical properties remain
unchanged.

Fresh0238 supplies actual two-marker auxiliary RGB after four original VLA
calls and664single50Hz integrations. All464post-grasp samples have hand-only
box support; minimum upright0.994124 and active sim/wall0.998474, zero misses
or Tick debt. Independent marker position errors are4.199mm(box) and4.489mm
(bin). Original model input arrays match the four original-camera PNGs exactly.
No target memory or approach/release is executed. Trace SHA256:
`e14ac690dfbe284011d19e5e464ce1866f5c4e2c4c7ae910728199352b8535c9`.

`g1_mobile_auxiliary_approach_diagnostic` is separately bounded to3150Ticks:
one actual auxiliary two-marker observation binds the existing coarse waypoint
(relative visual distance minus the declared0.65m reobservation margin), then
one new current observation checks the completed carry boundary. The original
transport arm posture is retained throughout. Auxiliary perception requires
current box proximity<=0.25m to the measured self-FK palm midpoint; this remains
a necessary visibility guard, not proof of secure grasp. There is no lowering,
raising, restoration, extra visibility turn, final approach or release here.

Fresh0240 executes one actual auxiliary-RGB coarse waypoint:1147single50Hz
integrations/four new original VLA calls,947post-grasp samples all hand-only.
Measured own-velocity carry displacement is1.03634m. New current marker errors
are0.583mm(box)/1.418mm(bin), with remaining visual distance0.645439m.
Active sim/wall0.998778, zero misses/debt; owned model closes. Trace SHA256:
`3fe6df070eb78771e841b3f6508e3fbddf8187f2a8cb0ea887917f4be05388d5`.

The separate3150Tick `g1_mobile_auxiliary_release_diagnostic` entry adds a
new actual coarse-boundary RGB fine waypoint with fixed0.12m margin, then a
new actual stationary placement image. Before any opening, current two-marker
poses and the pinned public bin floor part2/box collision vertices must place
all box vertices within its rectangular floor footprint with>=0.02m margin.
Estimated bottom-to-floor drop must be0.05–0.40m, visible bin upward cosine
>=0.98 and own root speed<=0.05m/s. Placement-only perception cannot admit
navigation, old target memory or original-camera substitution. Rejected
alignment leaves the owner paused. Admitted release uses the existing0.30m
commanded palm gap over100Ticks and125real physical settling Ticks; fingers,
clock and physics parameters remain unchanged. Pure actual-self-state opening
control0241 passes225numeric updates with maximum per-update joint change
0.002058rad and final commanded gap0.300000007m; it is not physical evidence.
This finite entry still requires independent grip-until-opening and strict
placement audit; it does not qualify the ten-episode suite or continuous1x.

Fresh0243 completes1443single50Hz integrations/four original VLA calls;
all1243post-grasp samples remain hand-only. Current placement RGB rejects
opening: estimated floor margin0.001379m<0.02m and self speed0.050051m/s>0.05.
Independent truth audit agrees the box is not wholly inside (margin−0.001423m).
No opening executes, no hand support is lost, upright>=0.992144, zero misses/debt;
active sim/wall0.999039 and owned model closes. Trace SHA256:
`0be6289b73c713d1b76b99b0215ab998fa88efa9e2ddf5a195725607acef1737`.
The fixed radial0.12m fine margin is an insufficient geometric approximation;
future fine targets must use the actual whole-object floor containment interval.
The completed100Tick navigation stop also retains measurable root motion;
additional standing must be a separately bounded real owner action and require
new current RGB afterward. Neither admission threshold is relaxed.

The release entry now derives its fine carry goal from the whole-object
containment interval along the current RGB navigation heading, choosing its
midpoint. It explicitly accounts for the existing0.05m navigation stop margin;
that controller is unchanged. At the actual0243coarse image, the physical
interval is[0.567917,0.695419]m, selected0.631668m, commanded0.581668m,
predicted floor margin0.083587m. This is a geometric prediction requiring
another current placement image, never automatic release admission.

Before that placement image, the same owner executes `ClassicalHold`: exactly
the existing stationary carry command, at least100 and at most250Ticks,
requiring20consecutive original self-root speed samples<=0.03m/s. It reads no
object/contact truth and integrates once per formal50HzTick. A timeout requires
explicit pause. Release requires a newer post-hold image. Mechanical control0245
repeats the1443physical prefix at identical native f32 bits/f64 simulation times
(JSON float encodings differ), then100real hold Ticks. All100retain hand-only
support,63stable samples and final self speed0.011316m/s; total1543Ticks,
zero fresh VLA/camera calls. Trace SHA256:
`92f43f4498d8796ea2d2b01497d0bf860555122fa38a3608aef2e5e090604bd6`.
No grip targets, gains, masses, friction or frequency are changed by holding.

Fresh0246 executes four original-camera VLA calls and1560single50Hz Ticks,
but loses hand-only support at1385 before any opening. The post-hold current
RGB correctly rejects release: box-to-measured-palm distance0.399150m exceeds
the0.25m necessary grip-region guard. No release Tick executes and independent
strict placement duration is0seconds. Active sim/wall0.999302, zero misses/debt;
this paused diagnostic is not continuous1x qualification. Trace SHA256:
`71836354cdc5d464847683d4127f35cb541a2aef9c04b6aa721fde2764b06a40`.
The driver session exits143 before writing final health/exit/cleanup fields;
those fields remain unknown. A separate timestamped recovery record confirms
the owned app/model processes and localhost5558 listener are absent, without
inventing a cleanup signal or changing the initial driver receipt.

Comparing0243 and0246 gives identical physical trajectories through1342Ticks.
Both already have more than25mm downward box motion relative to the measured
palm midpoint at1335, while walking.0246 first selects stand at1360 and loses
support25Ticks later. The initial slip therefore cannot be attributed solely
to stopping. Saved-action control0247 reproduces all1560physical body samples
at identical native f32 bits/f64 simulation times and adds only the existing
read-only completed-solve background auditor. There are zero non-floor
robot/background impulse contacts. No collision refresh, physics parameter
change, new VLA call or current rendered image is used. Completing its hold is
not successful gripping. Trace SHA256:
`5ce4f807da6b235c3f651d48c732082443f8096f2ab95682d8b76c39078b0751`.
The next finite causal comparison changes only the requested fine walking
speed; it does not promote a controller remedy or relax placement/grip gates.

Saved-prefix control0248 requests0.1m/s only during fine walking. The unchanged
progress guard stops it at1266Ticks for less than3cm forward progress in1second.
All1066post-grasp samples remain hand-only, but only44.5mm fine odometry is
accumulated and no hold/opening completes. The low-speed override remains
test-only; no speed sweep or weakened progress rule is adopted. Trace SHA256:
`19bbe3d2a02191e51d992ddb5abb697533318c38f10d03b8d8b372f76c1cdf31`.

Control0249 instead uses the unchanged0.3m/s navigator in four0.1m requested
segments (existing0.05m stop margin), each followed by its original100Tick
stop, then100Tick hold. All1918single50Hz Ticks complete, with1718post-grasp
hand-only samples and zero non-floor robot/background impulse contacts. The
first1147physical body samples match fresh0246 at native f32 bits. Fine root
displacement is[-0.077928,-0.572353,-0.000374]m and final self speed0.009834m/s.
Independent final floor margin is0.022907m, but the hands still support the
box: strict placement0seconds and opening0Ticks. Later suffix stamps are
explicitly synthetic self-clock fixture stamps, not new rendered observations.
This is a mechanical comparison with0freshVLA/0current-camera frames, not an
autonomous demonstration. It supports testing actual RGB after every bounded
segment; it does not justify blindly executing a saved four-segment sequence.
Trace SHA256:
`8f5b3d2cc46d7676444373f6aebf90cbdf8a9da19a56f18556463eaa98474b7f`.

The native release entry now reobserves at every completed fine segment,
with at most five segments. Each actual two-marker image derives a new
whole-box containment interval and current floor margin. If the current margin
already meets0.02m, no further walking command is admitted; otherwise one
0.1m requested step uses the unchanged navigator/0.05m stop margin and0.3m/s
command. A step beyond the far containment boundary, insufficient geometric
interval or exhausted segment budget leaves the owner paused. The protocol
explicitly distinguishes intermediate movement from achieved containment;
an intermediate prediction can still be outside the target and cannot admit
opening. Another current image is mandatory after each step. Only new current
containment admits the separate real hold and newer placement-only release
image. No saved/synthetic image stamp enters this runtime path.

Entry controls0250 pass41development tests/2ignored,12CLI tests,7navigation
tests, default compilation and a real saved-RGB local-worker round trip, with
zero physics/model calls. Five numeric-only geometry controls cover partial
advance, final step, already aligned, minimum-step blockage and a passed
containment interval. The0.02m release gate and strict physical acceptance
rules remain unchanged. Actual closed-loop physical release still requires
a new live test; these entry checks do not qualify it.

Fresh0251 executes four original VLA calls and four independently observed
fine segments. All1609post-grasp samples retain hand-only support through1809
single50Hz Ticks, with upright>=0.992144 and zero misses/debt. The final actual
image is captured, but the worker rejects its radial navigation distance below
0.1m before reporting containment. Independent whole-box floor margin is
0.079202m; the box remains held, so hold/opening0Ticks and strict placement
0seconds. The driver records the actual terminal app2/modelSIGINT−2 and closes
the owned model. Trace SHA256:
`b2358c6d7ed9ecda1cd2e950189b7f41a708b17ee00a6d5a11de3832176d4e13`.
This is a software arrival-gate failure after retained grip, not controlled
release success.

Entry0252 evaluates that exact final image/self-state: current full-box marker
geometry yields0.083087m estimated margin and an aligned result with no walking
goal or navigation proposal. This explicit geometry-only arrival may pass the
general image protocol before the normal motion-distance envelope. Movement
still requires the original0.1m minimum, and removing the pinned geometry still
rejects the same image. Foreign camera, incomplete containment and movement
under an aligned result remain rejected. The owned worker independently binds
the declared geometry to its pinned configuration before returning any reply.
The actual saved-RGB Rust/Python round trip and41development tests/2ignored
pass with0physics/models. Hold and newer placement-only image are still
required before opening; no criterion is relaxed.

Fresh0253 does not reach the repaired arrival gate: hand-only support ends at
987 during coarse walking, before any fine/hold/opening. Its current image1097
correctly rejects missing markers after the dropped box. All1097single50Hz
updates complete with zero misses/debt; strict placement0seconds. The first
original image differs from0251 by at most one RGB level, and the original
model outputs and resulting grasp state differ. This is not evidence that the
arrival fix failed. Trace SHA256:
`98a6427d0d331973fe7b1fb0d7c411dd158b63d465a5ecad5796ea927187964f`.

The exact saved0253 replay0254 reproduces all1097native f32 physical samples
and the987hand-support loss. Control0255 changes only one coarse walk into
three bounded segments with the same total requested physical path and the
unchanged0.3m/s navigator/100Tick stop. It still loses support1285, before
completing1373Ticks. Neither replay has non-floor robot/background impulse
contact. Segmentation delays failure but is not a sufficient grip remedy and
is not promoted to the runtime coarse approach. Both use0freshVLA/0newRGB;
later segmented stamps are explicitly self-clock fixtures. This also leaves
fresh-grasp robustness open instead of selecting a successful recorded grasp
as autonomous qualification.

A separate saved observed-prefix release test0256 matches all1809physical
samples from0251, then executes100real holding Ticks and225opening/settling
Ticks. All1709preopening post-grasp samples are hand-only, but the original
30cm commanded gap leaves both distal index links supporting the box through
2134: strict placement0seconds. No new RGB/VLA occurs; the hold/opening suffix
has explicitly synthetic self-clock stamps and cannot qualify live release.

The actual saved RGB/self-state and pinned original hand/box hulls provide a
zero-physics clearance calculation0257. Hulls are conservatively clipped to
the box footprint perpendicular to spreading; this requires more than the
30cm target for a1cm gap around relevant hand shapes. A single35cm goal within
the existing contract in0258 physically detaches the box, but the tilted spread
leaves one final corner0.000250m outside the frozen bin-floor footprint. It
correctly fails strict placement despite stable real bin support. No tolerance,
rate, gain, friction, mass or force limit is changed.

Control0259 changes only that35cm opening to spread perpendicular to named
self gravity, preserving each palm's original height, orientation and fingers.
The exact1909Tick physical prefix matches0258. Through the opening, real hand
contact ends at1990; strict placement is then sustained for2.52seconds at2134,
with upright>=0.992144. This passes the finite saved-prefix mechanical release
check, not fresh vision/Qwen/science-station/formal8of10 or continuous1x.
Trace SHA256:
`bf3ef15e4c7af6e284f139df5cd04f1a93aec7ee6e0b862486f17d0a12df0106`.

The current placement-only image protocol is now version2. It retains every
floor-margin/drop-height/upright/self-speed gate and independently binds both
original robot and task collision geometry. It predicts horizontal hand
clearance from the actual current box image and named self FK; required gap
beyond35cm rejects opening. An admitted command uses the existing35cm maximum
and100opening Ticks, followed by125physical settling Ticks. The prediction is
explicitly not proof of detachment: the separate contact/placement auditor
must still pass. No saved self-clock stamp enters the runtime release path.

Entry0260 passes the exact saved-RGB Rust/Python placement round trip with
required measured horizontal clearance gap0.297021m, admitted35cm goal and
no physics/models. The pure225-step release envelope also passes. A separate
self-command audit checks every100opening step: maximum gravity-height change
is1.25e-8m and palm rotation change4.46e-8, with fingers and zero navigation
preserved. Validation passes73simulation tests/38ignored,41development tests/
2ignored,12CLI tests and default compilation. A new unit test's initial numeric
type compile error is retained in the evidence alongside its corrected pass.
The new live full chain remains a separate required check.

Fresh0261 then completes the entire native development chain with four new
original N1.6 calls, current images before each of four fine moves, a new hold
image and a newer placement-only image admitting the horizontal35cm opening.
All1709post-grasp/preopening samples are hand-only. Real hand contact ends
inside the commanded opening, and strict whole-box containment/detachment/bin
support/low velocity/standing remains true for the final2.52seconds; final floor
margin0.094009m. All2134actual50Hz single integrations/control updates complete
with active sim/wall0.999187 and zero misses/debt. The owned model exits by
SIGINT−2 and the app exits0. No saved grasp or synthetic image stamp enters
this live chain. Trace SHA256:
`0434673574a9e2f18eb21dca88f6379edb76b6ae0944f868a331ce0235f766a0`.
This is a finite source-near native vision/classical-control development pass.
Qwen selection, science station, fresh-grasp robustness, continuous1x, the
frozen formal ten-trial suite, terrain/fault/coexistence and final video/runbook/
main merge remain required; formal T2 success is still false.

The next continuous-grasp preflight admits only the existing camera diagnostic
with exactly200Ticks and four original N1.6 action chunks. Each chunk still
contains50frames at20ms; the next actual image must come from at least25Ticks
into the running chunk. The existing bounded executor keeps the captured image's
original stamp and a separate future execution slot. A missed slot rejects;
it does not rebase an old action or pause the active physics clock. Original
source camera, source lighting, task/background geometry and the explicit4PGS
development factory are required. Auxiliary markers, classical carry/release,
task UI and other chunk budgets are excluded from this preflight. StaticApple
keeps its existing trigger and validation. Unit checks pass42development tests
with2ignored; live timing and physical-grasp outcomes remain separate evidence.
This bounded admission does not qualify continuous full-task1x operation.

Fresh0263 fails the first replacement at50Ticks before a second model call.
The renderer completes an already-pipelined Tick22scene for the Tick25 request;
that correctly stamped image is discarded, but another usable image does not
arrive by50. The50actual original frames have zero missed physics deadlines or
Tick debt; expired actions are not applied. Independent raw evidence contains
every50integration despite a stale false completeness flag after the last
counter observation. That reporting order is corrected without changing the
trace or physical outcome. Trace SHA256:
`a530b93694f7de785b1655b65074ed8be0c86d969b393393918cd6448f111139`.

The single camera slot now accepts a current-episode minimum physical Tick.
Prefetch registers this one bounded request ahead of its Tick25scene, avoiding
a late main-thread request followed by a wasted old-scene GPU copy. Rendering
waits until the actual native scene meets the floor; no image/self-state stamp
is rewritten and no extra buffer or queue is allocated. Reset clears the floor,
and foreign episodes are rejected. Existing unrestricted camera requests retain
their behavior. Request, extraction/copy/readback and consumption timing is
recorded separately; the original action start/deadline and trigger stay fixed.
Two camera regression tests check old-scene rejection, unchanged actual stamps
and reset isolation. Live timing remains to be checked in fresh0265.

Fresh0265 registers the next camera request at Tick7with an actual-scene floor
of25, and copies/discards no old scene. It still pauses at50before a second
image/model admission; all50original frames have zero missed deadlines/debt,
continuous boundary sim/wall1.000293 and complete evidence. This validates the
old-copy correction but leaves the remaining extraction/readback latency
unresolved; the paused outcome is not a continuous-grasp pass. Trace SHA256:
`c7a59d1a27aaa8ad8438e4d9f64d1d1e67e6569b29b64ec1465d869b8145855c`.
The failure receipt now snapshots read-only camera pipeline progress: last
considered/copy scene Tick, skipped old scenes and independent extraction/copy/
readback times. It retains the same single slot, image floor, physical behavior
and action deadlines. This instrumentation is acceptance evidence only and is
never passed to model/control input.

Fresh0267 resolves that missing timing: the requested scene is actually copied
at Tick30, extraction-to-copy124ms and copy-to-readback202ms. At the fixed
Tick50pause the camera is already completed, but the render/main pipeline has
not consumed it in time to issue another model request. Only one original call
occurs; all50physical frames complete without misses/debt. This is a camera/
consumer scheduling failure, not a failed second model inference. Trace SHA256:
`a0b6c869ddee597ac171344be93595d41baa1ba172cbe4c360444150d8b7e68c`.

The ego render target is now drawn only for its one outstanding eligible
capture request, instead of drawing a second view while idle, waiting for the
minimum scene, holding an in-flight copy or awaiting consumption. Its pose
continues to follow the exact native snapshot; original640x480/MSAA8/projection/
materials/lighting and the1920x1080main view are unchanged. The existing bounded
slot checks cover activation/deactivation and minimum-scene gating. Fresh0269
will measure the resulting latency with the same Tick25/50window; no deadline,
physics parameter or action horizon is relaxed.

The real station-fixture camera regression0268passes with actual640x480RGB and
1920x1080main output, zero physics/model calls and explicit StationFixture
provenance. Fresh0269still misses the same first replacement: its actual Tick29
image takes150ms extraction-to-copy and220ms copy-to-readback. One-shot gating
removes the old-scene draws/copies but does not make this original25/50window
sufficient. It stops after50real frames/one new model call with no misses/debt;
no late action or timestamp rebasing is admitted. Trace SHA256:
`772f420c3ed35fc814362f8294fc6e0e1581332b9f18b855a3327edcb8ff1994`.
Further trigger/physics tuning is not the next remedy.

An explicit traditional model-wait controller now follows a completed original
chunk or completed classical skill in the same owner/world. It preserves the
actually executed upper/finger/pelvis command, sets navigation to zero, and
performs one normal Homie/control/integration update per20ms. Its episode,
request and execution-start identity is an owner clock, not an image stamp.
The initial bounded interface permits at most100Ticks; nonfinite/foreign/
repeated/expired states, upright below0.98and excessive self stop odometry
reject. Model/image transport never runs in this controller. A separate
20consecutive self-speed<=0.03m/s gate reports observation readiness; reaching
the deadline alone does not grant it. It is not yet wired into the live model
driver and grants no task or continuous1x qualification.

Mechanical entry0270reproduces the exact native f32prefix after each of the four
fresh0261chunks, then requests50waiting Ticks. Prefix2correctly trips the first
5cm limit after21wait steps: its predecessor still commands0.289844m/sforward
and its actual root speed is moving. This is stopping displacement, not a fall
or an established grasp loss. The controller's finite stop envelope is therefore
separated from image readiness:15cm bounds the zero-navigation stopping
interval, while the low-speed consecutive gate still controls readiness. No
gain, friction, force, model bytes or physics rate changes.

Entry0271completes all four identical-prefix50Tick mechanical waits; all original
upper/finger/height targets and prefix physics bits are preserved. The largest
stop displacement is0.052750m after chunk2. After the completed fourth grasp,
all50wait samples retain hand-only support. The first two prefixes have not yet
met20stable samples and are explicitly not observation-ready. Entry0272checks
the existing100Tickmaximum: all four complete, all prefixes remain exact, and
the fourth grasp retains hand-only support through all100samples. Readiness
first occurs after99/75/27/20wait samples respectively; the first prefix thus
uses almost the entire initial budget just to settle. A live waiting contract
must budget settling and subsequent image/model work separately before it can
be integrated. These saved-prefix checks use zero new images/VLA calls, renew
only fixture wall freshness, and cannot qualify fresh autonomous behavior.
Validation passes76simulation tests/39ignored,42development tests/2ignored and
default compilation; the initial missing exhaustive test-enum arms are retained
beside the repaired results.

The handoff now reads the controller command from the owner's actually accepted
execution identity. It no longer decodes the incoming replacement as if it were
already executed while an earlier chunk is active. The scheduling regression
uses deliberately different pending targets and verifies that both the current
and replacement command identities stay correct. This does not change original
normalization, action bytes,50frame horizons or20ms action intervals.


Entry0273separates a maximum100Tick settling allowance from up to100additional
image/model waiting Ticks, as a finite200Tick traditional wait. All four frozen
prefixes complete this limit with an automatic owner transition, preserve every
previous100wait-state f32bit, and establish a continuous stationary interval
through their remaining budget. The postgrasp fourth prefix retains hand-only
support for all200wait samples. Total mechanical integrations are1300; new RGB
and fresh VLA calls are zero. These are mechanical evidence, not autonomous
success. Validation passes77simulation tests/39ignored,42development tests/
2ignored,13CLI tests and default compilation.

The separate `g1_mobile_wait_grasp_diagnostic` live entry requires a finite
maximum1000Ticks, the original T2 camera/material/light/body/scene identities,
four original50-frame chunks, and the existing4PGS candidate. It has no printed
fiducials or auxiliary camera. Each chunk transitions on the next real owner
boundary to explicitly recorded zero-navigation standing; no expired VLA frame
is held or repeated. A replacement is admitted at its actual owner time only
from the current consecutive stationary interval. Its image identity/time is
unchanged, and no previously assigned future action slot is rebased. Readiness,
waiting, original VLA frames and inference calls remain separate counters.
Waiting expiry, old/foreign images, motion during inference and service failure
produce an explicit stop/pause. This entry is a continuous timing/grasp preflight,
not Qwen, scientific-station, formal8/10or full-task qualification.

```bash
cargo build --bin bevy_sim2sim --features dev_tools,dev_tools_minigame/g1_constraint_diagnostic,dev_tools_minigame/g1_source_lighting
# Use a frozen matched source capture JSON with4calls and no prefetch field;
# the original local N1.6service must already be loaded on its declared endpoint.
/path/to/bevy_sim2sim --scene g1_mobile_wait_grasp_diagnostic --robot g1 --g1-config /path/to/frozen.json --g1-ticks 1000 --output /path/to/new_evidence
```


Fresh0274passes the bounded continuous original grasp/standing preflight: four
new official N1.6inferences,200unchanged original frames and518explicit waiting
integrations,718single50Hzsteps in total. New actual image Ticks are0/149/333/436;
actual owner chunk starts are0/185/365/468. Image admission ages are0/720/640/
640ms with no restamping or assigned-slot rebasing. The active sim/wall ratio
is0.999817943and the continuous first-to-last-boundary ratio is1.000085186,
with zero missed control deadlines/debt and a complete718record trace. Minimum
upright is0.993803382; the final200wait samples retain hand-only support. The
owned N1.6service closes bySIGINT. Trace SHA256:
`58c8ff426e01f5a98ecbae643811016652e65e0c2b72a0911604e0a2a6b3628a`.

This establishes continuous timing and physical grasp in the source-near
preflight. Classical carry/release image admission still uses the earlier
paused-boundary interface and must be upgraded before claiming continuous
full-task operation. Qwen/scientific station/frozen10episodes/faults/terrain/
coexistence/fullvideo/runbook remain open; neither goal nor main branch is
complete.


The continuous classical admission seam keeps the unchanged RGB observation
stamp and records a separate actual execution start. Its private proof is built
from at most201owner self samples during explicit stationary waiting, original
joint FK/IMU and velocity odometry. Maximum camera-to-owner displacement/palm
motion is5mm, rotation change0.01rad, and image age remains the frozen1second
limit. Relative navigation points are adjusted by measured self displacement;
public-map search headings remain absolute. Legacy paused constructors keep
requiring the exact current image boundary and preserve their old targets.

Read-only0275reuses actual0274self/RGB identities, with zero new images, model
calls, torque updates or world integrations. Three observations admit; the
Tick333image at actual owner365rejects because measured palm motion is
0.005016610m. It is not converted into a pass by relaxing5mm. A forged moved
past-image self state, old image and proof reused at another owner boundary
also reject. An initial missing test-only deserializer/dependency compile error,
and the wrong assumption that a zero-age image could be compared against its
own mutated current state, are retained beside the repaired explicit results.

`g1_mobile_continuous_release_diagnostic` is a separate3150Tickmaximum source
route candidate: four original VLA chunks, public-map search, current auxiliary
RGB coarse approach, at most five newly observed fine segments, actual aligned
hold, placement-only RGB, gravity-horizontal opening/settling, and explicitly
recorded standing between these operations. Marker visibility additionally
requires actual fourth-chunk completion; absoluteTick200does not enable markers
in its learned inputs. The owner transitions directly from every completed skill
to its finite200Tickwait, preserving actual upper/finger targets. Every new
classical skill must obtain the private image/current-self proof; expiry or
excess motion pauses explicitly. This candidate has not yet established fresh
continuous carry/release, Qwen, scientific-station or formal ten-episode success.
Validation passes78simulation tests/40ignored,35rendering/3ignored,
42development/2ignored,14CLI tests and default compilation. The original
17system-parameter compile error is repaired by grouping the two camera/gate
resources, not by changing scheduling or physics.

The first continuous full-route candidate0276 completes1078actual50Hz single
integrations, four fresh original model calls and200unchanged action frames.
Its active and uninterrupted boundary sim/wall ratios are0.999907and1.000262,
with zero control misses or Tick debt. It loses hand-only support at807during
the initial scan; the actual auxiliary image correctly rejects the absent box
marker. Strict placement remains0seconds. Trace SHA256:
`00d9b7bdd6281b838d14b04a60df3da875a185bb4d21f5a33f41e6e1414d36f4`.
The owner image proof was retained internally but omitted by the development
trace projection; subsequent traces now explicitly serialize it. This failed
run is not full-route or Qwen/science-station qualification.

A fixed saved-input comparison0277reproduces all1078native f32physical samples
and the807loss. Comparison0278preserves the actually executed original upper
command instead of applying the source palm-gap correction, with an identical
590Tick prefix and unchanged motors, contacts and navigation; support ends
earlier at776. The alternative is not promoted. Both have0freshRGB/VLA and
are mechanical diagnostics, not autonomous trials. Thus simply retaining the
narrower original target is not an established remedy.

Entry0279adds an explicitly selected timing candidate: the next original VLA
image may be acquired in the completed predecessor's second half, while its
unchanged50frames execute. A single model reply remains buffered until that
chunk completes; finite standing handles late arrival. Admission preserves
the real image identity, allows no active-chunk replacement or future slot,
and keeps the1second image-age bound. Classical geometry still requires the
separate stationary self-history proof with5mm/0.01rad bounds. The earlier
stationary-only and fixed scheduled diagnostics retain their original guards.
This opt-in pipeline is not qualified until fresh physical evidence exists.

Fresh0280validates that the second-half-image pipeline still consumes all200
original frames exactly once: images0/25/88/163admit at0/58/132/198, without
restamping. It completes332native integrations but the box remains supported
by the original shelf; the source-gap correction exceeds its unchanged0.1rad
joint-change envelope before the first turn. No classical motion is integrated
and strict placement is0seconds. Ratios are0.996934active/0.999634continuous;
the rejected owner step leaves1due Tick in the terminal snapshot, so the
strict zero-debt timing audit does not pass. Trace SHA256:
`fa57a90ebd733266b8561143c411685f0bd102a1dd9aa9a269bd2e8de66b6ab1`.

The second-half candidate is not promoted. Entry0281instead requests original
images at the completed full action boundary, before adding any stationary
settling requirement. A bounded next-chunk admission accepts only images from
the first25Ticks after that exact predecessor end, keeps the1second age limit,
and still starts the unchanged next50frames at the actual current owner Tick.
Late perception runs alongside explicit standing. This separates the original
VLA's moving self observation from stricter classical geometric admission,
while avoiding an image taken halfway through an unfinished learned action.
Neither pipeline grants grasp, full-route or formal task qualification.

Fresh0282uses completed-boundary images0/51/143/232and actual next-chunk starts
0/93/177/266. All200original frames are unchanged, and all581postgrasp samples
through897retain hand-only support, including442real scan Ticks. Its coarse
image846arrives at owner897:1.02seconds old, despite only3.58mm measured palm
motion and0.008245rad self rotation. The frozen1second bound correctly rejects
the carry before any carry integration. There is0strict placement,0deadline
misses and1due terminal Tick; the full-route/zero-debt checks remain failed.
Trace SHA256:
`2c0e4a282bf5c7341d6ef98f999d04a24bf6f8427a38033219a046f7a40b5c14`.

Entry0283loads a hash-bound single-episode CPU worker during initial scene/model
preparation. It wraps the unchanged pinned RGB/local-geometry implementation,
keeps one pending request and at most12images, and rejects repeated, foreign or
unbound observations. Physics never reads its pipes. The saved actual846image
returns a localization exactly equal to the original CLI result; CPU processing
is190.1ms after91.5ms preparation. Busy/repeated/old-episode inputs are rejected
and the owned process is reaped. This check uses0newRGB/physics/VLA, and does not
establish full-route success. Image age,5mm palm motion,0.01rad self rotation,
grip targets, source geometry, motors, contacts and50Hz physics remain unchanged.


The fresh persistent-worker route0284 performs1535actual single50Hz integrations,
four original VLA calls and three actual RGB localizations. All1227postgrasp
samples retain hand-only support; the box travels1.55478m and minimum upright
is0.992687. Its third image is rejected before another movement at0.8s age:
measured maximum palm motion0.00601613m exceeds the unchanged0.005m gate.
Strict placement is0seconds; the terminal rejected Tick remains pending, so
this run is neither task success nor a zero-debt timing pass. CPU/model children
are both explicitly reaped. Full trace SHA256 is
`f8f2feef99a815d57a59763d9bf4589537c184726cf25faed74b82c0090eb8ff`.

CPU profiling of that saved actual image locates0.135s of0.180s in repeated
parsing of frozen public JSON, while marker detection consumes0.005s.
`PinnedPublicVisionAssets` now parses the hash-bound public robot, task geometry
and fiducial calibration once during worker preparation. Observations and RGB
are still read afresh. The coarse and both fine results are exactly value-equal
to the original0284 outputs, at0.0102–0.0129s on this CPU-only comparison.
The cache cannot load an observation path. No physical parameters, image age,
self-motion limits, marker geometry or perception math changes. Saved self
history also shows the rejected image already exceeds5mm at0.34s; faster JSON
alone therefore does not prove the entire route will pass. Bounded reobservation
while the existing safe standing wait remains active is the next missing
recovery behavior; rejected inputs must never become an executed skill.


Bounded stale-image recovery0286 preserves the existing owner wait rather than
executing or restamping a rejected skill. A discarded observation remains
ineligible; a distinct new frame may be submitted. Waiting limits and all
self-motion/image gates stay unchanged. The RGB route permits at most two
reobservations and commits its next stage/fine-step count only after the native
owner publishes the corresponding accepted image proof. Wait exhaustion or
unsafe self state still halts explicitly.

A saved mechanical comparison uses three identical590Tick prefixes. The
rejection and reference owners then run138remaining standing Ticks with every
serialized body/control result equal. A repeated rejected command cannot extend
the200Tickwait; the next expired call halts without inference/torque/integration.
A third owner rejects the old frame for3Ticks, then admits the distinct original
saved scan image once. Counts are728/728/594=2050new native integrations, with
zero fresh RGB/VLA calls. This verifies rejection semantics only. Initial test
compile and an incorrect assertion about the halt latch are retained separately.
Current79simulation tests/42ignored and42development tests/3ignored pass, as does
the default app compile. Fresh perception/carry/release acceptance remains open.


The next fresh route0287 stops for a different cause:887single50Hz Ticks,
four fresh original calls, and grip support lost atTick816 during the scan,
before any coarse localization, carry or stale-image recovery. Active and
continuous clock ratios are0.999880/1.000095 with zero missed/due Ticks;
placement is0seconds. Full trace SHA256 is
`5725e0dd303d4e15dae469cf377707cce566f5a979ea2d8fb5f049617b7f1800`.
This failure is not evidence that cached perception or image rejection caused
the loss of grip.

A bounded mechanical comparison0289/0290 reuses the exact373Tick prefix and
464manual scan commands from0287. Both insert50zero-navigation preparation
Ticks after the same source-gap calibration. One keeps that calibrated command;
the other adds50bounded Cartesian increments totalling0.02604676m forward,
derived from the actual pre-scan box marker image and public native calibration.
Both937Tick trajectories retain hand-only support throughout. Thus forward
translation is not established as a remedy and is not installed in production.
The shared contact-settling time is the next causal candidate. Marker projection
and independent FK agree with the image; the pre-scan box estimate differs from
auditor truth by0.000204m. Auditor geometry stays outside the decision chain.
The reference calibration is a native50Hz replay of original source actions,
not a measurement of the upstream PhysX runtime.

The continuous development route now has a separately recorded `GripSettle`
phase before its first scan: fixed minimum50single20ms Ticks, zero navigation,
unchanged source-gap calibrated upper targets, and20consecutive named-self
velocity samples below0.03m/s. Failure to settle by250Ticks halts. It uses the
same private current-image/self-history admission as other observed skills.
After completion it enters the existing finite200Tick wait and requests a new
actual RGB frame before turning. The ordinary placement hold remains100minimum
Ticks. No force, friction, joint gain, frequency, hidden integration or forward
translation changes. This is disclosed traditional control; fresh continuous
carry/release, local Qwen, scientific station and frozen8of10 remain unqualified.


Fresh settled route0292 executes1460single50Hz Ticks and4fresh original calls.
The50Tick grip-settling phase and442Tick scan retain support, but support is lost
atTick1282 during the388Tick walking segment. Box net displacement is1.30671m;
strict placement is0seconds. Active/continuous clock ratios0.999861/1.000111 and
zero misses/debt pass this finite timing check only. Both owned inference children
are reaped. Full trace SHA256:
`81ed6a6a1d66321b1acd69124b752d6647c6688a76f706754a815f7fea7393e0`.
The handoff remedy does not establish walking robustness. No repeated frequency,
gain, force or material search follows from this result. Station/Qwen integration
continues independently while actual pre-drop grip/contact changes are audited.

The `20261003_0300` offline stop comparison reconstructs every physical command
of0292 without perception or queue admission. Its reference reproduces all1460
complete body steps exactly, including support loss at1282. One fixed30Tick
navigation deceleration changes only29nonzero navigation vectors starting1264;
all1263prefix body steps remain exact. The candidate loses support at1272,
while the original Homie walking policy is still selected. Therefore abrupt
walk-to-stand selection is not established as the sole cause, and this candidate
is not installed in navigation. The two1460Tick worlds use the original50Hz
integration and4nonintegratingPGS, unchanged gains/friction/mass/grip targets.
They supply causal mechanical evidence only, with0fresh RGB/VLA calls and no
real-time or task qualification. Inspect actual palm/box geometry and supporting
contacts before any further control change.

The native scientific-station foundation is now available to standalone Homie
standing diagnostics. The mobile owner receives the same immutable, hash-bound
prepared geometry as rendering and replaces its broad startup floor before
Tick0. Its original50Hz clock, actuator backend, gains, mass, limits and normal
one-PGS setting are unchanged. Each completed body step exposes the environment
identity/installation receipt for independent auditing; no environment truth
enters policy self state. Serialized inputs cannot inject prepared geometry.
This first mobile station entry rejects all task objects and assisted factories
before model loading, so an unvalidated original background cannot silently
overlap the station. Actual carrying in the station remains unqualified.

The first finite station stand (`0339`,fa4864e) records150actual native/WBC
ticks over3seconds, normal1PGS,0VLA/Qwen calls and2553installed station
colliders. All149positive-contact boundaries contain only source foot bodies7/14;
minimum pelvis height0.74955m and upright cosine0.998989 pass the frozen finite
entry guard. Clock ratios0.999474(active)/1.003309(boundaries),0misses and0pending
ticks pass. The actual1080p station/head images and complete150step trace are
retained. XY drift0.12838m and final horizontal velocity about0.25m/s are material
limits: this is neither long-term stationary stability nor a carrying result.
The retained Rapier solver-point arrays are empty after consumption, so their
positions are not claimed as measured contact geometry; subsequent output keeps
the actual positive impulse/body identity only. No task capability is enabled.

The single saved-walking convergence comparison (`0343`,9e4fcf5) retains all1460
commands and an identical996Tick grasp/hold prefix. The original4PGS reference
reproduces every retained0300body step exactly. Changing only nonintegrating
PGS to16 at997 loses all hand support at1038 instead of1282; it is rejected,
and neither normal nor live diagnostic solver settings change. Both worlds
complete1460native/WBC updates with one20ms integration per Tick,0fresh images,
VLA or Qwen calls. There is no solver-parameter grid or task qualification.

Read-only audits show the two palms' measured-joint FK agrees with actual body
positions within3.31e-7m; maximum joint-anchor discrepancy is below1.92e-7m.
Thus this trace does not support an assembly/anchor-drift explanation. During
the common bilateral-contact interval997..1035, arm tracking RMS medians are
0.072903rad(reference) and0.073814rad(candidate); the apparently lower candidate
error over the longer walking window includes its unloaded, dropped-box state.
Palm-origin gaps are not collision-surface clearance. In the reference, box
angular speed first exceeds2rad/s at the first stop Tick1264 (7.212rad/s); the
candidate reaches that threshold at1027 during walking. Contact-loss timing and
normal magnitudes alone cannot establish a friction/torque cause. Diagnostic
point records therefore preserve active contact identities, raw solver-basis
friction impulses and cached lever arms/anchors without feeding them to control.
Cached anchors and normals are explicitly distinct from fresh post-step shape
queries and from measured world-space friction forces.

Read-only point reproduction0344 (7c0add4) matches all1460complete legacy
records exactly after stripping only the added diagnostic output. It records
1460new native/WBC updates,0fresh RGB/VLA/Qwen calls and unchanged4PGS/50Hz.
All box/robot points use the supported pointwise Coulomb path; ordinary
rigid-body simplified friction explicitly reports pointwise actual tangent
impulses as unavailable instead of reporting a misleading zero.

At1281, the legacy cache still reports0.090455N.s on hand bodies32/30, while
the selected solver points report zero. Thus the earlier1282 "support loss"
was one Tick late. This is an evidence-aggregation defect, not a changed
physical trajectory. New diagnostic fields sum only current solver-contact
identities and keep legacy totals separately labelled; unavailable active
evidence is never replaced by the cache. Static/mobile placement auditors
require this active evidence for support and conservative release, retaining
geometry/speed/two-second thresholds. Eighteen adversarial Python checks and
the actual-engine sliding/stale-point regression pass. Old cached-only task
scores remain historical and require revalidation; they do not establish
present qualification. The matched SDK contact-force matrix audit is separate.

The fixed500Tick station continuation0345 (c2ed4b4) keeps the original command,
one-PGS setting and all150physical prefix samples from0339 exactly. It remains
upright for10simulated seconds (minimum0.993638,height0.743875m), but drifts
0.345226m by the end; the final2second horizontal-speed median is0.084182m/s.
Current active impulse evidence identifies only foot bodies7/14 when available;
unavailable point lists remain explicitly unverified. This unpaced mechanical
test establishes neither stationary placement nor real-time carrying.

One source-contract-preserving self-velocity feedback comparison0346 (0abcad5)
changes only navigation after the exact150Tick prefix. It rotates named self
velocity through measured yaw, sends its negative with unit gain and the existing
0.3m/s bound, and retains the original0.05stand/walk selection. No world position,
task/contact input, upper target or physical parameter changes. Final drift drops
to0.243029m, but final2second median speed rises to0.091339m/s: the frozen joint
improvement rule fails. The candidate is not installed or tuned further. Its
500native/WBC steps, original negative result and trace are retained; it grants
no task, stationary hold, model or real-time qualification.
