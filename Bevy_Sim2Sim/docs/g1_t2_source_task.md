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

The audit also retains two failed geometry-reader assumptions (unmeasured
dynamic background and lowercase robot path) and a failed released-stack import
root check. The latter was resolved by verifying the actual installed copied
source roots against all 3205 frozen source files, not by relaxing byte checks.

[contact_reports]: https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/physics/rb_physics.html
[cooking]: https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/extensions/runtime/source/omni.physx/docs/api/python.html
