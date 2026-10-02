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

[contact_reports]: https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/physics/rb_physics.html
