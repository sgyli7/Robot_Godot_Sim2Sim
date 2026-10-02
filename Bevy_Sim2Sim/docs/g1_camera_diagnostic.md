# Native G1 camera diagnostic

This explicit development entry renders the initialized native Homie G1 world.
Its floor matches the runner's 40 × 0.5 × 40 metre floor. It does not load the
science station and does not qualify standing, grasping or either task.

From the integration `Bevy_Sim2Sim` directory, prepare a new configuration:

```bash
python3 crates/dev_tools/python/scripts/unitree_g1_camera_config.py \
  --model-dir /home/ethan/models/unitree_g1/homie_v2 \
  --ort /home/ethan/models/unitree_g1/homie_v2/tools_env/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.30.0 \
  --output .scratch/g1_camera/config.json

CARGO_TARGET_DIR=/home/ethan/Projects/Sai_Lab/Bevy_Sim2Sim/target \
  flock /tmp/sai-g1-cargo.lock \
  cargo build --locked --offline --features dev_tools --bin bevy_sim2sim -j2

DISPLAY=:1 BEVY_SIM2SIM_ASSETS="$PWD/assets" \
  /home/ethan/Projects/Sai_Lab/Bevy_Sim2Sim/target/debug/bevy_sim2sim \
  --scene g1_camera_diagnostic --robot g1 \
  --g1-config .scratch/g1_camera/config.json --g1-ticks 0 \
  --output .scratch/g1_camera/render
```

The model cache must already exist. Both commands refuse to replace their output.
The shared Cargo target above is reserved for this integration worktree; sibling
worktrees must use their own target to preserve source identity.

The window exits after saving `ego_640x480.png`, `main_1920x1080.png`,
`ego_stamp.json` and `capture_receipt.json`, or on a 75-second diagnostic timeout.
The receipt records actual integration, torque and model-call counts. Tick zero
loads the pinned models but performs no model inference or physics integration.
The ego stamp binds acquisition time, camera pose, 53 native body poses and 43
measured joint positions/velocities from the same immutable boundary. GPU map
completion does not replace image acquisition time.

`--g1-ticks N` permits a bounded 1–150 Tick run of the unqualified standing
candidate; its failures remain failures. It never enables an accepted task mode.
External interruption can leave partial files; only a successful receipt confirms
the capture, and `task_qualified` always remains false here.

The 2026-10-02 initialization run completed in 2.32 seconds with 0 integrations,
0 torque updates, 0 inference attempts, a 27 ms image readback and matching
640×480 / 1920×1080 images. Nine camera contract tests passed. Camera comparisons
retain the 0.0001 radian rotation tolerance using a sign-invariant quaternion
chord, which avoids `acos` roundoff rejecting identical orientations.

A later actual Tick-150 run at `49eb9f6` captured the completed boundary after
150 inferences and 150 integrations. Active simulation/wall ratio was 0.998649,
with zero missed deadlines and pending ticks. Both PNGs retained 640×480 ego /
1920×1080 main and MSAA8; their immutable stamp contained the same 53 body poses
and 43 measured joint positions/velocities. Evidence is frozen under
`/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/g1_completed_boundary_camera_001/manifest.json`.
This proves a synchronized running native camera boundary for the floor-only
scene; it does not qualify science-station contact or task behavior.

## Live matched-policy diagnostic

The same entry also accepts an `ArenaTaskRunnerConfig` in `runner` and a
`policy` object with `endpoint`, `max_calls` and `timeout_ms`. Both must be
supplied together. The endpoint is IPv4 localhost `/infer`; the profile selects
the separate static or mobile client. This mode permits one to four complete
chunks only, and `--g1-ticks` must exactly equal the selected model's horizon
times `max_calls` (at most 200 ticks). The runner requires an explicit wall-clock image age limit.

Prepare and start the verified local service separately, for example:

```bash
/home/ethan/models/unitree_g1/envs/policy_onnx_cu13/bin/python \
  crates/dev_tools/python/scripts/unitree_g1_static_server.py \
  --receipt /home/ethan/Projects/Sai_Lab/.scratch/unitree_g1/policy/static_apple_files.json \
  --device cuda --port 5557 --capture-dir .scratch/NEW_CASE/server_captures
```

Only actual RGB and the measured named joint state enter inference. Each reply
resumes the same owner/world/controller history for its complete action chunk;
no reset, pose replay, object coordinate input or acquisition restamp occurs.
Between chunks the simulator explicitly pauses for camera and inference. This
is a live diagnostic, not qualification of continuous real-time task execution
or a physically verified holding/stopping controller. The procedural support
is displayed using the actual owner-world cuboid, with a disclosed diagnostic
material; original background renderer parity is not claimed.

The 2026-10-02 run at `ba42cab` completed three actual local VLA calls and 120
body-control calls, torque updates and integrations. Active simulation/wall
ratio was 0.99593, with zero pending ticks/deadline misses; pause intervals are
excluded. Matching RGB/self-state/action receipts and completed boundaries at
Ticks 40/80/120 were retained. G1 stayed upright, but the apple was not lifted.
This closes the live input-to-actuation seam only. All T1/T2 acceptance gates
remain open. `live_policy_*` counts are separate from WBC inference counts.
