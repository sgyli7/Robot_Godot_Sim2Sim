# Native G1 camera diagnostic

This explicit development entry renders the initialized native Homie G1 world.
Its floor matches the runner's 40 × 0.5 × 40 metre floor. It does not load the
science station and does not qualify standing, grasping or either task.

From the integration `Bevy_Sim2Sim` directory, prepare a new configuration:

```bash
python3 crates/dev_tools/python/unitree_g1_camera_config.py \
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
