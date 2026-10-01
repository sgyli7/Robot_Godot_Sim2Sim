# Native G1 visual decision probe

This development binary consumes an existing `ego_640x480.png` and its original
`ego_stamp.json`. It validates the exact native 53-body/43-joint snapshot, root
sensors, head-camera pose, episode and 50 Hz source boundary. It then submits one
real RGB request through `DecisionWorker` and `LocalQwenClient` to the existing
loopback service. It starts neither rendering, physics nor a Qwen service.

Run from the integration `Bevy_Sim2Sim` directory after the main agent builds
the approved source with its reserved warm Cargo target:

```bash
CARGO_TARGET_DIR=/home/ethan/Projects/Sai_Lab/Bevy_Sim2Sim/target \
  flock /tmp/sai-g1-cargo.lock \
  cargo build --locked --offline -p dev_tools_minigame \
  --features rendering_preview --bin g1_visual_decision_probe -j1

/home/ethan/Projects/Sai_Lab/Bevy_Sim2Sim/target/debug/g1_visual_decision_probe \
  .scratch/NEW_CAPTURE/ego_640x480.png \
  .scratch/NEW_CAPTURE/ego_stamp.json \
  .scratch/NEW_CAPTURE/decision_receipt.json
```

The output parent must exist; the receipt is reserved with `create_new`, so an
existing result is never replaced. The endpoint defaults to
`http://127.0.0.1:8002/v1`, and the model defaults to `qwen3.8-27b-fp8`.
`--endpoint`, `--model`, `--goal` and `--profile static_apple|mobile_box` are
explicit overrides. The client permits loopback HTTP only, disables proxying,
redirects and retries, requests non-thinking strict JSON, and uses a 512-token
output budget. Capability flags always remain false. Only `observe` and `stop`
can be admitted, and this binary executes neither request.

Live mode preserves `captured_at_unix_ms`, episode, source ticks and simulation
time. `ObservationStamp.frame_id` is the original `render_frame`; the original
capture sequence remains in the receipt. The simulation is explicitly assumed
paused at that same captured boundary. Wall time is read again for preparation,
expiration and admission. The original 30-second wall and simulation TTLs are
unchanged. Capture, load, decode and HTTP latency therefore all consume the wall
TTL. A stale saved image must fail; do not edit its stamp to make it fresh. A
new native capture is required for live admission after service warmup.

For isolated model/schema warmup, an older original capture may be used:

```bash
/home/ethan/Projects/Sai_Lab/Bevy_Sim2Sim/target/debug/g1_visual_decision_probe \
  .scratch/OLD_CAPTURE/ego_640x480.png \
  .scratch/OLD_CAPTURE/ego_stamp.json \
  .scratch/OLD_CAPTURE/warmup_receipt.json --warmup
```

Warmup uses `LocalQwenClient.decide` with a 120-second HTTP timeout and the same
original image, stamp and strict schema. It creates no `DecisionSession`, admits
no skill, and reports `warmup_only=true`, `live_admitted=false`. Live HTTP has a
30-second timeout and a bounded 32-second worker wait. Freshness expiration can
stop waiting earlier; a dropped worker may finish its already-running request
within the client timeout. The probe performs no second submission.

Receipts include input PNG/stamp hashes, compiled source hashes, executable path
and hash, the complete original native stamp, sent self measurements, re-encoded
wire PNG/request hashes, actual worker submission/result counts, observed client
outcome, latency, admitted request or session-generated safe stop. The client
does not expose a raw network-start counter or raw HTTP envelope, so a submitted
request is not mislabeled as a confirmed HTTP call before a reply arrives.
Joint q/dq and root velocity come directly from the native self state; projected
gravity is computed by inverse root-quaternion rotation of source `[0,0,-1]`.
No task-object poses or independent acceptance truth enter the model request.

Timeout, expired acquisition time, invalid pairing, malformed service output and
failed admission save a failure receipt and return nonzero. An admitted observe
or stop only validates this local decision seam: `qualified=false`, zero executor
actions and zero physics integrations remain explicit. Warmup and numerical
guard fixtures do not qualify standing, grasping, task success or a running game
loop.

The six guards for unchanged identity, mismatched native state, interpolation,
camera-pose forgery, expiration before/after HTTP, false capabilities, wrong
episode and output replacement passed in the integration tree at `6fbf817`.
At that source revision a new native Tick-zero capture reached the real local
Qwen service through the Rust worker: one returned request, 15.235 seconds HTTP,
18.109 seconds original image age at admission, and a valid `observe` decision.
The unchanged image subsequently expired; the actual probe refused it before
submission (zero client calls) and produced a safe-stop request. The model owner
was then stopped. None of these results executes a physical skill.

Frozen logs, receipts, source/binary hashes and original captures are in
`/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/g1_bevy_qwen_rust_001/manifest.json`.
This is a paused initial-state observation seam. Evolving task scenes, model
coexistence, task execution, latency percentiles and formal success rates remain
unverified.
