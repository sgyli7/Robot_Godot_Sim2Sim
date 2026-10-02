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

## Interactive native-camera window

The main development executable also provides an explicit `g1_task_lab` scene:

```bash
flock /tmp/sai-g1-cargo.lock env CARGO_BUILD_JOBS=2 \
  CARGO_TARGET_DIR=/home/ethan/Projects/Sai_Lab/Bevy_Sim2Sim/target \
  cargo build --features dev_tools,dev_tools_minigame/g1_constraint_diagnostic

/home/ethan/Projects/Sai_Lab/Bevy_Sim2Sim/target/debug/bevy_sim2sim \
  --scene g1_task_lab --robot g1 \
  --g1-config .scratch/NEW_LAB/config.json --output .scratch/NEW_LAB/native
```

The matched capture configuration must contain `task_lab.qwen` with `endpoint`,
`model`, `timeout_ms` and `max_output_tokens`. It uses the existing task owner,
original G1/prop visual assets and head camera. The output directory must be new.
Automatic `--g1-ticks`, a missing task owner, and a lab configuration used through
the ordinary capture scene are rejected before creating the output directory.
No service is started by this window. Verify `/health` and `/v1/models` on the
exact configured loopback port before opening it.

Chinese text entry, start, stop, reset, overview and grasp-detail controls feed
the runtime task module. Start rebuilds the scene and waits for the actual new
owner episode before acquiring RGB. Reset cancels the session/history and
rejects an old HTTP result without launching a second parallel request. Stop
explicitly pauses simulation. Runtime ports carry camera observations, owner
clock, execution intents and feedback; the decision module cannot access the
physical world. Only native RGB and named robot self measurements enter Qwen.
All three physical skill capabilities remain false in this development window,
which admits only observation/stop decisions and performs zero integrations.

`task_lab.smoke_reset_during_request: true` runs one finite reset injection through
the same UI intent path. It requires one old result discarded, one new decision
admitted in the new episode, and an actual window screenshot. The maximum run is
90 seconds, and failures return nonzero with retained receipts. HTTP attempt,
completion and discarded-generation counters are transport evidence; they are
not mislabeled as successful model inferences or task completion. The ordinary
manual window has no automatic termination.

The first GUI runs retained two real native frames, three actual owner resets
and one discarded in-flight HTTP result, but the fresh decision exceeded the
unchanged 30-second image/request limit. An explicit 60 Hz display pacing control
reduced the measured render rate from about 132 to 58 FPS and did not resolve
that failure. Resolution remains 1920×1080 with MSAA8; the independent physical
clock remains 50 Hz. This finite display control is recorded as
`diagnostic_render_hz: 60` and is absent by default.

A subsequent decode-graph service comparison initially used a mismatched
client/service port (8002/8003). Both sends failed before server inference; that
case is retained as a configuration failure and excluded from latency claims.

With the endpoint corrected, the decode-graph profile still exceeded the live
deadline. An isolated original-image warmup without rendering took 63.386
seconds of HTTP time and returned four detections plus a long explanation;
one reported box had reversed horizontal bounds. Warmup does not run admission
and is not a successful visual-location test. A compact prompt subsequently
returned within the GUI deadline but ordinary admission rejected an invalid
detection. Both failures are retained; bounding-box checks were not relaxed.

The generation schema now reflects the already closed physical capabilities:
when all are false it exposes only observe/stop, no unused target enumeration,
and a maximum 64-character reason. The model still receives the real image and
self measurements. When task capabilities are enabled, the existing target and
destination evidence remains required by ordinary admission. The generic wire
format and both robot action contracts are unchanged. An incompatible service
that ignores the generation schema still cannot bypass runtime capability
checks. This only bounds the observation/stop seam; it does not qualify grasping
or certify visual target localization.

Receipts retain the actual last UI failure as well as the finite-test outcome;
the test exits on a fresh decision failure instead of waiting until its overall
90-second budget expires. Chinese widgets use character wrapping. Existing
ICU4X segmentation warnings remain a disclosed dependency limitation rather
than being treated as a measured cause of model latency.

The bounded schema did not itself fix the GUI deadline. Exact Rust request/response
tracing through one owned loopback relay showed 512 completion tokens ending in
`finish_reason=length`: after opening the empty detection array, the decoder
repeated tabs until the budget ran out. The same failure occurred without
rendering. Removing the unreachable item grammar did not fix it; a separate
sorted-schema-order control completed 49 tokens in 8.605 seconds, so property
order was not established as the cause. These controls remain warmup-only and
admit no skill. The installed service reported vLLM 0.27.1 and XGrammar 0.2.3.

The next isolated service control uses explicit XGrammar
`disable_any_whitespace: true` with otherwise identical cached weights, image,
decode graphs, 4 GiB KV allocation and loopback port. This is a service decoder
setting, not a longer image lifetime, a host/GPU configuration change or an
alteration of the physical clock. vLLM documents the option in its
[structured-output configuration](https://docs.vllm.ai/en/latest/api/vllm/config/structured_outputs/).

That real GUI control passed: two completed local requests, one discarded old
generation and one admitted fresh `observe`, at 11639 ms service time and
11707 ms original image age. Both actual native camera frames retained their
own episode/frame identity. The owner performed zero integrations and all task
capabilities remained false. Workspace checks passed 223 tests with 33 ignored
diagnostics. The actual 1920×1080 window screenshot initially captured the
previous pending HUD; the screenshot injection now waits four completed display
updates after admission before requesting the native screenshot. This affects
only evidence capture, not model timestamps or physical execution.

The service control's frozen `service_creation.json` records the complete Docker
argv, immutable image identity, read-only cached checkpoint mount, owned cache
mount, owner/profile labels and compact decoder option. Recreate from that argv
only when its named container is absent. The existing lifecycle script can then
start/stop that owned container using the exact copied `service_profile.env`:

```bash
env QWEN_PROFILE=/ABSOLUTE/EVIDENCE/service_profile.env \
  /home/ethan/LocalServices/Local_Qwen/qwen38-27b-fp8-dgx-spark/g1-service.sh start
# After /health and /v1/models pass on the profile's port, use the lab command above.
env QWEN_PROFILE=/ABSOLUTE/EVIDENCE/service_profile.env \
  /home/ethan/LocalServices/Local_Qwen/qwen38-27b-fp8-dgx-spark/g1-service.sh stop
```

Do not use that script to create the compact profile from scratch: its original
creation path does not read `STRUCTURED_OUTPUTS_JSON`. The recorded Docker argv
is required for this isolated control. For a manual window set the lab smoke
flag to false in a new configuration, preserving all model/body/asset hashes and
using the exact service port. These commands reproduce the development seam;
the final science-station task runbook remains incomplete.

The corrected native screenshot run passed the same real reset/model seam:
two completed HTTP requests, one old result discarded, one fresh observation
admitted at 11499 ms service / 11561 ms image age, zero physical integrations.
Visual inspection confirmed that the saved HUD displays that actual decision,
latency and observation-only feedback. Prematurely closing a required smoke
window now returns failure instead of treating ordinary window closure as a
completed test. The owned inference container was stopped after both runs.
