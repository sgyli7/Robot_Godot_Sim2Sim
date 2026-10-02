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
the separate static or mobile client. This mode permits one to eight complete
chunks only, and `--g1-ticks` must exactly equal the selected model's horizon
times `max_calls` (at most 400 ticks). The runner requires an explicit wall-clock image age limit.

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

`exposure_ev100` optionally selects a finite diagnostic exposure in 0..=20.
At 11.7 the formerly clipped plate texture becomes visible; the three-call
comparison physically lifted the apple about 2.5 cm without completing placement.
The four-call original-limit run stopped at Tick 139 on the same index joint
limit guard found in the independent source-action replay.

For the isolated predictive-limit comparison, build with
`--features dev_tools,dev_tools_minigame/g1_constraint_diagnostic` and explicitly
set `predictive_limit_diagnostic: true` in the capture configuration. This
selects only a fresh static task owner and records factory-verified coverage of
43 original angular joints. Builds without the feature reject the option.
The native four-call comparison completed 160 real Ticks with no limit failure,
but still failed the grasp. Its active simulation/wall ratio was 0.9898, with
two missed deadlines and zero pending Ticks; camera/inference pauses remain
excluded and full continuous task execution remains unqualified.

The source procedural support is invisible. Its original visible shelf and
background are separate USD geometry. The disclosed visible diagnostic cuboid
therefore does not establish source image parity, even with matching object
meshes, UVs and camera calibration. This remaining input disparity must be
assessed alongside physical grasp/release contacts.

An optional `background_visual` object supplies `path` and `sha256` for the
original static background export. The import requires the original translated
T1 shelf profile. It adds rendering meshes only and hides the diagnostic floor
and support visuals; their actual collision bodies remain in the same world.
The measured export contains 16 material surfaces, 177889 vertices, 263703
triangles and seven original textures. All three shelf surfaces are included.
Thirty-three surfaces with unsupported/unbound material or authored-UV gaps
remain explicitly omitted. Bevy lighting and MDL compatibility remain different;
this does not establish source renderer or full background physics parity.

With that original shelf/background candidate, four live calls completed 160
Ticks but still failed grasping. The bounded eight-call extension completed
320 Ticks (6.4 active simulated seconds), covering the original six-second task
window while retaining complete 40-frame chunks. The robot remained upright
and no joint-limit guard fired, but the apple stayed at pickup. Active
simulation/wall ratio was approximately 0.9947, with zero missed/pending Ticks.
Camera/inference pauses are still excluded; this is not continuous 1× acceptance.

`diagnostic_constraint_sweeps: 16` is a separate explicit development comparison
and requires `predictive_limit_diagnostic: true`. No other count is accepted.
It selects the fresh static diagnostic factory and records its selection in
the receipt. These sweeps solve constraints without another temporal
integration. Normal configuration and loading remain unchanged at four sweeps.

The eight-call sixteen-sweep live run also completed 320 Ticks without a fall,
limit failure, missed deadline or accumulated Tick. Active simulation/wall ratio
was approximately 0.9928. It did not complete grasping, despite the improved
independent source-action release result. Local model ownership was closed
after the run.

`directional_shadow_maps: false` permits one disclosed camera-lighting control.
Default remains true, and light energy, exposure, geometry, materials and
physics are unchanged by this flag. A zero-integration capture visibly removed
the hard box shadow across the plate. The same bounded eight-call/320-Tick live
comparison still did not grasp; its active ratio was 0.9944 with no pending
Tick. Removing that shadow is therefore insufficient to resolve the task.
Neither flag changes the existing science-station lighting or establishes
equivalence to the original eight rectangular ceiling lights.

`render_only_environment_translation: [0, 0, 0.795]` is restricted to a fresh
legacy static apple/plate capture with zero requested Ticks, no physical shelf,
no policy service and no solver diagnostics. It allows the original background
to be rendered at externally measured source poses for the causal comparison
described in `g1_t1_source_task.md`. Normal background loading still requires
the frozen physical source-shelf profile. An actual zero-Tick preflight passed
all robot/prop pose checks; a nonzero-Tick invocation failed before creating a
capture or loading a world. Receipts label this mode as external-pose rendering.

The background exporter now retains otherwise unbound surfaces when the
original Gprim explicitly authors uniform opaque `displayColor`. No color is
inferred from images. Missing/nonuniform colors, translucent opacity and
unsupported color spaces remain rejected. This restores three complete black
box meshes and four black face subsets previously omitted; all existing sixteen
mesh/material/UV/normal records are exactly unchanged. Procedural MDL wall UVs
and unbound-surface shading parity remain disclosed limitations.

The background material also maps the original authored `albedo_add` before
linear color tint and PBR lighting, using a G1-specific material extension.
Old exports default to zero. Nondefault brightness/desaturation remains
unsupported. The source 6.0 OmniPBR/ClearCoat implementations were inspected;
the parameter is described in the [NVIDIA OmniPBR reference](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/templates/OmniPBR.html).
Unbound Gprim displayColor is rendered as diffuse color without an invented
specular material. Other MDL defaults/effects remain outside the parity claim.

`diagnostic_ambient_brightness` and `diagnostic_directional_illuminance` are
explicit finite camera-scene controls requiring the original background. Their
defaults remain 450 and 15000, and both actual values enter receipts. A bounded
calibration used two fixed shared background patches and actual zero-Tick
captures to solve two linear light coefficients: ambient 2328.264 and
directional 981.363 at unchanged EV11.7. It used no policy inference or task
success data. This is an approximate rendered-light calibration, not original
area-light/RTX/global-illumination parity. The science-station lighting is
unaffected, and native task performance remains unqualified.

Task-owner captures additionally save `owner_steps.jsonl`, one immutable real
completed step per integration, through an independently bounded 512-record
channel. File serialization happens on the development/render thread; the
physical owner only try-sends an `Arc`. Overflow/disconnection is counted and
fails capture completeness instead of delaying physics. The ordinary display
slot may still replace frames. Reset generations remain in evidence while old
display/model observations are rejected. Receipts report record count, dropped
records and coverage of actual integrations. Empty transport fixtures never
manufacture physical step records.

An eight-call native image-policy run preserved all 320 actual steps with zero
evidence drops. Independent containment checks use every original apple hull
vertex in the moving original plate's footprint, rather than object-center
placement. Contact-candidate presence alone is insufficient to identify touch:
a positive-distance, zero-force speculative pair may be retained. Moreover,
Rapier's cached solver distance belongs to the last full collision update.
Published contacts therefore also include a read-only Parry shape-distance
query at the exact completed body poses. Release requires positive current
geometric separation and zero last-solve robot normal impulse; missing or
unsupported queries remain release blockers. No clearance tolerance is added,
and cached distances cannot replace that query. This evidence never enters the
policy RGB/self-state wire and does not establish the formal ten-episode suite.

The repeated actual 320-Tick native run passed the diagnostic placement window
for 2.76 continuous seconds, with at least 0.02828 m footprint margin. Tick 280
retained one hand candidate with cached separation 0.01960 m, exact completed-
pose separation 0.01608 m and zero normal impulse; the original candidate-only
criterion consequently reported just 1.94 seconds. Both results are retained.
All 320 measured joint/root/prop pose and velocity arrays exactly matched the
run before adding the read-only geometry query. Positive hand force, actual
touch, out-of-target geometry, unsupported queries and missing-Tick negative
controls reject release/success. Active physics ratio was 0.98919, with four
missed boundaries and no remaining Tick debt; camera/inference pauses are
explicitly excluded and continuous 1× task operation remains unqualified.
