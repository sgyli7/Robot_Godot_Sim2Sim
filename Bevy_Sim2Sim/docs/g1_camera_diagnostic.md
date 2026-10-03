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

The optional static-only `policy.prefetch_after_ticks: 10` is a separate finite
asynchronous scheduling diagnostic, requiring 2..=8 whole original chunks.
Initial model/camera preparation is paused. During execution, a new real
RGB/self-state request starts after ten current-chunk Ticks while the owner
continues every original frame. One pending replacement may be admitted at the
exact 40-frame predecessor end; neither observation timestamps nor predicted
actions are rewritten. Whole-chunk limits and freshness are checked before
pending admission and again at the exact boundary. Late replies, mutated
sequences/schedules and unsupported future observations fail explicitly.
Normal immediate admission and ordinary physics factory defaults are unchanged.

Render extraction independently requires matching robot, prop and camera Tick
identities. A correctly stamped image completed before the requested boundary
is discarded and recaptured, never restamped or submitted to inference. This
also applies to the final paused capture. Retries are bounded to sixteen in the
receipt; continuous model/physical deadlines are not extended by retries.
`continuous_boundary_*_seconds` measures first-to-last actual integration
boundaries and therefore includes any intervening pause. It supplements the
existing active-interval ratio rather than silently excluding model waits.

The first continuous eight-call run executed all 320 actual WBC updates,
integrations and original action frames with zero evidence drops and no fall.
Its active ratio was 0.99948, and its raw 319-boundary ratio was 1.00152, with
seven missed boundaries and no remaining Tick debt. Nevertheless, it failed
placement: the apple was pushed away from the plate and the independent
successful-placement window was zero. Observation ages at replacement start
were approximately 0.58--0.60 seconds. The original final image arrived from
Tick 319 rather than 320 and correctly failed capture; that evidence is retained.
After adding the bounded final-image retry, a separate two-call/80-Tick real
model smoke passed exact final Tick 80 RGB/state identity, all 80 owner records,
zero missed/pending Ticks and two actual VLA calls. It only verifies the capture
fix, not a successful continuous task.

A fixed-input CUDA I/O-binding comparison retained the same five ONNX graphs,
three actual RGB/self-state inputs and original sequence seeds. All decoded
40-frame actions agreed with the ordinary path and archived actions at
`rtol=atol=1e-5`. Nine timed calls per path measured median 143.83 ms with CPU
intermediates and 142.85 ms with CUDA intermediates, without active rendering.
This failed the predeclared 20% speed improvement gate; the alternative was
not installed into the service. Online rendering runs measured roughly
277--305 ms of graph execution. The performance comparison does not establish
the cause of that difference or task success, and does not change precision,
inference steps, action timing, model identity or physics frequency.

`diagnostic_vsync: true` selects `AutoVsync` only in the development window and
records `render_present_mode`; the default remains `AutoNoVsync`. Resolution,
MSAA8, ego camera, lighting, original assets and the independent physical clock
are unchanged. A frozen two-call comparison failed before the first integration
because its acquired observation exceeded the existing 2000 ms owner wall TTL.
Only its first model call completed. The second-call performance gate could not
be evaluated, so this is not an adopted remedy and the deadline was not relaxed.

## Explicit finite static initialization

A development T1 capture may set `static_startup: true` together with
`predictive_limit_diagnostic: true` and `diagnostic_constraint_sweeps: 16`.
It requires the original static policy, camera diagnostic mode, and no prefetch
or standing-wait policy mode. For eight original40-frame chunks, pass
`--g1-ticks 380`:60actual startup ticks plus320actual task ticks. The normal
capture and all other profiles keep their existing behavior.

The single physical owner performs the startup with the original AGILE defaults,
zero navigation and0.75metre pelvis-height command. At least20consecutive self
velocity/IMU checks must pass by Tick60, otherwise the finite run stops. The
first original task image retains its actual Tick60or-later stamp. No RGB stamp,
body pose or saved VLA command is silently rebased. The receipt's `static_startup`
field records the gate; `owner_steps.jsonl` includes both typed startup and
original VLA records. The independent placement auditor consumes physical truth
only after execution. Neither startup nor successful camera capture qualifies T1.

Explicit native station lighting overrides are applied after its light entities
exist. The optional `native_station_illumination` receipt reads effective ECS
ambient color/brightness and directional color/energy/rotation/shadow settings.
The legacy scalar lighting fields also report those actual values in station
captures. With no override, the station's own defaults remain intact. Receipts
before this fix recorded requested capture parameters even when the station
branch skipped them;0308's separate application audit corrects that interpretation.


`g1_static_memory_place_diagnostic` is an explicit 715-Tick development entry:
60 original AGILE initialization ticks, two fresh unmarked N1.7 chunks,
current 140-Tick marked RGB localization, 250 classical transfer ticks,
then 325 disclosed grasp-memory placement ticks. It also records a distinct
actual 390-Tick near-table image and a final 715-Tick image. Only named joint,
IMU and self velocity telemetry is copied from the bounded owner subscription
into the memory input; task object poses and contacts are excluded. The original
140-Tick localization is never restamped as a current detection. Public robot
FK and a rigid-grasp assumption update the apple estimate; self odometry and a
static-target assumption update the plate estimate. Goal and input hashes,
origin/current timestamps and all 150 lowering/retraction FK receipts are saved
before placement admission. A current observation does not claim current object
detections. The owner remains explicitly paused at each image/decision boundary;
this entry does not qualify uninterrupted 1x, Qwen execution or the ten-case suite.

Use the frozen matched static JSON with `policy.max_calls=2` and valid static
marker worker identities, and invoke the ordinary app with
`--scene g1_static_memory_place_diagnostic --robot g1 --g1-config CONFIG
--g1-ticks 715 --output NEW_DIRECTORY`. The exact budget belongs only to this
entry; ordinary camera, task-lab and other transfer budgets remain unchanged.


The actual live placement at `7abf56b` completed 715 native/WBC ticks, two fresh
matched N1.7 calls and five distinct actual camera captures. All 390 prefix body
records matched the preceding held-transfer experiment (episode identity apart).
The 390-Tick grasp estimate used 5 simulated seconds / 5137 wall ms of disclosed
140-Tick memory; independent position errors were 8.175 mm (apple) and 1.031 mm
(plate). The unchanged independent v4 physical audit found 3.56 continuous seconds
of released, plate-supported, contained, slow placement and continuous standing.
All owned model/app/CPU children were reaped. This is one explicitly paused live
diagnostic, not Qwen execution, uninterrupted 1x or formal ten-case acceptance.

The actual final 715-Tick near-table image still has the forearm covering both
printed target patterns. The unchanged read-only CPU detector found zero target
roles. Physical success does not establish final visual feedback. This negative
result is retained; further camera-angle search is stopped. A bounded physical
observation withdrawal must be preflighted against current self state before
another fresh final visual observation can be evaluated. Existing evidence is
scratch case 0364 (live/physical) and 0365 (read-only final visibility).


`g1_static_memory_place_observe_diagnostic` has its own exact 840-Tick budget.
It retains the original 715-Tick entry and adds 100 fixed left-palm increments
of source +Y=1 mm (10 cm total), then 25 real hold ticks. Admission requires
completed typed placement at Tick 715, matching self identity, all left-hand
commands open, measured finger positions within 0.05 rad of open and speeds
within 0.02 rad/s. This self gate does not claim contact-based release. The
current 100-step FK path is checked before submitting the bounded command;
only one increment is checked within each physical Tick. Camera mounts, gains,
friction, mass, original policy outputs and the single20ms integration stay
unchanged. The purpose is to physically withdraw the occluding hand/forearm
before capturing the final actual near-table image. Independent truth must still
check uninterrupted physical placement; this movement does not qualify a task.


The first 840-Tick attempt at `49b7b76` failed before any native integration or
VLA inference: NVIDIA Vulkan reported inability to allocate presentation state,
then an unconfigured surface panic. All owned processes were reaped; no physical
parameter changed. One exact-code/profile retry completed 840 native/WBC ticks,
two fresh VLA calls and five actual images. The unchanged independent v4 audit
confirmed 6.06 seconds of continuous stable released placement and standing.
The fixed withdrawal did not interrupt the placement window. The final actual
image visibly exposes the apple and plate outline. The unchanged CPU localizer
recovers apple ID31 at 15.97px minimum edge, 0.054px reprojection RMS and 2.653mm
independent position error. Plate ID32 is covered by the placed apple, so no
current two-target precision pass is claimed. Failure/retry/visibility are
scratch 0367/0369/0370, respectively.

Read-only review of the previously frozen ten original traces found hand-only
support at the 140-Tick boundary in seven cases; three remain surface-supported.
This is independent failure classification, never controller input or an actual
new ten-case test. A fixed two-chunk grasp cannot simply be assumed to provide
8/10 acquisition. Actual visual grasp admission/recovery and the formal native
profile still require evidence before enabling the public execute-task action.


At `3b3ba8a`, the unchanged owned local Qwen profile cold-started in 236.35s,
performed one clearly separated old-image warmup, and remained resident during
one new 840-Tick native/VLA/camera run. All 840 body records matched the preceding
run apart from episode identity; its physical release window stayed 6.06s.
The existing typed decision probe then consumed the new final actual image and
43 bit-identical joint measurements with the original acquisition timestamp.
One loopback-only, non-thinking, structured response was live-admitted under
the unchanged 30s frame TTL: `observe`, reason "Apple visible on plate; no action
skills enabled." Service time was 12.451s and complete image age 16.708s.
No independent success/contact/object pose label was sent to Qwen. There were
zero Qwen executor actions; this is final semantic feedback, not Qwen-driven
physical task execution or formal acceptance. Owned Qwen/N1.7/app/CPU processes
were stopped/reaped immediately. Evidence: scratch 0371/0372.

The observation-withdrawal diagnostic now pauses at Tick 190, after the fixed
50 mm lift, before any horizontal transfer. One additional native head RGB
capture must detect apple31 through the same 8 px / 1 px image gates. The
original Tick 140 localization still requires both apple31 and plate32; neither
target identity nor gate is weakened. The two image poses and all 51 named self
samples from 140 through 190 verify at least 25 mm of apple and measured-palm
lift, with no more than 20 mm change in apple position relative to the measured
palm. The latter bound corresponds to the sum of the two previously calibrated
10 mm position-error bounds; it does not establish contact or final success.
Self velocity accounts for root translation. Current image age is limited to
2 seconds and origin age to 8 seconds, with original acquisition timestamps.
No contact/world truth enters this check. Missing, stale, reset, slipped or
stationary-object observations block horizontal motion and pause the simulation.
The initial and grip localization have separate immutable input directories and
receipt fields; the CPU worker permits at most these two captures. Other
diagnostic entries retain their prior single-capture route. This development
gate is not a recovered grasp, a formal task result, or a qualified UI skill.

The first actual lift check at `fa6ad75` stops after 190 native/WBC ticks:
apple31 is occluded by fingers in the original Arena camera, although plate32
passes. No horizontal transfer executes. A read-only geometry projection using
only actual140RGB and190self confirms the existing published near-table mount
places the expected whole apple-label border about90px from the frame edge,
versus31px in ArenaEgo. This predicts image coverage, not unoccluded visibility
or grasp. The lift check therefore uses that one already published fixed
`StaticPlacementOverview` mount, with two render frames after switching. The
original learned-policy images and initial two-target localization remain
ArenaEgo. The CPU reply is bound to the requested mount. No new camera angle,
search grid, control pose or precision relaxation is introduced. Cases0374/0375
retain the failed actual check and public-geometry-only projection separately.
