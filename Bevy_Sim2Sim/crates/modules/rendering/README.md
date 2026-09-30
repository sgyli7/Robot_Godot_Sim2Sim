# Science station rendering

`StationVisualPlugin` renders a loaded `StationScene` and can be combined with
`RobotVisualPlugin`. The development entry point
`dev_tools_minigame::visual_preview::run_preview_with_options(PreviewOptions)` creates a real window, waits for GPU
pipelines and font glyphs, and optionally captures the window. The preview has
no robot or physics simulation.

```text
cargo run --features dev_tools -- --scene science_station_preview --robot none \
  --view overview --frames 60 --capture /absolute/path/overview.png
```

Views are `arrival`, `overview`, `towers`, `samples`, `berth`, `hills`, and
`follow`. Tab cycles views; the right mouse button orbits the follow camera and
the wheel adjusts distance. Follow mode awaits a physical robot target and an
occlusion distance computed by the shared physics world.

## Shared scene definition

The asset root contains `game/dynamic_assets/game_data/science_station.ron`, the
matching `science_station_layout.ron`, and the SHA-256 checked GLB at
`game/arts/environment/models/science_station.glb`. The loader rejects missing
assets, mismatched hashes, inconsistent layout definitions, and invalid indices.
`BEVY_SIM2SIM_ASSETS` selects an explicit asset root.

The current static layout is `windpass_compound_v7`, revision 7. It defines
the main-building forecourt, three metre wide circulation lane, berth, open
skills area, four peripheral facility groups with workbenches, connectors,
equipment and route boards, six review cameras, three safe points, and the
2.8 metre wide graded route to the western observation station. The main
terrain, western route and all geological groups use identical displayed faces
for collision.
Fixture records identify their structural colliders by tag. Collision owners
and the six prop records preserve stable names for the physics consumer.
Revision 5 changes six directional labels and their layout identity, preserving
all scene geometry and collider data. Both root and the scene agent reviewed its
six actual GPU captures. Small equipment text and partial lamp occlusion remain;
this is limited static direction improvement, not complete text readability.
Revision 6 adds five typed, vector-drawn facility graphics on existing carrier
faces and changes three sample-cap material roles from blue to purple. Its asset
audit found the original 811,452 oriented surface triangles and 2,496 collider
payloads unchanged apart from the 36 approved cap role changes. Twenty-two
rendering library tests passed (two ignored). A frozen binary and asset archive
produced all six 1920×1080 GPU views, which root reviewed. These graphics make
local facilities easier to identify but do not yet resolve the mostly empty
central apron and distant landscape. A subsequent shader-only pigment pass adds
a quiet cool apron and four corner marks outside the open action area, plus a
continuous sand color fade toward the far dunes. All six new static GPU views
were reviewed; the promoted live shader reproduced the candidate overview PNG
exactly. The layout identity remains v6, while visual evidence binds the new
shader hash separately. The distant geometry silhouette remains unchanged.
Revision 7 organizes the west sample/plant facilities into one work band,
relocates the service cabin and both survey towers as complete display/collider
groups, and adds an open east service arcade. The central 11 by 9 metre action
area stays obstacle-free and gains a visual-only one-metre calibration lattice.
Five new 2.6 metre path segments mark a continuous route from the north ring
to the two relocated towers (eight path records in total); the three-metre ring and 2.8 metre ridge path are
unchanged. The old v6 layout is retained as a historical fixture. Source
geometry and collision records are preserved apart from the three audited
facility moves and 56 new, same-index display/collider pairs. This
new scene identity invalidates earlier scene-bound motion and video receipts.

Station coordinates are metres in a right-handed, Y-up frame. Static surfaces
and colliders use baked world coordinates; surfaces/colliders belonging to a
named prop use coordinates local to that prop. `StationPropVisual` identifies
the six prop root entities for body-pose updates. All live rigid-body poses must
come from the one simulation world; this module creates no physics world and
does not animate joints. The MJCF robot frame requires its own validated basis
conversion before rendering; station coordinates must not be used as that
conversion.

Layout rectangles store `[center_x, center_z, half_width, half_depth]`; loop
parameters store `[center_x, center_z, radius_x, radius_z]`. Ridge waypoints
include their Y elevations. `ridge_grade_plane` gives a continuous plane through
the three uphill control points, clamped at ground level before the slope and at the final landing height. Its
horizontal gradient is below six percent. Geometry tests check exact shared
triangles, the full three metre lane and 11 by 9 metre clear action area, and
full-width graded-route support/structure/geological clearance, including rounded bends and ends. These are
geometry checks and do not qualify real robot traversal.

## Rendering and evidence

The enamel material uses a finite palette, internal structural lines, a
distance-faded constant-pixel ink pass, derivative-filtered hatching, sky and
continuous low-frequency ground washes, glass highlights, and real shadow attenuation. The
preview currently uses eight-sample MSAA. Static screenshots do not validate
motion stability or sustained 60fps.

Original station geometry and labels come from the repository's Godot station.
The font is a source font face with its original license stored under
`assets/third_party/fonts/license.debian`. No UriWallpaper artwork is bundled.
The external reference is [Exploration 006](https://uriwallpaper.gumroad.com/l/exploration006).
Reference page metadata was available; its image pixels were unavailable to
the current browser tool. The URI style match remains an open visual gate.

Raw preview screenshots, logs, hashes, revisions, diagnostic experiments, and
fault reports are under `.scratch/visuals`. The revision archives preserve the
exact binary, assets, source, and configuration used for their six captures.
Changing scene geometry invalidates downstream motion/physics/video evidence.

## Robot initialization geometry preview

The non-default `rendering_preview` feature of `dev_tools_minigame` provides
`run_robot_initialization_preview`, which accepts `RobotInitializationPreviewResources`
and `PreviewOptions`. The caller supplies an explicit asset root and already
loaded `StationScene`, `RobotVisualModel`, `RobotVisualInput` and
`StationCameraControl`. This entry does not look up robot paths, invent appearance
or poses, create a physics world, run FK, integrate dynamics or infer policy.
It displays the supplied completed initialization snapshot with full compiled
source meshes and effective native untextured display normals. It is not qualified gameplay or a
controller evaluation.

The bound exporter and native receipts establish that the current leg/roller
meshes have no UV coordinates. For those meshes, the default reproduces the
MuJoCo 3.10 classic renderer's per-corner face-normal fallback; original compiled
normal arrays, face winding and geometry remain intact. UV meshes require a
separate supported path. The development CLI's optional `raw_normals` variant
retains the old raw-normal rendering solely for comparison. Three actual model
bindings passed 3,760,632 corner comparisons against a separately compiled
official-source C oracle with maximum component error zero. Eight actual GPU
comparison captures qualify this static display-rule correction only.

```rust,ignore
// The application owns definition/appearance verification and the sole world.
let frame = std::sync::Arc::new(assembly.pose_frame(&world.snapshot())?);
let model = rendering_minigame::RobotVisualModel::new(
    definition.clone(), appearance.clone(),
)?;
let input = rendering_minigame::RobotVisualInput::new(&definition, frame)?;
dev_tools_minigame::visual_preview::run_robot_initialization_preview(
    dev_tools_minigame::visual_preview::RobotInitializationPreviewResources {
        asset_root,
        scene, // StationScene::load from this explicit asset root
        model,
        input,
        camera, // supplied target/view/yaw/pitch/distance are preserved
    },
    dev_tools_minigame::visual_preview::PreviewOptions {
        capture_path: Some(capture_path),
        frames: Some(60),
        ..Default::default()
    },
)?;
```

The injected camera controls this entry; `PreviewOptions.view` retains its old
meaning for the standalone station entry only. A close view normally uses the
caller-selected `StationView::Follow` with its target set from the real body
snapshot. No automatic robot repositioning or camera target guessing occurs.

Before opening the window, the entry rejects zero frames, mismatched model/input
identity, missing snapshots, wrong body coverage/generations, nonfinite poses,
bad quaternions and invalid step counters. It reuses actual GPU shader and font
readiness, label texture freezing, screenshot-save acknowledgement, error drain
and timeouts. Capture admission runs after the current frame's robot validation;
a failed or absent status cannot use an earlier `Ready` result. Closing the
robot preview before readiness or before a requested screenshot was saved
returns an error. The original station preview's default behavior is preserved.

Material mapping reports retain original native appearance and explicitly record
sRGBA interpretation, derived PBR defaults and unsupported native planar
reflection. They do not assert MuJoCo pixel equivalence. CPU loader/buffer and
atomic synchronization tests do not establish actual GPU visibility, dynamic
trajectories, robot traversal, URI fidelity or sustained 60fps. The application
must preserve an independent binary/configuration/capture record when it builds
this entry; v4's prior frozen binary is a separate artifact.

The preview returns an error for missing resources, real shader compilation
errors, capture failures, and readiness/capture timeouts. Shader failures wait
for other queued work and drain submitted GPU work before exit. The v3 fault
run observed controlled exit code 2 for missing assets, early/late invalid WGSL,
and capture to a nonempty directory. A previous v2 late-WGSL experiment exited
139; its original failure log is retained. The v3 results cover those recorded
cases only.

## Development completed-pose sequence capture

`run_robot_pose_sequence_capture` opens the same station and verified robot
visuals, then saves one actual GPU PNG for each sampled input pose. The caller
must supply every consecutive completed `RobotPoseFrame` from one 60 Hz
episode, plus the same verified model/scene inputs as the initialization
preview. The renderer checks model identity, body coverage, assembly handles,
episode and contiguous step counters before opening the window. It refuses a
sequence whose sampled poses are all bit-identical. None of these checks proves
the caller's physics provenance or the stability of its controller.

```rust,ignore
let receipt = dev_tools_minigame::visual_preview::run_robot_pose_sequence_capture(
    dev_tools_minigame::visual_preview::RobotPoseSequenceCaptureResources {
        asset_root,
        scene,
        model,
        camera,
        poses_60hz: completed_world_poses, // one real completed frame per 60 Hz tick
    },
    dev_tools_minigame::visual_preview::RobotPoseSequenceCaptureOptions {
        output_dir,
        output_fps: 30,
    },
)?;
```

The output rate may be any integer from 1 through 60 fps. Output frame `n`
uses source tick `floor(n * 60 / output_fps)`; the renderer does not invent
intermediate poses or repeat one tick to fill a timeline. It advances to the
next pose only after the current screenshot is saved and checked. The output
directory gets `frame_000000.png` onward and `capture_manifest.json`, which
records each output frame's source tick, episode, pose SHA-256, and PNG path.
Existing output filenames are rejected to avoid mixing capture runs. Failed
runs may leave partial PNGs but no successful manifest.

This entry captures one continuous episode at a time. A development runner
can capture each skill separately and assemble the successful PNG sequences
into a video, for example with
`ffmpeg -framerate 30 -i frame_%06d.png -c:v libx264 -pix_fmt yuv420p skill.mp4`.
The renderer itself neither runs ONNX
inference nor steps physics.

`StationVisualPlugin` can be embedded in a runtime application. Font render
targets currently freeze through the preview lifecycle; runtime integration
must provide the same readiness/freeze lifecycle. Actual robot rendering,
dynamic prop synchronization, camera occlusion, interaction behavior, moving
camera antialiasing, sustained performance, and full physics traversal remain
separate integration and acceptance work.
