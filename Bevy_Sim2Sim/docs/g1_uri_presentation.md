# G1 URI spectator rendering

The G1 development window uses a URI spectator view with the official neutral
G1 finish: silver-gray torso and limbs, charcoal head, joints, hands and feet,
and a dark visor pigment on the original head surface. Restrained blue sky and
warm ground reflections are shading, not painted blue/yellow body panels.
Clean surface shading keeps the source robot mesh legible; ink shells are used
only for station surfaces. The source's default normal calculation drops tiny
faces in meter-scale unbound meshes. The presentation uses private mesh copies
with double-precision angle-weighted smooth normals and a shader fallback for
zero/invalid normals; all positions and indices are unchanged. Source normals
and materials stay untouched. The shell pigment is RGB 176/181/186, visibly
darker than the station's warm white. Robot ink shells are omitted. The existing
station gets clearer blue/white/orange color groups and a light flush floor grid. Geometry, body
proportions, task placements and the thin original support slab are retained.

The spectator camera is explicitly tagged `G1PresentationCamera`. Its layer 1
uses no additional tone mapping, so source sensor ACES calibration cannot
wash the neutral spectator gray back toward white. Sensor tone mapping is
unchanged. The layer's
copies share station mesh handles and use private robot meshes with repaired
shading normals. They copy the source global transforms and
inherited visibility after propagation, before visibility checks in the same
frame. Removed sources remove their copies. Copy materials are separate assets;
copies cannot cast or receive shadows. Source entities, materials, lights and
sensor cameras remain unchanged on layer 0. Task and fiducial textures retain
their original colors and texture assets.

At startup, a private 64 × 64 offscreen camera warms the source pipelines while
the spectator pipelines compile. Readiness requires the URI shader as well as
the existing source shaders. The unread camera is retained throughout the
run so later-revealed source materials keep specializing before one-shot
sensor captures; its image is released with the application. Hidden source standard-material variants, including later revealed
printed markers, are also queued through unread layer-2 copies during startup.
Those initial warmup meshes are excluded from presentation cloning and removed
after initial readiness. This prevents first sensor frames from missing newly revealed
textured materials while their pipelines compile. That image has no capture port, is never read back and cannot enter
a model observation. This avoids waiting for source shaders that otherwise
would not be queued until the first sensor capture.

`G1_URI_PRESENTATION=0` disables this development presentation plugin and
restores the source spectator view. The default is enabled for G1 capture and
task-lab development entries. It changes the appearance of the main-window
screenshots only; head and fixed auxiliary sensor captures retain the source
renderer. No model request, timestamp, observation/action schema, physical
parameter, initialization, control frequency or controller is modified.

The source warehouse's dedicated `BackgroundMaterial` is also copied into
the spectator layer. Its existing floor, shelf, cart and clutter remain
visible; missing material-type coverage must not leave a blank backdrop.

The rendering remains an interpretation of the [URI art direction](g1_uri_art_direction.md),
constrained to existing meshes. It cannot add the concept illustration's
imagined fine shell seams or hardware. The illustration is not loaded by the
renderer and cannot replace real simulation captures or VLA observations.

The user supplied a further [G1 field-operations poster](../assets/game/arts/entity/textures/g1_uri_finish_reference.jpg)
on 2026-10-04 (SHA256 `0c61f1a76ea53919c2708fa1a59655cc73d56721177a0b193aa8527f7da48598`). It guides
the URI environment and finish treatment, not a custom body repaint. The user's
clarification keeps the robot predominantly gray. The neutral factory finish
was visually checked against the [official G1 page](https://www.unitree.com/g1/)
and its [official product image](https://www.unitree.com/images/773f6a21c6764bc8bfd9da5f9d714ecf_3840x3600.jpg)
on 2026-10-04. The site's product image shows silver shells, a charcoal head and
black joints/hands/feet, with a blue face light. No blue helmet or orange
shoulder paint is applied. It does not replace the five fixed
URI style references or authorize different geometry, tools, people, dirt,
branding, marker graphics or scene layout. The earlier concept illustration
retains its original blue-torso version as a separate provenance record.

## Validation

Run the same frozen configuration at zero ticks with presentation disabled and
enabled, then compare decoded `ego_640x480.png` RGB bytes and the camera/body/joint
fields in `ego_stamp.json`, excluding acquisition/readback wall timestamps.
Both receipts must report zero integrations and zero model inferences.
Inspect the actual main-window image separately for style and retained geometry.
This check validates visual isolation at that boundary, not task completion.

The focused rendering test checks same-frame global transforms, source material
preservation, disjoint sensor/spectator layers, shadow exclusion, hidden-source
behavior and removal. Existing camera contract tests continue to cover sensor
mounts, source frame stamps and reset handling.

On 2026-10-04, the final zero-Tick pair had identical decoded 640 × 480
head-camera RGB, camera pose, body/joints, sequence and simulation stamps.
Both captures had zero integrations and model calls. Decoded RGB SHA256:
`f340fba1c9436c53374f09e4a3c1316ab0fb3f68eae98ae047c700f48a383f5c`.
The fixed chest ROI's near-black coverage fell from 67.35% to 0%; the
regression test demonstrates Bevy's zero normals on a real 1 mm triangle,
then verifies finite presentation normals without changing source geometry.

Fresh recording episodes 20719/20720 used the same frozen task/body models,
camera calibration, preprocessing and controllers as episodes 20619/20620.
All 1,046/2,134 physical steps, model observation arrays and action arrays
matched. The six saved station sensor frames were byte-identical. All four
fresh mobile VLA frames were also byte-identical; some later mobile diagnostic
PNGs differed at 1–3 of 307,200 pixels by one 8-bit channel level. The strict
all-PNG byte-equality assertion therefore failed and is preserved alongside
the difference metrics. These measurements do not establish universal
bitwise GPU equality. Independent placement checks still passed at 6.02/2.52 s.

The recording process used display-only `diagnostic_render_hz=60` and CPU
affinity `5-9,15-19` to avoid concurrent host load. Physics remained 50 Hz,
one integration per Tick, with the original timeout and controller settings.
The final CPU workspace suite, structure checker and actual GPU shader
startup were exercised; receipts and full raw recordings are archived at
`/home/ethan/ProjectBackups/2026-10-04/Sai_Lab/g1_uri_presentation_delivery_001/`.
See [recording identities and limitations](g1_milestone.md).

The station box-grasp replay (episode 20724) also matched all 200 physical
steps, five saved sensor PNGs and four fresh VLA observation/action pairs
from episode 20622. Its 7.04 s homepage GIF is a fresh continuous window
capture with the same medium-gray finish; it is not an extended carry result.
