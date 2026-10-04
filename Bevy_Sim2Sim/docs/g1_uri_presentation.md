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
and materials stay untouched. Robot ink shells are omitted. The existing station gets clearer
blue/white/orange color groups and a light flush floor grid. Geometry, body
proportions, task placements and the thin original support slab are retained.

The spectator camera is explicitly tagged `G1PresentationCamera`. Its layer 1
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
