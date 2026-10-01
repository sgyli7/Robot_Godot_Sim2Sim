# Native G1 task visuals and real camera registration

The CPU exporter `unitree_g1_task_visual_export.py` reads the four frozen
original task USDs and texture dependencies. Task scale overrides match the
original collision/mass conversion. Each visible mesh preserves source
triangulated faces, authored normals and indexed/face-varying UVs. The analytic
box has explicit per-face UVs. glTF and OmniPBR albedo, normal, ORM, UV scale,
roughness and metallic inputs are mapped to Bevy PBR; this is not MDL execution
or proof of source renderer parity.

The frozen plate has 36 zero/denormal authored vertex normals. The initial
native visual preflight rejected these before loading a model or GPU world.
Visual v2 substitutes area-weighted adjacent original-face normals, using the
largest original supporting face where opposite faces cancel. All triangle
corner positions, UVs and material inputs were compared against v1 and remained
unchanged. All valid authored normals remain in use. This changes no collision,
mass, contact material or physics parameter.

Original apple albedo is JPEG. The existing Bevy version's JPEG feature is now
enabled; `Cargo.lock` adds only its cached `zune-core 0.5.3` and `zune-jpeg 0.5.15`
decoder dependencies. No existing dependency version was upgraded.

`G1TaskVisualPlugin` binds a visual file SHA and its matching physics definition
SHA at startup. It verifies every original asset and texture hash, finite
geometry, normal lengths and attribute coverage. Textures are decoded from
those verified bytes. Runtime input contains only immutable completed object
poses for rendering. Whole-frame validation rejects foreign definitions,
duplicate objects, invalid poses and stale episode/Tick identity before changing
any pose. It supplies no task truth to Qwen or VLA.

The existing `g1_camera_diagnostic` accepts distinct complete AGILE or Homie
runner configurations and creates the corresponding typed background worker.
It never converts one body's policy actions into the other contract. When task
objects are configured, matching task visual path/hash are required together.
Robot, measured self-state and object frames must match at the completed boundary
before an image can be captured. The ego-camera stamp format is unchanged;
object truth is recorded separately in the diagnostic receipt.

At source `66bb7e0366e3a8e97cd9e1c1c97ec178df4a19b4`, actual T1 AGILE rendering
and physics completed 150 Ticks:

- 150 real model attempts/accepted outputs and 150 native integrations at 50 Hz.
- 3.0 simulated seconds / 3.00493233 active wall seconds, ratio about 0.9983586,
  zero control deadlines missed and zero pending Ticks.
- Ego RGB 640×480 and main window 1920×1080, MSAA 8 retained. Original textured
  apple and plate appear in the inspected ego image. The main camera's framing
  does not include the entire diagnostic object area.
- Robot, objects and image match episode 22 / Tick 150. Four original object
  meshes and eight original textures were loaded; only the two T1 objects were
  visible. The native world contained 56 bodies/55 colliders/52 multibody handles.
- The owner was paused at its explicit 150-Tick command endpoint. No VLA task
  action or Qwen decision was executed during this capture.

Evidence with preflight failure, both derived visual files, normal-repair audit,
exact executable/config/source hashes, receipts and original-resolution images:
`/home/ethan/ProjectBackups/2026-10-02/Sai_Lab/g1_task_visual_native_001/manifest.json`.

This capture uses the documented flat diagnostic floor and separated object
placements. It does not qualify source task layout, grasping, T1/T2 success,
science-station terrain or full source appearance. Source shelf/table placement,
matched source task rollout, the interactive local Qwen/VLA executor and final
acceptance remain required.

## Reproduction

Visual v2 cache:
`/home/ethan/models/unitree_g1/task_assets/20261001_frozen/native_task_visual_v2.json`,
SHA `b961c2633e5cd2063ef73f5705430230dd2881c466983c0100faa1df11f412c0`.
The accompanying original USDs and eight textures stay outside ordinary Git.

```bash
flock /tmp/sai-g1-cargo.lock env \
  CARGO_TARGET_DIR=/home/ethan/Projects/Sai_Lab/Bevy_Sim2Sim/target \
  cargo build --locked --offline --features dev_tools -j 2
```

Use the frozen `task_visual_native_capture_v2_20261001_2150/config.json`, SHA
`e2e1e2de998d8668e40c280564abfad8ff3abfcba822305f4fc7c05c7b3e28ca`,
with a fresh output directory:

```bash
/home/ethan/Projects/Sai_Lab/Bevy_Sim2Sim/target/debug/bevy_sim2sim \
  --scene g1_camera_diagnostic --robot g1 \
  --g1-config /absolute/path/to/frozen/config.json \
  --g1-ticks 150 --output /absolute/path/to/new/output
```

The receipt records profile-specific model identities, actual physics/model
counts, task visual readiness, object state and synchronized ego stamp. It
always keeps `task_qualified=false` for this diagnostic.
