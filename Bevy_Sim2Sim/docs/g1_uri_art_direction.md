# G1 URI art direction

This image is an AI style-transfer concept based on a real main-camera screenshot. It is art guidance only, not a simulation screenshot, physics result, evaluation receipt, manufacturing reference, or VLA camera input. It is not loaded by the renderer or control pipeline. No hardware, physics, object placement, robot control, observation/action contract, or VLA code is changed by this asset.

The generated asset is intended at `assets/game/arts/environment/textures/g1_uri_art_direction.png`. Generation uses the built-in image generator with `transparent_background=false`.

## Current runtime finish

The user clarified on 2026-10-04 that the poster preserves the factory G1
colors and gets its blue from environment reflection. The earlier illustration's
blue torso/head and yellow joints are superseded for runtime use. Runtime uses
neutral silver-gray shells and charcoal mechanics, checked against the
[official G1 product images](https://www.unitree.com/g1/). The original concept
and prompt remain here only as an accurate generation record; see
[G1 URI presentation](g1_uri_presentation.md) for the implemented finish.

## Reference roles

All five fixed URI v5 originals were viewed at original resolution in the specified order. The actual five-image tool input reserves one slot for the scene screenshot and uses fixed style anchors 1, 3, 5, then interface reference 4. Reference 2 (optical pod) is omitted to respect the five-input limit and avoid importing optical-pod geometry.

| Input | Absolute path | Role |
|---|---|---|
| 1 | `/home/ethan/Projects/Sai_Rotbots/.scratch/g1_uri_before.png` | EDIT_TARGET, SUBJECT, PROPORTION, GEOMETRY, COMPOSITION |
| 2 | `/home/ethan/Projects/uri-style-skill/uri-style/assets/style_01_race_car_hangar.png` | STYLE |
| 3 | `/home/ethan/Projects/uri-style-skill/uri-style/assets/style_03_tracked_mobile_station.png` | STYLE |
| 4 | `/home/ethan/Projects/uri-style-skill/uri-style/assets/style_05_white_blue_racer.png` | STYLE |
| 5 | `/home/ethan/Projects/uri-style-skill/uri-style/assets/style_04_service_entry.png` | STYLE, SURFACE_CLEANLINESS |

The exact source scene constrains all geometry. The style references control ink, materials, color and graphical order only. New objects, structure changes and table supports are excluded. The source table is a plain unsupported slab, which remains a plain slab in the art direction.

## Complete actual prompt

```text
URI / v5 — USER-SELECTED FIVE REFERENCES, VIVID BLUE / YELLOW-ORANGE / WARM WHITE

The only fixed STYLE reference pool is the five images explicitly supplied by the user on 2026-10-03: (1) blue racing car in a hangar, (2) blue optical pod, (3) tracked mobile station, (4) blue service entry with warm-white interior, (5) white-and-blue racer. Use their actual visual relationships, not previous headphone, recorder, waterfront, person, control-room or architectural references. Treat each selected image as STYLE unless another role is explicitly stated.

PALETTE — Anchor the painted surfaces in the clearly colored medium/sky blues visible in the supplied images. Pair them with saturated golden yellow to yellow-orange blocks and accents, with warm off-white and neutral gray as supporting surfaces. Keep blue recognizable in both lit and shaded planes. Image 5 also supports a warm-white main body with substantial clear-blue areas and yellow-orange accents. Preserve distinct local color blocks and their relative strength; do not wash everything with a warm beige filter. Red or pink details in image 2 are minor incidental accents, not a replacement for the blue/yellow-orange relationship. Avoid turning the default palette into powder/pastel blue, mint/teal, lavender, desaturated cream or brown sepia. Do not sample color from the orange uri title, watermark or screenshot controls. Natural skin, plants and other organic subjects retain appropriate local colors; apply the palette to relevant clothes, equipment and surroundings. Explicit user color requirements take priority.

STYLE controls rendering language only: confident dark ink contours with readable line-weight variation, richly assembled technical details, broad distinct color planes, compatible rounded industrial shells and coherent metal/glass/material layers. The design brief controls geometry, topology, component layout, function and framing. Do not transplant car silhouettes, wheel arrangements, the optical pod's lens layout, tracked chassis, service doors, hangar composition, branding, numbers, captions, watermarks, borders or screenshot UI into a new subject unless requested. No collage/page-layout imitation.

A SURFACE_CLEANLINESS reference controls surface integrity, coherent seams and graphical-noise control only. A PROPORTION reference controls only explicitly approved body regions and coarse ratios. Keep these roles separate, even when supplied by previous generated images. Preserve the user's required machine type and task action when developing functional structure.

Create a refined science-fiction illustration that is clean, elegant and richly constructed. Richness comes from complete, precisely fitted assemblies at meaningful scales: connected load-bearing parts, organized joints, layered covers, seals, hinges, tool interfaces, optical modules and service components appropriate to the subject. A detail belongs because it performs a function or completes a readable assembly. Use orderly groups and continuous contours so many parts remain individually intelligible.

For the default clean finish, maintain continuous painted surfaces and graceful large shapes around the assemblies. Preserve intentional technical ink lines and complete material boundaries; do not erase the drawing language in an attempt to clean the image. The supplied originals show some wear, but dirt, chipped paint and random distress are not required STYLE attributes. Fasteners appear at real removable cover edges or structural connections. Functional openings are bounded and correctly located. Rich detail does not require arbitrary circles, holes, black dots or scratches over every plate. Both physical dirt and nonfunctional graphical noise count as a dirty result.

Use clear illustrated volume and controlled reflections in enamel, glass, machined metal, rubber and cloth. Let near-black or deep neutral contours separate assemblies; dark recesses remain bounded rather than swallowing all the detail. Readable broad shapes and blue/yellow-orange/white color planes must survive at thumbnail size while closer viewing reveals the finer construction. Lighting follows the scene brief and retains the anchored local colors. Match the supplied technical illustration language rather than a plain icon, blank toy or photographic product render.

For objects/buildings, derive functional structure from the task, then express it with rounded shells, filleted boxes, cylinders and coherent hardware. For people retain natural anatomy, identity and skin, with fine hair, folds, seams and relevant accessories. For animals/plants keep organic structure; machinery appears only when requested. For scenes organize perspective and broad environmental areas around richly described local structures. Explicit subject constraints determine exact forms, weather, lighting and any palette override.

Inspect separately for palette fidelity, functional richness, surface integrity, visual order, source-object leakage and subject requirements. A clean design retains rich purposeful construction; a simplified blank toy also misses the target.

SUBJECT / DESIGN BRIEF
[Define purpose, component geometry, actual reference roles, framing and the specific features to retain. For engineering concepts distinguish design targets from validated performance.]
ACTUAL SUBJECT / DESIGN BRIEF — STYLE-TRANSFER EDIT
Use case: style-transfer.
Asset type: one URI art-direction illustration only; NOT a simulation screenshot, execution trace, physics result, VLA input, or hardware engineering validation.
Primary request: edit the attached real G1 scene screenshot into a refined URI technical illustration. Improve material/color hierarchy and line clarity while strictly retaining the original scene geometry and framing.

INPUT ROLES IN EXACT INPUT ORDER:
Image 1 (/home/ethan/Projects/Sai_Rotbots/.scratch/g1_uri_before.png): EDIT_TARGET + SUBJECT + PROPORTION + GEOMETRY + COMPOSITION. This is the absolute source of the subject and layout. Retain the exact G1 skeleton proportions, head/torso/body silhouette, limbs, hand shape, joint positions and axes, existing leaning stance, feet and floor contact positions, table plane and silhouette, the two small task object positions and identities, the science-station silhouettes, open floor, perspective and crop. The apparent table is a very simple unsupported rectangular slab: keep that original simple slab exactly; do not invent table legs, supports, pedestal, machine base or more complex geometry to explain it. Change only illustrated surface appearance, purposeful material boundaries already implicit in existing assemblies, lighting and ink treatment.
Image 2 (/home/ethan/Projects/uri-style-skill/uri-style/assets/style_01_race_car_hangar.png): STYLE only. Use its clear sky-blue / yellow-orange / warm-white relationship, confident precise dark ink contours and broad colored planes. Do not inherit vehicle shapes, hangar arches, composition, branding or wear.
Image 3 (/home/ethan/Projects/uri-style-skill/uri-style/assets/style_03_tracked_mobile_station.png): STYLE only. Use its organized complete mechanical layers, restrained dark recesses and vivid blue with white/gray layers. Do not add a tracked base, antennas, mobile station, machine, vehicle or any new scene object.
Image 4 (/home/ethan/Projects/uri-style-skill/uri-style/assets/style_05_white_blue_racer.png): STYLE only. Use the readable warm-white limb-like planes, substantial recognizable vivid blue color blocks, small saturated yellow-orange accents, clear continuous outlines and clean enamel finish. Do not transplant a cockpit, wheels, racer proportions or hangar.
Image 5 (/home/ethan/Projects/uri-style-skill/uri-style/assets/style_04_service_entry.png): STYLE + SURFACE_CLEANLINESS only. Use the orderly interface/cover seams and complete blue painted surfaces with warm-white and yellow-orange grouping. Its cleanliness role controls integrity and graphical-noise control only. Do not transplant doors, components, a new entry, decals, text or room layout.

RETAIN / REDESIGN:
Retain every physical shape, position and coarse proportion from Image 1. The robot remains exactly the G1 visible there, including its slim head above the existing narrow torso, exposed mechanics, slim limbs and large table occlusion of pelvis/thighs. Do not create a different humanoid, bulky mecha, enlarged head, shoulder armor, torso proportions or a new gesture.
Redesign surface color and illustrated finish only: vivid medium/sky-blue torso and existing head panels; warm-white limb shell surfaces; neutral charcoal/dark-gray organized joint mechanics; small saturated golden yellow-orange joint/service accents at actual boundaries. Preserve the existing hand and finger articulation silhouettes, without adding fingers or reshaping them. Increase clarity of EXISTING mechanical separations and layered assemblies with precise dark ink edges, continuous refined seams and material contrast. Do not achieve detail through random holes, screws, spots, printed graphics or mechanical decoration on flat plates.
Retain the two tiny task objects and table arrangement. Keep their identities and original local colors readable; do not transform them into tools, robot parts or larger props. The table top remains a thin plain rectangular slab in exactly its current perspective; render it with a refined neutral dark charcoal-gray surface and clean edge definition, without adding a giant orange stripe, surface graphics, bevel geometry, new fixtures or extra objects.
Retain the original science-station walls, windows, doorway, lower equipment/couch-like shapes and small existing cabinet silhouette. Express only those existing architectural panels in blue, warm-white, gray and selective yellow-orange zones. Refine their existing seam and panel hierarchy without building a new station or filling the open floor.

SCENE / LIGHTING:
Keep the source screenshot's oblique main camera angle, frame coverage and approximate 880:720 landscape aspect. Preserve all overlaps and occlusions, especially the robot behind the table slab. Keep the open tiled floor light neutral warm gray with subtle consistent grid lines. Clean soft directional illustrated volume, bounded charcoal recesses and modest controlled enamel/metal highlights. Legible blue remains blue in shadow. Use only subtle physically coherent floor contact shading at the original feet; do not add shadows that imply new table legs, hidden pedestals or supports. Do not give an unsupported slab a misleading invented floor support shadow. Keep background less contrast-heavy so existing G1 and task area stay readable.

CONSTRAINTS / AVOID:
One polished URI technical illustration. No text, title, watermark, logos, letters, numerals, borders, UI or captions. No new machines, buildings, obstacles, furniture, personnel, plants, lighting fixtures or mechanical appendages. No silhouette, topology, joint-layout, object-placement, camera or terrain change. No photographic product render, flat icon, blank toy, pastel blue, mint, purple, overall beige filter, dirt, chipped paint, rust, scratches, random black dots, scattered decorative fasteners, broken contour noise or black masses swallowing the mechanics. No apparent simulation/evaluation evidence. This is visual art guidance only; all physics, hardware contracts, robot controls and VLA remain unchanged.
```

## Generation and inspection

The built-in generator completed one image on 2026-10-04. Its original output remains at
`/home/ethan/.codex/generated_images/01a104f3-b07f-7270-9e9c-b3cac87686a0/exec-a26b9911-1663-4655-b139-8304193e9db3.png`.
The selected copy is [g1_uri_art_direction.png](../assets/game/arts/environment/textures/g1_uri_art_direction.png), 1387 × 1134, RGB PNG, 1,627,346 bytes.
SHA256: `ac4dddfc897f3cc63b6b0dffb4ea13bf9b26ddee0d49749b9dc13c0e3a626fa3`.

The saved copy was inspected at original resolution. Visual checks are qualitative:

- Palette: the blue head/torso, warm-white limbs, yellow-orange joint/service accents and neutral charcoal mechanics are distinct. Blue stays recognizable in shadow; no overall beige, mint or purple cast is present.
- Detail: the existing robot silhouette gains readable joint rings, layered shells, wrist/hand articulation and continuous material boundaries. The station uses grouped panel/interface detail. Broad forms still read clearly.
- Cleanliness: continuous painted surfaces, complete edges and orderly seams; no dirt, distress, scratches or scattered decorative black-hole noise. Most small dark marks follow mechanical boundaries.
- Visual order: the illustrated volume is clear, foreground mechanics remain readable and the background uses larger quieter color planes. The table stays a plain slab without invented legs or a support shadow. No new obstacle or machine appears on the open floor.
- Subject and composition: the robot's broad stance, head/limb proportions, feet, table occlusion, two task props and science-station layout visually follow the source. There is no visible source-car, tracked-base, cockpit or service-door transplantation.

Remaining limitations: image generation interprets fine robot shells, fingers, seam geometry and marker graphics; they are not metrically identical to the real meshes or task textures. The warm-white/blue background and orange wall/floor zones are art-direction suggestions. The source slab remains visually unsupported because adding supports would change geometry. This illustration must not replace a task texture, a calibrated marker, an ego-camera frame, a VLA observation, an engineering mesh, or actual simulation evidence. Runtime visual changes must be implemented and checked separately against the frozen geometry and camera/control contracts.

No code files were edited for this asset. The result is a non-runtime concept image plus this prompt/provenance document.
