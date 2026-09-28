//! Engine-neutral, immutable station geometry shared by display and collision.
//!
//! Coordinates are right-handed Y-up metres, inherited from the station source.
//! This is already the Bevy world frame: never apply the robot MJCF conversion here.

use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

/// One indexed triangle surface. Positions and normals are in its owner's frame.
#[derive(Clone, Debug)]
pub struct StationSurface {
    pub role: String,
    pub owner: String,
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
}
/// Collision geometry retains the source shape, including open furniture and bins.
#[derive(Clone, Debug)]
pub enum StationCollisionShape {
    Convex {
        vertices: Vec<[f32; 3]>,
    },
    Triangles {
        vertices: Vec<[f32; 3]>,
        indices: Vec<[u32; 3]>,
    },
}
/// Static station owner or local geometry belonging to a named movable prop.
#[derive(Clone, Debug)]
pub struct StationCollider {
    pub owner: String,
    /// Authored fixture identity, or `source` for retained source structure.
    pub tag: String,
    pub shape: StationCollisionShape,
}
/// Initial prop pose and contact parameters in SI units; the backend owns later motion.
#[derive(Clone, Debug, Deserialize)]
pub struct StationProp {
    pub name: String,
    pub position: [f32; 3],
    /// Quaternion x, y, z, w.
    pub rotation: [f32; 4],
    pub mass: f32,
    pub friction: f32,
    pub restitution: f32,
}
/// Original signage retained for subsequent text rendering.
#[derive(Clone, Debug, Deserialize)]
pub struct StationLabel {
    pub text: String,
    pub position: [f32; 3],
    /// Euler degrees from the source station.
    pub rotation: [f32; 3],
    pub em: f32,
    pub role: String,
}
/// One authored review camera, in the same world frame as the terrain.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct StationReviewCamera {
    pub name: String,
    pub eye: [f32; 3],
    pub target: [f32; 3],
    pub vertical_fov: f32,
}
/// Source-to-world placement provenance for a complete station facility.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct StationZone {
    pub name: String,
    pub source_anchor: [f32; 3],
    pub target_anchor: [f32; 3],
    pub yaw_degrees: f32,
}
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct StationFixture {
    pub name: String,
    pub zone: String,
    pub position: [f32; 3],
    pub collision_tag: String,
}
/// One approved code-native graphic lying on an existing carrier face.
///
/// Unknown fields are rejected. v4 and v5 omit the list; v6 requires the
/// five approved records exactly.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct StationSurfaceGraphic {
    pub id: String,
    pub kind: String,
    pub carrier: String,
    pub center: [f32; 3],
    pub size: [f32; 2],
    pub normal: [f32; 3],
    pub depth: f32,
    pub role: String,
}
/// Immutable circulation plan used both to regrade terrain and to shade its markings.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct StationLayout {
    pub identity: String,
    pub revision: u32,
    /// X/Z centre followed by half extents.
    pub plaza: [f32; 4],
    /// X/Z centre and the two ellipse radii.
    pub loop_center_radii: [f32; 4],
    pub loop_width: f32,
    /// X/Z centre followed by half extents.
    pub berth: [f32; 4],
    pub skills: [f32; 4],
    pub safe_points: Vec<[f32; 3]>,
    /// X/Z start followed by X/Z end. Each branch is 2.6 metres wide.
    pub paths: Vec<[f32; 4]>,
    pub cameras: Vec<StationReviewCamera>,
    pub zones: Vec<StationZone>,
    pub ridge_waypoints: Vec<[f32; 3]>,
    pub ridge_width: f32,
    /// Continuous approach plane; clamp height between zero and the last waypoint Y.
    pub ridge_grade_plane: [f32; 3],
    pub fixtures: Vec<StationFixture>,
    /// Empty unless this is the v6 functional-graphic revision.
    #[serde(default)]
    pub graphics: Vec<StationSurfaceGraphic>,
}
/// Complete immutable station layout. No engine handles or physics world are stored.
#[derive(Clone, Debug)]
pub struct StationGeometry {
    pub model_sha256: String,
    /// Includes prop initial state, labels and palette alongside geometry references.
    pub manifest_sha256: String,
    pub source: String,
    pub surfaces: Vec<StationSurface>,
    pub colliders: Vec<StationCollider>,
    pub props: Vec<StationProp>,
    pub labels: Vec<StationLabel>,
    pub palette: BTreeMap<String, [f32; 4]>,
    pub safe_points: Vec<[f32; 3]>,
    pub layout: StationLayout,
    pub layout_sha256: String,
}
#[derive(Deserialize)]
struct SurfaceRecord {
    mesh: usize,
    role: String,
    owner: String,
}
#[derive(Deserialize)]
struct ColliderRecord {
    mesh: usize,
    kind: String,
    owner: String,
    #[serde(default)]
    tag: String,
}
#[derive(Deserialize)]
struct PaletteRecord {
    role: String,
    srgba: [f32; 4],
}
#[derive(Deserialize)]
struct StationManifest {
    schema_version: u32,
    coordinate_frame: String,
    source: String,
    model_sha256: String,
    surfaces: Vec<SurfaceRecord>,
    colliders: Vec<ColliderRecord>,
    props: Vec<StationProp>,
    labels: Vec<StationLabel>,
    palette: Vec<PaletteRecord>,
    safe_points: Vec<[f32; 3]>,
    layout: StationLayout,
    layout_sha256: String,
}

struct ApprovedSurfaceGraphic {
    id: &'static str,
    kind: &'static str,
    carrier: &'static str,
    center: [f32; 3],
    size: [f32; 2],
    normal: [f32; 3],
    depth: f32,
    role: &'static str,
    carrier_role: &'static str,
    carrier_center: [f64; 3],
    carrier_size: [f64; 3],
    fixture_anchor: [f32; 3],
    signage: Option<&'static str>,
}

const APPROVED_SURFACE_GRAPHICS: [ApprovedSurfaceGraphic; 5] = [
    ApprovedSurfaceGraphic {
        id: "sample_vial_glyph",
        kind: "surface_vector",
        carrier: "sample_record_board",
        center: [-14.95, 0.94, 5.024],
        size: [0.13, 0.14],
        normal: [0.0, 0.0, 1.0],
        depth: 0.0015,
        role: "purple",
        carrier_role: "paper",
        carrier_center: [-14.6, 0.93, 5.0],
        carrier_size: [1.0, 0.47, 0.045],
        fixture_anchor: [-14.6, 0.0, 5.0],
        signage: Some("SAMPLES\n04"),
    },
    ApprovedSurfaceGraphic {
        id: "service_power_glyph",
        kind: "surface_vector",
        carrier: "service_connector",
        center: [20.2, 0.67, 4.825],
        size: [0.14, 0.18],
        normal: [0.0, 0.0, 1.0],
        depth: 0.0015,
        role: "yellow",
        carrier_role: "graphite",
        carrier_center: [20.2, 0.67, 4.811],
        carrier_size: [0.29, 0.25, 0.025],
        fixture_anchor: [20.2, 0.0, 4.6],
        signage: None,
    },
    ApprovedSurfaceGraphic {
        id: "service_tool_glyph",
        kind: "surface_vector",
        carrier: "service_parts_rack",
        center: [19.5, 0.49, 2.2165],
        size: [0.20, 0.16],
        normal: [0.0, 0.0, 1.0],
        depth: 0.0015,
        role: "blue",
        carrier_role: "paper",
        carrier_center: [19.5, 0.49, 2.0],
        carrier_size: [0.37, 0.28, 0.43],
        fixture_anchor: [19.8, 0.0, 2.0],
        signage: None,
    },
    ApprovedSurfaceGraphic {
        id: "survey_wave_glyph",
        kind: "surface_vector",
        carrier: "observation_survey",
        center: [7.6, 0.78, -15.491],
        size: [0.32, 0.12],
        normal: [0.0, 0.0, 1.0],
        depth: 0.0015,
        role: "paper",
        carrier_role: "glass",
        carrier_center: [7.6, 0.78, -15.52],
        carrier_size: [0.42, 0.24, 0.055],
        fixture_anchor: [7.6, 0.0, -15.8],
        signage: None,
    },
    ApprovedSurfaceGraphic {
        id: "field_route_glyph",
        kind: "surface_vector",
        carrier: "field_route_board",
        center: [-9.9, 0.94, 9.424],
        size: [0.12, 0.18],
        normal: [0.0, 0.0, 1.0],
        depth: 0.0015,
        role: "blue",
        carrier_role: "paper",
        carrier_center: [-9.5, 0.93, 9.4],
        carrier_size: [1.0, 0.47, 0.045],
        fixture_anchor: [-9.5, 0.0, 9.4],
        signage: Some("RIDGE \u{2190}\n09"),
    },
];

fn close_enough(left: f32, right: f32, tolerance: f32) -> bool {
    (left - right).abs() <= tolerance
}

fn vector_close(left: &[f32], right: &[f32], tolerance: f32) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(a, b)| close_enough(*a, *b, tolerance))
}

/// Front corners of an authored box, matching the f32 values stored in the station GLB.
fn carrier_front_corners(center: [f64; 3], size: [f64; 3]) -> [[f32; 3]; 4] {
    let half_x = size[0] * 0.5;
    let half_y = size[1] * 0.5;
    let front_z = center[2] + size[2] * 0.5;
    [
        [center[0] - half_x, center[1] - half_y, front_z],
        [center[0] + half_x, center[1] - half_y, front_z],
        [center[0] + half_x, center[1] + half_y, front_z],
        [center[0] - half_x, center[1] + half_y, front_z],
    ]
    .map(|point| [point[0] as f32, point[1] as f32, point[2] as f32])
}

fn signage_quad_metres(label: &StationLabel) -> [f32; 2] {
    let width = label
        .text
        .lines()
        .map(|line| {
            line.chars()
                .map(|c| if c.is_ascii() { 0.65 } else { 1.05 })
                .sum::<f32>()
        })
        .fold(1.0, f32::max);
    let pixels_w = (width * 128.0 + 32.0).ceil().clamp(128.0, 2048.0) as u32;
    let pixels_h = (label.text.lines().count().max(1) as f32 * 160.0 + 16.0).ceil() as u32;
    [
        pixels_w as f32 * label.em / 128.0,
        pixels_h as f32 * label.em / 128.0,
    ]
}

fn ranges_overlap(min_a: f32, max_a: f32, min_b: f32, max_b: f32) -> bool {
    min_a < max_b && min_b < max_a
}

fn validate_carrier_rectangle(
    surfaces: &[StationSurface],
    role: &str,
    corners: [[f32; 3]; 4],
) -> Result<(), String> {
    let mut matched = Vec::new();
    for surface in surfaces
        .iter()
        .filter(|surface| surface.owner == "station" && surface.role == role)
    {
        if surface.positions.len() != surface.normals.len() {
            return Err("station graphic carrier face is absent or changed".into());
        }
        for triangle in surface.indices.chunks_exact(3) {
            let mut corner_ids = [0_usize; 3];
            let mut on_face = true;
            for (slot, index) in triangle.iter().enumerate() {
                let index = *index as usize;
                if index >= surface.positions.len() {
                    on_face = false;
                    break;
                }
                let position = surface.positions[index];
                let normal = surface.normals[index];
                if normal[2] < 0.999 || normal[0].abs() > 1.0e-4 || normal[1].abs() > 1.0e-4 {
                    on_face = false;
                    break;
                }
                let Some(corner) = corners.iter().position(|corner| {
                    corner
                        .iter()
                        .zip(position)
                        .all(|(expected, actual)| (expected - actual).abs() <= 5.0e-5)
                }) else {
                    on_face = false;
                    break;
                };
                corner_ids[slot] = corner;
            }
            if on_face {
                matched.push(corner_ids);
            }
        }
    }
    if matched.len() != 2 {
        return Err("station graphic carrier face is absent or changed".into());
    }
    // Four corners alone do not prove a complete rectangle: two triangles
    // sharing an outer edge can overlap and leave an uncovered region.
    let shared: BTreeSet<_> = matched[0]
        .iter()
        .copied()
        .filter(|corner| matched[1].contains(corner))
        .collect();
    if shared != BTreeSet::from([0, 2]) && shared != BTreeSet::from([1, 3]) {
        return Err("station graphic carrier face is absent or changed".into());
    }
    let mut covered = [false; 4];
    for triangle in matched {
        if triangle[0] == triangle[1] || triangle[1] == triangle[2] || triangle[0] == triangle[2] {
            return Err("station graphic carrier face is absent or changed".into());
        }
        let [a, b, c] = [
            corners[triangle[0]],
            corners[triangle[1]],
            corners[triangle[2]],
        ];
        let area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
        if area <= 1.0e-6 {
            return Err("station graphic carrier face is absent or changed".into());
        }
        for corner in triangle {
            covered[corner] = true;
        }
    }
    if covered.iter().any(|present| !present) {
        return Err("station graphic carrier face is absent or changed".into());
    }
    Ok(())
}

fn validate_one_surface_graphic(
    graphic: &StationSurfaceGraphic,
    layout: &StationLayout,
    surfaces: &[StationSurface],
    palette: &BTreeMap<String, [f32; 4]>,
    labels: &[StationLabel],
    seen: &mut BTreeSet<String>,
) -> Result<(), String> {
    if !graphic
        .center
        .iter()
        .chain(&graphic.size)
        .chain(&graphic.normal)
        .all(|value| value.is_finite())
        || !graphic.depth.is_finite()
    {
        return Err(format!("station graphic is not finite: {}", graphic.id));
    }
    let Some(spec) = APPROVED_SURFACE_GRAPHICS
        .iter()
        .find(|spec| spec.id == graphic.id)
    else {
        return Err(format!("unknown station graphic: {}", graphic.id));
    };
    if !seen.insert(graphic.id.clone()) {
        return Err(format!("duplicate station graphic: {}", graphic.id));
    }
    if graphic.kind != spec.kind {
        return Err(format!("unknown station graphic kind: {}", graphic.kind));
    }
    if graphic.carrier.is_empty() || graphic.carrier != spec.carrier {
        return Err(format!(
            "station graphic carrier is missing or wrong: {}",
            graphic.id
        ));
    }
    if graphic.size[0] <= 0.0 || graphic.size[1] <= 0.0 {
        return Err(format!("station graphic size is invalid: {}", graphic.id));
    }
    if graphic.size[0] > spec.carrier_size[0] as f32 + 1.0e-4
        || graphic.size[1] > spec.carrier_size[1] as f32 + 1.0e-4
    {
        return Err(format!(
            "station graphic size is out of bounds: {}",
            graphic.id
        ));
    }
    if !vector_close(&graphic.size, &spec.size, 1.0e-4) {
        return Err(format!(
            "station graphic size does not match the approved graphic: {}",
            graphic.id
        ));
    }
    if graphic.role != spec.role || !palette.contains_key(&graphic.role) {
        return Err(format!("station graphic role is wrong: {}", graphic.id));
    }
    if !vector_close(&graphic.normal, &spec.normal, 1.0e-4) {
        return Err(format!("station graphic normal is wrong: {}", graphic.id));
    }
    if !close_enough(graphic.depth, spec.depth, 5.0e-5) {
        return Err(format!("station graphic depth is wrong: {}", graphic.id));
    }
    if !vector_close(&graphic.center, &spec.center, 1.0e-4) {
        return Err(format!("station graphic center is wrong: {}", graphic.id));
    }
    let expected_z =
        (spec.carrier_center[2] + spec.carrier_size[2] * 0.5 + f64::from(spec.depth)) as f32;
    if !close_enough(graphic.center[2], expected_z, 5.0e-5) {
        return Err(format!(
            "station graphic depth does not meet the carrier face: {}",
            graphic.id
        ));
    }
    let fixtures: Vec<_> = layout
        .fixtures
        .iter()
        .filter(|fixture| fixture.name == graphic.carrier)
        .collect();
    if fixtures.len() != 1
        || fixtures[0].collision_tag != graphic.carrier
        || !vector_close(&fixtures[0].position, &spec.fixture_anchor, 1.0e-4)
    {
        return Err(format!(
            "station graphic carrier anchor does not match the fixture: {}",
            graphic.id
        ));
    }
    let face_center_x = spec.carrier_center[0] as f32;
    let face_center_y = spec.carrier_center[1] as f32;
    let face_half_x = (spec.carrier_size[0] * 0.5) as f32;
    let face_half_y = (spec.carrier_size[1] * 0.5) as f32;
    let min_x = graphic.center[0] - graphic.size[0] * 0.5;
    let max_x = graphic.center[0] + graphic.size[0] * 0.5;
    let min_y = graphic.center[1] - graphic.size[1] * 0.5;
    let max_y = graphic.center[1] + graphic.size[1] * 0.5;
    if min_x < face_center_x - face_half_x - 1.0e-4
        || max_x > face_center_x + face_half_x + 1.0e-4
        || min_y < face_center_y - face_half_y - 1.0e-4
        || max_y > face_center_y + face_half_y + 1.0e-4
    {
        return Err(format!(
            "station graphic is outside its carrier face: {}",
            graphic.id
        ));
    }
    validate_carrier_rectangle(
        surfaces,
        spec.carrier_role,
        carrier_front_corners(spec.carrier_center, spec.carrier_size),
    )
    .map_err(|error| format!("{error}: {}", graphic.id))?;
    if let Some(text) = spec.signage {
        let hits: Vec<_> = labels.iter().filter(|label| label.text == text).collect();
        if hits.len() != 1 {
            return Err(format!(
                "station graphic signage anchor is missing: {}",
                graphic.id
            ));
        }
        let quad = signage_quad_metres(hits[0]);
        let overlaps = ranges_overlap(
            min_x,
            max_x,
            hits[0].position[0] - quad[0] * 0.5,
            hits[0].position[0] + quad[0] * 0.5,
        ) && ranges_overlap(
            min_y,
            max_y,
            hits[0].position[1] - quad[1] * 0.5,
            hits[0].position[1] + quad[1] * 0.5,
        );
        if overlaps {
            return Err(format!("station graphic overlaps signage: {}", graphic.id));
        }
    }
    Ok(())
}

pub(crate) fn validate_surface_graphics(
    layout: &StationLayout,
    surfaces: &[StationSurface],
    palette: &BTreeMap<String, [f32; 4]>,
    labels: &[StationLabel],
) -> Result<(), String> {
    let historical = matches!(
        (layout.revision, layout.identity.as_str()),
        (4, "windpass_courtyard_v4") | (5, "windpass_courtyard_v5")
    );
    if historical {
        if layout.graphics.is_empty() {
            return Ok(());
        }
        return Err("Historical station layout cannot contain surface graphics".into());
    }
    if layout.revision != 6 || layout.identity != "windpass_courtyard_v6" {
        return Err("Unsupported station circulation plan".into());
    }
    if layout.graphics.len() != APPROVED_SURFACE_GRAPHICS.len() {
        return Err("Station v6 requires exactly five surface graphics".into());
    }
    let mut seen = BTreeSet::new();
    for graphic in &layout.graphics {
        validate_one_surface_graphic(graphic, layout, surfaces, palette, labels, &mut seen)?;
    }
    Ok(())
}

/// Read the shipped source-derived geometry. Missing, corrupt, or incompatible data fails.
pub fn load_station_geometry(asset_root: &Path) -> Result<StationGeometry, String> {
    let manifest_path = asset_root.join("game/dynamic_assets/game_data/science_station.ron");
    let text = fs::read_to_string(&manifest_path)
        .map_err(|e| format!("{}: {e}", manifest_path.display()))?;
    let manifest: StationManifest =
        ron::from_str(&text).map_err(|e| format!("Invalid station manifest: {e}"))?;
    if manifest.schema_version != 1 || manifest.coordinate_frame != "right_handed_y_up_meters" {
        return Err("Unsupported station schema or coordinate frame".into());
    }
    let layout_path = asset_root.join("game/dynamic_assets/game_data/science_station_layout.ron");
    let layout_bytes =
        fs::read(&layout_path).map_err(|e| format!("{}: {e}", layout_path.display()))?;
    if format!("{:x}", Sha256::digest(&layout_bytes)) != manifest.layout_sha256 {
        return Err("Station layout does not match its manifest SHA256".into());
    }
    let layout: StationLayout =
        ron::de::from_bytes(&layout_bytes).map_err(|e| format!("Invalid station layout: {e}"))?;
    if layout != manifest.layout || layout.safe_points != manifest.safe_points {
        return Err("Station manifest and circulation layout disagree".into());
    }
    for name in ["arrival", "overview", "towers", "samples", "berth", "hills"] {
        let matches: Vec<_> = layout.cameras.iter().filter(|c| c.name == name).collect();
        if matches.len() != 1
            || !matches[0]
                .eye
                .iter()
                .chain(&matches[0].target)
                .all(|v| v.is_finite())
            || !(15. ..=120.).contains(&matches[0].vertical_fov)
        {
            return Err(format!("Invalid authored station camera: {name}"));
        }
    }
    if !matches!(
        (layout.revision, layout.identity.as_str()),
        (4, "windpass_courtyard_v4") | (5, "windpass_courtyard_v5") | (6, "windpass_courtyard_v6")
    ) || layout.paths.is_empty()
        || layout.paths.len() > 8
        || layout.loop_width < 2.5
        || layout.ridge_waypoints.len() != 4
        || layout.ridge_width < 2.5
        || layout.ridge_grade_plane[0].hypot(layout.ridge_grade_plane[1]) >= 0.06
        || !layout.ridge_grade_plane.iter().all(|v| v.is_finite())
    {
        return Err("Unsupported station circulation plan".into());
    }
    let model_path = asset_root.join("game/arts/environment/models/science_station.glb");
    let bytes = fs::read(&model_path).map_err(|e| format!("{}: {e}", model_path.display()))?;
    let actual_sha256 = format!("{:x}", Sha256::digest(&bytes));
    if actual_sha256 != manifest.model_sha256 {
        return Err("Station GLB does not match its manifest SHA256".into());
    }
    let gltf = gltf::Gltf::from_slice(&bytes).map_err(|e| format!("Invalid station GLB: {e}"))?;
    let blob = gltf
        .blob
        .as_ref()
        .ok_or("Station GLB has no binary buffer")?;
    let source_meshes: Vec<_> = gltf.meshes().collect();
    let read = |index: usize| -> Result<(Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<u32>), String> {
        let mesh = source_meshes
            .get(index)
            .ok_or_else(|| format!("Station mesh {index} missing"))?;
        let primitive = mesh.primitives().next().ok_or("Empty station mesh")?;
        let reader = primitive.reader(|buffer| match buffer.source() {
            gltf::buffer::Source::Bin => Some(blob.as_slice()),
            gltf::buffer::Source::Uri(_) => None,
        });
        let positions: Vec<_> = reader
            .read_positions()
            .ok_or("Station positions missing")?
            .collect();
        let normals: Vec<_> = reader
            .read_normals()
            .map(|v| v.collect())
            .unwrap_or_default();
        let indices = reader
            .read_indices()
            .map(|v| v.into_u32().collect())
            .unwrap_or_default();
        if positions.iter().flatten().any(|v| !v.is_finite()) {
            return Err("Non-finite station positions".into());
        }
        Ok((positions, normals, indices))
    };
    let mut surfaces = Vec::new();
    for record in manifest.surfaces {
        let (positions, normals, indices) = read(record.mesh)?;
        if positions.len() != normals.len()
            || indices.len() % 3 != 0
            || indices.iter().any(|i| *i as usize >= positions.len())
        {
            return Err("Invalid station triangle surface".into());
        }
        surfaces.push(StationSurface {
            role: record.role,
            owner: record.owner,
            positions,
            normals,
            indices,
        });
    }
    let mut colliders = Vec::new();
    for record in manifest.colliders {
        let (vertices, _, indices) = read(record.mesh)?;
        let shape = match record.kind.as_str() {
            "convex" => StationCollisionShape::Convex { vertices },
            "triangles"
                if indices.len() % 3 == 0
                    && indices.iter().all(|i| (*i as usize) < vertices.len()) =>
            {
                StationCollisionShape::Triangles {
                    vertices,
                    indices: indices
                        .chunks_exact(3)
                        .map(|i| [i[0], i[1], i[2]])
                        .collect(),
                }
            }
            _ => return Err(format!("Invalid collider kind or indices: {}", record.kind)),
        };
        colliders.push(StationCollider {
            owner: record.owner,
            tag: if record.tag.is_empty() {
                "source".into()
            } else {
                record.tag
            },
            shape,
        });
    }
    if manifest.props.len() != 6 || manifest.safe_points.len() != 3 {
        return Err("Station props/safe points are incomplete".into());
    }
    let palette: BTreeMap<_, _> = manifest
        .palette
        .into_iter()
        .map(|p| (p.role, p.srgba))
        .collect();
    let valid_owner =
        |owner: &str| owner == "station" || manifest.props.iter().any(|p| p.name == owner);
    if surfaces
        .iter()
        .any(|s| !palette.contains_key(&s.role) || !valid_owner(&s.owner))
        || colliders.iter().any(|c| !valid_owner(&c.owner))
    {
        return Err("Station surface palette or owner mapping is invalid".into());
    }
    for fixture in &layout.fixtures {
        if !colliders.iter().any(|c| c.tag == fixture.collision_tag) {
            return Err(format!(
                "Authored fixture has no physical structure: {}",
                fixture.name
            ));
        }
    }
    for prop in &manifest.props {
        let norm: f32 = prop.rotation.iter().map(|v| v * v).sum();
        if prop
            .position
            .iter()
            .chain(&prop.rotation)
            .any(|v| !v.is_finite())
            || (norm - 1.).abs() > 1e-4
            || !prop.mass.is_finite()
            || prop.mass <= 0.
            || !prop.friction.is_finite()
            || prop.friction < 0.
            || !prop.restitution.is_finite()
            || !(0. ..=1.).contains(&prop.restitution)
        {
            return Err(format!(
                "Invalid station prop pose/contact parameters: {}",
                prop.name
            ));
        }
    }
    validate_surface_graphics(&layout, &surfaces, &palette, &manifest.labels)?;
    Ok(StationGeometry {
        model_sha256: actual_sha256,
        manifest_sha256: format!("{:x}", Sha256::digest(text.as_bytes())),
        source: manifest.source,
        surfaces,
        colliders,
        props: manifest.props,
        labels: manifest.labels,
        palette,
        safe_points: manifest.safe_points,
        layout,
        layout_sha256: manifest.layout_sha256,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    fn asset_root() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../assets")
    }
    fn key(p: [f32; 3]) -> [u32; 3] {
        p.map(|v| if v == 0. { 0 } else { v.to_bits() })
    }
    #[test]
    fn carrier_requires_two_triangles_on_a_diagonal() {
        let corners = carrier_front_corners([0.0, 0.0, 1.0], [1.0, 1.0, 0.1]);
        let mut face = StationSurface {
            role: "paper".into(),
            owner: "station".into(),
            positions: corners.to_vec(),
            normals: vec![[0.0, 0.0, 1.0]; 4],
            indices: vec![0, 1, 2, 0, 2, 3],
        };
        assert!(validate_carrier_rectangle(&[face.clone()], "paper", corners).is_ok());
        face.indices = vec![0, 1, 2, 0, 1, 3];
        assert!(validate_carrier_rectangle(&[face], "paper", corners).is_err());
    }
    #[test]
    fn v6_graphics_fail_when_carrier_missing_or_metadata_unknown() {
        let scene = load_station_geometry(&asset_root()).expect("v6 station");
        let mut stripped = scene.surfaces.clone();
        stripped.retain(|surface| surface.role != "paper");
        assert!(
            validate_surface_graphics(&scene.layout, &stripped, &scene.palette, &scene.labels)
                .unwrap_err()
                .contains("carrier face")
        );
        let text = fs::read_to_string(
            asset_root().join("game/dynamic_assets/game_data/science_station_layout.ron"),
        )
        .unwrap();
        let changed = text.replace(
            "id:\"sample_vial_glyph\"",
            "id:\"sample_vial_glyph\",unexpected_field:1",
        );
        assert_ne!(changed, text);
        assert!(ron::from_str::<StationLayout>(&changed).is_err());
    }
    #[test]
    fn archived_v4_and_v5_layouts_still_parse_without_graphics() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
        let v4 = root.join(".scratch/visuals/v4_archive/assets/game/dynamic_assets/game_data/science_station_layout.ron");
        let v5 = root.join(".scratch/visuals/v5_signage/before/science_station_layout.ron");
        let scene = load_station_geometry(&asset_root()).expect("v6 station");
        for path in [v4, v5] {
            let layout: StationLayout = ron::from_str(&fs::read_to_string(path).unwrap()).unwrap();
            assert!(layout.graphics.is_empty());
            assert!(
                validate_surface_graphics(&layout, &scene.surfaces, &scene.palette, &scene.labels)
                    .is_ok()
            );
        }
    }
    #[test]
    fn source_terrain_triangles_are_identical_for_display_and_collision() {
        let scene = load_station_geometry(&asset_root()).unwrap();
        let displayed: HashSet<_> = scene
            .surfaces
            .iter()
            .filter(|s| s.owner == "station")
            .flat_map(|s| {
                s.indices.chunks_exact(3).map(|i| {
                    [
                        key(s.positions[i[0] as usize]),
                        key(s.positions[i[1] as usize]),
                        key(s.positions[i[2] as usize]),
                    ]
                })
            })
            .collect();
        let mut checked = 0;
        for collider in &scene.colliders {
            if collider.owner != "station" {
                continue;
            }
            if let StationCollisionShape::Triangles { vertices, indices } = &collider.shape {
                for i in indices {
                    let triangle = [
                        key(vertices[i[0] as usize]),
                        key(vertices[i[1] as usize]),
                        key(vertices[i[2] as usize]),
                    ];
                    assert!(
                        displayed.contains(&triangle),
                        "A static physics triangle is absent from the visible geometry"
                    );
                    checked += 1;
                }
            }
        }
        assert!(
            checked >= 61440,
            "Central 48x40m quarter-metre terrain is incomplete"
        );
        assert_eq!(scene.props.len(), 6);
        assert_eq!(
            scene.safe_points,
            [[0., 0., 6.8], [-12., 0., 1.5], [15., 0., 4.]]
        );
    }
    #[test]
    fn authored_ring_and_skill_area_have_flat_shared_terrain() {
        let scene = load_station_geometry(&asset_root()).unwrap();
        let terrain: Vec<_> = scene
            .surfaces
            .iter()
            .filter(|s| s.owner == "station" && s.role == "sand")
            .flat_map(|s| s.positions.iter())
            .collect();
        let mut checks = Vec::new();
        let c = scene.layout.loop_center_radii;
        for step in 0..72 {
            let angle = step as f32 * std::f32::consts::TAU / 72.;
            let normal = [c[3] * angle.cos(), c[2] * angle.sin()];
            let length = normal[0].hypot(normal[1]);
            for across in [-0.5, 0., 0.5] {
                let offset = across * scene.layout.loop_width;
                checks.push([
                    c[0] + angle.cos() * c[2] + normal[0] / length * offset,
                    c[1] + angle.sin() * c[3] + normal[1] / length * offset,
                ]);
            }
        }
        for x in -11..=11 {
            for z in -9..=9 {
                checks.push([x as f32 * 0.5, -2.5 + z as f32 * 0.5]);
            }
        }
        for point in checks {
            let nearby: Vec<_> = terrain
                .iter()
                .filter(|p| (p[0] - point[0]).abs() <= 0.26 && (p[2] - point[1]).abs() <= 0.26)
                .collect();
            assert!(
                !nearby.is_empty(),
                "Circulation plan has no actual terrain near {point:?}"
            );
            assert!(
                nearby.iter().all(|p| p[1].abs() < 1e-6),
                "Prepared circulation terrain is not flat near {point:?}"
            );
        }
        let skills = scene.layout.skills;
        for c in scene.colliders.iter().filter(|c| c.owner == "station") {
            if let StationCollisionShape::Convex { vertices } = &c.shape {
                let low: [f32; 3] = std::array::from_fn(|i| {
                    vertices.iter().map(|v| v[i]).fold(f32::INFINITY, f32::min)
                });
                let high: [f32; 3] = std::array::from_fn(|i| {
                    vertices
                        .iter()
                        .map(|v| v[i])
                        .fold(f32::NEG_INFINITY, f32::max)
                });
                let intrudes = high[1] > 0.03
                    && low[1] < 1.5
                    && high[0] >= skills[0] - skills[2]
                    && low[0] <= skills[0] + skills[2]
                    && high[2] >= skills[1] - skills[3]
                    && low[2] <= skills[1] + skills[3];
                assert!(!intrudes, "Structure {} enters the open skill area", c.tag);
            }
        }
    }
    #[test]
    fn ridge_route_has_real_shared_support_at_authored_grade() {
        let scene = load_station_geometry(&asset_root()).unwrap();
        let terrain: Vec<_> = scene
            .surfaces
            .iter()
            .filter(|s| s.owner == "station" && s.role == "sand")
            .flat_map(|s| s.positions.iter())
            .collect();
        for pair in scene.layout.ridge_waypoints.windows(2) {
            let a = pair[0];
            let b = pair[1];
            let length = ((b[0] - a[0]).powi(2) + (b[2] - a[2]).powi(2)).sqrt();
            assert!(
                (b[1] - a[1]).abs() / length < 0.06,
                "Ridge approach exceeds six percent grade"
            );
            for step in 0..=32 {
                let t = step as f32 / 32.;
                let p = std::array::from_fn::<_, 3, _>(|i| a[i] + (b[i] - a[i]) * t);
                for across in [-0.5, -0.25, 0., 0.25, 0.5] {
                    let offset = across * scene.layout.ridge_width;
                    let q = [
                        p[0] - (b[2] - a[2]) / length * offset,
                        p[1],
                        p[2] + (b[0] - a[0]) / length * offset,
                    ];
                    let grade = scene.layout.ridge_grade_plane;
                    let expected_y = (grade[0] * q[0] + grade[1] * q[2] + grade[2])
                        .clamp(0., scene.layout.ridge_waypoints.last().unwrap()[1]);
                    let near: Vec<_> = terrain
                        .iter()
                        .filter(|v| (v[0] - q[0]).abs() <= 0.26 && (v[2] - q[2]).abs() <= 0.26)
                        .collect();
                    assert!(!near.is_empty(), "Ridge route has no support near {q:?}");
                    assert!(
                        near.iter().all(|v| (v[1] - expected_y).abs() < 0.035),
                        "Ridge route differs from authored full-width grade near {q:?}"
                    );
                }
                for collider in scene.colliders.iter().filter(|c| {
                    scene
                        .layout
                        .fixtures
                        .iter()
                        .any(|f| f.collision_tag == c.tag)
                }) {
                    if let StationCollisionShape::Convex { vertices } = &collider.shape {
                        let low: [f32; 3] = std::array::from_fn(|i| {
                            vertices.iter().map(|v| v[i]).fold(f32::INFINITY, f32::min)
                        });
                        let high: [f32; 3] = std::array::from_fn(|i| {
                            vertices
                                .iter()
                                .map(|v| v[i])
                                .fold(f32::NEG_INFINITY, f32::max)
                        });
                        let intersects = low[1] < p[1] + 0.5
                            && high[1] > p[1] + 0.01
                            && p[0] >= low[0] - 0.9
                            && p[0] <= high[0] + 0.9
                            && p[2] >= low[2] - 0.9
                            && p[2] <= high[2] + 0.9;
                        assert!(
                            !intersects,
                            "New facility {} obstructs the ridge approach near {p:?}",
                            collider.tag
                        );
                    }
                }
            }
        }
        for fixture in &scene.layout.fixtures {
            assert!(
                scene
                    .colliders
                    .iter()
                    .any(|c| c.tag == fixture.collision_tag)
            );
        }
        for center in &scene.layout.ridge_waypoints {
            for step in 0..16 {
                let angle = step as f32 * std::f32::consts::TAU / 16.;
                let q = [
                    center[0] + angle.cos() * scene.layout.ridge_width * 0.5,
                    center[1],
                    center[2] + angle.sin() * scene.layout.ridge_width * 0.5,
                ];
                let grade = scene.layout.ridge_grade_plane;
                let expected_y = (grade[0] * q[0] + grade[1] * q[2] + grade[2])
                    .clamp(0., scene.layout.ridge_waypoints.last().unwrap()[1]);
                let near: Vec<_> = terrain
                    .iter()
                    .filter(|v| (v[0] - q[0]).abs() <= 0.26 && (v[2] - q[2]).abs() <= 0.26)
                    .collect();
                assert!(
                    !near.is_empty(),
                    "Rounded ridge bend/end has no support near {q:?}"
                );
                assert!(
                    near.iter().all(|v| (v[1] - expected_y).abs() < 0.035),
                    "Rounded ridge bend/end has inconsistent support near {q:?}"
                );
            }
        }
    }

    // Clip actual projected geometry against each full-width route strip,
    // rather than sampling only its centre line. Convex structure uses a
    // conservative bounding rectangle; geological triangles retain their Y.
    fn clipped_strip(mut polygon: Vec<[f32; 3]>, length: f32, half_width: f32) -> Vec<[f32; 3]> {
        for (axis, value, greater) in [
            (0, 0., true),
            (0, length, false),
            (1, -half_width, true),
            (1, half_width, false),
        ] {
            if polygon.is_empty() {
                return polygon;
            }
            let mut output = Vec::new();
            let mut a = *polygon.last().unwrap();
            for &b in &polygon {
                let inside = |p: [f32; 3]| {
                    if greater {
                        p[axis] >= value
                    } else {
                        p[axis] <= value
                    }
                };
                if inside(a) != inside(b) {
                    let t = (value - a[axis]) / (b[axis] - a[axis]);
                    let mut p = std::array::from_fn(|i| a[i] + (b[i] - a[i]) * t);
                    p[axis] = value;
                    output.push(p);
                }
                if inside(b) {
                    output.push(b);
                }
                a = b;
            }
            polygon = output;
        }
        polygon
    }

    fn clipped_disk(mut polygon: Vec<[f32; 3]>, center: [f32; 3], radius: f32) -> Vec<[f32; 3]> {
        // Tangent half-planes contain the actual circular cap conservatively.
        for step in 0..16 {
            if polygon.is_empty() {
                return polygon;
            }
            let angle = step as f32 * std::f32::consts::TAU / 16.;
            let distance = |p: [f32; 3]| {
                (p[0] - center[0]) * angle.cos() + (p[1] - center[2]) * angle.sin() - radius
            };
            let mut output = Vec::new();
            let mut a = *polygon.last().unwrap();
            let mut ad = distance(a);
            for &b in &polygon {
                let bd = distance(b);
                if (ad <= 0.) != (bd <= 0.) {
                    let t = ad / (ad - bd);
                    output.push(std::array::from_fn(|i| a[i] + (b[i] - a[i]) * t));
                }
                if bd <= 0. {
                    output.push(b);
                }
                a = b;
                ad = bd;
            }
            polygon = output;
        }
        polygon
    }

    #[test]
    fn full_ridge_width_is_clear_of_structure_and_geological_faces() {
        let scene = load_station_geometry(&asset_root()).unwrap();
        let mut checked_rocks = 0;
        for pair in scene.layout.ridge_waypoints.windows(2) {
            let a = pair[0];
            let b = pair[1];
            let dx = b[0] - a[0];
            let dz = b[2] - a[2];
            let length = dx.hypot(dz);
            let local = |p: [f32; 3]| {
                [
                    ((p[0] - a[0]) * dx + (p[2] - a[2]) * dz) / length,
                    (-(p[0] - a[0]) * dz + (p[2] - a[2]) * dx) / length,
                    p[1],
                ]
            };
            let clearance = |p: [f32; 3]| p[2] - a[1] - (b[1] - a[1]) * p[0] / length;
            for collider in scene.colliders.iter().filter(|c| c.owner == "station") {
                match &collider.shape {
                    StationCollisionShape::Triangles { vertices, indices }
                        if collider.tag.starts_with("geology_") =>
                    {
                        checked_rocks += 1;
                        for face in indices {
                            let poly = clipped_strip(
                                face.iter().map(|&i| local(vertices[i as usize])).collect(),
                                length,
                                scene.layout.ridge_width * 0.5,
                            );
                            assert!(
                                poly.iter().all(|&p| clearance(p) <= 0.081),
                                "Geology {} intrudes into the full ridge width",
                                collider.tag
                            );
                        }
                    }
                    StationCollisionShape::Convex { vertices } => {
                        let low: [f32; 3] = std::array::from_fn(|i| {
                            vertices.iter().map(|v| v[i]).fold(f32::INFINITY, f32::min)
                        });
                        let high: [f32; 3] = std::array::from_fn(|i| {
                            vertices
                                .iter()
                                .map(|v| v[i])
                                .fold(f32::NEG_INFINITY, f32::max)
                        });
                        if low[1] > a[1].max(b[1]) + 1.2 || high[1] <= a[1].min(b[1]) + 0.08 {
                            continue;
                        }
                        let poly = clipped_strip(
                            vec![
                                local([low[0], high[1], low[2]]),
                                local([high[0], high[1], low[2]]),
                                local([high[0], high[1], high[2]]),
                                local([low[0], high[1], high[2]]),
                            ],
                            length,
                            scene.layout.ridge_width * 0.5,
                        );
                        assert!(
                            poly.is_empty() || poly.iter().all(|&p| clearance(p) <= 0.081),
                            "Structure {} intrudes into the full ridge width at {poly:?}",
                            collider.tag
                        );
                    }
                    _ => {}
                }
            }
        }
        let grade = scene.layout.ridge_grade_plane;
        let clearance = |p: [f32; 3]| {
            p[2] - (grade[0] * p[0] + grade[1] * p[1] + grade[2])
                .clamp(0., scene.layout.ridge_waypoints.last().unwrap()[1])
        };
        for &center in &scene.layout.ridge_waypoints {
            for c in scene.colliders.iter().filter(|c| c.owner == "station") {
                match &c.shape {
                    StationCollisionShape::Triangles { vertices, indices }
                        if c.tag.starts_with("geology_") =>
                    {
                        for face in indices {
                            let poly = clipped_disk(
                                face.iter()
                                    .map(|&i| {
                                        let p = vertices[i as usize];
                                        [p[0], p[2], p[1]]
                                    })
                                    .collect(),
                                center,
                                scene.layout.ridge_width * 0.5,
                            );
                            assert!(
                                poly.iter().all(|&p| clearance(p) <= 0.081),
                                "Geology {} enters a rounded ridge bend/end",
                                c.tag
                            );
                        }
                    }
                    StationCollisionShape::Convex { vertices } => {
                        let low: [f32; 3] = std::array::from_fn(|i| {
                            vertices.iter().map(|v| v[i]).fold(f32::INFINITY, f32::min)
                        });
                        let high: [f32; 3] = std::array::from_fn(|i| {
                            vertices
                                .iter()
                                .map(|v| v[i])
                                .fold(f32::NEG_INFINITY, f32::max)
                        });
                        if low[1] > center[1] + 1.2 || high[1] <= center[1] + 0.08 {
                            continue;
                        }
                        let poly = clipped_disk(
                            vec![
                                [low[0], low[2], high[1]],
                                [high[0], low[2], high[1]],
                                [high[0], high[2], high[1]],
                                [low[0], high[2], high[1]],
                            ],
                            center,
                            scene.layout.ridge_width * 0.5,
                        );
                        assert!(
                            poly.is_empty() || poly.iter().all(|&p| clearance(p) <= 0.081),
                            "Structure {} enters a rounded ridge bend/end at {poly:?}",
                            c.tag
                        );
                    }
                    _ => {}
                }
            }
        }
        assert!(checked_rocks >= 45, "Geological physics faces are missing");
    }

    fn scratch(rel: &str) -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../.scratch/visuals")
            .join(rel)
    }

    fn stage_assets(name: &str, manifest: &Path, layout: &Path, glb: &Path) -> std::path::PathBuf {
        let root =
            std::env::temp_dir().join(format!("bevy_sim2sim_v6_{name}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let data = root.join("game/dynamic_assets/game_data");
        let models = root.join("game/arts/environment/models");
        fs::create_dir_all(&data).unwrap();
        fs::create_dir_all(&models).unwrap();
        fs::copy(manifest, data.join("science_station.ron")).unwrap();
        fs::copy(layout, data.join("science_station_layout.ron")).unwrap();
        std::os::unix::fs::symlink(glb, models.join("science_station.glb")).unwrap();
        root
    }

    fn write_layout_pair(root: &Path, mut layout: String) {
        if !layout.ends_with('\n') {
            layout.push('\n');
        }
        let body = layout.trim_end_matches('\n');
        let digest = format!("{:x}", Sha256::digest(layout.as_bytes()));
        let data = root.join("game/dynamic_assets/game_data");
        fs::write(data.join("science_station_layout.ron"), &layout).unwrap();
        let manifest = fs::read_to_string(data.join("science_station.ron")).unwrap();
        let start = manifest.find("layout:").unwrap() + "layout:".len();
        let end = manifest.rfind(",layout_sha256:").unwrap();
        let mut updated = String::new();
        updated.push_str(&manifest[..start]);
        updated.push_str(body);
        updated.push_str(&manifest[end..]);
        let token = "layout_sha256:\"";
        let sha_at = updated.find(token).unwrap() + token.len();
        let sha_end = sha_at + updated[sha_at..].find('"').unwrap();
        updated.replace_range(sha_at..sha_end, &digest);
        fs::write(data.join("science_station.ron"), updated).unwrap();
    }

    fn cap_vertex_keys(center: [f64; 3], size: [f64; 3]) -> HashSet<([u32; 3], [u32; 3])> {
        let half = [size[0] * 0.5, size[1] * 0.5, size[2] * 0.5];
        let mut keys = HashSet::new();
        for axis in 0..3 {
            for sign in [-1.0_f64, 1.0] {
                let u = (axis + 1) % 3;
                let v = (axis + 2) % 3;
                let mut normal = [0.0_f64; 3];
                normal[axis] = sign;
                for (along_u, along_v) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                    let mut position = center;
                    position[axis] += sign * half[axis];
                    position[u] += along_u * half[u];
                    position[v] += along_v * sign * half[v];
                    keys.insert((
                        position.map(|value| (value as f32).to_bits()),
                        normal.map(|value| (value as f32).to_bits()),
                    ));
                }
            }
        }
        keys
    }

    fn triangles_on_keys(surface: &StationSurface, keys: &HashSet<([u32; 3], [u32; 3])>) -> usize {
        surface
            .indices
            .chunks_exact(3)
            .filter(|triangle| {
                triangle.iter().all(|index| {
                    let index = *index as usize;
                    let position = surface.positions[index].map(f32::to_bits);
                    let normal = surface.normals[index].map(f32::to_bits);
                    keys.contains(&(position, normal))
                })
            })
            .count()
    }

    #[test]
    fn v6_graphics_and_sample_caps_match_the_approved_station() {
        let scene = load_station_geometry(&asset_root()).unwrap();
        assert_eq!(scene.layout.identity, "windpass_courtyard_v6");
        assert_eq!(scene.layout.revision, 6);
        assert_eq!(scene.labels.len(), 38);
        assert_eq!(
            scene
                .layout
                .graphics
                .iter()
                .map(|graphic| graphic.id.as_str())
                .collect::<Vec<_>>(),
            vec![
                "sample_vial_glyph",
                "service_power_glyph",
                "service_tool_glyph",
                "survey_wave_glyph",
                "field_route_glyph",
            ]
        );
        let caps = [
            ([-16.45, 0.88, 3.85], [0.24, 0.025, 0.26]),
            ([-16.0, 0.88, 3.85], [0.24, 0.025, 0.26]),
            ([-15.55, 0.88, 3.85], [0.24, 0.025, 0.26]),
        ];
        for (center, size) in caps {
            let keys = cap_vertex_keys(center, size);
            assert_eq!(keys.len(), 24);
            let purple = scene
                .surfaces
                .iter()
                .find(|surface| surface.role == "purple" && surface.owner == "station")
                .unwrap();
            let blue = scene
                .surfaces
                .iter()
                .find(|surface| surface.role == "blue" && surface.owner == "station")
                .unwrap();
            assert_eq!(triangles_on_keys(purple, &keys), 12);
            assert_eq!(triangles_on_keys(blue, &keys), 0);
        }
        let mut changed = scene.surfaces.clone();
        for surface in &mut changed {
            if surface.owner == "station" && surface.role == "paper" {
                for position in &mut surface.positions {
                    position[0] += 0.05;
                }
            }
        }
        let missing_face =
            validate_surface_graphics(&scene.layout, &changed, &scene.palette, &scene.labels)
                .unwrap_err();
        assert!(missing_face.contains("carrier face"), "{missing_face}");
        let mut layout = scene.layout.clone();
        layout.graphics[0].center[0] = f32::NAN;
        assert!(
            validate_surface_graphics(&layout, &scene.surfaces, &scene.palette, &scene.labels)
                .unwrap_err()
                .contains("not finite")
        );
        layout.graphics[0].center = scene.layout.graphics[0].center;
        layout.graphics[0].size = [0.0, 0.14];
        assert!(
            validate_surface_graphics(&layout, &scene.surfaces, &scene.palette, &scene.labels)
                .unwrap_err()
                .contains("size is invalid")
        );
        layout.graphics[0].size = [4.0, 4.0];
        assert!(
            validate_surface_graphics(&layout, &scene.surfaces, &scene.palette, &scene.labels)
                .unwrap_err()
                .contains("out of bounds")
        );
        layout.graphics[0] = scene.layout.graphics[0].clone();
        layout.graphics[0].role = "ink".into();
        assert!(
            validate_surface_graphics(&layout, &scene.surfaces, &scene.palette, &scene.labels)
                .unwrap_err()
                .contains("role")
        );
        layout.graphics[0] = scene.layout.graphics[0].clone();
        layout.graphics[0].center[0] += 0.2;
        assert!(
            validate_surface_graphics(&layout, &scene.surfaces, &scene.palette, &scene.labels)
                .unwrap_err()
                .contains("center")
        );
        layout.graphics[0] = scene.layout.graphics[0].clone();
        layout.graphics[0].depth = 0.02;
        assert!(
            validate_surface_graphics(&layout, &scene.surfaces, &scene.palette, &scene.labels)
                .unwrap_err()
                .contains("depth")
        );
        layout.graphics[0] = scene.layout.graphics[0].clone();
        layout.graphics[0].normal = [0.0, 1.0, 0.0];
        assert!(
            validate_surface_graphics(&layout, &scene.surfaces, &scene.palette, &scene.labels)
                .unwrap_err()
                .contains("normal")
        );
        layout.graphics[0] = scene.layout.graphics[0].clone();
        layout.graphics[0].kind = "decal".into();
        assert!(
            validate_surface_graphics(&layout, &scene.surfaces, &scene.palette, &scene.labels)
                .unwrap_err()
                .contains("kind")
        );
        layout.graphics[0] = scene.layout.graphics[0].clone();
        layout.graphics[0].id = "not_approved".into();
        assert!(
            validate_surface_graphics(&layout, &scene.surfaces, &scene.palette, &scene.labels)
                .unwrap_err()
                .contains("unknown")
        );
        layout.graphics[0] = scene.layout.graphics[0].clone();
        layout.graphics[0].carrier.clear();
        assert!(
            validate_surface_graphics(&layout, &scene.surfaces, &scene.palette, &scene.labels)
                .unwrap_err()
                .contains("carrier")
        );
        layout.graphics[1].id = layout.graphics[0].id.clone();
        layout.graphics[0] = scene.layout.graphics[0].clone();
        assert!(
            validate_surface_graphics(&layout, &scene.surfaces, &scene.palette, &scene.labels)
                .unwrap_err()
                .contains("duplicate")
        );
        let mut labels = scene.labels.clone();
        labels
            .iter_mut()
            .find(|label| label.text == "SAMPLES\n04")
            .unwrap()
            .position[0] = -14.95;
        assert!(
            validate_surface_graphics(&scene.layout, &scene.surfaces, &scene.palette, &labels)
                .unwrap_err()
                .contains("overlaps signage")
        );
    }

    #[test]
    fn historical_v4_and_v5_load_without_graphics_and_reject_v6_records() {
        let v4 = scratch("v4_archive/assets");
        let v4_scene = load_station_geometry(&v4).unwrap();
        assert_eq!(v4_scene.layout.identity, "windpass_courtyard_v4");
        assert!(v4_scene.layout.graphics.is_empty());
        assert_eq!(v4_scene.labels.len(), 38);
        let glb = v4.join("game/arts/environment/models/science_station.glb");
        let v5_root = stage_assets(
            "v5_historical",
            &scratch("v5_signage/after/science_station.ron"),
            &scratch("v5_signage/after/science_station_layout.ron"),
            &glb,
        );
        let v5_scene = load_station_geometry(&v5_root).unwrap();
        assert_eq!(v5_scene.layout.identity, "windpass_courtyard_v5");
        assert!(v5_scene.layout.graphics.is_empty());
        let live_layout = fs::read_to_string(
            asset_root().join("game/dynamic_assets/game_data/science_station_layout.ron"),
        )
        .unwrap();
        let graphics = live_layout.trim_end().rsplit_once(",graphics:").unwrap().1;
        let graphics = graphics.trim_end_matches(')');
        for (name, manifest, layout) in [
            (
                "v4_with_graphics",
                v4.join("game/dynamic_assets/game_data/science_station.ron"),
                v4.join("game/dynamic_assets/game_data/science_station_layout.ron"),
            ),
            (
                "v5_with_graphics",
                scratch("v5_signage/after/science_station.ron"),
                scratch("v5_signage/after/science_station_layout.ron"),
            ),
        ] {
            let root = stage_assets(name, &manifest, &layout, &glb);
            let old = fs::read_to_string(
                root.join("game/dynamic_assets/game_data/science_station_layout.ron"),
            )
            .unwrap();
            let mut injected = old.trim_end().to_string();
            injected.pop();
            injected.push_str(",graphics:");
            injected.push_str(graphics);
            injected.push(')');
            write_layout_pair(&root, injected);
            let err = load_station_geometry(&root).unwrap_err();
            assert!(
                err.contains("Historical station layout cannot contain surface graphics"),
                "{err}"
            );
        }
        let live_manifest = asset_root().join("game/dynamic_assets/game_data/science_station.ron");
        let live_layout_path =
            asset_root().join("game/dynamic_assets/game_data/science_station_layout.ron");
        let live_glb = asset_root().join("game/arts/environment/models/science_station.glb");
        let missing = stage_assets(
            "v6_missing_graphics",
            &live_manifest,
            &live_layout_path,
            &live_glb,
        );
        let stripped = fs::read_to_string(
            missing.join("game/dynamic_assets/game_data/science_station_layout.ron"),
        )
        .unwrap();
        let stripped = stripped
            .trim_end()
            .rsplit_once(",graphics:")
            .unwrap()
            .0
            .to_string()
            + ")\n";
        write_layout_pair(&missing, stripped);
        let err = load_station_geometry(&missing).unwrap_err();
        assert!(err.contains("exactly five"), "{err}");
        let unknown = stage_assets(
            "v6_unknown_field",
            &live_manifest,
            &live_layout_path,
            &live_glb,
        );
        let text = fs::read_to_string(
            unknown.join("game/dynamic_assets/game_data/science_station_layout.ron"),
        )
        .unwrap();
        let text = text.replacen("role:\"purple\")", "role:\"purple\",extra:1)", 1);
        write_layout_pair(&unknown, text);
        let err = load_station_geometry(&unknown).unwrap_err();
        assert!(
            err.to_lowercase().contains("unknown") || err.contains("extra"),
            "{err}"
        );
        let wrong_role = stage_assets(
            "v6_wrong_role",
            &live_manifest,
            &live_layout_path,
            &live_glb,
        );
        let text = fs::read_to_string(
            wrong_role.join("game/dynamic_assets/game_data/science_station_layout.ron"),
        )
        .unwrap();
        let text = text.replacen("role:\"purple\"", "role:\"ink\"", 1);
        write_layout_pair(&wrong_role, text);
        let err = load_station_geometry(&wrong_role).unwrap_err();
        assert!(err.contains("role is wrong"), "{err}");
    }
}
