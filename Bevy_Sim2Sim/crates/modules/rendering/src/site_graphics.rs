//! Code-native vector marks on existing station faces.
//!
//! Each graphic is a triangle mesh in the carrier's XY plane, offset 1.5 mm
//! in front of that face. Strokes stay inside the typed rectangle. These marks
//! are static illustrations and do not add collision, text, or a new material role.

use crate::StationScene;
use crate::geometry::StationSurfaceGraphic;
use bevy::{asset::RenderAssetUsages, prelude::*};

#[derive(Debug)]
pub(crate) struct SurfaceGraphicMesh {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
}

fn sub(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

fn add(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [a[0] + b[0], a[1] + b[1]]
}

fn mul(a: [f32; 2], scale: f32) -> [f32; 2] {
    [a[0] * scale, a[1] * scale]
}

fn normalize(value: [f32; 2]) -> Option<[f32; 2]> {
    let length = value[0].hypot(value[1]);
    if length < 1.0e-6 {
        None
    } else {
        Some(mul(value, 1.0 / length))
    }
}

fn left_normal(direction: [f32; 2]) -> [f32; 2] {
    [-direction[1], direction[0]]
}

fn push_triangle(mesh: &mut SurfaceGraphicMesh, triangle: [[f32; 2]; 3], center: [f32; 3]) {
    let base = mesh.positions.len() as u32;
    for point in triangle {
        mesh.positions
            .push([center[0] + point[0], center[1] + point[1], center[2]]);
        mesh.normals.push([0.0, 0.0, 1.0]);
    }
    mesh.indices.extend([base, base + 1, base + 2]);
}

fn side_offsets(
    points: &[[f32; 2]],
    closed: bool,
    half_stroke: f32,
) -> Result<Vec<([f32; 2], [f32; 2])>, String> {
    if points.len() < 2 {
        return Err("station graphic stroke is degenerate".into());
    }
    let mut sides = Vec::with_capacity(points.len());
    for index in 0..points.len() {
        let previous = if index == 0 {
            if closed {
                Some(points[points.len() - 1])
            } else {
                None
            }
        } else {
            Some(points[index - 1])
        };
        let next = if index + 1 == points.len() {
            if closed { Some(points[0]) } else { None }
        } else {
            Some(points[index + 1])
        };
        let incoming = match previous {
            Some(point) => Some(
                normalize(sub(points[index], point))
                    .ok_or_else(|| "station graphic stroke is degenerate".to_string())?,
            ),
            None => None,
        };
        let outgoing = match next {
            Some(point) => Some(
                normalize(sub(point, points[index]))
                    .ok_or_else(|| "station graphic stroke is degenerate".to_string())?,
            ),
            None => None,
        };
        let normal = match (incoming.map(left_normal), outgoing.map(left_normal)) {
            (Some(start), Some(end)) => {
                let sum = add(start, end);
                match normalize(sum) {
                    Some(direction) => {
                        let alignment =
                            (start[0] * direction[0] + start[1] * direction[1]).max(0.35);
                        mul(direction, (half_stroke / alignment).min(half_stroke * 1.8))
                    }
                    None => mul(start, half_stroke),
                }
            }
            (Some(normal), None) | (None, Some(normal)) => mul(normal, half_stroke),
            (None, None) => return Err("station graphic stroke is degenerate".into()),
        };
        sides.push((
            add(points[index], mul(normal, -1.0)),
            add(points[index], normal),
        ));
    }
    Ok(sides)
}

fn add_polyline(
    mesh: &mut SurfaceGraphicMesh,
    points: &[[f32; 2]],
    closed: bool,
    stroke: f32,
    center: [f32; 3],
) -> Result<(), String> {
    let sides = side_offsets(points, closed, stroke * 0.5)?;
    let segments = if closed { sides.len() } else { sides.len() - 1 };
    for index in 0..segments {
        let next = (index + 1) % sides.len();
        let (right_start, left_start) = sides[index];
        let (right_end, left_end) = sides[next];
        push_triangle(mesh, [right_start, right_end, left_end], center);
        push_triangle(mesh, [right_start, left_end, left_start], center);
    }
    Ok(())
}

fn add_disk(mesh: &mut SurfaceGraphicMesh, local_center: [f32; 2], radius: f32, center: [f32; 3]) {
    let segments = 10;
    let mut ring = Vec::with_capacity(segments);
    for step in 0..segments {
        let angle = step as f32 * std::f32::consts::TAU / segments as f32;
        ring.push([
            local_center[0] + radius * angle.cos(),
            local_center[1] + radius * angle.sin(),
        ]);
    }
    for step in 0..segments {
        let next = (step + 1) % segments;
        push_triangle(mesh, [local_center, ring[step], ring[next]], center);
    }
}

fn stroke_width(id: &str) -> Result<f32, String> {
    match id {
        "sample_vial_glyph" | "survey_wave_glyph" | "field_route_glyph" => Ok(0.008),
        "service_power_glyph" | "service_tool_glyph" => Ok(0.010),
        _ => Err(format!("unknown station graphic: {id}")),
    }
}

fn vial_loops() -> Vec<Vec<[f32; 2]>> {
    let margin = 0.008;
    let gap = 0.008;
    let width = 0.13;
    let height = 0.14;
    let outer_w = (width - 2.0 * margin - 2.0 * gap) / 3.0;
    let outer_h = height - 2.0 * margin;
    let inset = 0.004;
    let mut loops = Vec::new();
    for index in 0..3 {
        let outer_left = -width * 0.5 + margin + index as f32 * (outer_w + gap);
        let left = outer_left + inset;
        let right = outer_left + outer_w - inset;
        let bottom = -height * 0.5 + margin + inset;
        let top = bottom + outer_h - 2.0 * inset;
        let mid_x = (left + right) * 0.5;
        let neck_half = (right - left) * 0.28;
        let shoulder = bottom + (top - bottom) * 0.72;
        let neck = bottom + (top - bottom) * 0.80;
        loops.push(vec![
            [left, bottom],
            [right, bottom],
            [right, shoulder],
            [mid_x + neck_half, neck],
            [mid_x + neck_half, top],
            [mid_x - neck_half, top],
            [mid_x - neck_half, neck],
            [left, shoulder],
        ]);
    }
    loops
}

fn wrench_loop() -> Vec<[f32; 2]> {
    vec![
        [-0.086, -0.015],
        [0.012, -0.015],
        [0.028, -0.052],
        [0.084, -0.052],
        [0.084, -0.024],
        [0.046, -0.024],
        [0.046, 0.024],
        [0.084, 0.024],
        [0.084, 0.052],
        [0.028, 0.052],
        [0.012, 0.015],
        [-0.086, 0.015],
    ]
}

fn power_bolt() -> Vec<[f32; 2]> {
    vec![
        [0.036, 0.072],
        [-0.040, 0.006],
        [0.024, -0.008],
        [-0.036, -0.072],
    ]
}

fn survey_marks() -> (Vec<[f32; 2]>, [[f32; 2]; 3]) {
    let mut wave = Vec::new();
    for step in 0..=8 {
        let t = step as f32 / 8.0;
        wave.push([
            -0.145 + t * 0.290,
            0.012 + (t * std::f32::consts::TAU).sin() * 0.028,
        ]);
    }
    (wave, [[-0.145, -0.050], [0.0, -0.050], [0.145, -0.050]])
}

pub(crate) fn project_ridge_route(
    waypoints: &[[f32; 3]],
    size: [f32; 2],
) -> Result<Vec<[f32; 2]>, String> {
    if waypoints.len() < 2
        || waypoints
            .iter()
            .any(|point| point.iter().any(|value| !value.is_finite()))
    {
        return Err("station graphic route is missing waypoints".into());
    }
    let margin = 0.012_f64;
    let usable_w = f64::from(size[0]) - 2.0 * margin;
    let usable_h = f64::from(size[1]) - 2.0 * margin;
    if usable_w <= 0.0 || usable_h <= 0.0 {
        return Err("station graphic size is invalid".into());
    }
    let min_x = waypoints
        .iter()
        .map(|point| f64::from(point[0]))
        .fold(f64::INFINITY, f64::min);
    let max_x = waypoints
        .iter()
        .map(|point| f64::from(point[0]))
        .fold(f64::NEG_INFINITY, f64::max);
    let min_z = waypoints
        .iter()
        .map(|point| f64::from(point[2]))
        .fold(f64::INFINITY, f64::min);
    let max_z = waypoints
        .iter()
        .map(|point| f64::from(point[2]))
        .fold(f64::NEG_INFINITY, f64::max);
    let range_x = (max_x - min_x).max(1.0e-6);
    let range_z = (max_z - min_z).max(1.0e-6);
    let scale = (usable_w / range_x).min(usable_h / range_z);
    let origin_x = -range_x * scale * 0.5;
    let origin_y = -range_z * scale * 0.5;
    Ok(waypoints
        .iter()
        .map(|point| {
            [
                (origin_x + (f64::from(point[0]) - min_x) * scale) as f32,
                (origin_y + (f64::from(point[2]) - min_z) * scale) as f32,
            ]
        })
        .collect())
}

fn audit_surface_graphic(
    mesh: &SurfaceGraphicMesh,
    graphic: &StationSurfaceGraphic,
) -> Result<(), String> {
    if mesh.positions.len() != mesh.normals.len()
        || mesh.indices.is_empty()
        || mesh.indices.len() % 3 != 0
        || mesh
            .indices
            .iter()
            .any(|index| *index as usize >= mesh.positions.len())
    {
        return Err(format!("station graphic mesh is invalid: {}", graphic.id));
    }
    let half_x = graphic.size[0] * 0.5;
    let half_y = graphic.size[1] * 0.5;
    for (position, normal) in mesh.positions.iter().zip(&mesh.normals) {
        if !position.iter().chain(normal).all(|value| value.is_finite()) {
            return Err(format!(
                "station graphic mesh is not finite: {}",
                graphic.id
            ));
        }
        if (position[0] - graphic.center[0]).abs() > half_x + 1.0e-4
            || (position[1] - graphic.center[1]).abs() > half_y + 1.0e-4
            || (position[2] - graphic.center[2]).abs() > 1.0e-4
        {
            return Err(format!(
                "station graphic mesh leaves its bounds: {}",
                graphic.id
            ));
        }
        if normal[0].abs() > 1.0e-5 || normal[1].abs() > 1.0e-5 || (normal[2] - 1.0).abs() > 1.0e-5
        {
            return Err(format!(
                "station graphic mesh normal is not +Z: {}",
                graphic.id
            ));
        }
    }
    let mut area = 0.0_f32;
    for triangle in mesh.indices.chunks_exact(3) {
        let a = mesh.positions[triangle[0] as usize];
        let b = mesh.positions[triangle[1] as usize];
        let c = mesh.positions[triangle[2] as usize];
        let cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
        if cross <= 1.0e-8 {
            return Err(format!(
                "station graphic mesh is degenerate: {}",
                graphic.id
            ));
        }
        area += cross * 0.5;
    }
    let limit = graphic.size[0] * graphic.size[1];
    if area <= 1.0e-4 || area > limit * 0.75 {
        return Err(format!(
            "station graphic mesh is not an outline: {}",
            graphic.id
        ));
    }
    Ok(())
}

pub(crate) fn build_surface_graphic(
    graphic: &StationSurfaceGraphic,
    ridge_waypoints: &[[f32; 3]],
) -> Result<SurfaceGraphicMesh, String> {
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
    if graphic.size[0] <= 0.0 || graphic.size[1] <= 0.0 {
        return Err(format!("station graphic size is invalid: {}", graphic.id));
    }
    if graphic.normal[2] < 0.5 {
        return Err(format!("station graphic normal must be +Z: {}", graphic.id));
    }
    let stroke = stroke_width(&graphic.id)?;
    if !(0.008..=0.012).contains(&stroke) {
        return Err(format!(
            "station graphic stroke is out of range: {}",
            graphic.id
        ));
    }
    let center = graphic.center;
    let mut mesh = SurfaceGraphicMesh {
        positions: Vec::new(),
        normals: Vec::new(),
        indices: Vec::new(),
    };
    match graphic.id.as_str() {
        "sample_vial_glyph" => {
            for loop_points in vial_loops() {
                add_polyline(&mut mesh, &loop_points, true, stroke, center)?;
            }
        }
        "service_power_glyph" => add_polyline(&mut mesh, &power_bolt(), false, stroke, center)?,
        "service_tool_glyph" => add_polyline(&mut mesh, &wrench_loop(), true, stroke, center)?,
        "survey_wave_glyph" => {
            let (wave, ticks) = survey_marks();
            add_polyline(&mut mesh, &wave, false, stroke, center)?;
            for tick_x in ticks {
                add_polyline(
                    &mut mesh,
                    &[[tick_x[0], -0.030], [tick_x[0], tick_x[1]]],
                    false,
                    stroke,
                    center,
                )?;
            }
        }
        "field_route_glyph" => {
            let mut route = project_ridge_route(ridge_waypoints, graphic.size)?;
            let endpoint = *route.last().expect("route has an endpoint");
            let previous = route[route.len() - 2];
            let direction = normalize(sub(endpoint, previous))
                .ok_or_else(|| "station graphic stroke is degenerate".to_string())?;
            let radius = 0.006;
            *route.last_mut().expect("route has an endpoint") =
                sub(endpoint, mul(direction, radius));
            add_polyline(&mut mesh, &route, false, stroke, center)?;
            add_disk(&mut mesh, endpoint, radius, center);
        }
        _ => return Err(format!("unknown station graphic: {}", graphic.id)),
    }
    audit_surface_graphic(&mesh, graphic)?;
    Ok(mesh)
}

pub(crate) fn setup_surface_graphics(
    mut commands: Commands,
    scene: Res<StationScene>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    for graphic in &scene.0.layout.graphics {
        let built = build_surface_graphic(graphic, &scene.0.layout.ridge_waypoints)
            .expect("validated station graphic mesh");
        let color = scene
            .0
            .palette
            .get(&graphic.role)
            .copied()
            .expect("station graphic palette checked by loader");
        let material = materials.add(StandardMaterial {
            base_color: Color::srgba(color[0], color[1], color[2], color[3]),
            unlit: true,
            alpha_mode: if color[3] < 1.0 {
                AlphaMode::Blend
            } else {
                AlphaMode::Opaque
            },
            cull_mode: None,
            ..default()
        });
        let mut mesh = Mesh::new(
            bevy::mesh::PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, built.positions);
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, built.normals);
        mesh.insert_indices(bevy::mesh::Indices::U32(built.indices));
        commands.spawn((
            Name::new(format!("Station graphic: {}", graphic.id)),
            Mesh3d(meshes.add(mesh)),
            MeshMaterial3d(material),
            Transform::default(),
            bevy::light::NotShadowCaster,
            bevy::light::NotShadowReceiver,
        ));
    }
    info!(
        graphics = scene.0.layout.graphics.len(),
        "STATION_SURFACE_GRAPHICS_CREATED"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::load_station_geometry;
    use std::path::Path;

    fn asset_root() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../assets")
    }

    fn graphic(id: &str, center: [f32; 3], size: [f32; 2]) -> StationSurfaceGraphic {
        StationSurfaceGraphic {
            id: id.to_string(),
            kind: "surface_vector".to_string(),
            carrier: "carrier".to_string(),
            center,
            size,
            normal: [0.0, 0.0, 1.0],
            depth: 0.0015,
            role: "blue".to_string(),
        }
    }

    #[test]
    fn approved_graphics_are_finite_ccw_outlines_inside_their_bounds() {
        let scene = load_station_geometry(&asset_root()).expect("v6 station");
        assert_eq!(scene.layout.graphics.len(), 5);
        for graphic in &scene.layout.graphics {
            let mesh = build_surface_graphic(graphic, &scene.layout.ridge_waypoints)
                .unwrap_or_else(|error| panic!("{error}"));
            assert!(mesh.indices.len() >= 6, "{}", graphic.id);
        }
    }

    #[test]
    fn ridge_route_keeps_world_axis_order_without_a_camera() {
        let scene = load_station_geometry(&asset_root()).expect("v6 station");
        let local = project_ridge_route(&scene.layout.ridge_waypoints, [0.12, 0.18]).unwrap();
        assert_eq!(local.len(), scene.layout.ridge_waypoints.len());
        for i in 0..local.len() {
            for j in 0..local.len() {
                if scene.layout.ridge_waypoints[i][0] < scene.layout.ridge_waypoints[j][0] {
                    assert!(local[i][0] < local[j][0]);
                }
                if scene.layout.ridge_waypoints[i][2] < scene.layout.ridge_waypoints[j][2] {
                    assert!(local[i][1] < local[j][1]);
                }
            }
        }
        let graphic = scene
            .layout
            .graphics
            .iter()
            .find(|graphic| graphic.id == "field_route_glyph")
            .unwrap();
        let mesh = build_surface_graphic(graphic, &scene.layout.ridge_waypoints).unwrap();
        let endpoint = [
            graphic.center[0] + local[local.len() - 1][0],
            graphic.center[1] + local[local.len() - 1][1],
            graphic.center[2],
        ];
        assert!(mesh.positions.iter().any(|position| {
            position
                .iter()
                .zip(endpoint)
                .all(|(actual, expected)| (actual - expected).abs() < 1.0e-5)
        }));
    }

    #[test]
    fn graphic_meshes_reject_empty_size_nonfinite_input_and_a_short_route() {
        let mut zero = graphic("sample_vial_glyph", [-14.95, 0.94, 5.024], [0.0, 0.14]);
        assert!(
            build_surface_graphic(&zero, &[])
                .unwrap_err()
                .contains("size is invalid")
        );
        zero.size = [0.13, 0.14];
        zero.center[0] = f32::NAN;
        assert!(
            build_surface_graphic(&zero, &[])
                .unwrap_err()
                .contains("not finite")
        );
        let mut turned = graphic("service_power_glyph", [20.2, 0.67, 4.825], [0.14, 0.18]);
        turned.normal = [0.0, 1.0, 0.0];
        assert!(
            build_surface_graphic(&turned, &[])
                .unwrap_err()
                .contains("+Z")
        );
        let route = graphic("field_route_glyph", [-9.9, 0.94, 9.424], [0.12, 0.18]);
        assert!(
            build_surface_graphic(&route, &[[-12.0, 0.0, 1.5]])
                .unwrap_err()
                .contains("waypoints")
        );
        assert!(
            build_surface_graphic(&graphic("not_a_graphic", [0.0, 0.0, 0.0], [0.1, 0.1]), &[])
                .unwrap_err()
                .contains("unknown")
        );
    }
}
