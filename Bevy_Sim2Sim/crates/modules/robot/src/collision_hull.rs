//! Native MuJoCo collision hull topology, without a second hull approximation.
//!
//! Graph layout: https://mujoco.readthedocs.io/en/3.9.0/APIreference/APItypes.html#convex-hulls

use crate::{RobotError, definition::RobotDefinition};
use std::collections::{HashMap, HashSet};

pub struct NativeCollisionHull {
    /// Source geom-local, compiler-centered coordinates, in meters and Z-up.
    pub vertices: Vec<[f64; 3]>,
    /// Native outward triangles, remapped from full-mesh indices to this hull.
    pub triangles: Vec<[u32; 3]>,
}

pub fn from_definition(
    definition: &RobotDefinition,
    mesh: usize,
) -> Result<NativeCollisionHull, RobotError> {
    let model = definition.model();
    let fields = &model.fields;
    let addresses = fields
        .mesh_graphadr
        .as_ref()
        .ok_or_else(|| invalid("missing native collision hull addresses"))?;
    let graph = fields
        .mesh_graph
        .as_ref()
        .ok_or_else(|| invalid("missing native collision hull data"))?;
    if addresses.len() != model.counts.nmesh || mesh >= model.counts.nmesh {
        return Err(invalid("native hull mesh/address count mismatch"));
    }
    let start = usize::try_from(addresses[mesh])
        .map_err(|_| invalid("collision mesh has no native hull"))?;
    let sizes = span(graph, start, 2)?;
    let vertices_count =
        usize::try_from(sizes[0]).map_err(|_| invalid("negative hull vertex count"))?;
    let faces_count = usize::try_from(sizes[1]).map_err(|_| invalid("negative hull face count"))?;
    if vertices_count < 4 || faces_count < 4 || vertices_count > u32::MAX as usize {
        return Err(invalid("invalid native hull size"));
    }
    let twice_vertices = vertices_count
        .checked_mul(2)
        .ok_or_else(|| invalid("hull size overflow"))?;
    let face_entries = faces_count
        .checked_mul(3)
        .ok_or_else(|| invalid("hull size overflow"))?;
    let edge_entries = vertices_count
        .checked_add(face_entries)
        .ok_or_else(|| invalid("hull size overflow"))?;
    let vertex_start = start
        .checked_add(2)
        .ok_or_else(|| invalid("hull address overflow"))?;
    let global_start = vertex_start
        .checked_add(vertices_count)
        .ok_or_else(|| invalid("hull address overflow"))?;
    let edge_start = vertex_start
        .checked_add(twice_vertices)
        .ok_or_else(|| invalid("hull address overflow"))?;
    let face_start = edge_start
        .checked_add(edge_entries)
        .ok_or_else(|| invalid("hull address overflow"))?;
    let edge_addresses = span(graph, vertex_start, vertices_count)?;
    let global_ids = span(graph, global_start, vertices_count)?;
    let edge_ids = span(graph, edge_start, edge_entries)?;
    let face_ids = span(graph, face_start, face_entries)?;
    let mesh_points = span(
        &fields.mesh_vert,
        fields.mesh_vertadr[mesh],
        fields.mesh_vertnum[mesh],
    )?;
    let mut global_to_local = HashMap::new();
    let mut vertices = Vec::with_capacity(vertices_count);
    for (local, global) in global_ids.iter().enumerate() {
        let global = usize::try_from(*global).map_err(|_| invalid("negative hull vertex index"))?;
        let point = mesh_points
            .get(global)
            .ok_or_else(|| invalid("hull vertex exceeds mesh"))?;
        if global_to_local.insert(global, local as u32).is_some() {
            return Err(invalid("duplicate native hull vertex"));
        }
        vertices.push(*point);
    }
    // Every adjacency record is bounded and terminated. Preserve native graph
    // topology as evidence even though the backend consumes outward triangles.
    let mut observed_edges = HashSet::new();
    for (local, address) in edge_addresses.iter().enumerate() {
        let address =
            usize::try_from(*address).map_err(|_| invalid("negative adjacency address"))?;
        let record = edge_ids
            .get(address..)
            .ok_or_else(|| invalid("adjacency address exceeds graph"))?;
        let end = record
            .iter()
            .position(|entry| *entry == -1)
            .ok_or_else(|| invalid("unterminated native adjacency"))?;
        if end < 3 {
            return Err(invalid("native hull vertex has insufficient edges"));
        }
        for neighbor in &record[..end] {
            let neighbor =
                usize::try_from(*neighbor).map_err(|_| invalid("negative adjacency vertex"))?;
            if neighbor >= vertices_count
                || neighbor == local
                || !observed_edges.insert((local, neighbor))
            {
                return Err(invalid("invalid or duplicate native hull edge"));
            }
        }
    }
    if observed_edges.len() != face_entries
        || observed_edges
            .iter()
            .any(|(a, b)| !observed_edges.contains(&(*b, *a)))
    {
        return Err(invalid(
            "native hull adjacency is not a closed triangular graph",
        ));
    }
    let mut triangles = Vec::with_capacity(faces_count);
    let mut triangle_edges: HashMap<(u32, u32), usize> = HashMap::new();
    let mut unique_faces = HashSet::new();
    for face in face_ids.chunks_exact(3) {
        let mut triangle = [0; 3];
        for (slot, global) in face.iter().enumerate() {
            let global = usize::try_from(*global).map_err(|_| invalid("negative face vertex"))?;
            triangle[slot] = *global_to_local
                .get(&global)
                .ok_or_else(|| invalid("face vertex absent from native hull"))?;
        }
        if triangle[0] == triangle[1] || triangle[1] == triangle[2] || triangle[0] == triangle[2] {
            return Err(invalid("native hull face has repeated vertices"));
        }
        let mut sorted = triangle;
        sorted.sort_unstable();
        if !unique_faces.insert(sorted) {
            return Err(invalid("duplicate native hull face"));
        }
        for (a, b) in [
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ] {
            if !observed_edges.contains(&(a as usize, b as usize)) {
                return Err(invalid("native triangle edge absent from adjacency"));
            }
            *triangle_edges.entry((a, b)).or_default() += 1;
        }
        triangles.push(triangle);
    }
    if triangle_edges.len() != face_entries
        || triangle_edges
            .iter()
            .any(|((a, b), count)| *count != 1 || triangle_edges.get(&(*b, *a)) != Some(&1))
    {
        return Err(invalid(
            "native hull triangles are not consistently oriented and closed",
        ));
    }
    Ok(NativeCollisionHull {
        vertices,
        triangles,
    })
}

fn span<T>(data: &[T], start: usize, length: usize) -> Result<&[T], RobotError> {
    let end = start
        .checked_add(length)
        .ok_or_else(|| invalid("native hull span overflow"))?;
    data.get(start..end)
        .ok_or_else(|| invalid("native hull span exceeds compiled data"))
}

fn invalid(message: &str) -> RobotError {
    RobotError::Contract(message.into())
}
