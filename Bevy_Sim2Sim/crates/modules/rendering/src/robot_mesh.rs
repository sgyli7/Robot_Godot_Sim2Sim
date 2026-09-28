//! Complete compiled visual meshes and hash-bound native appearance data.
//!
//! Positions use compiled mesh vertices, not source STL coordinates or native
//! collision hulls. Mesh compiler transforms must never be applied a second time.

use robot_minigame::{basis::source_to_engine_vector, definition::RobotDefinition};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::HashMap, fmt, fs, path::Path};

/// A terminal display-boundary failure, independent of control qualification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RobotVisualError(pub String);
impl fmt::Display for RobotVisualError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}
impl std::error::Error for RobotVisualError {}

fn invalid(message: impl Into<String>) -> RobotVisualError {
    RobotVisualError(message.into())
}

/// Native material ID is its position in the document's material array.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RobotMaterialAppearance {
    pub name: Option<String>,
    pub rgba: [f32; 4],
    pub emission: f32,
    pub specular: f32,
    pub shininess: f32,
    pub reflectance: f32,
    /// MuJoCo's -1 sentinel is preserved, not rejected or silently overwritten.
    pub metallic: f32,
    pub roughness: f32,
}

/// Compiled native normals, with mesh-local corner indices for every face.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RobotMeshNormals {
    pub mesh_id: usize,
    pub normals: Vec<[f32; 3]>,
    pub face_normals: Vec<[usize; 3]>,
}

/// Export from the same BAM-edited compilation as the bound RobotDefinition.
/// These are source data; the mutable deserialization object is never installed
/// as runtime configuration. Only VerifiedRobotAppearance may enter the plugin.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RobotAppearanceDocument {
    pub schema: String,
    pub model_file_sha256: String,
    pub visible_geom_group: u32,
    pub visible_geom_ids: Vec<usize>,
    /// This first importer supports the actual untextured MicroDuck models only.
    pub texture_count: usize,
    pub geom_group: Vec<u32>,
    pub geom_rgba: Vec<[f32; 4]>,
    pub geom_matid: Vec<i32>,
    pub materials: Vec<RobotMaterialAppearance>,
    pub meshes: Vec<RobotMeshNormals>,
}

/// Validated native appearance remains immutable for the model's lifetime.
pub struct VerifiedRobotAppearance {
    document: RobotAppearanceDocument,
    file_sha256: String,
}
impl VerifiedRobotAppearance {
    pub fn load_json(
        path: &Path,
        expected_file_sha256: &str,
        definition: &RobotDefinition,
    ) -> Result<Self, RobotVisualError> {
        let (bytes, file_sha256) = read_bound_bytes(path, expected_file_sha256)?;
        let document = serde_json::from_slice(&bytes).map_err(|e| invalid(e.to_string()))?;
        Self::validated(document, file_sha256, definition)
    }

    pub fn load_ron(
        path: &Path,
        expected_file_sha256: &str,
        definition: &RobotDefinition,
    ) -> Result<Self, RobotVisualError> {
        let (bytes, file_sha256) = read_bound_bytes(path, expected_file_sha256)?;
        let document = ron::de::from_bytes(&bytes).map_err(|e| invalid(e.to_string()))?;
        Self::validated(document, file_sha256, definition)
    }

    fn validated(
        document: RobotAppearanceDocument,
        file_sha256: String,
        definition: &RobotDefinition,
    ) -> Result<Self, RobotVisualError> {
        validate_appearance(&document, definition)?;
        Ok(Self {
            document,
            file_sha256,
        })
    }

    pub fn document(&self) -> &RobotAppearanceDocument {
        &self.document
    }
    pub fn file_sha256(&self) -> &str {
        &self.file_sha256
    }
}

fn read_bound_bytes(
    path: &Path,
    expected_file_sha256: &str,
) -> Result<(Vec<u8>, String), RobotVisualError> {
    let bytes = fs::read(path).map_err(|e| invalid(e.to_string()))?;
    let file_sha256 = format!("{:x}", Sha256::digest(&bytes));
    if file_sha256 != expected_file_sha256 {
        return Err(invalid("robot appearance file SHA256 mismatch"));
    }
    Ok((bytes, file_sha256))
}

fn rgba_valid(rgba: &[f32; 4]) -> bool {
    rgba.iter()
        .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
}

/// Checks coverage, identity and native indexing; this does not establish that
/// a producer used the correct compilation. The workflow supplies that evidence.
pub fn validate_appearance(
    document: &RobotAppearanceDocument,
    definition: &RobotDefinition,
) -> Result<(), RobotVisualError> {
    let model = definition.model();
    if document.schema != "microduck_visual_v1"
        || document.model_file_sha256 != definition.file_sha256()
        || document.visible_geom_group != 2
        || document.texture_count != 0
        || document.geom_group.len() != model.counts.ngeom
        || document.geom_rgba.len() != model.counts.ngeom
        || document.geom_matid.len() != model.counts.ngeom
        || document.meshes.len() != model.counts.nmesh
    {
        return Err(invalid(
            "robot appearance identity, coverage or texture mismatch",
        ));
    }
    let expected_visible: Vec<_> = document
        .geom_group
        .iter()
        .enumerate()
        .filter_map(|(id, group)| (*group == 2).then_some(id))
        .collect();
    if expected_visible.is_empty() || document.visible_geom_ids != expected_visible {
        return Err(invalid(
            "visible geom IDs must exactly cover source group 2",
        ));
    }
    for id in &document.visible_geom_ids {
        if model.fields.geom_bodyid[*id] == 0 || model.fields.geom_type[*id] != 7 {
            return Err(invalid(
                "robot visual geom has no supported robot body/mesh",
            ));
        }
    }
    for (rgba, matid) in document.geom_rgba.iter().zip(&document.geom_matid) {
        if !rgba_valid(rgba) || *matid < -1 || *matid >= document.materials.len() as i32 {
            return Err(invalid("invalid source geom RGBA or material ID"));
        }
    }
    for material in &document.materials {
        let unit_fields = [
            material.emission,
            material.specular,
            material.shininess,
            material.reflectance,
        ];
        if !rgba_valid(&material.rgba)
            || !unit_fields
                .iter()
                .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
            || ![material.metallic, material.roughness]
                .iter()
                .all(|v| v.is_finite() && (*v == -1.0 || (0.0..=1.0).contains(v)))
        {
            return Err(invalid("invalid source material properties"));
        }
    }
    for (id, mesh) in document.meshes.iter().enumerate() {
        if mesh.mesh_id != id
            || mesh.normals.is_empty()
            || mesh.face_normals.len() != model.fields.mesh_facenum[id]
            || mesh
                .face_normals
                .iter()
                .flatten()
                .any(|n| *n >= mesh.normals.len())
            || mesh.normals.iter().any(|normal| {
                !normal.iter().all(|v| v.is_finite())
                    || (normal.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() - 1.0).abs()
                        > 5e-4
            })
        {
            return Err(invalid(format!("invalid native normals for mesh {id}")));
        }
    }
    Ok(())
}

/// MuJoCo's local RGBA takes precedence only when different from its default.
/// https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-geom-rgba
pub fn effective_geom_rgba(document: &RobotAppearanceDocument, geom_id: usize) -> [f32; 4] {
    let rgba = document.geom_rgba[geom_id];
    let material = document.geom_matid[geom_id];
    if rgba != [0.5, 0.5, 0.5, 1.0] || material < 0 {
        rgba
    } else {
        document.materials[material as usize].rgba
    }
}

/// The native classic renderer corrects untextured corner normals at sharp edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum RobotNormalPolicy {
    NativeUntextured,
    /// A diagnostic for comparing the old importer; never a source visual claim.
    RawCompiledDiagnostic,
}

/// Every compiled source triangle, with the selected display normals.
#[derive(Debug)]
pub struct RobotMeshBuffers {
    pub mesh_id: usize,
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
    pub native_face_normal_fallback_corners: usize,
}

/// Split vertex/normal pairs, retaining sharp source edges and face winding.
/// B has determinant +1, so no triangle reversal or compiler transform is needed.
pub fn build_robot_mesh_buffers(
    definition: &RobotDefinition,
    appearance: &VerifiedRobotAppearance,
) -> Result<Vec<RobotMeshBuffers>, RobotVisualError> {
    build_robot_mesh_buffers_with_normal_policy(
        definition,
        appearance,
        RobotNormalPolicy::NativeUntextured,
    )
}

/// Raw appearance arrays remain immutable under either display policy.
pub fn build_robot_mesh_buffers_with_normal_policy(
    definition: &RobotDefinition,
    appearance: &VerifiedRobotAppearance,
    policy: RobotNormalPolicy,
) -> Result<Vec<RobotMeshBuffers>, RobotVisualError> {
    if appearance.document.model_file_sha256 != definition.file_sha256() {
        return Err(invalid(
            "mesh builder received an appearance for another model",
        ));
    }
    let fields = &definition.model().fields;
    appearance
        .document
        .meshes
        .iter()
        .map(|mesh| {
            let id = mesh.mesh_id;
            let vertices = &fields.mesh_vert
                [fields.mesh_vertadr[id]..fields.mesh_vertadr[id] + fields.mesh_vertnum[id]];
            let faces = &fields.mesh_face
                [fields.mesh_faceadr[id]..fields.mesh_faceadr[id] + fields.mesh_facenum[id]];
            build_mesh(
                id,
                vertices,
                faces,
                &mesh.normals,
                &mesh.face_normals,
                policy,
            )
        })
        .collect()
}

fn build_mesh(
    mesh_id: usize,
    vertices: &[[f64; 3]],
    faces: &[[usize; 3]],
    normals: &[[f32; 3]],
    face_normals: &[[usize; 3]],
    policy: RobotNormalPolicy,
) -> Result<RobotMeshBuffers, RobotVisualError> {
    let mut result = RobotMeshBuffers {
        mesh_id,
        positions: Vec::new(),
        normals: Vec::new(),
        indices: Vec::with_capacity(faces.len() * 3),
        native_face_normal_fallback_corners: 0,
    };
    let mut corner_ids = HashMap::new();
    for (face, normal_face) in faces.iter().zip(face_normals) {
        let points = face.map(|v| vertices[v].map(|v| v as f32));
        if points.iter().flatten().any(|v| !v.is_finite()) {
            return Err(invalid("compiled vertex cannot be represented as f32"));
        }
        let face_normal = native_untextured_face_normal(points);
        for corner in 0..3 {
            let raw = normals[normal_face[corner]];
            // MuJoCo 3.10.0 mjr_uploadMesh uses this per-corner fallback only
            // when mesh_texcoordadr < 0. The audited native exporter rejects UVs
            // and textures rather than dropping unsupported source data.
            // https://github.com/google-deepmind/mujoco/blob/3.10.0/src/render/classic/render_context.c#L229-L267
            let dot = raw[0] * face_normal[0] + raw[1] * face_normal[1] + raw[2] * face_normal[2];
            let fallback = policy == RobotNormalPolicy::NativeUntextured && f64::from(dot) < 0.8;
            let normal = if fallback { face_normal } else { raw };
            result.native_face_normal_fallback_corners += usize::from(fallback);
            // A source vertex/normal pair can acquire a different effective
            // face normal in another triangle; preserve that sharp boundary.
            let key = (face[corner], normal.map(f32::to_bits));
            let index = if let Some(index) = corner_ids.get(&key) {
                *index
            } else {
                let index = u32::try_from(result.positions.len())
                    .map_err(|_| invalid("compiled visual mesh exceeds u32 indexing"))?;
                result
                    .positions
                    .push(source_to_engine_vector(points[corner]));
                result.normals.push(source_to_engine_vector(normal));
                corner_ids.insert(key, index);
                index
            };
            result.indices.push(index);
        }
    }
    Ok(result)
}

/// Mirror mjr_makeNormal/mjr_normalizeVec's f32 arithmetic and degenerate rule.
fn native_untextured_face_normal(points: [[f32; 3]; 3]) -> [f32; 3] {
    let u: [f32; 3] = std::array::from_fn(|i| points[1][i] - points[0][i]);
    let v: [f32; 3] = std::array::from_fn(|i| points[2][i] - points[0][i]);
    let normal = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    let length = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
    if length < 1e-10 {
        [0., 0., 1.]
    } else {
        let scale = 1. / length;
        normal.map(|v| v * scale)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_corner_normals_preserve_a_sharp_edge_and_right_handed_winding() {
        let vertices = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]];
        let mesh = build_mesh(
            7,
            &vertices,
            &[[0, 1, 2], [0, 3, 1]],
            &[[0., 0., 1.], [0., 1., 0.]],
            &[[0, 0, 0], [1, 1, 1]],
            RobotNormalPolicy::NativeUntextured,
        )
        .unwrap();
        assert_eq!(mesh.indices.len(), 6);
        assert_eq!(mesh.positions.len(), 6);
        assert_ne!(mesh.indices[0], mesh.indices[3]);
        for triangle in mesh.indices.chunks_exact(3) {
            let p: Vec<_> = triangle
                .iter()
                .map(|i| mesh.positions[*i as usize])
                .collect();
            let u = std::array::from_fn::<_, 3, _>(|i| p[1][i] - p[0][i]);
            let v = std::array::from_fn::<_, 3, _>(|i| p[2][i] - p[0][i]);
            let cross = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ];
            assert_eq!(cross, mesh.normals[triangle[0] as usize]);
        }
    }

    #[test]
    fn untextured_normals_use_native_sharp_edge_fallback_without_changing_faces() {
        let vertices = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]];
        let faces = [[0, 1, 2], [0, 3, 1]];
        let raw = [[0., 0., -1.], [0., 0.6, 0.8]];
        let corners = [[0, 0, 0], [1, 1, 1]];
        let native = build_mesh(
            0,
            &vertices,
            &faces,
            &raw,
            &corners,
            RobotNormalPolicy::NativeUntextured,
        )
        .unwrap();
        let old = build_mesh(
            0,
            &vertices,
            &faces,
            &raw,
            &corners,
            RobotNormalPolicy::RawCompiledDiagnostic,
        )
        .unwrap();
        assert_eq!(native.native_face_normal_fallback_corners, 6);
        assert_eq!(old.native_face_normal_fallback_corners, 0);
        for corner in 0..6 {
            assert_eq!(
                native.positions[native.indices[corner] as usize],
                old.positions[old.indices[corner] as usize]
            );
            assert_eq!(
                native.normals[native.indices[corner] as usize],
                source_to_engine_vector(if corner < 3 {
                    [0., 0., 1.]
                } else {
                    [0., 1., 0.]
                })
            );
        }
        assert_eq!(native_untextured_face_normal([[0.; 3]; 3]), [0., 0., 1.]);
        let kept = build_mesh(
            0,
            &vertices,
            &faces[..1],
            &raw[1..],
            &[[0, 0, 0]],
            RobotNormalPolicy::NativeUntextured,
        )
        .unwrap();
        assert_eq!(kept.native_face_normal_fallback_corners, 0);
        assert_eq!(kept.normals[0], source_to_engine_vector(raw[1]));
    }

    #[test]
    #[ignore = "requires actual compiled models and the separately compiled MuJoCo C normal oracle"]
    fn actual_native_display_normals_match_independent_c_oracle() {
        let path = std::env::var("ROBOT_NORMAL_ORACLE").expect("ROBOT_NORMAL_ORACLE");
        let receipt: serde_json::Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
        for model in receipt["models"].as_array().unwrap() {
            let definition = RobotDefinition::load_json(
                Path::new(model["definition"]["path"].as_str().unwrap()),
                model["definition"]["sha256"].as_str().unwrap(),
            )
            .unwrap();
            let appearance = VerifiedRobotAppearance::load_json(
                Path::new(model["appearance"]["path"].as_str().unwrap()),
                model["appearance"]["sha256"].as_str().unwrap(),
                &definition,
            )
            .unwrap();
            let (bytes, _) = read_bound_bytes(
                Path::new(model["oracle"]["path"].as_str().unwrap()),
                model["oracle"]["sha256"].as_str().unwrap(),
            )
            .unwrap();
            assert_eq!(bytes.len() % 12, 0);
            let normals: Vec<[f32; 3]> = bytes
                .chunks_exact(12)
                .map(|row| {
                    std::array::from_fn(|i| {
                        f32::from_le_bytes(row[i * 4..i * 4 + 4].try_into().unwrap())
                    })
                })
                .collect();
            let buffers = build_robot_mesh_buffers(&definition, &appearance).unwrap();
            let mut offset = 0;
            let mut fallback = 0;
            let mut maximum = 0_f32;
            for buffer in buffers {
                for index in &buffer.indices {
                    let actual = buffer.normals[*index as usize];
                    for component in 0..3 {
                        assert!(
                            actual[component].is_finite() && normals[offset][component].is_finite()
                        );
                        maximum =
                            maximum.max((actual[component] - normals[offset][component]).abs());
                    }
                    offset += 1;
                }
                fallback += buffer.native_face_normal_fallback_corners;
            }
            assert_eq!(offset, normals.len());
            assert_eq!(fallback as u64, model["fallback_corners"].as_u64().unwrap());
            assert!(maximum <= 2e-7, "C oracle normal error {maximum}");
            println!(
                "actual C oracle: {} corners={offset} fallback={fallback} max={maximum}",
                definition.model().family
            );
        }
    }
}
