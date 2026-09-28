//! Toon enamel and a constant pixel-width inverted hull ink pass.
use bevy::material::descriptor::RenderPipelineDescriptor;
use bevy::{
    mesh::MeshVertexBufferLayoutRef,
    pbr::{ExtendedMaterial, MaterialExtension, MaterialPipeline, MaterialPipelineKey},
    prelude::*,
    render::render_resource::{AsBindGroup, Face, SpecializedMeshPipelineError},
    shader::ShaderRef,
};
pub type StationMaterial = ExtendedMaterial<StandardMaterial, StationEnamel>;
#[derive(Asset, AsBindGroup, Reflect, Debug, Clone)]
pub struct StationEnamel {
    #[uniform(100)]
    hatch_strength: f32,
    #[uniform(100)]
    material_kind: u32,
    #[uniform(100)]
    padding: Vec2,
    #[uniform(100)]
    plaza: Vec4,
    #[uniform(100)]
    loop_center_radii: Vec4,
    #[uniform(100)]
    berth: Vec4,
    #[uniform(100)]
    skills: Vec4,
    #[uniform(100)]
    paths: [Vec4; 8],
    #[uniform(100)]
    loop_width: f32,
    #[uniform(100)]
    path_count: u32,
    #[uniform(100)]
    tail_padding: Vec2,
    #[uniform(100)]
    ridge_waypoints: [Vec4; 4],
    #[uniform(100)]
    ridge_width: f32,
    #[uniform(100)]
    ridge_padding: Vec3,
}
impl StationEnamel {
    pub(crate) fn with_hatch_strength(mut self, strength: f32) -> Self {
        self.hatch_strength = strength;
        self
    }
    pub fn new(material_kind: u32, layout: &crate::geometry::StationLayout) -> Self {
        Self {
            hatch_strength: 0.075,
            material_kind,
            padding: Vec2::ZERO,
            plaza: Vec4::from_array(layout.plaza),
            loop_center_radii: Vec4::from_array(layout.loop_center_radii),
            berth: Vec4::from_array(layout.berth),
            skills: Vec4::from_array(layout.skills),
            paths: std::array::from_fn(|i| {
                layout
                    .paths
                    .get(i)
                    .copied()
                    .map(Vec4::from_array)
                    .unwrap_or(Vec4::ZERO)
            }),
            loop_width: layout.loop_width,
            path_count: layout.paths.len() as u32,
            tail_padding: Vec2::ZERO,
            ridge_waypoints: std::array::from_fn(|i| {
                let p = layout.ridge_waypoints[i];
                Vec4::new(p[0], p[1], p[2], 0.)
            }),
            ridge_width: layout.ridge_width,
            ridge_padding: Vec3::ZERO,
        }
    }
}
impl MaterialExtension for StationEnamel {
    fn fragment_shader() -> ShaderRef {
        "game/shaders/station_enamel.wgsl".into()
    }
}
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct InkMaterial {
    #[uniform(0)]
    color: LinearRgba,
    #[uniform(0)]
    pixels: f32,
    #[uniform(0)]
    fade_start: f32,
    #[uniform(0)]
    fade_end: f32,
    #[uniform(0)]
    padding: f32,
}
impl Default for InkMaterial {
    fn default() -> Self {
        Self {
            color: Color::srgb(0.075, 0.071, 0.085).to_linear(),
            pixels: 0.72,
            fade_start: 35.,
            fade_end: 150.,
            padding: 0.,
        }
    }
}
impl InkMaterial {
    pub(crate) fn with_pixels(mut self, pixels: f32) -> Self {
        self.pixels = pixels;
        self
    }
}
impl Material for InkMaterial {
    fn vertex_shader() -> ShaderRef {
        "game/shaders/station_ink.wgsl".into()
    }
    fn fragment_shader() -> ShaderRef {
        "game/shaders/station_ink.wgsl".into()
    }
    fn specialize(
        _: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _: &MeshVertexBufferLayoutRef,
        _: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        descriptor.primitive.cull_mode = Some(Face::Front);
        Ok(())
    }
}
