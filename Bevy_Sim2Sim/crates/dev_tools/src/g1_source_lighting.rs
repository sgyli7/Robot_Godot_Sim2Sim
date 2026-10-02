//! Fixed source RectLight quadrature for one native renderer diagnostic.
//!
//! Bevy has no rectangular emitter here. Four point samples per source panel
//! preserve its measured location, area, color and on-axis luminous intensity.
//! A generated linear cubemap supplies its Lambertian emission envelope.
//! Finite quadrature and raster shadows are declared approximations,
//! not RTX/Lambertian-area-light equivalence. Physics never enters this module.

use std::{fs::File, io::Read, path::Path};

use bevy::{
    asset::RenderAssetUsages,
    camera::primitives::CubemapLayout,
    light::PointLightTexture,
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

const QUERY_SHA256: &str = "d23f8c54bcbdfcc394efb0a23b0273c3c7379152b66361dc773e4733f33a482b";
const LIGHT_PREFIX: &str =
    "/World/envs/env_0/galileo_locomanip/Lights/ceiling_light_bars/RectLight";
const SAMPLES_PER_PANEL: usize = 4;

#[derive(Clone, Debug, Serialize)]
pub(super) struct SourceLightingReceipt {
    pub query_sha256: String,
    pub source_panel_count: usize,
    pub native_point_count: usize,
    pub samples_per_panel: usize,
    pub source_luminance_cd_per_m2: f32,
    pub source_panel_area_m2: f32,
    pub point_luminous_power_parameter_lm: f32,
    pub source_color_linear: [f32; 3],
    pub source_environment_translation: [f32; 3],
    pub angular_emission_approximation: &'static str,
    pub source_renderer_parity_proven: bool,
}

pub(super) struct SourceRectLighting {
    samples: Vec<Transform>,
    pub receipt: SourceLightingReceipt,
}

impl SourceRectLighting {
    pub fn load(path: &Path, expected_sha256: &str) -> Result<Self, String> {
        if expected_sha256 != QUERY_SHA256 {
            return Err("source lighting comparison requires its frozen actual SDK query".into());
        }
        let mut bytes = Vec::new();
        File::open(path)
            .map_err(|e| e.to_string())?
            .take(512 * 1024 + 1)
            .read_to_end(&mut bytes)
            .map_err(|e| e.to_string())?;
        if bytes.len() > 512 * 1024 || format!("{:x}", Sha256::digest(&bytes)) != expected_sha256 {
            return Err("source light query bytes/identity changed".into());
        }
        let query: Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        if query["schema"] != "g1_released_t2_actual_render_inputs_v1"
            || query["runtime_build"] != "6.0.0-rc.22+release.33481.407f3ea1.gl"
            || query["physics_integrations"] != 0
            || query["usd_writes"] != 0
            || query["render_setting_writes"] != 0
        {
            return Err("source lighting query belongs to another scene/runtime".into());
        }
        let lights = query["lights"]
            .as_array()
            .ok_or("missing original lights")?;
        if lights.len() != 8 {
            return Err("source lighting requires all eight original panels".into());
        }
        let mut samples = Vec::new();
        let color = [0.96163684, 0.9597079, 0.94933975];
        let width = 200. * 0.0098;
        let height = 28. * 0.003;
        let area = width * height;
        let luminance = 200_000.;
        // Bevy divides its PointLight power parameter by4pi to obtain candela.
        // L*A/N is the source patch's on-axis candela, not an exposure fit.
        let intensity = 4. * std::f32::consts::PI * luminance * area / SAMPLES_PER_PANEL as f32;
        for (index, light) in lights.iter().enumerate() {
            let suffix = if index == 0 {
                String::new()
            } else {
                format!("_{index:02}")
            };
            if light["path"] != format!("{LIGHT_PREFIX}{suffix}") || light["type"] != "RectLight" {
                return Err("original light paths/order changed".into());
            }
            for (name, value) in [("intensity", luminance), ("width", 200.), ("height", 28.)] {
                let attr = &light["attributes"][name];
                if attr["authored"] != true || number(&attr["value"])? != value as f64 {
                    return Err(format!("legacy source light {name} changed"));
                }
            }
            let source_color = light["attributes"]["color"]["value"]
                .as_array()
                .ok_or("missing light color")?;
            if source_color.len() != 3
                || source_color
                    .iter()
                    .zip(color)
                    .any(|(v, c)| match number(v) {
                        Ok(v) => (v - c as f64).abs() > 1e-7,
                        Err(_) => true,
                    })
            {
                return Err("original warm light color changed".into());
            }
            let matrix = light["world_transform"]
                .as_array()
                .ok_or("missing light transform")?;
            if matrix.len() != 4
                || matrix
                    .iter()
                    .any(|r| r.as_array().is_none_or(|r| r.len() != 4))
            {
                return Err("original light transform dimensions changed".into());
            }
            for row in 0..4 {
                for col in 0..4 {
                    let value = number(&matrix[row][col])?;
                    if row != 3 || col == 3 {
                        let expected = match (row, col) {
                            (0, 0) => 0.0098,
                            (1, 1) => 0.003,
                            (2, 2) => 0.00720000010728836,
                            (3, 3) => 1.,
                            _ => 0.,
                        };
                        if (value - expected).abs() > 1e-12 {
                            return Err("original ceiling light basis changed".into());
                        }
                    }
                }
            }
            let center_source = Vec3::new(
                number(&matrix[3][0])? as f32,
                number(&matrix[3][1])? as f32,
                number(&matrix[3][2])? as f32 + 0.795,
            );
            for u in [-0.25, 0.25] {
                for v in [-0.25, 0.25] {
                    let source = center_source + Vec3::new(u * width, v * height, 0.);
                    let engine = Vec3::new(source.x, source.z, -source.y);
                    samples.push(Transform::from_translation(engine).looking_to(-Vec3::Y, Vec3::Z));
                }
            }
        }
        Ok(Self {
            samples,
            receipt: SourceLightingReceipt {
                query_sha256: expected_sha256.into(),
                source_panel_count: 8,
                native_point_count: 32,
                samples_per_panel: SAMPLES_PER_PANEL,
                source_luminance_cd_per_m2: luminance,
                source_panel_area_m2: area,
                point_luminous_power_parameter_lm: intensity,
                source_color_linear: color,
                source_environment_translation: [0., 0., 0.795],
                angular_emission_approximation: "fixed2x2pointquadrature;64px_linear_lambertian_cube_mask;finite20mrange;raster_shadowmaps;noRTXGI",
                source_renderer_parity_proven: false,
            },
        })
    }

    pub fn spawn(&self, commands: &mut Commands, images: &mut Assets<Image>) {
        let mask = images.add(lambertian_cube_mask());
        for (index, transform) in self.samples.iter().enumerate() {
            let [r, g, b] = self.receipt.source_color_linear;
            commands.spawn((
                Name::new(format!("source_rect_light_sample_{index}")),
                PointLight {
                    color: Color::linear_rgba(r, g, b, 1.),
                    intensity: self.receipt.point_luminous_power_parameter_lm,
                    range: 20.,
                    radius: (self.receipt.source_panel_area_m2 / (4. * std::f32::consts::PI))
                        .sqrt(),
                    shadow_maps_enabled: true,
                    ..default()
                },
                PointLightTexture {
                    image: mask.clone(),
                    cubemap_layout: CubemapLayout::SequenceVertical,
                },
                *transform,
            ));
        }
    }
}

fn lambertian_cube_mask() -> Image {
    const SIZE: u32 = 64;
    let mut bytes = Vec::new();
    // Exact inverse of Bevy's cubemap_uv face order/coordinate convention.
    for face in 0..6 {
        for y in 0..SIZE {
            for x in 0..SIZE {
                let u = 2. * (x as f32 + 0.5) / SIZE as f32 - 1.;
                let v = 2. * (y as f32 + 0.5) / SIZE as f32 - 1.;
                let direction = match face {
                    0 => Vec3::new(1., -v, u),
                    1 => Vec3::new(-1., -v, -u),
                    2 => Vec3::new(u, 1., -v),
                    3 => Vec3::new(u, -1., v),
                    4 => Vec3::new(u, -v, -1.),
                    5 => Vec3::new(u, v, 1.),
                    _ => unreachable!(),
                }
                .normalize();
                let value = ((-direction.z).max(0.) * 255.).round() as u8;
                bytes.extend_from_slice(&[value, value, value, 255]);
            }
        }
    }
    Image::new(
        Extent3d {
            width: SIZE,
            height: SIZE * 6,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        bytes,
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::default(),
    )
}

fn number(value: &Value) -> Result<f64, String> {
    value
        .as_f64()
        .filter(|v| v.is_finite())
        .ok_or_else(|| "nonfinite/missing source light input".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lambertian_mask_emits_forward_without_srgb_or_backlight() {
        let image = lambertian_cube_mask();
        assert_eq!(image.texture_descriptor.format, TextureFormat::Rgba8Unorm);
        let data = image.data.unwrap();
        let center = |face: usize| data[((face * 64 + 32) * 64 + 32) * 4];
        assert_eq!(center(4), 255);
        assert_eq!(center(5), 0);
        assert!((0..4).all(|face| center(face) < 5));
        assert!(data.chunks_exact(4).all(|pixel| pixel[3] == 255));
    }

    #[test]
    #[ignore = "requires frozen original SDK light query; no GPU or physics"]
    fn actual_source_rect_light_area_energy_basis() {
        let path = std::env::var("G1_SOURCE_LIGHT_QUERY").expect("actual query path");
        let light = SourceRectLighting::load(Path::new(&path), QUERY_SHA256).unwrap();
        assert_eq!(light.samples.len(), 32);
        assert!((light.receipt.source_panel_area_m2 - 0.16464).abs() < 1e-7);
        let total_on_axis_cd =
            light.receipt.point_luminous_power_parameter_lm * 4. / (4. * std::f32::consts::PI);
        assert!((total_on_axis_cd - 32928.).abs() < 0.01);
        assert!((light.samples[8].translation.y - 2.94999998).abs() < 1e-6);
        assert!(
            light
                .samples
                .iter()
                .all(|s| s.forward().as_vec3().abs_diff_eq(-Vec3::Y, 1e-6))
        );
        assert!(SourceRectLighting::load(Path::new(&path), &"0".repeat(64)).is_err());
    }
}
