// Bounded original OmniPBR albedo offset, evaluated before PBR lighting.
// Original texture lookup is linear; the scalar is added before color tint.
#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::{alpha_discard, apply_pbr_lighting, main_pass_post_lighting_processing},
    forward_io::{VertexOutput, FragmentOutput},
}

struct BackgroundAlbedo {
    tint_and_add: vec4<f32>,
}
@group(#{MATERIAL_BIND_GROUP}) @binding(100) var<uniform> albedo: BackgroundAlbedo;

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    var pbr = pbr_input_from_standard_material(in, is_front);
    pbr.material.base_color = vec4(
        max(pbr.material.base_color.rgb + albedo.tint_and_add.rgb * albedo.tint_and_add.a, vec3(0.)),
        pbr.material.base_color.a,
    );
    pbr.material.base_color = alpha_discard(pbr.material, pbr.material.base_color);
    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr);
    out.color = main_pass_post_lighting_processing(pbr, out.color);
    return out;
}
