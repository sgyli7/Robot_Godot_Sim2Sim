#import bevy_pbr::{pbr_fragment::pbr_input_from_standard_material, pbr_functions::alpha_discard, forward_io::{VertexOutput, FragmentOutput}, mesh_view_bindings::view, mesh_functions}

struct UriPigment { parameters: vec4<f32>, plaza: vec4<f32> }
@group(#{MATERIAL_BIND_GROUP}) @binding(100) var<uniform> pigment: UriPigment;

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    let pbr = pbr_input_from_standard_material(in, is_front);
    let base = alpha_discard(pbr.material, pbr.material.base_color);
    let p = in.world_position.xyz;
    // Source meter-scale meshes can contain zero/invalid smooth normals. Never
    // normalize those into NaNs: derive the same triangle's geometric normal.
    let geometric = cross(dpdx(p), dpdy(p));
    var normal = in.world_normal;
    if !(dot(normal, normal) > 1e-8) {
        normal = geometric * select(-1., 1., is_front);
    }
    var n = vec3(0., 1., 0.);
    if dot(normal, normal) > 1e-20 {
        n = normalize(normal);
    }
    let eye = normalize(view.world_position - p);
    let kind = pigment.parameters.x;
    var color = base.rgb;
    if kind == 1. {
        // Flush floor pigment only: no additional floor or physical surface.
        let q = p.xz;
        let distance = abs(fract(q + vec2(.5)) - vec2(.5));
        let width = max(fwidth(q), vec2(.001));
        let line = 1. - min(smoothstep(.008, .008 + width.x, distance.x), smoothstep(.008, .008 + width.y, distance.y));
        let inside = all(abs(q - pigment.plaza.xy) < pigment.plaza.zw) && n.y > .94;
        if inside {
            color = mix(base.rgb, vec3(.36, .43, .49), line * .28);
        } else {
            color = vec3(.76, .42, .16) * (.8 + .2 * max(n.y, 0.));
        }
    } else if kind == 2. {
        let height = smoothstep(-.05, .6, normalize(p - view.world_position).y);
        color = mix(vec3(.54, .72, .84), vec3(.08, .29, .54), height);
    } else if kind != 3. {
        let key = normalize(vec3(-.45, .8, .55));
        let diffuse = dot(n, key);
        let shade = mix(.76 + .24 * max(diffuse, 0.), .91 + .09 * max(diffuse, 0.), pigment.parameters.y);
        let edge = smoothstep(.02, .23, abs(dot(n, eye)));
        let highlight = pow(max(dot(n, normalize(key + eye)), 0.), 64.);
        color = base.rgb * shade * mix(.93, 1., edge) + vec3(highlight * .035);
        if kind == 5. || kind == 6. || kind == 7. {
            // Neutral factory finishes, with restrained sky and warm ground reflections.
            // The blue belongs to reflected environment light, never the body pigment.
            let sky = max(dot(n, normalize(vec3(.65, .25, -.7))), 0.);
            let bounce = max(dot(n, normalize(vec3(-.5, -.6, .4))), 0.);
            let fresnel = pow(1. - abs(dot(n, eye)), 3.);
            color += vec3(.012, .035, .060) * sky * (.25 + .75 * fresnel)
                + vec3(.022, .012, .003) * bounce * fresnel;
        }
        if kind == 5. {
            // Pigment on the existing front face; no visor mesh or changed silhouette.
            let local = (mesh_functions::get_local_from_world(in.instance_index) * in.world_position).xyz;
            let face = smoothstep(.035, .048, local.x)
                * smoothstep(.344, .354, local.y)
                * (1. - smoothstep(.472, .483, local.y));
            color = mix(color, vec3(.012, .023, .034) + vec3(highlight * .09), face);
        }
        if kind == 4. {
            color += vec3(.03, .09, .14) * pow(1. - abs(dot(n, eye)), 3.);
        }
    }
    var out: FragmentOutput;
    // Spectator pigment is independent of source light/exposure calibration.
    out.color = vec4(color, base.a);
    return out;
}
