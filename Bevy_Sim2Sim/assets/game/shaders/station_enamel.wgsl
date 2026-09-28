#import bevy_pbr::{pbr_fragment::pbr_input_from_standard_material, pbr_functions::{alpha_discard, apply_pbr_lighting, main_pass_post_lighting_processing}, forward_io::{VertexOutput, FragmentOutput}, mesh_view_bindings::view}

struct StationPigment {
    hatch_strength: f32, material_kind: u32, _padding: vec2<f32>,
    plaza: vec4<f32>, loop_center_radii: vec4<f32>, berth: vec4<f32>, skills: vec4<f32>,
    paths: array<vec4<f32>, 8>, loop_width: f32, path_count: u32, tail_padding: vec2<f32>, ridge_waypoints: array<vec4<f32>,4>, ridge_width: f32, ridge_padding: vec3<f32>
}
@group(#{MATERIAL_BIND_GROUP}) @binding(100) var<uniform> pigment: StationPigment;

fn rect(p: vec2<f32>, c: vec2<f32>, h: vec2<f32>, r: f32) -> f32 {
    let q=abs(p-c)-h+vec2(r);
    return length(max(q,vec2(0.)))+min(max(q.x,q.y),0.)-r;
}
fn segment(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let d=b-a; return length(p-a-d*clamp(dot(p-a,d)/dot(d,d),0.,1.));
}
fn aa_edge(d: f32) -> f32 { let w=max(fwidth(d)*.7,.001); return smoothstep(-w,w,d); }
fn ink(d: f32, width: f32) -> f32 { return 1.-smoothstep(width,width+max(fwidth(d),.002),abs(d)); }
fn hash2(p: vec2<f32>) -> f32 { return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453); }
fn grain(p: vec2<f32>) -> f32 {
    let cell=floor(p); let t=fract(p); let u=t*t*(3.-2.*t);
    return mix(mix(hash2(cell),hash2(cell+vec2(1.,0.)),u.x),mix(hash2(cell+vec2(0.,1.)),hash2(cell+vec2(1.,1.)),u.x),u.y);
}
fn ground_color(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let q=p.xz;
    // Continuous analytic washes avoid the cell and slope thresholds that
    // produced conspicuous polygonal patches in the previous sand pigment.
    let broad=.50+.18*sin(q.x*.031+q.y*.016)+.12*sin(q.y*.023-q.x*.011+1.7);
    var color=mix(vec3(.79,.34,.12),vec3(.94,.55,.25),broad);
    // Broad, continuous mineral fade gives the open desert a near/mid/far
    // hierarchy without putting any object or wall on a traversable surface.
    let site_radius=length(q-vec2(0.,-3.5));
    let far_dust=smoothstep(18.,78.,site_radius);
    color=mix(color,vec3(.80,.64,.53),far_dust*.50);
    let wash=.5+.5*sin(q.x*.067-q.y*.035+sin(q.y*.018)*1.8);
    color=mix(color,vec3(.93,.61,.30),wash*.12);
    let ripple=sin(q.x*2.1+q.y*.22+sin(q.y*.062)*1.7);
    color*=1.-ink(ripple,.009)*.010*(1.-smoothstep(.2,.7,fwidth(ripple)));
    let c=pigment.loop_center_radii;
    let radial=abs(length((q-c.xy)/c.zw)-1.)*min(c.z,c.w);
    var road=radial-pigment.loop_width*.5;
    for (var i=0u;i<pigment.path_count;i++) { let path=pigment.paths[i]; road=min(road,segment(q,path.xy,path.zw)-1.3); }
    for (var i=0u;i<3u;i++) { road=min(road,segment(q,pigment.ridge_waypoints[i].xz,pigment.ridge_waypoints[i+1u].xz)-pigment.ridge_width*.5); }
    let plaza=rect(q,pigment.plaza.xy,pigment.plaza.zw,.65);
    let berth=rect(q,pigment.berth.xy,pigment.berth.zw,.3);
    // The wing door gets a narrow flush apron onto the old court ground.
    // This changes pigment only, never support height or a collision face.
    let wing_apron=segment(q,vec2(-7.8,-13.9),vec2(-7.8,-12.3))-1.12;
    let prepared=min(min(road,plaza),min(berth,wing_apron));
    let slab=1.-aa_edge(prepared);
    let mottling=(sin(q.x*.19+q.y*.087)+sin(q.y*.23-q.x*.055))*.004;
    let enamel=vec3(.84,.80,.67)+vec3(mottling);
    color=mix(color,enamel,slab);
    // Compact surface washes gather existing equipment without new geometry.
    let sample_yard=rect(q,vec2(-16.2,3.5),vec2(4.2,2.8),.65);
    let survey_yard=rect(q,vec2(11.8,-18.0),vec2(7.0,4.8),.9);
    let service_yard=rect(q,vec2(18.6,1.8),vec2(3.2,2.7),.55);
    let work_yard=min(sample_yard,min(survey_yard,service_yard));
    let yard=(1.-smoothstep(-.28,.72,work_yard))*(1.-slab);
    color=mix(color,vec3(.88,.69,.46),yard*.19);
    let yard_edge=ink(work_yard,.02)*(1.-slab);
    color=mix(color,vec3(.44,.49,.46),yard_edge*.07);
    // Short role-colored work bars sit beside existing equipment only.
    let sample_bar=ink(q.y-6.35,.095)*(1.-smoothstep(3.15,3.45,abs(q.x+16.1)))*(1.-slab);
    let survey_bar=ink(q.y+19.8,.095)*(1.-smoothstep(2.45,2.75,abs(q.x-7.7)))*(1.-slab);
    let service_bar=ink(q.y+1.30,.095)*(1.-smoothstep(2.45,2.75,abs(q.x-18.7)))*(1.-slab);
    let field_bar=ink(q.y-9.6,.085)*(1.-smoothstep(2.0,2.4,abs(q.x+8.2)))*(1.-slab);
    color=mix(color,vec3(.47,.37,.55),sample_bar*.29);
    color=mix(color,vec3(.61,.55,.31),survey_bar*.27);
    color=mix(color,vec3(.27,.49,.59),service_bar*.30);
    color=mix(color,vec3(.33,.53,.57),field_bar*.25);
    // The quiet tint is restricted to the two side branches of the existing
    // 3 m lane; the action court and its entry remain open and unpainted.
    let side_mask=smoothstep(5.5,7.0,abs(q.x));
    let side_lane=(1.-aa_edge(abs(radial)-1.32))*slab*side_mask;
    color=mix(color,vec3(.70,.76,.73),side_lane*.11);
    let loop_trace=ink(radial-1.12,.018)*slab*side_mask;
    color=mix(color,vec3(.51,.62,.63),loop_trace*.15);
    // Only an apron and four quiet corner brackets mark the 11 x 9 m
    // action court. The interior stays visually and physically empty.
    let action=rect(q,pigment.skills.xy,pigment.skills.zw,.55);
    let apron=(1.-aa_edge(action-1.05))*aa_edge(action);
    color=mix(color,vec3(.71,.76,.75),apron*slab*.27);
    let corner=ink(action-.78,.03)
      *smoothstep(4.20,5.0,abs(q.x-pigment.skills.x))
      *smoothstep(2.9,3.8,abs(q.y-pigment.skills.y));
    color=mix(color,vec3(.31,.53,.63),corner*slab*.43);
    // A few construction joints serve the main entrance; no tiled grid or
    // skill-area outline is painted through the open action space.
    let entrance_joint=ink(q.y+9.6,.009)*(1.-aa_edge(plaza))*(1.-smoothstep(6.0,8.0,abs(q.x)));
    let side_joint=ink(q.x+4.4,.008)*(1.-smoothstep(-8.0,-6.5,q.y))*(1.-aa_edge(plaza));
    color=mix(color,vec3(.46,.43,.36),max(entrance_joint,side_joint)*.11);
    let bay=ink(rect(q,pigment.berth.xy,pigment.berth.zw-vec2(.22),.25),.022)*smoothstep(1.8,2.4,abs(q.y-pigment.berth.y));
    color=mix(color,vec3(.10,.33,.46),bay*.27);
    return color;
}

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    var pbr=pbr_input_from_standard_material(in,is_front);
    let n=normalize(in.world_normal);
    let p=in.world_position.xyz;
    var q=p.xy;
    if abs(n.y)>abs(n.x) && abs(n.y)>abs(n.z) { q=p.xz; }
    else if abs(n.x)>abs(n.z) { q=p.zy; }
    let coordinate=(q.x+q.y*.62)*85.;
    let footprint=fwidth(coordinate);
    let stroke=1.-smoothstep(.035,.035+footprint,abs(fract(coordinate)-.5));
    let fade=1.-smoothstep(.16,.55,footprint);
    let shade=1.-smoothstep(.1,.65,dot(n,normalize(vec3(-.394,.743,.541))));
    var base=pbr.material.base_color;
    if pigment.material_kind==1u { base=vec4(ground_color(p,n),1.); }
    if pigment.material_kind==2u {
        let eye=normalize(view.world_position-p);
        let rim=pow(1.-abs(dot(n,eye)),3.);
        base=vec4(mix(vec3(.018,.025,.045),vec3(.14,.22,.30),rim),1.);
        pbr.material.emissive=vec4(vec3(.15,.24,.28)*pow(max(dot(reflect(-eye,n),normalize(vec3(-.4,.8,.3))),0.),90.)*.28,0.);
    }
    if pigment.material_kind==3u {
        let eye=normalize(p-view.world_position);
        let h=smoothstep(-.03,.48,eye.y);
        var sky: FragmentOutput;
        sky.color=vec4(mix(vec3(.17,.48,.70),vec3(.012,.14,.43),h),1.);
        return sky;
    }
    if pigment.material_kind!=1u { base=vec4(base.rgb*(1.-stroke*fade*shade*pigment.hatch_strength),base.a); }
    pbr.material.base_color=alpha_discard(pbr.material,base);
    var out: FragmentOutput;
    out.color=apply_pbr_lighting(pbr);
    // Preserve shadow attenuation while grouping the illuminated enamel into quiet values.
    let ratio=dot(out.color.rgb,vec3(.2126,.7152,.0722))/max(dot(base.rgb,vec3(.2126,.7152,.0722)),.005);
    let middle=smoothstep(.12,.42,ratio);
    let bright=smoothstep(.42,.78,ratio);
    let band=.19+.43*middle+.30*bright;
    let cool=mix(vec3(.73,.73,.88),vec3(1.,.98,.92),bright);
    out.color=vec4(base.rgb*band*cool,out.color.a);
    if pigment.material_kind==4u {
        let distance=length(view.world_position-p);
        let haze=smoothstep(30.,140.,distance)*.48;
        out.color=vec4(mix(out.color.rgb*.77,vec3(.40,.39,.36),haze),out.color.a);
    }
    out.color=main_pass_post_lighting_processing(pbr,out.color);
    return out;
}
