#import bevy_pbr::{forward_io::{Vertex,VertexOutput},mesh_functions,mesh_view_bindings::view}
struct Ink { color: vec4<f32>, pixels: f32, fade_start: f32, fade_end: f32, _padding: f32 }
@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> ink: Ink;
@vertex
fn vertex(v: Vertex) -> VertexOutput {
    var out: VertexOutput;
    let world_from_local=mesh_functions::get_world_from_local(v.instance_index);
    out.world_position=mesh_functions::mesh_position_local_to_world(world_from_local,vec4(v.position,1.));
    out.world_normal=mesh_functions::mesh_normal_local_to_world(v.normal,v.instance_index);
    out.position=view.clip_from_world*out.world_position;
    let n=normalize((view.view_from_world*vec4(out.world_normal,0.)).xyz);
    let projected=(view.clip_from_view*vec4(n,0.)).xy;
    let direction=projected/max(length(projected),.0001);
    let depth=-(view.view_from_world*out.world_position).z;
    let fade=1.-smoothstep(ink.fade_start,ink.fade_end,depth);
    out.position=vec4(out.position.xy+direction*ink.pixels*fade*2./view.viewport.zw*out.position.w,out.position.zw);
#ifdef VERTEX_UVS_A
    out.uv=v.uv;
#endif
#ifdef VERTEX_UVS_B
    out.uv_b=v.uv_b;
#endif
#ifdef VERTEX_OUTPUT_INSTANCE_INDEX
    out.instance_index=v.instance_index;
#endif
    return out;
}
@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> { return ink.color; }
