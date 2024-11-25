struct GlobalsUniform {
    time: f32,
}

struct Camera {
    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
}

struct Light {
    position: vec3<f32>,
    color: vec3<f32>,
}

struct InstanceInput {
    @location(5) model_matrix_0: vec4<f32>,
    @location(6) model_matrix_1: vec4<f32>,
    @location(7) model_matrix_2: vec4<f32>,
    @location(8) model_matrix_3: vec4<f32>,
    @location(9) normal_matrix_0: vec3<f32>,
    @location(10) normal_matrix_1: vec3<f32>,
    @location(11) normal_matrix_2: vec3<f32>,
}

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) tangent: vec3<f32>,
    @location(4) bitangent: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) world_position: vec3<f32>,
}

@group(3) @binding(0)
var<uniform> globals: GlobalsUniform;

@group(2) @binding(0)
var<uniform> light: Light;

@group(1) @binding(0)
var<uniform> camera: Camera;

// Vertex shader

@vertex
fn vs_main(
    model: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    let model_matrix = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3,
    );
    let normal_matrix = mat3x3<f32>(
        instance.normal_matrix_0,
        instance.normal_matrix_1,
        instance.normal_matrix_2,
    );
    var out: VertexOutput;
    var world_position: vec4<f32> = model_matrix * vec4<f32>(model.position, 1.0);
    
    out.tex_coords = model.tex_coords;
    out.world_normal = normal_matrix * model.normal;
    out.world_position = world_position.xyz;
    out.clip_position = camera.view_proj * world_position;
    
    return out;
}

// Fragment shader
@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;
@group(0) @binding(2)
var t_normal: texture_2d<f32>;
@group(0) @binding(3)
var s_normal: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let texture_color: vec4<f32> = textureSample(t_diffuse, s_diffuse, in.tex_coords);
    
    let light_direction = light.position - in.world_position;

    let normal = normalize(in.world_normal.xyz);

    let unit_light_direction = normalize(light_direction);
    let eye_direction = normalize(-(in.world_position.xyz - camera.view_pos.xyz));
    let reflected_direction = normalize(-reflect(unit_light_direction, normal));

    let diffuse_intensity = dot(normal, unit_light_direction);

    let diffuseTemp = vec4<f32>(clamp(texture_color.xyz * light.color * diffuse_intensity, vec3<f32>(0.0), vec3<f32>(1.0)), texture_color.a);

    let result = (diffuseTemp) * texture_color;

    return result;
}