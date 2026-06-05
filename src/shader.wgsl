struct VertexInput {
    @location(0) pos: vec3<f32>,
    @location(1) dim: u32,
    @location(2) uv: u32,
    @location(3) color: u32,
    @location(4) content_type_with_srgb: u32,
    @location(5) depth: f32,
}

struct VertexOutput {
    @invariant @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) @interpolate(flat) content_type: u32,
};

struct Params {
    screen_resolution: vec2<u32>,
    _pad: vec2<u32>,
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var color_atlas_texture: texture_2d<f32>;

@group(0) @binding(1)
var mask_atlas_texture: texture_2d<f32>;

@group(0) @binding(2)
var atlas_sampler: sampler;

@group(1) @binding(0)
var<uniform> params: Params;

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        return c / 12.92;
    } else {
        return pow((c + 0.055) / 1.055, 2.4);
    }
}

@vertex
fn vs_main(in_vert: VertexInput) -> VertexOutput {
    let color = in_vert.color;
    let uv = vec2<u32>(in_vert.uv & 0xffffu, (in_vert.uv & 0xffff0000u) >> 16u);

    var vert_output: VertexOutput;

    // Position is pre-transformed on CPU: pos.xy = clip x/y, pos.z = clip w, depth = clip z
    // Keep text depth centered to avoid clipping when camera zooms very close.
    vert_output.position = vec4<f32>(in_vert.pos.xy, 0.0, in_vert.pos.z);

    let content_type = in_vert.content_type_with_srgb & 0xffffu;
    let srgb = (in_vert.content_type_with_srgb & 0xffff0000u) >> 16u;

    switch srgb {
        case 0u: {
            vert_output.color = vec4<f32>(
                f32((color & 0x00ff0000u) >> 16u) / 255.0,
                f32((color & 0x0000ff00u) >> 8u) / 255.0,
                f32(color & 0x000000ffu) / 255.0,
                f32((color & 0xff000000u) >> 24u) / 255.0,
            );
        }
        case 1u: {
            vert_output.color = vec4<f32>(
                srgb_to_linear(f32((color & 0x00ff0000u) >> 16u) / 255.0),
                srgb_to_linear(f32((color & 0x0000ff00u) >> 8u) / 255.0),
                srgb_to_linear(f32(color & 0x000000ffu) / 255.0),
                f32((color & 0xff000000u) >> 24u) / 255.0,
            );
        }
        default: {}
    }

    var atlas_dim: vec2<u32> = vec2(0u);
    switch content_type {
        case 0u: {
            atlas_dim = textureDimensions(color_atlas_texture);
            break;
        }
        case 1u: {
            atlas_dim = textureDimensions(mask_atlas_texture);
            break;
        }
        default: {}
    }

    vert_output.content_type = content_type;

    // UVs are provided as integer texel coordinates; offset to texel centers.
    vert_output.uv = (vec2<f32>(uv) + vec2<f32>(0.5, 0.5)) / vec2<f32>(atlas_dim);

    return vert_output;
}

@fragment
fn fs_main(in_frag: VertexOutput) -> @location(0) vec4<f32> {
    switch in_frag.content_type {
        case 0u: {
            return textureSampleLevel(color_atlas_texture, atlas_sampler, in_frag.uv, 0.0);
        }
        case 1u: {
            return vec4<f32>(in_frag.color.rgb, in_frag.color.a * textureSampleLevel(mask_atlas_texture, atlas_sampler, in_frag.uv, 0.0).x);
        }
        default: {
            return vec4<f32>(0.0);
        }
    }
}
