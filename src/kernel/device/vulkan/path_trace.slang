// SPDX-FileCopyrightText: 2011-2022 Blender Foundation
// SPDX-License-Identifier: Apache-2.0

/* M4 monolithic path-tracing entry point (one workgroup per tile).
 *
 * Dispatch contract (see vulkan_pipeline.cpp):
 *   gl_WorkGroupSize = (8, 8, 1)
 *   one thread per pixel of the current tile; grid = ceil(tile/8)^2 groups
 *   The kernel loops `samples_per_pixel` times internally and accumulates
 *   radiance into film_buffer as float4 (rgb + weight), matching Cycles'
 *   tile accumulation format so mem_copy_from can blit it straight out.
 *
 * Scope for first image: camera rays -> BVH2 closest hit -> constant world
 * background on miss / grey lambert-ish shading stub on hit. No BSDFs yet
 * (M5-M6). This keeps the pipeline surface minimal while exercising the
 * whole dispatch + film readback path. */

#include "vulkan_data.slang"
#include "bvh2.slang"

/* Parameter buffer: single push-constant-style storage buffer bound at
 * set 0, binding 0. Using a storage buffer (not uniform) because we exceed
 * the 16KB uniform limit is not a concern here (~300 bytes) but pointer
 * chasing with PhysicalStorageBuffer needs addresses anyway. */
ParameterBlock<VulkanKernelParams> params : register(b0);

/* PCG32 RNG — same stream layout as the spike so CPU reference matches. */
struct Pcg32
{
    uint64 state;
    uint64 inc;

    void init(uint64 seed, uint64 seq)
    {
        state = 0u;
        inc = (seq << 1u) | 1u;
        next();
        state += seed;
        next();
    }

    uint next()
    {
        uint64 oldstate = state;
        state = oldstate * 6364136223846793005ULL + inc;
        uint xorshifted = (uint)(((oldstate >> 18u) ^ oldstate) >> 27u);
        uint rot = (uint)(oldstate >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31u));
    }

    float next_float()
    {
        /* 24-bit mantissa precision is plenty for path tracing. */
        return float(next() >> 8) * (1.0f / 16777216.0f);
    }
};

float3 camera_ray_direction(float2 pixel_center)
{
    /* Normalized device coords in [-1,1] scaled by aspect, then pushed through
     * focal length / sensor like Cycles' camera.cu generate_camera_ray. */
    float ndc_x = pixel_center.x * 2.0f - 1.0f;
    float ndc_y = pixel_center.y * 2.0f - 1.0f;

    /* half frame size in mm mapped to world units at focal distance */
    float scale = cam_sensor_half_width();
    float3 dir = normalize(cam_forward
                         + ndc_x * scale * params.cam_aspect * cam_right
                         + ndc_y * scale * cam_up);
    return dir;
}

float cam_sensor_half_width()
{
    /* sensor fit horizontal assumed (matches instance2.xml camera). */
    return 0.5f * params.cam_sensor_width / params.cam_focal_length;
}

float3 shade_stub(RayGPU ray, uint prim)
{
    /* M4 stub: hit anything -> flat albedo 0.5 grey lit by a fake hemisphere
     * term from the geometric normal. Replaced by real BSDF sampling in M5. */
    let tris = cast<uint4[]>(params.tri_indices);
    let verts = cast<float4[]>(params.vert_co);
    uint4 idx = tris[prim];
    float3 v0 = verts[idx.x].xyz;
    float3 v1 = verts[idx.y].xyz;
    float3 v2 = verts[idx.z].xyz;
    float3 n = normalize(cross(v1 - v0, v2 - v0));
    float hemi = 0.5f + 0.5f * dot(n, -ray.dir);
    return float3(0.5f) * hemi;
}

[numthreads(8, 8, 1)]
compute void path_trace_kernel()
{
    uint tx = gl_GlobalInvocationID.x;
    uint ty = gl_GlobalInvocationID.y;

    if (tx >= params.tile_width || ty >= params.tile_height)
        return;

    uint film_idx = ty * params.tile_width + tx;
    let film = cast<float4[]>(params.film_buffer);

    float3 acc = float3(0.0f, 0.0f, 0.0f);

    Pcg32 rng;
    uint64 seed = ((uint64)params.rng_seed << 32u)
                | (uint64)((ty * params.film_width + tx) * 1973u + 911u);
    rng.init(seed, ((uint64)tx << 32u) | (uint64)ty);

    for (uint s = 0; s < params.samples_per_pixel; ++s)
    {
        /* Sub-pixel jitter within this pixel. Pixel centers are computed in
         * top-left origin space; Cycles flips Y when writing the final image,
         * we keep flip here too so readback matches CPU reference directly. */
        float jx = (float(tx) + rng.next_float()) / float(params.tile_width);
        float jy = (float(ty) + rng.next_float()) / float(params.tile_height);
        /* global pixel coord including tile offset, normalized over full film */
        float px = (float(params.tile_offset_x) + float(tx) + rng.next_float()) / float(params.film_width);
        float py = (float(params.tile_offset_y) + float(ty) + rng.next_float()) / float(params.film_height);

        RayGPU ray;
        ray.origin = params.cam_origin;
        ray.dir = camera_ray_direction(float2(px, 1.0f - py)); // flip Y
        ray.inv_dir = float3(1.0f / ray.dir.x, 1.0f / ray.dir.y, 1.0f / ray.dir.z);
        ray.tmin = 1e30f;
        ray.prim = INVALID_PRIM;

        bvh2_intersect(ray, params.bvh_nodes, params.bvh_prim_offsets,
                       params.bvh_prim_count, params.tri_indices, params.vert_co);

        if (ray.prim != INVALID_PRIM)
            acc += shade_stub(ray, ray.prim);
        else
            acc += params.bg_color;
    }

    float w = float(params.samples_per_pixel);
    film[film_idx] = float4(acc, w);
}
