// SPDX-FileCopyrightText: 2011-2022 Blender Foundation
// SPDX-License-Identifier: Apache-2.0

/* M4 kernel ABI shared between the C++ host (vulkan_pipeline.cpp) and the
 * Slang kernels (path_trace.slang).
 *
 * Layout rules for this struct (std430, scalar-packed is NOT used; we keep
 * explicit natural alignment so both sides agree without surprises):
 *   - every field is uint/float sized (4 bytes) unless noted
 *   - device pointers are stored as two u32 halves (lo, hi) because SPIR-V
 *     PhysicalStorageBuffer 64-bit addresses must be constructed from parts
 *     when coming from a plain storage buffer on many drivers
 *   - total size MUST stay a multiple of 16; pad with `reserved` at the end
 *
 * If you add a field here, mirror it in VulkanKernelParams in
 * src/device/vulkan/vulkan_pipeline.h. The static_assert at the bottom of
 * that header catches size drift at compile time. */

struct VulkanKernelParams
{
    // --- film ----------------------------------------------------------
    uint film_width;          // full image width in pixels
    uint film_height;         // full image height in pixels
    uint tile_offset_x;       // left edge of current tile
    uint tile_offset_y;       // top edge of current tile
    uint tile_width;          // current tile width  (<= TILE_SIZE)
    uint tile_height;         // current tile height (<= TILE_SIZE)
    uint samples_per_pixel;   // path budget per pixel
    uint max_bounces;         // diffuse-only M4 uses 1..2
    uint64_t film_buffer;     // float4 accumulation buffer, tile-local layout
                              // index = (y * tile_width + x)

    // --- camera --------------------------------------------------------
    float3 cam_origin;        // camera position (world space)
    float  cam_aspect;        // width / height
    float3 cam_forward;       // normalized view direction
    float  cam_focal_length;  // focal length in mm (sensor fit applied)
    float3 cam_right;         // normalized right vector
    float  cam_sensor_width;  // sensor width in mm
    float3 cam_up;            // normalized up vector
    float  pad_cam;

    // --- scene ---------------------------------------------------------
    uint64_t bvh_nodes;       // BVH2Node array (Cycles layout, see bvh2.slang)
    uint64_t bvh_prim_offsets;// uint per interior node: first primitive of subtree
    uint64_t bvh_prim_count;  // uint per interior node: primitive count
    uint64_t tri_indices;     // make_uint3 per triangle (vertex indices)
    uint64_t vert_co;         // float3 per vertex
    uint64_t object_transforms; // float4x4-ish: 12 floats (row-major 3x4) per object
    uint64_t light_prim_offsets; // uint per light: first primitive
    uint64_t light_prim_counts;  // uint per light: primitive count
    uint64_t light_emission;     // float3 per primitive (triangle emission color)

    uint  num_triangles;
    uint  num_objects;
    uint  num_lights;
    uint  pad_scene;

    // --- world ---------------------------------------------------------
    float3 bg_color;          // background radiance (constant environment)
    float  pad_bg;

    // --- rng seed ------------------------------------------------------
    uint rng_seed;            // per-render pass base seed
    uint reserved_0;
    uint reserved_1;
    uint reserved_2;
};

/* Tile size used by the kernel. Must match TILE_SIZE in vulkan_pipeline.h. */
static const uint VULKAN_TILE_SIZE = 64u;
