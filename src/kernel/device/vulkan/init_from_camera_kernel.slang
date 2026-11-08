// SPDX-FileCopyrightText: 2011-2022 Blender Foundation
// SPDX-License-Identifier: Apache-2.0

/* init_from_camera_kernel.slang - the PRODUCTION entry point of the
 * DEVICE_KERNEL_INTEGRATOR_INIT_FROM_CAMERA device kernel (M4f slice 5,
 * dispatch plumbing). New file: the slices 1-4 probe SPVs include
 * init_from_camera.slang but NOT this file, so their bytes and gates stay
 * untouched.
 *
 * Cycles source (verbatim contract): src/kernel/device/gpu/kernel.h:72-114
 *
 *   ccl_gpu_kernel(...) ccl_gpu_kernel_signature(integrator_init_from_camera,
 *       ccl_global KernelWorkTile *tiles, const int num_tiles,
 *       ccl_global float *render_buffer, const int max_tile_work_size,
 *       const ccl_global int *path_index_array, const int num_active_paths)
 *   {
 *     const int work_index = ccl_gpu_global_id_x();
 *     if (tiles) {                                   // -> has_tiles flag
 *       if (work_index >= max_tile_work_size * num_tiles) return;
 *       const int tile_index = work_index / max_tile_work_size;
 *       const int tile_work_index = work_index - tile_index * max_tile_work_size;
 *       const ccl_global KernelWorkTile *tile = &tiles[tile_index];
 *       if (tile_work_index >= tile->work_size) return;
 *       const int state = tile->path_index_offset + tile_work_index;
 *       uint x, y, sample;
 *       get_work_pixel(tile, tile_work_index, &x, &y, &sample);
 *       integrator_init_from_camera(nullptr, state, tile, render_buffer, x, y, sample);
 *     }
 *     else {                                        // -> has_path_index_array flag
 *       if (work_index >= num_active_paths) return;
 *       const int state = (path_index_array) ? path_index_array[work_index] : work_index;
 *       integrator_init_from_camera(nullptr, state, nullptr, render_buffer, 0, 0, 0);
 *     }
 *   }
 *
 * Mapping (AGENTS.md sections 4-5):
 *   - The host marshals the six enqueue() arguments into InitPush exactly
 *     as PathTraceWorkGPU passes them (source: path_trace_work_gpu.cpp:
 *     942-945; DeviceKernelArguments types at device/queue.h:23-58).
 *   - `if (tiles)` / `path_index_array ?` become HOST-WIRED uint flags
 *     (has_tiles, has_path_index_array), each set from the very pointer the
 *     host passes. A shader-side pointer==null comparison lowers to
 *     OpConvertPtrToU and crashes SwiftShader (film_queue.slang header
 *     constraint 3); the flag cannot disagree with the pointer because the
 *     same code computes both - same spelling as render_buffer_valid.
 *   - kg -> DispatchParams (host staging buffer, pointer in push). The host
 *     fills it from its TYPED KernelData (host_kernel_data(),
 *     device_impl.h:70) - the shader never parses the opaque 2560-byte
 *     blob (vulkan_data.slang:35-43; AGENTS.md section 4 opaque-blob rule).
 *     Fields: the MathParams camera/rng subset (math_core.slang), the 5
 *     film pass offsets + kernel_features + background.{object_index,
 *     volume_shader} (the body's explicit cited scalars, init_from_camera.
 *     slang:187-196) and render_buffer_valid (platform constraint 3).
 *     scrambling_distance rides in MathParams (mp) and feeds get_work_pixel.
 *   - render_buffer travels TWICE (float* reads, uint* atomics): platform
 *     constraint 4, film_queue.slang header - no in-shader pointer bitcast.
 *   - tile -> the KernelWorkTile mirror below (offsets locked by host
 *     static_asserts in device/vulkan/kernels.cpp).
 *   - The body is the slices-1..4 gated port (init_from_camera.slang:223).
 *
 * Launch config: [numthreads(64,1,1)], grid = ceil(work_size/64) where
 * work_size = max_tile_work_size * num_tiles (the enqueue() n). 64 = this
 * backend's entry width; every branch guards its work_index, so the block
 * width cannot change the result (see integrator_reset.slang header).
 *
 * Host side: loaded from CYCLES_VULKAN_INIT_FROM_CAMERA_KERNEL_SPV, else
 * ./init_from_camera_kernel.spv, else
 * src/kernel/device/vulkan/init_from_camera_kernel.spv (probe loading
 * precedent). Compiled by hand with slangc (NOT part of the build);
 * SPIR-V entry point name is "main" (kernels.cpp:451-456). */

#include "init_from_camera.slang"  // the gated body (slice 4) + its
                                   // dependencies: math_core (MathParams +
                                   // sampler + camera_sample), integrator_state,
                                   // path_state, film_queue

/* KernelWorkTile mirror. source: src/kernel/types.h:1688-1701. Eleven
 * 4-byte members = 44 bytes, no padding, alignment 4 - every member is a
 * uint/int so host and Slang cannot disagree on layout (AGENTS.md section
 * 4). Offsets locked by static_asserts in device/vulkan/kernels.cpp. */
struct KernelWorkTile
{
    uint x;                    // offset   0
    uint y;                    // offset   4
    uint w;                    // offset   8
    uint h;                    // offset  12
    uint start_sample;         // offset  16
    uint num_samples;          // offset  20
    uint sample_offset;        // offset  24
    int  offset;               // offset  28  (KernelWorkTile.offset)
    uint stride;               // offset  32
    int  path_index_offset;    // offset  36
    int  work_size;            // offset  40
};                             // size 44

/* DispatchParams: the kg-derived scalars the entry + body need, staged by
 * the host per dispatch (host fill in device/vulkan/kernels.cpp, layout
 * locked there with static_asserts). MathParams first (its offset-0..
 * layout is the shared probe mirror); the appended body scalars are the
 * EXACT tail of the body signature (init_from_camera.slang:230-240), so
 * the entry can forward them 1:1. */
struct DispatchParams
{
    MathParams mp;                       // 256 B (math_core.slang) - carries
                                         // scrambling_distance too (appended
                                         // integrator sampler subset), which
                                         // get_work_pixel reads from mp below
    int pass_stride;                     // kernel_data.film.pass_stride
    int pass_adaptive_aux_buffer;        // kernel_data.film.pass_adaptive_aux_buffer
    int pass_sample_count;               // kernel_data.film.pass_sample_count
    int pass_volume_majorant;            // kernel_data.film.pass_volume_majorant
    int pass_volume_majorant_sample_count; // kernel_data.film.pass_volume_majorant_sample_count
    uint kernel_features;                // kernel_data.kernel_features (RAW, no probe overrides)
    int background_object_index;         // kernel_data.background.object_index
    int background_volume_shader;        // kernel_data.background.volume_shader
    uint render_buffer_valid;            // 1: the session always passes the real
                                         // buffer (platform constraint 3)
};                                       // 256 + 9 * 4 = 292 payload bytes
                                         // (host mirror sizeof = 296: tail pad
                                         // to the 8-byte alignment the pointer
                                         // member brings)

/* Push block: 6 marshalled args + params/staging pointers + counts/flags =
 * 72 bytes (< 128-byte spec minimum). */
struct InitPush
{
    KernelParamsVulkan* params;          //   0: ist = &params->integrator_state
    KernelWorkTile*     tiles;           //   8: enqueue() arg 1 (BDA)
    float*              render_buffer;   //  16: enqueue() arg 3, float view
    uint*               render_buffer_u; //  24: arg 3 again, uint view (constraint 4)
    DispatchParams*     dp;              //  32: host staging (kg-derived scalars)
    int*                path_index_array;//  40: enqueue() arg 5 (often 0)
    int                 num_tiles;       //  48: arg 2
    int                 max_tile_work_size; // 52: arg 4
    int                 num_active_paths;//  56: arg 6
    uint                has_tiles;       //  60: host 1:1 from tiles != 0
    uint                has_path_index_array; // 64: host 1:1 from arg 5 != 0
    uint                pad0;            //  68: -> 72 bytes total
};                                       // 6*8 + 6*4 = 72 (locked by host
                                         // static_assert)

[[vk::push_constant]]
ConstantBuffer<InitPush> pc;

/* source: src/kernel/device/gpu/work_stealing.h:14-41 (get_work_pixel).
 * kg access (kernel_data.integrator.scrambling_distance at :22) arrives as
 * an explicit argument - the same "kg -> explicit cited scalars" mapping
 * the body uses (init_from_camera.slang:187-196). Out-params become `out`
 * parameters (AGENTS.md section 4: no pointers to function-local memory).
 *
 * No zero-divisor guards: the entry below returns early whenever
 * tile_work_index >= tile->work_size (:93 of kernel.h), and work_size is
 * computed as w * h * num_samples by the scheduler (path_trace_work_gpu.
 * cpp:924), so any tile that reaches this function has w >= 1, h >= 1 and
 * num_samples >= 1 - all three divisors. Same unreachable-guard reasoning
 * as the body's width==0 note (init_from_camera.slang:260-264). */
void get_work_pixel(KernelWorkTile* tile, const uint global_work_index,
                    const float scrambling_distance,
                    out uint x, out uint y, out uint sample)
{
    uint sample_offset, pixel_offset;

    if (scrambling_distance < 0.9f) {
        /* source: work_stealing.h:22-26 - keep threads of one sample together */
        const uint tile_pixels = tile->w * tile->h;
        sample_offset = global_work_index / tile_pixels;
        pixel_offset = global_work_index - sample_offset * tile_pixels;
    }
    else {
        /* source: work_stealing.h:31-32 - keep threads of one pixel together */
        sample_offset = global_work_index % tile->num_samples;
        pixel_offset = global_work_index / tile->num_samples;
    }

    /* source: work_stealing.h:35-40 */
    const uint y_offset = pixel_offset / tile->w;
    const uint x_offset = pixel_offset - y_offset * tile->w;
    x = tile->x + x_offset;
    y = tile->y + y_offset;
    sample = tile->start_sample + sample_offset;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void init_from_camera_kernel_main(uint3 tid : SV_DispatchThreadID)
{
    /* source: kernel.h:81 */
    const int work_index = (int)tid.x;

    /* source: kernel.h:83 `if (tiles)` -> host-wired flag (header mapping). */
    if (pc.has_tiles != 0u) {
        /* source: kernel.h:84-86 - total-work guard */
        const int total_work = pc.max_tile_work_size * pc.num_tiles;
        if (work_index >= total_work) {
            return;
        }

        /* source: kernel.h:88-89 */
        const int tile_index = work_index / pc.max_tile_work_size;
        const int tile_work_index = work_index - tile_index * pc.max_tile_work_size;

        /* source: kernel.h:91. Take the array pointer into a local FIRST,
         * then index it: a dynamic index straight on the struct member
         * loaded from a push-constant block is the shape the AGENTS.md
         * SLANG RULE warns about (dynamic index on a fixed array reached
         * through a device pointer always returns element 0). The local
         * pointer form is the proven probe pattern. */
        KernelWorkTile* tiles = pc.tiles;
        KernelWorkTile* tile = &tiles[tile_index];

        /* source: kernel.h:93-95 */
        if (tile_work_index >= tile->work_size) {
            return;
        }

        /* source: kernel.h:97 */
        const int state = tile->path_index_offset + tile_work_index;

        uint x, y, sample;
        /* source: kernel.h:100 -> work_stealing.h:14-41 */
        get_work_pixel(tile, (uint)tile_work_index, pc.dp->mp.scrambling_distance, x, y, sample);

        /* source: kernel.h:102-103 -> the gated body (init_from_camera.
         * slang:223-243) with the tile branch (has_tile = true): the tile
         * scalars come from the tile itself, the kg scalars from staging. */
        IntegratorStateGPU* ist = &pc.params->integrator_state;
        integrator_init_from_camera(
            ist,
            (uint)state,
            pc.dp->mp,
            true,                       // has_tile (Cycles: tile != nullptr)
            tile->offset,               // int
            tile->stride,               // uint
            tile->sample_offset,        // uint
            pc.dp->pass_stride,
            pc.dp->pass_adaptive_aux_buffer,
            pc.dp->pass_sample_count,
            pc.dp->pass_volume_majorant,
            pc.dp->pass_volume_majorant_sample_count,
            pc.dp->kernel_features,
            pc.dp->background_object_index,
            pc.dp->background_volume_shader,
            pc.render_buffer,
            pc.render_buffer_u,
            pc.dp->render_buffer_valid,
            (int)x,
            (int)y,
            (int)sample);
    }
    else {
        /* source: kernel.h:105-108 - restart-from-miss branch */
        if (work_index >= pc.num_active_paths) {
            return;
        }

        /* source: kernel.h:110 `state = path_index_array ? path_index_array[
         * work_index] : work_index` -> host-wired flag (header mapping). */
        int state = work_index;
        if (pc.has_path_index_array != 0u) {
            int* path_index_array = pc.path_index_array;
            state = path_index_array[work_index];
        }

        /* source: kernel.h:112-113 -> body restart branch: tile == nullptr
         * becomes has_tile = false and x/y/scheduled_sample = 0; the body
         * reconstructs x/y/sample from the state itself
         * (init_from_camera.slang:249-275). */
        IntegratorStateGPU* ist = &pc.params->integrator_state;
        integrator_init_from_camera(
            ist,
            (uint)state,
            pc.dp->mp,
            false,                      // has_tile (Cycles: tile == nullptr)
            0,                          // tile_offset
            0u,                         // tile_stride
            0u,                         // tile_sample_offset
            pc.dp->pass_stride,
            pc.dp->pass_adaptive_aux_buffer,
            pc.dp->pass_sample_count,
            pc.dp->pass_volume_majorant,
            pc.dp->pass_volume_majorant_sample_count,
            pc.dp->kernel_features,
            pc.dp->background_object_index,
            pc.dp->background_volume_shader,
            pc.render_buffer,
            pc.render_buffer_u,
            pc.dp->render_buffer_valid,
            0,                          // x_   (kernel.h:113)
            0,                          // y_
            0);                         // scheduled_sample
    }
}
