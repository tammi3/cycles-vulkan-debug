// SPDX-FileCopyrightText: 2011-2022 Blender Foundation
// SPDX-License-Identifier: Apache-2.0

/* init_from_camera_body_probe.slang : M4f SLICE 4 gate probe (host env
 * CYCLES_VULKAN_INIT_FROM_CAMERA_BODY_PROBE=1, SPV via ..._SPV or
 * ./init_from_camera_body_probe.spv).
 *
 * WHAT IT PROVES: the ported integrator_init_from_camera BODY
 * (init_from_camera.slang, source src/kernel/integrator/init_from_camera.h:
 * 57-143) produces exactly what the REAL Cycles integrator_init_from_camera
 * produces when the host calls it: the full per-state writes (path state,
 * camera ray INCLUDING the ray differentials dP/dD), the film sample-count
 * write, the returned pixel-finished flag, and the queue counter net sums.
 * The host reference is the real function compiled into the binary -
 * same method as the slices 1-3 probes.
 *
 * PER CASE (= one SoA state, one unique render_pixel_index in tile modes):
 * the shader runs ONE call to the ported body per thread; everything the
 * body writes IS the probe output (no side-channel result struct beyond the
 * returned bool):
 *   - state slices: all members the real body writes (ray.*, path.sample,
 *     path.rng_pixel, path.queued_kernel, throughput, bounce counters, ...)
 *     compared against the same members of the host's IntegratorStateCPU
 *     after the REAL call;
 *   - render buffer: the film sample-count rows film_write_sample bumps,
 *     plus the aux.w prefill film_need_sample_pixel reads;
 *   - queue counters: net sums of the GPU-branch atomics (the CPU branch
 *     omits counter updates - the same documented caveat as slice 3);
 *   - out.ret: the returned pixel-finished bool.
 *
 * VARIANTS (whole-dispatch configs; one CLEAN run each, params.has_tile /
 * pre-seeded state / aux prefill select the body branch under test):
 *   V0 tile:        has_tile, aux.w=0 -> need=true, is_inside_volume=0
 *                   (path_state_init, film_need, film_write_sample, path_init)
 *   V1 restart:     has_tile=0, SoA pre-seeded path.render_pixel_index +
 *                   path.sample -> x/y/sample reconstruction, no film access
 *                   (init_from_camera.h:69-75, path_next)
 *   V2 tile+volume: like V0, is_inside_volume=1 -> INTERSECT_VOLUME_STACK
 *                   instead of INTERSECT_CLOSEST (init_from_camera.h:134-138)
 *   V3 need=false:  aux.w=1 -> film_need_sample_pixel returns false -> early
 *                   `return false` before any camera/film/queue work
 *                   (init_from_camera.h:83-86)
 * The cache_miss (:99-106) and is_zero(T) (:108-114) branches are UNREACHABLE
 * in the covered PERSPECTIVE config (file header of init_from_camera.slang)
 * and are not exercised here - reported uncovered, never faked.
 *
 * NEGATIVES (one fault per dispatch, applied OUTSIDE the ported code: the
 * body has no switch; host reference is ALWAYS the CLEAN variant):
 *   N1 base V0: tile_offset +1        -> render_pixel_index and the film row
 *                                        it addresses may differ
 *   N2 base V1: pre-seeded path.sample +1 (SoA INPUT payload) -> sample,
 *                                        rng_pixel and the rng-derived ray
 *                                        members may differ
 *   N3 base V0: is_inside_volume flipped -> queued_kernel + counters may
 *                                        differ (next-kernel choice)
 *   N4 base V0: tile_sample_offset +1 -> film_write_sample's return feeds
 *                                        sample/rng_pixel/ray, may differ
 * ray.tmin is EXCLUDED from the rng masks: it is constant 0.0f
 * (camera.h:169) and must stay bit-identical under N2/N4.
 * A negative is DETECTED only if a masked region differs AND every other
 * region is identical AND all canaries are intact (host logic, kernels.cpp).
 *
 * Comparison policy (user ruling, this slice): integer / passthrough words
 * RAW-BIT (film rows, counters, out, u8/u16/u32 state members, canaries);
 * computed float words |gpu-ref| <= 1e-5*max(1,|ref|) (ray.*, throughput,
 * ... the established GPU-vs-CPU float policy at kernels.cpp:2494/:3065).
 *
 * Layouts locked against kernels.cpp: IfcBodyCase 20 B, IfcBodyParams 264 B
 * (MathParams 212 + 13 scalars), IfcBodyOut 4 B, IfcBodyPush 56 B. */

#include "init_from_camera.slang"  // the body under test + its dependencies
                                   // (math_core, integrator_state, path_state,
                                   //  film_queue - all pulled in by that file)

/* One test case. ALL 4-byte scalars, layout provable. */
struct BodyCase
{
    uint  x, y;               // pixel coordinates (tile modes: body's x_/y_;
                              // restart mode: ignored by the body, kept for
                              // the host reference's call shape)
    int   scheduled_sample;   // init_from_camera.h:89 (tile modes only)
    uint  rpi_seed;           // SoA INPUT path.render_pixel_index seed
                              // (V1 restart reads it; 0 elsewhere so the
                              //  tile-mode write is provably a write)
    int   sample_seed;        // SoA INPUT path.sample seed (V1 reads it, V3
                              // keeps it as passthrough; V0/V2 overwrite it)
};                            // 5 * 4 = 20 bytes

/* Gate params: the MathParams the body's camera half needs, plus the exact
 * scalars the body signature takes (kg -> explicit cited args, same mapping
 * as init_from_camera.slang's signature comment). */
struct BodyParams
{
    MathParams mp;            // 212 B: cam/w (restart x,y reconstruction),
                              // is_inside_volume (next-kernel choice), rng
    int has_tile;             // Cycles' tile != nullptr (0/1)
    int tile_offset;          // KernelWorkTile.offset      (types.h:1695)
    uint tile_stride;         // KernelWorkTile.stride      (types.h:1696)
    int tile_sample_offset;   // KernelWorkTile.sample_offset (types.h:1693)
    int pass_stride;          // kernel_data.film.pass_stride
    int pass_adaptive_aux_buffer;
    int pass_sample_count;
    int pass_volume_majorant;
    int pass_volume_majorant_sample_count;
    uint kernel_features;     // kernel_data.kernel_features (+ forced bits,
                              // mirrored on the host - slice-2 precedent)
    int background_object_index;
    int background_volume_shader;
    uint render_buffer_valid; // platform constraint 3 (non-null in this gate)
};                            // 212 + 13 * 4 = 264 bytes

/* Per-case return value: the pixel-finished bool of init_from_camera.h:85. */
struct BodyOut
{
    uint ret;                 // 0 = pixel finished (need==false), 1 = queued
};                            // 4 bytes

/* Push block: 6 device pointers + count (+ pad) = 56 bytes. The render
 * buffer travels twice (float* reads + uint* atomics) - platform constraint
 * 4 in film_queue.slang: no in-shader pointer bitcasts (RX 570 miscompiles
 * them to the buffer base / SwiftShader crashes on them). */
struct BodyPush
{
    BodyCase*        cases;
    BodyOut*         out;
    BodyParams*      params;
    IntegratorStateGPU* ist;
    float*           render_buffer;    // float READS (film_need_sample_pixel)
    uint*            render_buffer_u;  // ATOMIC path (film_write_sample)
    uint             num_cases;
    uint             pad0;
};                            // 6 * 8 + 4 + 4 = 56 bytes

[[vk::push_constant]]
ConstantBuffer<BodyPush> pc;

[shader("compute")]
[numthreads(64, 1, 1)]
void init_from_camera_body_main(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= pc.num_cases)
        return;

    BodyCase c = pc.cases[i];
    BodyParams bp = *(pc.params);

    /* The ONE call under test: the ported integrator_init_from_camera with
     * the same argument mapping the production kernel will use (the queue.cpp
     * dispatch wiring lands in the slice after this one). All faults (N1/N3/
     * N4) are already inside bp's values; N2 is inside the SoA input payload.
     * The ported body contains no probe switch - AGENTS.md section 6. */
    BodyOut o;
    o.ret = integrator_init_from_camera(pc.ist,
                                        i,
                                        bp.mp,
                                        bp.has_tile != 0,
                                        bp.tile_offset,
                                        bp.tile_stride,
                                        bp.tile_sample_offset,
                                        bp.pass_stride,
                                        bp.pass_adaptive_aux_buffer,
                                        bp.pass_sample_count,
                                        bp.pass_volume_majorant,
                                        bp.pass_volume_majorant_sample_count,
                                        bp.kernel_features,
                                        bp.background_object_index,
                                        bp.background_volume_shader,
                                        pc.render_buffer,
                                        pc.render_buffer_u,
                                        bp.render_buffer_valid,
                                        (int)c.x,
                                        (int)c.y,
                                        c.scheduled_sample)
                ? 1u
                : 0u;
    pc.out[i] = o;
}
