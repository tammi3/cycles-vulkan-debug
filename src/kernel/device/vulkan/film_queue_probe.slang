// SPDX-FileCopyrightText: 2011-2022 Blender Foundation
// SPDX-License-Identifier: Apache-2.0

/* film_queue_probe.slang : M4f SLICE 3 gate probe (host env
 * CYCLES_VULKAN_FILM_QUEUE_PROBE=1, SPV via ..._SPV or ./film_queue_probe.spv).
 *
 * WHAT IT PROVES: the ported film + queue cluster (film_queue.slang) produces
 * EXACTLY what the REAL Cycles functions produce. Like slice 2 there is no
 * dedicated "result" for most outputs: the film-buffer writes, the queue
 * counter atomics and the queued_kernel slice ARE the outputs, and the host
 * compares them against a reference built from the real Cycles host calls.
 * film_need_sample_pixel's and film_write_sample's RETURN values go to a
 * small per-case out buffer (the real kernel consumes them at
 * init_from_camera.h:84-89).
 *
 * PER CASE (= one SoA state, one unique render_pixel_index):
 *   1. path_state_init(...)                  [slice-2 port, integration]
 *   2. probe writes the case's path.flag / path.optical_depth (input data)
 *   3. need = film_need_sample_pixel(...)
 *   4. if need: out.sample = film_write_sample(...)
 *      else    out.sample = 0xFFFFFFFF (sentinel: pixel skipped)
 *   5. queue op by mode:
 *        A: integrator_path_init(next_kernel)          [+1 to num_queued[next]]
 *        B: integrator_path_init(0) then
 *           integrator_path_cache_miss()               [cache_miss = 1]
 *        C: integrator_path_terminate(current_kernel)  [optical-depth film
 *           writes when the volume flag is set AND render_buffer_valid != 0;
 *           -1 to num_queued[current]]
 *        D: integrator_path_next(0, next_kernel)       [-1 [0], +1 [next]]
 *
 * PROBE-STRUCTURE NOTE (documented so it cannot be mistaken for a shortcut):
 * film_write_sample is called only when need == true (mirrors
 * init_from_camera.h:84-89), but the queue op runs unconditionally per mode.
 * The real kernel's early-out also skips the queue ops; that control-flow
 * coupling is gated when the full integrator_init_from_camera is assembled.
 * This slice gates each function's OUTPUTS, per the slice definition.
 *
 * ref_mode / neg_mode: the host reference is ALWAYS built from CLEAN inputs;
 * neg_mode is a probe-side INPUT fault (the ported code has no switch):
 *   1 -> sample_offset fed to film_write_sample is +1
 *        -> ONLY out.sample may differ;
 *   2 -> film pass_stride fed to every film function is +1
 *        -> ONLY out.{need,sample} + the render buffer may differ;
 *   3 -> pass_sample_count fed to film_write_sample is +1
 *        -> ONLY out.sample + the render buffer may differ;
 *   4 -> next_kernel fed to the queue functions is +1
 *        -> ONLY queue_counter + queued_kernel may differ.
 * A negative is DETECTED only if its target group differs AND every other
 * group is byte-identical AND all canaries are intact.
 *
 * The CLEAN dispatch itself exercises write_optical_depth's null branch
 * (platform constraint 3 in film_queue.slang): every mode-C case carries a
 * per-case render_buffer_valid. When it is 0, the host reference calls the
 * REAL Cycles integrator_path_terminate with a literal nullptr render_buffer
 * (exactly what intersect_subsurface.h:21 does), while the probe runs the
 * ported terminate with valid=0 - the optical-depth slots of that case's
 * pixel row must keep their initial fill value (no writes happened). Cases
 * with valid=1 compare the real optical-depth film writes. The probe's film
 * steps 3-4 still run on the shared real buffer for every case, so the
 * expected bytes for a valid=0 case are the initial fill plus those steps'
 * writes and nothing else; the reference computes the same combination.
 *
 * One fault mode per dispatch (one neg_mode in params), same as slice 2. */

#include "path_state.slang"    // path_state_init (slice 2) + accessors
#include "film_queue.slang"    // the slice-3 ports (uses #pragma once guard)

/* One test case. Layout locked against FilmQueueCase in kernels.cpp.
 * ALL SCALAR 4-byte members, no float3: layout must be provable. */
struct FilmQueueCase
{
    uint  x, y;               // pixel coordinates (path_state_init)
    int   scheduled_sample;   // init_from_camera.h:89 `scheduled_sample`
    int   sample_offset;      // KernelWorkTile.sample_offset (:89)
    uint  mode;               // A=0, B=1, C=2, D=3 (see header)
    uint  next_kernel;        // DeviceKernel value for init/next
    uint  current_kernel;     // DeviceKernel value for next/terminate
    uint  flag;               // -> path.flag before terminate (optical branch)
    float optical_depth;      // -> path.optical_depth before terminate
    uint  render_buffer_valid;// mode C: 0 = null buffer (constraint 3)
};                            // 10 * 4 = 40 bytes

/* Gate params. Layout locked against FilmQueueParams in kernels.cpp. */
struct FilmQueueParams
{
    int film_stride;                  // kernel_data.film.pass_stride
    int pass_sample_count;            // kernel_data.film.pass_sample_count
    int pass_adaptive_aux_buffer;     // kernel_data.film.pass_adaptive_aux_buffer
    int pass_volume_majorant;         // kernel_data.film.pass_volume_majorant
    int pass_volume_majorant_sample_count;  // kernel_data.film.pass_volume_majorant_sample_count
    int tile_offset;                  // KernelWorkTile.offset (path_state.h:32)
    uint tile_stride;                 // KernelWorkTile.stride
    int neg_mode;                     // input fault switch (see header)
};                                    // 8 * 4 = 32 bytes

/* Per-case return values. Layout locked against FilmQueueOut in kernels.cpp. */
struct FilmQueueOut
{
    uint need;    // film_need_sample_pixel return (0/1)
    uint sample;  // film_write_sample return, 0xFFFFFFFF when need == 0
};                // 8 bytes

/* Push block: 5 device pointers + count (+ pad) = 48 bytes. */
struct FilmQueuePush
{
    FilmQueueCase*       cases;
    IntegratorStateGPU*  ist;
    FilmQueueParams*     params;
    float*               render_buffer;
    FilmQueueOut*        out;
    uint                 num_cases;
    uint                 pad0;
};                         // 5 * 8 + 4 + 4 = 48 bytes

[[vk::push_constant]]
ConstantBuffer<FilmQueuePush> pc;

[shader("compute")]
[numthreads(64, 1, 1)]
void film_queue_main(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= pc.num_cases)
        return;

    FilmQueueCase c = pc.cases[i];
    FilmQueueParams p = *(pc.params);

    /* ---- probe-side input faults (header): the ported code has NO switch ---- */
    int  sample_offset_arg = c.sample_offset;
    int  film_stride_arg   = p.film_stride;
    int  sample_count_arg  = p.pass_sample_count;
    uint next_kernel_arg   = c.next_kernel;
    if (p.neg_mode == 1) sample_offset_arg += 1;
    if (p.neg_mode == 2) film_stride_arg   += 1;
    if (p.neg_mode == 3) sample_count_arg  += 1;
    if (p.neg_mode == 4) next_kernel_arg   += 1u;

    /* ---- 1. slice-2 integration: writes render_pixel_index + queued=0 ---- */
    path_state_init(pc.ist, i, p.tile_offset, p.tile_stride, (int)c.x, (int)c.y);

    /* ---- 2. input data into the state (both sides get the same values) ---- */
    write_path_flag(pc.ist, i, c.flag);
    write_path_optical_depth(pc.ist, i, c.optical_depth);

    /* ---- 3. film_need_sample_pixel (adaptive_sampling.h:13-25) ---- */
    bool need = film_need_sample_pixel(pc.ist,
                                       i,
                                       pc.render_buffer,
                                       p.pass_adaptive_aux_buffer,
                                       film_stride_arg);

    /* ---- 4. film_write_sample (light_passes.h:158-173), gated on `need`
     * exactly as init_from_camera.h:84-89 gates it ---- */
    FilmQueueOut o;
    o.need = need ? 1u : 0u;
    if (need) {
        o.sample = (uint)film_write_sample(pc.ist,
                                           i,
                                           pc.render_buffer,
                                           c.scheduled_sample,
                                           sample_offset_arg,
                                           sample_count_arg,
                                           film_stride_arg);
    }
    else {
        o.sample = 0xFFFFFFFFu;  // sentinel: pixel not sampled this pass
    }
    pc.out[i] = o;

    /* ---- 5. queue op by mode (state_flow.h GPU branch:71-105) ---- */
    if (c.mode == 0u) {
        /* A: normal continuation, init_from_camera.h:135-138 */
        integrator_path_init(pc.ist, i, next_kernel_arg);
    }
    else if (c.mode == 1u) {
        /* B: cache-miss restart, init_from_camera.h:101-104 */
        integrator_path_init(pc.ist, i, 0u);
        integrator_path_cache_miss(pc.ist, i);
    }
    else if (c.mode == 2u) {
        /* C: terminate, init_from_camera.h:110-111 (optical-depth writes;
         * render_buffer_valid selects the null branch, see header) */
        integrator_path_terminate(pc.ist,
                                  i,
                                  pc.render_buffer,
                                  c.render_buffer_valid,
                                  c.current_kernel,
                                  p.pass_volume_majorant,
                                  p.pass_volume_majorant_sample_count,
                                  film_stride_arg);
    }
    else {
        /* D: restart continuation, init_from_camera.h:122-126 */
        integrator_path_next(pc.ist, i, 0u, next_kernel_arg);
    }
}
