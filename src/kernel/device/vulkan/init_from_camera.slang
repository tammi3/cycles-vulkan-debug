// SPDX-FileCopyrightText: 2011-2022 Blender Foundation
// SPDX-License-Identifier: Apache-2.0

/* init_from_camera.slang - M4f: the camera-sample helper AND the kernel body
 * of the INIT_FROM_CAMERA device kernel, both ported faithfully.
 *
 * WHAT IS PORTED (AGENTS.md section 3A; slice 4 adds the second function):
 *   integrate_camera_sample()      src/kernel/integrator/init_from_camera.h:19-52
 *     (M4f slice 1; slice 4 threads the ray differentials dP/dD through it,
 *     so the body can hand them to integrator_state_write_ray.)
 *   integrator_init_from_camera()  src/kernel/integrator/init_from_camera.h:57-143
 *     (M4f slice 4 - was a DEFERRED-marker skeleton; every dependency landed
 *     in slices 1-3: state init (path_state.slang), film + queue
 *     (film_queue.slang), and ray differentials (math_core.slang slice 4).)
 *     Nothing is dispatched yet: the gate is init_from_camera_body_probe.slang
 *     (host reference = the REAL Cycles function compiled into the binary);
 *     the production dispatch wiring lands with queue.cpp's enqueue plumbing
 *     in the slice after this one.
 *
 * BRANCHES NOT REACHABLE IN THE COVERED CONFIG (ported faithfully, but the
 * gate cannot exercise them without fabricating an unsupported camera type -
 * they are reported as uncovered, not faked):
 *   - the cache_miss branch (init_from_camera.h:99-106): only
 *     camera_sample_custom can set r_cache_miss (camera.h:335) and only for
 *     CAMERA_CUSTOM cameras (camera.h:520-523), which the math_core
 *     camera_sample port does not cover;
 *   - the is_zero(T) branch (:108-114): the covered perspective branch
 *     returns one_spectrum() = (1,1,1) (camera.h:172); zero T only comes
 *     from panorama/custom cameras (camera.h:336,340,366,404).
 *
 * DEPENDENCIES ALREADY PORTED AND GATED (math_core.slang, M4e STEP 1a; gate =
 * the math probe, host reference = the REAL Cycles headers):
 *   path_rng_2D / path_rng_3D / path_rng_pixel_init  kernel/sample/pattern.h
 *   camera_sample / camera_sample_perspective        kernel/camera/camera.h
 *
 * CONFIG COVERED (inherited from the math_core camera_sample port): a
 * PERSPECTIVE camera with aperturesize == 0 (no DOF), shuttertime == -1 (no
 * motion blur), have_perspective_motion == 0, num_motion_steps == 0,
 * interocular_offset == 0, and pixel_jitter != FLT_MAX (no filter table).
 * Orthographic / panorama / custom cameras are NOT covered: camera_sample
 * leaves `ok` false there and this port cannot produce their Spectrum. The
 * probe refuses to run the gate outside the covered config (same guard as the
 * math probe, kernels.cpp).
 *
 * OBSERVATION PARAMETERS (not in the Cycles signature, results unchanged):
 *   Cycles' integrate_camera_sample returns Spectrum T and fills Ray +
 *   r_cache_miss. In the covered config the two random draws this function
 *   makes - and the way lines 47-48 repack them into rand_time / rand_lens -
 *   are NOT consumed downstream: filter_uv is read only on the FLT_MAX
 *   filter-table branch, rand_lens only by the DOF branch, rand_time only by
 *   the shutter-table branch (camera.h:461, 75, 478). Comparing ray/T/
 *   cache_miss alone would therefore exercise none of THIS function's own
 *   logic. The draws and the packing are also forwarded through `out`
 *   parameters so the gate can compare them with the host reference. They are
 *   pure observations: no computation, no branch and no returned value
 *   depends on them.
 */

#include "math_core.slang"         // MathParams, path_rng_*, camera_sample
#include "integrator_state.slang"  // IntegratorStateGPU + accessors (#pragma once)
#include "path_state.slang"        // slice 2: path_state_init/_integrator,
                                   // integrator_state_write_ray (body calls them)
#include "film_queue.slang"        // slice 3: film_need_sample_pixel,
                                   // film_write_sample, integrator_path_* (body)

/* Sampling dimensions for the INIT bounce.
 * source: src/kernel/types.h:74-75 (enum PathTraceDimension) */
static const int PRNG_FILTER = 0;
static const int PRNG_LENS_TIME = 1;

/* source: src/kernel/types.h (enum CameraType: CAMERA_PERSPECTIVE,
 * CAMERA_ORTHOGRAPHIC, CAMERA_PANORAMA, CAMERA_CUSTOM -> CUSTOM is 3) */
static const int CAMERA_CUSTOM = 3;

/* ========================================================================= *
 *  integrate_camera_sample - FAITHFUL PORT
 *  source: src/kernel/integrator/init_from_camera.h:19-52
 * ========================================================================= *
 * Returns the Spectrum T (Cycles: Spectrum, util/types_spectrum.h:13 - the RGB
 * build uses `float3`) and fills the camera ray + r_cache_miss, exactly as the
 * Cycles function does; the two `out` draws are the gate observations
 * documented in the file header.
 *
 * No virtual dispatch, no recursion, no allocation: the two calls go to the
 * already-ported math_core functions, and `kernel_data.cam.*` reads are
 * mapped to the host-filled MathParams scalars (the same mapping the S1a
 * camera port uses). */
float3 integrate_camera_sample(MathParams p,
                               const int sample,
                               const int x,
                               const int y,
                               const uint rng_pixel,
                               out float3 ray_P,
                               out float3 ray_D,
                               out float ray_tmin,
                               out float ray_tmax,
                               out float ray_time,
                               out float ray_dP,
                               out float ray_dD,
                               out int r_cache_miss,
                               out float2 obs_rand_filter,
                               out float3 obs_rand_time_lens,
                               out float obs_rand_time,
                               out float2 obs_rand_lens)
{
    /* Filter sampling.
     * source: init_from_camera.h:28-29 - sample 0 is a fixed 0.5/0.5 draw
     * (the first sample needs no jitter), every other sample draws from the
     * PRNG_FILTER dimension. */
    /* The 5th argument is the math_core fault hook (M4e); the faithful
     * call site always passes false (no fault). */
    float2 rand_filter = (sample == 0) ? float2(0.5f, 0.5f) :
                                         path_rng_2D(p, rng_pixel, (uint)sample, PRNG_FILTER, false);

    /* Motion blur (time) and depth of field (lens) sampling.
     * source: init_from_camera.h:32-34 - kernel_data.cam.* -> MathParams. */
    bool use_motionblur = (p.shuttertime != -1.0f);
    bool use_dof = (p.aperturesize > 0.0f);
    bool use_custom_cam = (p.cam_type == CAMERA_CUSTOM);

    /* source: init_from_camera.h:35-37 - the three-dimensional draw is only
     * made when one of the three features is on, otherwise zero_float3().
     * The comment at init_from_camera.h:39-46 explains WHY x is time and y,z
     * are the lens: Sobol stratification converges better that way. */
    float3 rand_time_lens = (use_motionblur || use_dof || use_custom_cam) ?
                                path_rng_3D(p, rng_pixel, (uint)sample, PRNG_LENS_TIME, false) :
                                float3(0.0f, 0.0f, 0.0f);

    /* source: init_from_camera.h:47-48 */
    float rand_time = rand_time_lens.x;
    float2 rand_lens = float2(rand_time_lens.y, rand_time_lens.z);

    /* Generate camera ray.
     * source: init_from_camera.h:51 - camera_sample's Slang signature splits
     * the Ray into scalars and reports "camera type supported" as `ok`
     * (math_core.slang:498). r_cache_miss is filled below. */
    bool ok_covered;
    camera_sample(p, x, y, rand_filter, rand_time, rand_lens, ray_P, ray_D,
                  ray_tmin, ray_tmax, ray_time, ray_dP, ray_dD, ok_covered);

    /* source: camera.h:493 - camera_sample clears r_cache_miss BEFORE the
     * camera-type dispatch; only camera_sample_custom (an OSL camera,
     * camera.h:506+) can set it back to true, and custom cameras are NOT
     * covered by the math_core port. So the covered config always leaves 0. */
    r_cache_miss = 0;

    /* Gate observations (file header): surface the draws and the packing
     * above unchanged, so the host can check lines 28-29, 35-37, 47-48
     * separately (in the covered config none of them reaches the ray). */
    obs_rand_filter = rand_filter;
    obs_rand_time_lens = rand_time_lens;
    obs_rand_time = rand_time;
    obs_rand_lens = rand_lens;

    /* source: init_from_camera.h:51 + camera.h:172 - the covered perspective
     * branch returns one_spectrum() (util/types_spectrum.h:21 ->
     * one_float3() -> (1,1,1)). DEFERRED: orthographic / panorama return
     * their own Spectrum and camera_sample_custom can return a zero one and
     * set cache_miss - those types are NOT covered (math_core.slang:534), so
     * this placeholder is never compared: the probe refuses to run the gate
     * outside the covered config and reports the clean pass as failed there
     * instead. */
    bool covered = ok_covered;
    return covered ? float3(1.0f, 1.0f, 1.0f) : float3(0.0f, 0.0f, 0.0f);
}

/* DeviceKernel values the body writes into queued_kernel / queue counters.
 * source: src/kernel/types.h:1744-1752 (enum DeviceKernel) - counted from
 * DEVICE_KERNEL_INTEGRATOR_INIT_FROM_CAMERA = 0: INIT_FROM_BAKE = 1,
 * INTERSECT_CLOSEST = 2, INTERSECT_SHADOW = 3, INTERSECT_SUBSURFACE = 4,
 * INTERSECT_VOLUME_STACK = 5. */
static const uint DEVICE_KERNEL_INTEGRATOR_INIT_FROM_CAMERA = 0u;
static const uint DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST = 2u;
static const uint DEVICE_KERNEL_INTEGRATOR_INTERSECT_VOLUME_STACK = 5u;

/* source: src/util/math_float3.h:576-582 (is_zero(const float3), non-SSE
 * build path): component-wise comparison against 0.0f. */
bool is_zero(float3 a)
{
    return (a.x == 0.0f && a.y == 0.0f && a.z == 0.0f);
}

/* ========================================================================= *
 *  integrator_init_from_camera - FAITHFUL PORT
 *  source: src/kernel/integrator/init_from_camera.h:57-143
 * ========================================================================= *
 * Signature mapping (AGENTS.md section 4: no storage class left to guess):
 *   kg                    -> MathParams p PLUS explicit cited scalars, the
 *                            same "kg -> explicit cited int args" spelling
 *                            slices 2/3 used (path_state.slang kernel_features
 *                            etc., film_queue.slang's 5 film pass offsets).
 *                            The body reads: kernel_data.cam.width
 *                            (MathParams.width), cam.is_inside_volume
 *                            (MathParams.is_inside_volume), the 5 film pass
 *                            fields, kernel_features, background.object_index
 *                            and background.volume_shader.
 *   state                 -> uint state (the work-item index; the caller
 *                            derives it from tid.x, as Cycles' GPU wrapper
 *                            does at device/gpu/kernel.h:97).
 *   tile                  -> has_tile + tile_offset/tile_stride/
 *                            tile_sample_offset scalars (KernelWorkTile has
 *                            no Slang type; types.h:1688-1701:
 *                            int offset, uint stride, uint sample_offset -
 *                            the exact types path_state_init already takes).
 *                            has_tile == false is Cycles' tile == nullptr
 *                            restart branch (:69-75).
 *   render_buffer         -> float* render_buffer (reads: film_need_sample_
 *                            pixel) AND uint* render_buffer_u (writes:
 *                            film_write_sample / integrator_path_terminate),
 *                            both the SAME device address, plus
 *                            render_buffer_valid for the terminate null check
 *                            (platform constraints 3+4, film_queue.slang
 *                            header: an in-shader pointer bitcast lands at
 *                            the buffer base on the RX 570 / SwiftShader
 *                            crashes on any pointer bitcast).
 *   x_/y_/scheduled_sample -> unchanged (ints).
 *
 * NOT covered in this port, exactly as the slices before it: PROFILING_INIT
 * (init_from_camera.h:65, no profiling infra in this backend - same DEFERRED
 * note as film_queue.slang). The cache_miss and is_zero(T) branches are
 * ported faithfully but unreachable in the gate's covered config (file
 * header). */
bool integrator_init_from_camera(IntegratorStateGPU* ist,
                                 uint state,
                                 MathParams p,
                                 const bool has_tile,
                                 const int tile_offset,
                                 const uint tile_stride,
                                 const uint tile_sample_offset,
                                 const int pass_stride,
                                 const int pass_adaptive_aux_buffer,
                                 const int pass_sample_count,
                                 const int pass_volume_majorant,
                                 const int pass_volume_majorant_sample_count,
                                 const uint kernel_features,
                                 const int background_object_index,
                                 const int background_volume_shader,
                                 float* render_buffer,
                                 uint* render_buffer_u,
                                 const uint render_buffer_valid,
                                 const int x_,
                                 const int y_,
                                 const int scheduled_sample)
{
    /* PROFILING_INIT (init_from_camera.h:65) DEFERRED - file header. */

    int x, y, sample;

    /* source: init_from_camera.h:69-75 - the restart-from-miss branch
     * (Cycles: tile == nullptr): x, y and sample are reconstructed from the
     * state itself; no film access happens on this path. */
    if (!has_tile) {
        /* source: :71 */
        const uint pixel_index = read_path_render_pixel_index(ist, state);

        /* source: :72-73 - reproduce the exact C++ conversions:
         *   x = pixel_index % (int)kernel_data.cam.width;
         * pixel_index is uint, so the int modulus converts to uint
         * (usual arithmetic conversions) -> one signed->unsigned cast and
         * an UNSIGNED % and /. DEFENSIVE width==0 guard: Cycles would
         * divide by zero (host crash) and SPIR-V leaves it undefined; real
         * cam.width >= 1 and the probe checks the value host-side, so the
         * guard cannot fire in the gate (same defensive style as
         * queue_kernel_index_valid in film_queue.slang). */
        int width_i = (int)p.width;
        if (width_i == 0) {
            return false;
        }
        uint width_u = (uint)width_i;
        x = (int)(pixel_index % width_u);
        y = (int)(pixel_index / width_u);

        /* source: :74 */
        sample = (int)read_path_sample(ist, state);
    }
    else {
        x = x_;
        y = y_;

        /* source: init_from_camera.h:80-81 */
        path_state_init(ist, state, tile_offset, tile_stride, x, y);

        /* source: init_from_camera.h:83-86 - adaptive-sampling early-out: a
         * converged pixel is not sampled again and the kernel reports
         * "nothing scheduled" (false). */
        if (!film_need_sample_pixel(ist, state, render_buffer,
                                    pass_adaptive_aux_buffer, pass_stride)) {
            return false;
        }

        /* source: init_from_camera.h:88-89 - the ONLY producer of `sample`
         * in this branch: the atomic sample counter returns the OLD count,
         * plus tile->sample_offset. */
        sample = film_write_sample(ist, state, render_buffer_u, scheduled_sample,
                                   (int)tile_sample_offset, pass_sample_count,
                                   pass_stride);
    }

    /* source: init_from_camera.h:93 - the 5th argument is the math_core
     * fault hook; faithful call sites always pass false (no fault). */
    const uint rng_pixel = path_rng_pixel_init(p, sample, x, y, false);

    /* source: init_from_camera.h:95-98 - generate camera ray. */
    float3 ray_P, ray_D;
    float ray_tmin, ray_tmax, ray_time, ray_dP, ray_dD;
    int cache_miss;
    /* integrate_camera_sample's observation outs (slice-1 file header):
     * unconsumed by this body - only T, the ray and cache_miss are used. */
    float2 obs_rand_filter;
    float3 obs_rand_time_lens;
    float obs_rand_time;
    float2 obs_rand_lens;
    float3 T = integrate_camera_sample(p, sample, x, y, rng_pixel,
                                       ray_P, ray_D, ray_tmin, ray_tmax, ray_time,
                                       ray_dP, ray_dD, cache_miss,
                                       obs_rand_filter, obs_rand_time_lens,
                                       obs_rand_time, obs_rand_lens);

    /* source: init_from_camera.h:99-106 - cache-miss reschedule. NOT
     * reachable in the covered PERSPECTIVE config (camera.h:509 clears
     * r_cache_miss before the camera-type dispatch; only
     * camera_sample_custom can set it, camera.h:335) - ported faithfully,
     * reported uncovered (file header). */
    if (cache_miss != 0) {
        /* source: :100-103 (tile != nullptr half) */
        if (has_tile) {
            integrator_path_init(ist, state, DEVICE_KERNEL_INTEGRATOR_INIT_FROM_CAMERA);
            write_path_sample(ist, state, (uint)sample);
        }
        /* source: :104 */
        integrator_path_cache_miss(ist, state);
        return true;
    }

    /* source: init_from_camera.h:108-114 - zero-throughput early-out. NOT
     * reachable in the covered config (the perspective branch returns
     * one_spectrum() = (1,1,1), camera.h:172) - ported faithfully, reported
     * uncovered (file header). */
    if (is_zero(T)) {
        /* source: :109-112 (tile == nullptr half) - terminate also writes
         * optical depth into the film (state_flow.h:51-67). */
        if (!has_tile) {
            integrator_path_terminate(ist, state, render_buffer_u,
                                      render_buffer_valid,
                                      DEVICE_KERNEL_INTEGRATOR_INIT_FROM_CAMERA,
                                      pass_volume_majorant,
                                      pass_volume_majorant_sample_count,
                                      pass_stride);
        }
        return true;
    }

    /* source: init_from_camera.h:116-117 - all seven ray members; dP/dD are
     * the compact floats the slice-4 differential port produced. */
    integrator_state_write_ray(ist, state, ray_P, ray_dP, ray_D, ray_dD,
                               ray_tmin, ray_tmax, ray_time);

    /* source: init_from_camera.h:119-140 */
    if (!has_tile) {
        /* source: :120-126 - restart half: re-init, then CONTINUE
         * (integrator_path_next subtracts the current kernel's queue slot
         * and adds the next kernel's, state_flow.h:78-86). */
        path_state_init_integrator(ist, state, kernel_features,
                                   background_object_index,
                                   background_volume_shader,
                                   sample, rng_pixel, T);
        integrator_path_next(ist, state,
                             DEVICE_KERNEL_INTEGRATOR_INIT_FROM_CAMERA,
                             p.is_inside_volume != 0 ?
                                 DEVICE_KERNEL_INTEGRATOR_INTERSECT_VOLUME_STACK :
                                 DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST);
    }
    else {
        /* source: :130-133 - fresh tile half: initialize, then queue the
         * next kernel below. */
        path_state_init_integrator(ist, state, kernel_features,
                                   background_object_index,
                                   background_volume_shader,
                                   sample, rng_pixel, T);

        /* source: init_from_camera.h:134-139 - volume stack optionally
         * initialized before intersect_closest. */
        if (p.is_inside_volume != 0) {
            integrator_path_init(ist, state, DEVICE_KERNEL_INTEGRATOR_INTERSECT_VOLUME_STACK);
        }
        else {
            integrator_path_init(ist, state, DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST);
        }
    }

    /* source: init_from_camera.h:142 */
    return true;
}
