// SPDX-FileCopyrightText: 2011-2022 Blender Foundation
// SPDX-License-Identifier: Apache-2.0

/* init_from_camera.slang - M4f SLICE 1: the camera-sample helper of the
 * INIT_FROM_CAMERA device kernel, ported faithfully, plus the Slang SKELETON
 * of the kernel that owns it.
 *
 * WHAT IS PORTED (one Cycles function - AGENTS.md section 3A):
 *   integrate_camera_sample()      src/kernel/integrator/init_from_camera.h:19-52
 *
 * WHAT IS A SKELETON ONLY (compile-checked, never dispatched):
 *   integrator_init_from_camera() src/kernel/integrator/init_from_camera.h:57-143
 *   Every part that needs the film buffer, a KernelWorkTile or the state-flow
 *   queue is a DEFERRED marker naming the Cycles call it will replace. The
 *   split was agreed with the user: slice 1 = this file, slice 2 = the state
 *   initialization (path_state_init / path_state_init_integrator /
 *   integrator_state_write_ray, plus the missing path.rng_pixel accessor),
 *   slice 3 = film + queue.
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
#include "integrator_state.slang"  // IntegratorStateGPU + accessors (skeleton)

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
                  ray_tmin, ray_tmax, ray_time, ok_covered);

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

/* ========================================================================= *
 *  integrator_init_from_camera - SKELETON ONLY (compile-checked)
 *  source: src/kernel/integrator/init_from_camera.h:57-143
 * ========================================================================= *
 * Signature mapping (AGENTS.md section 4: no storage class left to guess):
 *   kg              -> MathParams p   (host-filled from the REAL kernel_data,
 *                                      as in math_core.slang / the math probe)
 *   state           -> IntegratorStateGPU* ist (SoA slice pointers; the
 *                                      accessors in integrator_state.slang)
 *   tile            -> DEFERRED (no KernelWorkTile type in this backend yet)
 *   render_buffer   -> DEFERRED (no film buffer yet, slice 3)
 *   x_/y_/scheduled_sample -> unchanged (ints)
 *
 * Nothing here is dispatched: this exists so the file compiles, so the DEFERRED
 * map is executable documentation, and so slice 2/3 can fill it in place. */
bool integrator_init_from_camera(IntegratorStateGPU* ist,
                                 MathParams p,
                                 const int x_,
                                 const int y_,
                                 const int scheduled_sample)
{
    /* ---- init_from_camera.h:67-90 : NOT PORTED (slice 2 + slice 3) ----
     *   tile == nullptr restart branch:
     *     pixel_index = INTEGRATOR_STATE(state, path, render_pixel_index)
     *     x = pixel_index % kernel_data.cam.width;  y = pixel_index / width
     *     sample = INTEGRATOR_STATE(state, path, sample)      (:71-74)
     *     needs kernel_data.cam.width, which MathParams does not carry yet.
     *   tile != nullptr branch:
     *     path_state_init(state, tile, x, y)                   (:81)
     *       = path_state.h:27 + path_state_init_queues (path_state.h:16,
     *       queued_kernel = 0 - the accessor exists and is gated) but it also
     *       needs tile->offset / tile->stride.
     *     if (!film_need_sample_pixel(...)) return false;      (:84)
     *       = film/adaptive_sampling.h:13, needs the render buffer.
     *     sample = film_write_sample(..., scheduled_sample,
     *                                 tile->sample_offset);    (:89)
     *       = film/light_passes.h:158, needs the render buffer; this is the
     *       ONLY producer of `sample`, which is why the lines below stay
     *       DEFERRED too - inventing a stand-in for `sample` would be a
     *       fabricated result (AGENTS.md section 5). */

    /* ---- init_from_camera.h:92-98 : DEFERRED (blocked on `sample`) ----
     *   rng_pixel = path_rng_pixel_init(kg, sample, x, y);     (:93)
     *   Ray ray; int cache_miss = 0;                            (:96-97)
     *   Spectrum T = integrate_camera_sample(kg, sample, x, y,
     *                                        rng_pixel, &ray, cache_miss); (:98)
     * The ported half of this pair is integrate_camera_sample above; its gate
     * is init_from_camera_probe.slang, which calls it directly. */

    /* ---- init_from_camera.h:99-106 : DEFERRED (queue layer, slice 3) ----
     *   if (cache_miss) { integrator_path_init(state, INIT_FROM_CAMERA);
     *     INTEGRATOR_STATE_WRITE(state, path, sample) = sample;
     *     integrator_path_cache_miss(state, ...); return true; }
     *   source: state_flow.h:71-105 (queue_counter atomics + queued_kernel) */

    /* ---- init_from_camera.h:108-114 : DEFERRED (film, slice 3) ----
     *   if (is_zero(T)) { integrator_path_terminate(kg, state,
     *                       render_buffer, INIT_FROM_CAMERA); return true; }
     *   source: state_flow.h:88-98 (also writes optical_depth to the film) */

    /* ---- init_from_camera.h:116-117 : DEFERRED (slice 2) ----
     *   integrator_state_write_ray(state, &ray);  source: state_util.h:25-57
     *   writes ray.P/D/tmin/tmax/time AND ray.dP/dD; the math_core
     *   camera_sample port does not produce dP/dD (math_core.slang:451), so
     *   this cannot land before ray differentials exist. */

    /* ---- init_from_camera.h:119-140 : DEFERRED (slice 2 + slice 3) ----
     *   path_state_init_integrator(kg, state, sample, rng_pixel, T)  (:121,130)
     *     source: path_state.h:40-... - needs the path.rng_pixel accessor,
     *     which integrator_state.slang does not have yet (see its "MEMBERS
     *     NOT DONE" list: state_template.h:35).
     *   integrator_path_next / integrator_path_init(... volume_stack ?
     *     INTERSECT_VOLUME_STACK : INTERSECT_CLOSEST)            (:122-139)
     *     source: state_flow.h:78-86, needs kernel_data.cam.is_inside_volume
     *     and the queue_counter buffer. */

    /* The parameters are only read by those DEFERRED parts; the body is a
     * stub until slice 2/3 land. Do NOT dispatch this kernel. */
    (void)ist;
    (void)p;
    (void)x_;
    (void)y_;
    (void)scheduled_sample;
    return true;  // STUB: "pixel not finished" - no path is terminated here.
}
