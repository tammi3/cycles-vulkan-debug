// SPDX-FileCopyrightText: 2011-2022 Blender Foundation
// SPDX-License-Identifier: Apache-2.0

/* path_state.slang - M4f SLICE 2: the state-initialization cluster of the
 * INIT_FROM_CAMERA device kernel, ported faithfully (GPU semantics).
 *
 * WHAT IS PORTED (one small cluster, AGENTS.md section 3A):
 *   path_state_init_queues()            src/kernel/integrator/path_state.h:16-23
 *   path_state_init()                   src/kernel/integrator/path_state.h:27-37
 *   path_state_init_integrator()        src/kernel/integrator/path_state.h:40-108
 *   integrator_state_write_ray()        src/kernel/integrator/state_util.h:25-60
 *
 * GATE: state_init_probe.slang + vulkan_state_init_probe (kernels.cpp); the
 * host reference calls these REAL Cycles functions on a zeroed
 * IntegratorStateCPU (state.h:227: on the host IntegratorState is an
 * IntegratorStateCPU*, and INTEGRATOR_STATE_WRITE is a direct member
 * assignment, state.h:234-240), then compares every written member against
 * the raw SoA slice bytes the GPU wrote through the accessors.
 *
 * SIGNATURE MAPPING (AGENTS.md section 4: no storage class left to guess):
 *   kg                  -> three explicit values it is only used to read:
 *                          kernel_features, background.object_index,
 *                          background.volume_shader (types.h:1369,
 *                          data_template.h:44/:27). MathParams is frozen at
 *                          176 bytes by the math-probe harness assert, so
 *                          these are arguments, not a params struct.
 *   tile                -> explicit tile_offset (int) + tile_stride (uint):
 *                          there is no KernelWorkTile type in this backend
 *                          yet (types.h:1688 is the host struct). Only
 *                          offset/stride are read by path_state_init.
 *   state               -> IntegratorStateGPU* ist + uint state (SoA slice
 *                          pointers via integrator_state.slang accessors)
 *   throughput          -> float3 (Spectrum = float3, types_spectrum.h:13)
 *
 * NOT PORTED (with reasons, so it cannot be mistaken for an oversight):
 *   - the __PATH_GUIDING__ block (path_state.h:66-78): features.h:156-161
 *     defines __PATH_GUIDING__ only #ifndef __KERNEL_GPU__ (CPU builds), and
 *     WITH_PATH_GUIDING appears nowhere in this build (CMakeCache has no
 *     such option), so the host reference does not compile it either. The
 *     runtime guard kernel_features & KERNEL_FEATURE_PATH_GUIDING would skip
 *     it anyway; if a scene ever had that feature on, the gate would FAIL
 *     loudly rather than silently diverge.
 *   - path_state_init_queues' shadow/ao writes (path_state.h:19-22): those
 *     lines are #ifndef __KERNEL_GPU__ = CPU-only queue layout. The GPU
 *     path writes only path.queued_kernel (path_state.h:18).
 */

#include "integrator_state.slang"  // accessors (incl. the M4f slice-2 ones)

/* ---------------------------------------------------------------------------
 * Constants used by the ported functions. Values cited from the real enums;
 * declared as `static const` file-scope so Slang needs no C preprocessor and
 * no duplicate-name collisions with other Slang files (each file keeps its
 * own scope in the probe translation units).
 * ------------------------------------------------------------------------- */
// source: src/kernel/types.h:118 (enum PathTraceDimension, PRNG_BOUNCE_NUM = 16)
static const uint PRNG_BOUNCE_NUM = 16u;
// source: src/kernel/types.h:154 (PATH_RAY_VISIBILITY_CAMERA = 1U << 0)
static const uint PATH_RAY_VISIBILITY_CAMERA = 1u << 0u;
// source: src/kernel/types.h:215 (PATH_RAY_MIS_SKIP = 1U << 7)
static const uint PATH_RAY_MIS_SKIP = 1u << 7u;
// source: src/kernel/types.h:221 (PATH_RAY_TRANSPARENT_BACKGROUND = 1U << 9)
static const uint PATH_RAY_TRANSPARENT_BACKGROUND = 1u << 9u;
// source: src/kernel/types.h:247 (PATH_RAY_DENOISING_FEATURES = 1U << 17)
static const uint PATH_RAY_DENOISING_FEATURES = 1u << 17u;
// source: src/kernel/types.h:47 (#define OBJECT_NONE (~0)) -> int -1
static const int OBJECT_NONE_I = -1;
// source: src/kernel/types.h:48 (#define PRIM_NONE (~0)) -> int -1
static const int PRIM_NONE_I = -1;
// source: src/kernel/types.h:46 (#define SHADER_NONE (~0)) -> int -1
static const int SHADER_NONE_I = -1;
// source: src/kernel/types.h:686 (PRIMITIVE_NONE = 0)
static const int PRIMITIVE_NONE_I = 0;
// source: src/kernel/features.h:30 (KERNEL_FEATURE_NODE_PORTAL = 1U << 10)
static const uint KERNEL_FEATURE_NODE_PORTAL = 1u << 10u;
// source: src/kernel/features.h:49 (KERNEL_FEATURE_VOLUME = 1U << 18)
static const uint KERNEL_FEATURE_VOLUME = 1u << 18u;
// source: src/kernel/features.h:76 (KERNEL_FEATURE_LIGHT_LINKING = 1U << 28)
static const uint KERNEL_FEATURE_LIGHT_LINKING = 1u << 28u;
// source: src/kernel/features.h:80 (KERNEL_FEATURE_DENOISING = 1U << 30)
static const uint KERNEL_FEATURE_DENOISING = 1u << 30u;

/* =========================================================================== *
 *  path_state_init_queues - FAITHFUL PORT
 *  source: src/kernel/integrator/path_state.h:16-23
 * =========================================================================== *
 * The shadow/ao queue writes at path_state.h:19-22 are #ifndef __KERNEL_GPU__
 * (CPU-only state layout) and are correctly absent on this GPU port; see the
 * file header. */
void path_state_init_queues(IntegratorStateGPU* ist, uint state)
{
    /* source: path_state.h:18 */
    write_path_queued_kernel(ist, state, 0u);
}

/* =========================================================================== *
 *  path_state_init - FAITHFUL PORT
 *  source: src/kernel/integrator/path_state.h:27-37
 * =========================================================================== *
 * tile -> explicit offset/stride (see file header). C++ promotes the
 * expression to unsigned the moment `y * tile->stride` is evaluated (stride
 * is uint, types.h:1695); the cast order below reproduces that. */
void path_state_init(IntegratorStateGPU* ist, uint state,
                     int tile_offset, uint tile_stride, const int x, const int y)
{
    /* source: path_state.h:32
     *   (uint)tile->offset + x + y * tile->stride */
    uint render_pixel_index = (uint)tile_offset + (uint)x + (uint)y * tile_stride;

    /* source: path_state.h:34 */
    write_path_render_pixel_index(ist, state, render_pixel_index);

    /* source: path_state.h:36 */
    path_state_init_queues(ist, state);
}

/* =========================================================================== *
 *  path_state_init_integrator - FAITHFUL PORT
 *  source: src/kernel/integrator/path_state.h:40-108
 * =========================================================================== *
 * Write ORDER matters for the gate: flag is written at :60 and then OR-ed at
 * :98 (denoising), so the read-modify-write keeps its position. The
 * __PATH_GUIDING__ block (:66-78) is omitted - file header explains why
 * (features.h:156-161 makes it CPU-only; WITH_PATH_GUIDING absent from this
 * build, so the host reference has it too). */
void path_state_init_integrator(IntegratorStateGPU* ist,
                                uint state,
                                const uint kernel_features,
                                const int background_object_index,
                                const int background_volume_shader,
                                const int sample,
                                const uint rng_pixel,
                                const float3 throughput)
{
    /* source: path_state.h:46-53 - the bounce counters, all 0. */
    write_path_sample(ist, state, (uint)sample);
    write_path_bounce(ist, state, 0u);
    write_path_diffuse_bounce(ist, state, 0u);
    write_path_glossy_bounce(ist, state, 0u);
    write_path_transmission_bounce(ist, state, 0u);
    write_path_transparent_bounce(ist, state, 0u);
    write_path_volume_bounce(ist, state, 0u);
    write_path_volume_bounds_bounce(ist, state, 0u);

    /* source: path_state.h:54-56 - runtime guard, no compile guard. */
    if ((kernel_features & KERNEL_FEATURE_NODE_PORTAL) != 0u) {
        write_path_portal_bounce(ist, state, 0u);
    }

    /* source: path_state.h:57-65 */
    write_path_rng_pixel(ist, state, rng_pixel);
    write_path_rng_offset(ist, state, PRNG_BOUNCE_NUM);
    write_path_visibility(ist, state, PATH_RAY_VISIBILITY_CAMERA);
    write_path_flag(ist, state, PATH_RAY_MIS_SKIP | PATH_RAY_TRANSPARENT_BACKGROUND);
    write_path_mis_ray_pdf(ist, state, 0.0f);
    /* FLT_MAX bit pattern 0x7F7FFFFF - same spelling the Slang math core
     * uses (math_core.slang:511). */
    write_path_min_ray_pdf(ist, state, asfloat(0x7F7FFFFFu));
    write_path_continuation_probability(ist, state, 1.0f);
    write_path_throughput(ist, state, throughput);
    write_path_optical_depth(ist, state, 0.0f);

    /* DEFERRED: __PATH_GUIDING__ block, path_state.h:66-78 - file header. */

    /* source: path_state.h:79-81 - __MNEE__ is unconditionally defined
     * (features.h:127; the __KERNEL_FEATURES__ per-kernel undef does not
     * exist in this backend) and the host TU compiles it too, so the write is
     * unconditional - exactly as written in Cycles. */
    write_path_mnee(ist, state, 0u);

    /* source: path_state.h:83-85 */
    write_isect_object(ist, state, OBJECT_NONE_I);
    write_isect_prim(ist, state, PRIM_NONE_I);
    write_isect_type(ist, state, PRIMITIVE_NONE_I);

    /* source: path_state.h:87-94 - volume stack slots 0 and 1. */
    if ((kernel_features & KERNEL_FEATURE_VOLUME) != 0u) {
        write_volume_stack_object(ist, 0u, state, background_object_index);
        write_volume_stack_shader(ist, 0u, state, background_volume_shader);
        write_volume_stack_object(ist, 1u, state, OBJECT_NONE_I);
        write_volume_stack_shader(ist, 1u, state, SHADER_NONE_I);
    }

    /* source: path_state.h:96-101 - read-modify-write of flag at :98, then
     * the denoising throughput (one_spectrum() -> (1,1,1),
     * types_spectrum.h:21). */
    if ((kernel_features & KERNEL_FEATURE_DENOISING) != 0u) {
        uint flag = read_path_flag(ist, state);
        write_path_flag(ist, state, flag | PATH_RAY_DENOISING_FEATURES);
        write_path_denoising_feature_throughput(ist, state, float3(1.0f, 1.0f, 1.0f));
    }

    /* source: path_state.h:103-107 */
    if ((kernel_features & KERNEL_FEATURE_LIGHT_LINKING) != 0u) {
        write_path_mis_ray_object(ist, state, background_object_index);
    }
}

/* =========================================================================== *
 *  integrator_state_write_ray - FAITHFUL PORT
 *  source: src/kernel/integrator/state_util.h:25-60
 * =========================================================================== *
 * The Cycles function has two branches: the packed-state one
 * (state_util.h:26-47) is compiled only under
 * `defined(__INTEGRATOR_GPU_PACKED_STATE__) && defined(__KERNEL_GPU__)`, and
 * __INTEGRATOR_GPU_PACKED_STATE__ is defined for Apple ARM64 only
 * (kernel/types.h:626-628; see lastsession's S1b-5 record) - not this
 * Windows/MSVC target. This port therefore follows the NON-PACKED branch,
 * which writes the seven members individually.
 *
 * Ray -> explicit scalars: this backend has no Ray type yet (math_core's
 * camera_sample also passes split scalars). dP/dD are floats in the state
 * (state_template.h:81,83), matching the non-packed C++ writes at
 * state_util.h:57-58. */
void integrator_state_write_ray(IntegratorStateGPU* ist, uint state,
                                const float3 P, const float dP,
                                const float3 D, const float dD,
                                const float tmin, const float tmax,
                                const float time)
{
    /* source: state_util.h:52-58 (the #else / non-packed branch) */
    write_ray_P(ist, state, P);
    write_ray_D(ist, state, D);
    write_ray_tmin(ist, state, tmin);
    write_ray_tmax(ist, state, tmax);
    write_ray_time(ist, state, time);
    write_ray_dP(ist, state, dP);
    write_ray_dD(ist, state, dD);
}
