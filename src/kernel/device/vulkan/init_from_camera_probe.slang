// SPDX-FileCopyrightText: 2011-2022 Blender Foundation
// SPDX-License-Identifier: Apache-2.0

/* init_from_camera_probe.slang : M4f SLICE 1 gate probe (host env
 * CYCLES_VULKAN_INIT_FROM_CAMERA_PROBE=1, SPV via
 * CYCLES_VULKAN_INIT_FROM_CAMERA_PROBE_SPV or ./init_from_camera_probe.spv).
 *
 * WHAT IT PROVES: the ported integrate_camera_sample() (init_from_camera.slang)
 * produces exactly what the REAL Cycles function produces on the host
 * (src/kernel/integrator/init_from_camera.h:19-52 compiled into the binary).
 *
 * 64 threads per workgroup, one test case per thread: (sample, x, y) drawn
 * over a 64x48 tile x samples {0,1,5,15,255,1023}. rng_pixel is computed on
 * BOTH sides with the same real expression (init_from_camera.h:93) and is
 * reported back so the host can compare it bit-exactly.
 *
 * Reference split (be precise about what the host compares):
 *   - the PUBLIC outputs (ray P/D/tmin/tmax/time, T, r_cache_miss) come from
 *     the real integrate_camera_sample() call on the host;
 *   - the two OBSERVATION outputs (rand_filter, rand_time_lens) are compared
 *     against the real path_rng_2D / path_rng_3D calls plus the four packing
 *     lines of init_from_camera.h:28-48, which the host reproduces verbatim
 *     (each line cited at the comparison site). Those four lines cannot be
 *     read out of the real function because they are locals.
 * Floats: |gpu-ref| <= 1e-5 * max(1,|ref|) (same tolerance as the math probe).
 *
 * neg_mode (MathParams, gate-only, NOT in Cycles) applies ONE probe-side
 * INPUT fault per pass so the gate can prove each comparison is live. The
 * ported function contains no fault switch:
 *   1 -> rng_pixel fed to the function is XOR 1 (reported value stays clean)
 *        -> the PRNG draws must disagree for every sample != 0 case;
 *   2 -> the function is fed sample+1 (reported rng_pixel stays clean)
 *        -> for sample == 0 the 0.5/0.5 filter draw must disagree;
 *   3 -> rastertocamera transposed in the params copy (same fault as the math
 *        probe's neg 3) -> the ray must disagree.
 * The clean pass (neg_mode 0) must MATCH for all three seeds. */

#include "init_from_camera.slang"

/* Case kind - only one kind exists here; kept for layout symmetry with the
 * math probe (host enum in kernels.cpp must match). */
static const uint KIND_INIT_CAM = 0;

/* One test case. Layout locked against InitCamCase in kernels.cpp. */
struct InitCamCase
{
    int  sample;  // scheduled sample (init_from_camera.h:63)
    uint x, y;    // pixel coordinates
};                // 12 bytes

/* One result. Layout locked against InitCamResult in kernels.cpp.
 *
 * f[] layout (host MUST match, see the comparison in kernels.cpp):
 *   [0],[1]        rand_filter.x, .y     init_from_camera.h:28-29
 *   [2],[3],[4]    rand_time_lens.x/y/z  init_from_camera.h:35-37
 *   [5]            rand_time             init_from_camera.h:47
 *   [6],[7]        rand_lens.x, .y       init_from_camera.h:48
 *   [8],[9],[10]   ray.P                 init_from_camera.h:51 -> camera_sample
 *   [11],[12],[13] ray.D
 *   [14],[15]      ray.tmin, ray.tmax
 *   [16]           ray.time
 *   [17],[18],[19] T (Spectrum = float3, util/types_spectrum.h:13)
 *   [20..23]       unused (written 0) */
struct InitCamResult
{
    uint u0;      // rng_pixel passed INTO the function (bit-exact check)
    uint u1;      // r_cache_miss (bit-exact check)
    uint nfloat;  // number of valid floats in f[] (20)
    uint pad;
    float f[24];
};                // 16 + 96 = 112 bytes

/* Push block: 3 device pointers + count = 32 bytes (matches the host struct;
 * same shape as the math probe's push, so the push-constant range is 32 B). */
struct Push
{
    InitCamCase*  cases;
    InitCamResult* out;
    MathParams*   params;
    uint          num_cases;
};

[[vk::push_constant]]
ConstantBuffer<Push> pc;

[shader("compute")]
[numthreads(64, 1, 1)]
void init_from_camera_main(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= pc.num_cases)
        return;

    InitCamCase c = pc.cases[i];
    MathParams p = *(pc.params);

    /* ---- probe-side input faults (see header): the port has NO switch ---- */
    int sample_arg = c.sample;
    if (p.neg_mode == 2)
        sample_arg += 1;

    /* init_from_camera.h:93 on the GPU side, computed with the clean sample
     * so neg 1 perturbs only what is FEED into the function (the reported
     * u0 stays clean and the mismatch must show up in the draws). */
    uint rng_reported = path_rng_pixel_init(p, c.sample, (int)c.x, (int)c.y, false);
    uint rng_fed = rng_reported;
    if (p.neg_mode == 1)
        rng_fed ^= 1u;

    /* neg 3: transpose the 4x4 rastertocamera on the scalar storage
     * (same construction as math_probe.slang:150-158). */
    MathParams pf = p;
    if (p.neg_mode == 3) {
        pf.rtc_x_x = p.rtc_x_x; pf.rtc_x_y = p.rtc_y_x; pf.rtc_x_z = p.rtc_z_x; pf.rtc_x_w = p.rtc_w_x;
        pf.rtc_y_x = p.rtc_x_y; pf.rtc_y_y = p.rtc_y_y; pf.rtc_y_z = p.rtc_z_y; pf.rtc_y_w = p.rtc_w_y;
        pf.rtc_z_x = p.rtc_x_z; pf.rtc_z_y = p.rtc_y_z; pf.rtc_z_z = p.rtc_z_z; pf.rtc_z_w = p.rtc_w_z;
        pf.rtc_w_x = p.rtc_x_w; pf.rtc_w_y = p.rtc_y_w; pf.rtc_w_z = p.rtc_z_w; pf.rtc_w_w = p.rtc_w_w;
    }

    /* ---- the port under test ---- */
    float3 ray_P, ray_D;
    float ray_tmin, ray_tmax, ray_time;
    int r_cache_miss;
    float2 obs_rand_filter;
    float3 obs_rand_time_lens;
    float obs_rand_time;
    float2 obs_rand_lens;
    float3 T = integrate_camera_sample(pf, sample_arg, (int)c.x, (int)c.y, rng_fed,
                                       ray_P, ray_D, ray_tmin, ray_tmax, ray_time,
                                       r_cache_miss,
                                       obs_rand_filter, obs_rand_time_lens,
                                       obs_rand_time, obs_rand_lens);

    InitCamResult r;
    r.u0 = rng_reported;
    r.u1 = (uint)(r_cache_miss != 0 ? 1 : 0);
    r.nfloat = 20u;
    r.pad = 0u;
    r.f[0] = obs_rand_filter.x;
    r.f[1] = obs_rand_filter.y;
    r.f[2] = obs_rand_time_lens.x;
    r.f[3] = obs_rand_time_lens.y;
    r.f[4] = obs_rand_time_lens.z;
    r.f[5] = obs_rand_time;       // rand_time (init_from_camera.h:47)
    r.f[6] = obs_rand_lens.x;     // rand_lens.x (init_from_camera.h:48)
    r.f[7] = obs_rand_lens.y;     // rand_lens.y
    r.f[8]  = ray_P.x;  r.f[9]  = ray_P.y;  r.f[10] = ray_P.z;
    r.f[11] = ray_D.x;  r.f[12] = ray_D.y;  r.f[13] = ray_D.z;
    r.f[14] = ray_tmin; r.f[15] = ray_tmax;
    r.f[16] = ray_time;
    r.f[17] = T.x; r.f[18] = T.y; r.f[19] = T.z;
    r.f[20] = 0.0f; r.f[21] = 0.0f; r.f[22] = 0.0f; r.f[23] = 0.0f;

    pc.out[i] = r;
}
