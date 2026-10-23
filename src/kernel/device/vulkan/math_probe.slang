// SPDX-FileCopyrightText: 2011-2022 Blender Foundation
// SPDX-License-Identifier: Apache-2.0

/* math_probe.slang : M4e STEP 1a gate probe (env CYCLES_VULKAN_MATH_PROBE=1).
 *
 * MANY threads (64 per workgroup, ceil(num_cases/64) workgroups). Each thread
 * handles ONE test case drawn from a host-built case array covering pixels over
 * a 64x48 tile, samples {0,1,5,15,255,1023}, seeds {0,1,large}, and dimensions
 * (low/mid/high) for hash / sobol / path_rng / camera. The camera matrices and
 * sampling config come from a host-filled MathParams buffer taken from the REAL
 * kernel data of the loaded scene (NOT from the KernelParamsVulkan blob).
 *
 * The host compares each case against the REAL Cycles headers compiled on the
 * host (same technique as the M4d closure probe). Integer outputs (hash_iqnt2d,
 * path_rng_pixel_init) are compared bit-exact; sobol/path_rng/camera floats are
 * compared with |a-b| <= 1e-5 * max(1,|b|) (and sobol floats are additionally
 * bit-compared as an informational integer check, since uint_to_float_excl is a
 * single FMA-free multiply).
 *
 * neg_mode (params) applies ONE fault so the gate can prove it is detected:
 *   1 -> perturb one sobol table entry, 2 -> wrong seed XOR in pixel init,
 *   3 -> transposed rastertocamera. 0 -> faithful. */

#include "math_core.slang"

// Case kinds (must match the host's enum in kernels.cpp).
static const uint KIND_HASH       = 0;
static const uint KIND_PIXEL_INIT = 1;
static const uint KIND_SOBOL1D    = 2;
static const uint KIND_SOBOL2D    = 3;
static const uint KIND_SOBOL3D    = 4;
static const uint KIND_SOBOL4D    = 5;
static const uint KIND_RNG1D      = 6;
static const uint KIND_RNG2D      = 7;
static const uint KIND_RNG3D      = 8;
static const uint KIND_RNG4D      = 9;
static const uint KIND_CAMERA     = 10;

// One test case. Layout locked against the host struct ProbeCase (kernels.cpp).
struct ProbeCase
{
    uint kind;
    int  sample;
    uint x, y;
    uint dim;    // dimension / dimension_set (sobol + rng)
    uint seed;   // sobol scramble seed (sobol kinds)
    uint mask;   // shuffled_index_mask (sobol kinds)
    uint index;  // sobol index (sobol kinds)
    float fx, fy, fz, fw; // camera: filter_uv.x, filter_uv.y, time (lens hardcoded 0,0)
};

// One result. Layout locked against the host struct ProbeResult (kernels.cpp).
struct ProbeResult
{
    uint u0;      // integer output (bit-exact), valid if has_u != 0
    uint has_u;   // 0/1
    uint nfloat;  // number of valid floats in f[]
    uint pad;
    float f[12];  // float outputs
};

struct Push
{
    ProbeCase*  cases;
    ProbeResult* out;
    MathParams* params;
    uint        num_cases;
};

[[vk::push_constant]]
ConstantBuffer<Push> pc;

[shader("compute")]
[numthreads(64, 1, 1)]
void math_probe_main(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= pc.num_cases)
        return;

    ProbeCase* cases = pc.cases;
    ProbeResult* out = pc.out;
    MathParams p = *(pc.params);

    // Per-family fault switches (only one is active per dispatch).
    bool neg_sobol = (p.neg_mode == 1);
    bool neg_seed  = (p.neg_mode == 2);
    bool neg_cam   = (p.neg_mode == 3);

    ProbeCase c = cases[i];

    ProbeResult r;
    r.u0 = 0u; r.has_u = 0u; r.nfloat = 0u; r.pad = 0u;
    [unroll]
    for (int k = 0; k < 12; k++)
        r.f[k] = 0.0f;

    if (c.kind == KIND_HASH) {
        r.u0 = hash_iqnt2d(c.x, c.y);          // integer, bit-exact
        r.has_u = 1u;
    }
    else if (c.kind == KIND_PIXEL_INIT) {
        r.u0 = path_rng_pixel_init(p, c.sample, (int)c.x, (int)c.y, neg_seed); // integer, bit-exact
        r.has_u = 1u;
    }
    else if (c.kind == KIND_SOBOL1D) {
        r.f[0] = sobol_burley_sample_1D(c.index, c.dim, c.seed, c.mask, neg_sobol);
        r.nfloat = 1u;
    }
    else if (c.kind == KIND_SOBOL2D) {
        float2 v = sobol_burley_sample_2D(c.index, c.dim, c.seed, c.mask, neg_sobol);
        r.f[0] = v.x; r.f[1] = v.y;
        r.nfloat = 2u;
    }
    else if (c.kind == KIND_SOBOL3D) {
        float3 v = sobol_burley_sample_3D(c.index, c.dim, c.seed, c.mask, neg_sobol);
        r.f[0] = v.x; r.f[1] = v.y; r.f[2] = v.z;
        r.nfloat = 3u;
    }
    else if (c.kind == KIND_SOBOL4D) {
        float4 v = sobol_burley_sample_4D(c.index, c.dim, c.seed, c.mask, neg_sobol);
        r.f[0] = v.x; r.f[1] = v.y; r.f[2] = v.z; r.f[3] = v.w;
        r.nfloat = 4u;
    }
    else if (c.kind == KIND_RNG1D) {
        uint rng_pixel = path_rng_pixel_init(p, c.sample, (int)c.x, (int)c.y, false);
        r.f[0] = path_rng_1D(p, rng_pixel, (uint)c.sample, (int)c.dim, neg_sobol);
        r.nfloat = 1u;
    }
    else if (c.kind == KIND_RNG2D) {
        uint rng_pixel = path_rng_pixel_init(p, c.sample, (int)c.x, (int)c.y, false);
        float2 v = path_rng_2D(p, rng_pixel, (uint)c.sample, (int)c.dim, neg_sobol);
        r.f[0] = v.x; r.f[1] = v.y;
        r.nfloat = 2u;
    }
    else if (c.kind == KIND_RNG3D) {
        uint rng_pixel = path_rng_pixel_init(p, c.sample, (int)c.x, (int)c.y, false);
        float3 v = path_rng_3D(p, rng_pixel, (uint)c.sample, (int)c.dim, neg_sobol);
        r.f[0] = v.x; r.f[1] = v.y; r.f[2] = v.z;
        r.nfloat = 3u;
    }
    else if (c.kind == KIND_RNG4D) {
        uint rng_pixel = path_rng_pixel_init(p, c.sample, (int)c.x, (int)c.y, false);
        float4 v = path_rng_4D(p, rng_pixel, (uint)c.sample, (int)c.dim, neg_sobol);
        r.f[0] = v.x; r.f[1] = v.y; r.f[2] = v.z; r.f[3] = v.w;
        r.nfloat = 4u;
    }
    else if (c.kind == KIND_CAMERA) {
        // Optional fault: transpose rastertocamera before feeding camera_sample.
        MathParams pc_params = p;
        if (neg_cam) {
            // Transpose the 4x4 rastertocamera: new row j = (old col j).
            // Operates on the scalar storage (rtc_*_<row>_<col>).
            pc_params.rtc_x_x = p.rtc_x_x; pc_params.rtc_x_y = p.rtc_y_x; pc_params.rtc_x_z = p.rtc_z_x; pc_params.rtc_x_w = p.rtc_w_x;
            pc_params.rtc_y_x = p.rtc_x_y; pc_params.rtc_y_y = p.rtc_y_y; pc_params.rtc_y_z = p.rtc_z_y; pc_params.rtc_y_w = p.rtc_w_y;
            pc_params.rtc_z_x = p.rtc_x_z; pc_params.rtc_z_y = p.rtc_y_z; pc_params.rtc_z_z = p.rtc_z_z; pc_params.rtc_z_w = p.rtc_w_z;
            pc_params.rtc_w_x = p.rtc_x_w; pc_params.rtc_w_y = p.rtc_y_w; pc_params.rtc_w_z = p.rtc_z_w; pc_params.rtc_w_w = p.rtc_w_w;
        }
        float2 filter_uv = float2(c.fx, c.fy);
        float2 lens_uv = float2(0.0f, 0.0f);
        float3 P, D; float tmin, tmax, time; bool ok;
        camera_sample(pc_params, (int)c.x, (int)c.y, filter_uv, c.fz, lens_uv, P, D, tmin, tmax, time, ok);
        if (ok) {
            r.f[0] = P.x; r.f[1] = P.y; r.f[2] = P.z;
            r.f[3] = D.x; r.f[4] = D.y; r.f[5] = D.z;
            r.f[6] = tmin; r.f[7] = tmax; r.f[8] = time;
            r.nfloat = 9u;
        }
        else {
            r.nfloat = 0u; // unsupported camera config -> host flags it.
        }
    }

    out[i] = r;
}
