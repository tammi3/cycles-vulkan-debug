// SPDX-FileCopyrightText: 2011-2022 Blender Foundation
// SPDX-License-Identifier: Apache-2.0

/* math_core.slang : M4e STEP 1a — faithful Slang port of the Cycles sampler /
 * hash / camera MATH CORE. Functions only, no entry point, no integrator state,
 * no queue. Used by math_probe.slang (host: src/device/vulkan/kernels.cpp
 * vulkan_math_probe).
 *
 * Every function carries a `// source: <path>:<line>` citation. This is a
 * FAITHFUL port (AGENTS.md §4): the Cycles logic is reproduced as written, no
 * divergence. Integer/hash outputs are bit-exact with the host; float outputs
 * are compared with a relative tolerance (GPU FMA vs MSVC is not bit-identical).
 *
 * Covered by the S1a gate scene (perspective camera, aperture 0, shuttertime
 * -1, pixel_jitter != FLT_MAX, SAMPLING_PATTERN_SOBOL_BURLEY):
 *   - hash_iqnt2d, hash_hp_uint/_seeded, uint_to_float_excl
 *   - sobol_burley (+ integer form), sobol_burley_sample_1D/2D/3D/4D
 *   - blue_noise_indexing, path_rng_pixel_init, path_rng_1D/2D/3D/4D
 *   - camera_sample_perspective, camera_sample (perspective branch + clipping)
 * NOT covered (stated, deferred): SAMPLING_PATTERN_TABULATED_SOBOL /
 *   BLUE_NOISE_* (tabulated_sobol table + morton path not exercised), DOF
 *   (aperturesize > 0), motion blur / shutter table (shuttertime != -1),
 *   panorama / orthographic / custom cameras, the FLT_MAX pixel-filter-table
 *   branch (needs kernel_data.tables.lookup_table, not passed in S1a), stereo,
 *   and ray differentials (dP/dD are not part of the S1a comparison). */

/* ---------------------------------------------------------------------------
 * Bit / math primitives (util/math_base.h, util/hash.h).
 * Implemented portably (no HW intrinsics) so the integer results are bit-exact
 * with the host MSVC build and no Slang intrinsic name has to be guessed.
 * ------------------------------------------------------------------------- */

// source: util/hash.h:14 (uint_to_float_excl)
float uint_to_float_excl(uint n)
{
    /* [0, uint_max] -> [0.0, 1.0). Divide by 4294967808, not 2^32, to keep the
     * top of the range below 1.0 (see comment in Cycles). */
    return (float)n * (1.0f / 4294967808.0f);
}

// source: util/math_base.h:813 (reverse_integer_bits, portable fallback path).
// Bit reversal is exact; the host MSVC path (_byteswap_ulong) yields the same
// bits, so this is bit-identical on both sides.
uint reverse_integer_bits(uint x)
{
    // Flip pairwise.
    x = ((x & 0x55555555u) << 1) | ((x & 0xAAAAAAAAu) >> 1);
    // Flip pairs.
    x = ((x & 0x33333333u) << 2) | ((x & 0xCCCCCCCCu) >> 2);
    // Flip nibbles.
    x = ((x & 0x0F0F0F0Fu) << 4) | ((x & 0xF0F0F0F0u) >> 4);
    // Flip bytes.
    x = ((x & 0x00FF00FFu) << 8) | ((x & 0xFF00FF00u) >> 8);
    return (x << 16) | (x >> 16);
}

// source: util/math_base.h:718 (count_leading_zeros, MSVC _BitScanReverse path:
// 31 - msb_index). Portable loop; x != 0 on all call sites (Cycles asserts this).
uint count_leading_zeros(uint x)
{
    if (x == 0u)
        return 32u;
    uint n = 0u;
    for (int i = 31; i >= 0; --i) {
        if (((x >> i) & 1u) != 0u)
            break;
        n++;
    }
    return n;
}

// source: util/hash.h:511 (hash_hp_uint)
uint hash_hp_uint(uint i)
{
    i ^= i >> 16;
    i *= 0x21f0aaadu;
    i ^= i >> 15;
    i *= 0xd35a2d97u;
    i ^= i >> 15;
    return i ^ 0xe6fe3bebu;
}

// source: util/hash.h:526 (hash_hp_seeded_uint)
uint hash_hp_seeded_uint(uint i, uint seed)
{
    seed ^= seed << 19;
    return hash_hp_uint(i ^ seed);
}

// source: util/hash.h:625 (hash_iqnt2d)
uint hash_iqnt2d(uint x, uint y)
{
    uint qx = 1103515245u * ((x >> 1u) ^ y);
    uint qy = 1103515245u * ((y >> 1u) ^ x);
    uint n = 1103515245u * (qx ^ (qy >> 3u));
    return n;
}

/* ---------------------------------------------------------------------------
 * Owen-scrambling / morton helpers (kernel/sample/util.h).
 * ------------------------------------------------------------------------- */

// source: kernel/sample/util.h:18 (reversed_bit_owen)
uint reversed_bit_owen(uint n, uint seed)
{
    n ^= n * 0x3d20adeau;
    n += seed;
    n *= (seed >> 16) | 1u;
    n ^= n * 0x05526c56u;
    n ^= n * 0x53a22864u;
    return n;
}

// source: kernel/sample/util.h:35 (reversed_bit_owen_base4)
uint reversed_bit_owen_base4(uint n, uint seed)
{
    n ^= n * 0x3d20adeau;
    n ^= (n >> 1) & (n << 1) & 0x55555555u;
    n += seed;
    n *= (seed >> 16) | 1u;
    n ^= (n >> 1) & (n << 1) & 0x55555555u;
    n ^= n * 0x05526c56u;
    n ^= n * 0x53a22864u;
    return n;
}

// source: kernel/sample/util.h:51 (nested_uniform_scramble)
uint nested_uniform_scramble(uint i, uint seed)
{
    return reverse_integer_bits(reversed_bit_owen(reverse_integer_bits(i), seed));
}

// source: kernel/sample/util.h:59 (nested_uniform_scramble_base4)
uint nested_uniform_scramble_base4(uint i, uint seed)
{
    return reverse_integer_bits(reversed_bit_owen_base4(reverse_integer_bits(i), seed));
}

// source: kernel/sample/util.h:64 (expand_bits)
uint expand_bits(uint x)
{
    x &= 0x0000ffffu;
    x = (x ^ (x << 8)) & 0x00ff00ffu;
    x = (x ^ (x << 4)) & 0x0f0f0f0fu;
    x = (x ^ (x << 2)) & 0x33333333u;
    x = (x ^ (x << 1)) & 0x55555555u;
    return x;
}

// source: kernel/sample/util.h:74 (morton2d)
uint morton2d(uint x, uint y)
{
    return (expand_bits(x) << 1) | expand_bits(y);
}

/* ---------------------------------------------------------------------------
 * Sobol sampler (kernel/sample/sobol_burley.h + kernel/tables.h).
 *
 * The table is embedded as a static const array (4 x 32 = 128 uints = 512
 * bytes). Rationale: it is a small compile-time constant that SPIR-V embeds in
 * the module as a readonly object, so it needs no descriptor set, no buffer and
 * no device address — exactly like Cycles' `ccl_inline_constant`
 * (kernel/tables.h:81). A storage buffer would add an ABI surface and a bind
 * for no benefit at this size.
 * ------------------------------------------------------------------------- */

// source: kernel/tables.h:81 (sobol_burley_table[4][32])
static const uint sobol_burley_table[4][32] = {
    {
        0x00000001u, 0x00000002u, 0x00000004u, 0x00000008u,
        0x00000010u, 0x00000020u, 0x00000040u, 0x00000080u,
        0x00000100u, 0x00000200u, 0x00000400u, 0x00000800u,
        0x00001000u, 0x00002000u, 0x00004000u, 0x00008000u,
        0x00010000u, 0x00020000u, 0x00040000u, 0x00080000u,
        0x00100000u, 0x00200000u, 0x00400000u, 0x00800000u,
        0x01000000u, 0x02000000u, 0x04000000u, 0x08000000u,
        0x10000000u, 0x20000000u, 0x40000000u, 0x80000000u,
    },
    {
        0x00000001u, 0x00000003u, 0x00000005u, 0x0000000fu,
        0x00000011u, 0x00000033u, 0x00000055u, 0x000000ffu,
        0x00000101u, 0x00000303u, 0x00000505u, 0x00000f0fu,
        0x00001111u, 0x00003333u, 0x00005555u, 0x0000ffffu,
        0x00010001u, 0x00030003u, 0x00050005u, 0x000f000fu,
        0x00110011u, 0x00330033u, 0x00550055u, 0x00ff00ffu,
        0x01010101u, 0x03030303u, 0x05050505u, 0x0f0f0f0fu,
        0x11111111u, 0x33333333u, 0x55555555u, 0xffffffffu,
    },
    {
        0x00000001u, 0x00000003u, 0x00000006u, 0x00000009u,
        0x00000017u, 0x0000003au, 0x00000071u, 0x000000a3u,
        0x00000116u, 0x00000339u, 0x00000677u, 0x000009aau,
        0x00001601u, 0x00003903u, 0x00007706u, 0x0000aa09u,
        0x00010117u, 0x0003033au, 0x00060671u, 0x000909a3u,
        0x00171616u, 0x003a3939u, 0x00717777u, 0x00a3aaaau,
        0x01170001u, 0x033a0003u, 0x06710006u, 0x09a30009u,
        0x16160017u, 0x3939003au, 0x77770071u, 0xaaaa00a3u,
    },
    {
        0x00000001u, 0x00000003u, 0x00000004u, 0x0000000au,
        0x0000001fu, 0x0000002eu, 0x00000045u, 0x000000c9u,
        0x0000011bu, 0x000002a4u, 0x0000079au, 0x00000b67u,
        0x0000101eu, 0x0000302du, 0x00004041u, 0x0000a0c3u,
        0x0001f104u, 0x0002e28au, 0x000457dfu, 0x000c9baeu,
        0x0011a105u, 0x002a7289u, 0x0079e7dbu, 0x00b6dba4u,
        0x0100011au, 0x030002a7u, 0x0400079eu, 0x0a000b6du,
        0x1f001001u, 0x2e003003u, 0x45004004u, 0xc900a00au,
    },
};

// source: kernel/sample/sobol_burley.h:37 (sobol_burley). `neg` is a gate-only
// hook (NOT in Cycles): when true it perturbs ONE table entry
// (sobol_burley_table[1][0]) so the S1a negative test can prove a single-table-
// fault is detected. It is always false in the faithful path.
uint sobol_burley_uint(uint rev_bit_index, uint dimension, uint scramble_seed, bool neg)
{
    uint result = 0u;
    if (dimension == 0u) {
        // Fast-path for dimension 0 (Van der Corput). sobol_burley.h:43-48.
        result = reverse_integer_bits(rev_bit_index);
    }
    else {
        uint i = 0u;
        while (rev_bit_index != 0u) {
            uint j = count_leading_zeros(rev_bit_index);
            uint entry = sobol_burley_table[dimension][i + j];
            if (neg && dimension == 1u && i == 0u)
                entry ^= 1u; // NEGATIVE TEST: perturb the first dimension-1 table read.
            result ^= entry;
            i += j + 1u;
            // Can't do "<<= j+1" (overflow). sobol_burley.h:56-59.
            rev_bit_index <<= j;
            rev_bit_index <<= 1u;
        }
    }
    // Apply Owen scrambling. sobol_burley.h:64.
    result = reverse_integer_bits(reversed_bit_owen(result, scramble_seed));
    return result;
}

// source: kernel/sample/sobol_burley.h:37-67 (sobol_burley) = uint_to_float_excl of the above.
float sobol_burley(uint rev_bit_index, uint dimension, uint scramble_seed, bool neg)
{
    return uint_to_float_excl(sobol_burley_uint(rev_bit_index, dimension, scramble_seed, neg));
}

// source: kernel/sample/sobol_burley.h:99 (sobol_burley_sample_1D)
float sobol_burley_sample_1D(uint index, uint dimension, uint seed, uint shuffled_index_mask, bool neg)
{
    seed ^= hash_hp_uint(dimension);
    index = reversed_bit_owen(reverse_integer_bits(index), seed ^ 0xbff95bfeu);
    index &= shuffled_index_mask;
    return sobol_burley(index, 0u, seed ^ 0x635c77bdu, neg);
}

// source: kernel/sample/sobol_burley.h:126 (sobol_burley_sample_2D)
float2 sobol_burley_sample_2D(uint index, uint dimension_set, uint seed, uint shuffled_index_mask, bool neg)
{
    seed ^= hash_hp_uint(dimension_set);
    index = reversed_bit_owen(reverse_integer_bits(index), seed ^ 0xf8ade99au);
    index &= shuffled_index_mask;
    return float2(sobol_burley(index, 0u, seed ^ 0xe0aaaf76u, neg),
                  sobol_burley(index, 1u, seed ^ 0x94964d4eu, neg));
}

// source: kernel/sample/sobol_burley.h:154 (sobol_burley_sample_3D)
float3 sobol_burley_sample_3D(uint index, uint dimension_set, uint seed, uint shuffled_index_mask, bool neg)
{
    seed ^= hash_hp_uint(dimension_set);
    index = reversed_bit_owen(reverse_integer_bits(index), seed ^ 0xcaa726acu);
    index &= shuffled_index_mask;
    return float3(sobol_burley(index, 0u, seed ^ 0x9e78e391u, neg),
                  sobol_burley(index, 1u, seed ^ 0x67c33241u, neg),
                  sobol_burley(index, 2u, seed ^ 0x78c395c5u, neg));
}

// source: kernel/sample/sobol_burley.h:183 (sobol_burley_sample_4D)
float4 sobol_burley_sample_4D(uint index, uint dimension_set, uint seed, uint shuffled_index_mask, bool neg)
{
    seed ^= hash_hp_uint(dimension_set);
    index = reversed_bit_owen(reverse_integer_bits(index), seed ^ 0xc2c1a055u);
    index &= shuffled_index_mask;
    return float4(sobol_burley(index, 0u, seed ^ 0x39468210u, neg),
                  sobol_burley(index, 1u, seed ^ 0xe9d8a845u, neg),
                  sobol_burley(index, 2u, seed ^ 0x5f32b482u, neg),
                  sobol_burley(index, 3u, seed ^ 0x1524cc56u, neg));
}

/* ---------------------------------------------------------------------------
 * Probe parameter blocks (host-filled from the REAL kernel data of the loaded
 * scene; NOT read from the KernelParamsVulkan blob in S1a — see report).
 * ------------------------------------------------------------------------- */

// Camera + sampling fields the S1a math reads, host-filled from
// kernel_data.cam / kernel_data.integrator (source: kernel/types.h:1162,121).
//
// ALL members are 4-byte scalars (no float2/float4/float3): this struct is
// shared with the host through a device-address buffer, and scalar-only layout
// is identical in Slang and C++ (contiguous 4-byte fields, no alignment
// padding), so the two sides can never disagree on offsets (AGENTS.md section
// 4: "Prefer scalar members ... so C++ and Slang cannot disagree on padding").
// Matrices are stored row-by-row as scalars; the camera functions rebuild the
// float4 rows locally (local variables carry no cross-ABI layout).
struct MathParams
{
    // -- kernel_data.cam subset (KernelCamera, types.h:1162) ---------------
    int cam_type;                  // cam.type
    int have_perspective_motion;   // cam.have_perspective_motion
    float aperturesize;            // cam.aperturesize
    int num_motion_steps;          // cam.num_motion_steps
    float interocular_offset;      // cam.interocular_offset
    float shuttertime;             // cam.shuttertime
    int rolling_shutter_type;      // cam.rolling_shutter_type
    float nearclip;                // cam.nearclip
    float cliplength;              // cam.cliplength
    // integrator.pixel_jitter (see report: overridden != FLT_MAX)
    float pixel_jitter_x;
    float pixel_jitter_y;
    // cam.cameratoworld (Transform, 3 rows x,y,z as float4)
    float ctw_x_x, ctw_x_y, ctw_x_z, ctw_x_w;
    float ctw_y_x, ctw_y_y, ctw_y_z, ctw_y_w;
    float ctw_z_x, ctw_z_y, ctw_z_z, ctw_z_w;
    // cam.rastertocamera (ProjectionTransform, 4 rows x,y,z,w as float4)
    float rtc_x_x, rtc_x_y, rtc_x_z, rtc_x_w;
    float rtc_y_x, rtc_y_y, rtc_y_z, rtc_y_w;
    float rtc_z_x, rtc_z_y, rtc_z_z, rtc_z_w;
    float rtc_w_x, rtc_w_y, rtc_w_z, rtc_w_w;
    // -- kernel_data.integrator subset -------------------------------------
    int sampling_pattern;          // integrator.sampling_pattern (S1a: SOBOL_BURLEY)
    uint seed;                     // integrator.seed (xored into rng_pixel)
    uint sobol_index_mask;         // integrator.sobol_index_mask
    uint blue_noise_sequence_length; // integrator.blue_noise_sequence_length (unused in SOBOL_BURLEY)
    // -- gate-only negative-test switch (NOT in Cycles) --------------------
    int neg_mode;                  // 0 clean, 1 sobol table, 2 seed xor, 3 transpose rastertocamera
};

/* ---------------------------------------------------------------------------
 * pattern.h (kernel/sample/pattern.h). SOBOL_BURLEY path ported; the
 * TABULATED_SOBOL / BLUE_NOISE_* branches are not exercised by the gate scene
 * and are reported as not covered.
 * ------------------------------------------------------------------------- */

// source: kernel/sample/pattern.h:34 (blue_noise_indexing). SOBOL_BURLEY
// branch is the one the gate uses; the two blue-noise branches are ported
// faithfully (they need blue_noise_sequence_length, morton2d, and
// nested_uniform_scramble_base4, all present above) but are not exercised.
uint3 blue_noise_indexing(MathParams p, uint pixel_index, uint sample)
{
    if (p.sampling_pattern == 0) { // SAMPLING_PATTERN_SOBOL_BURLEY (types.h:122)
        // pattern.h:40
        return uint3(sample, pixel_index, p.sobol_index_mask);
    }
    if (p.sampling_pattern == 2) { // SAMPLING_PATTERN_BLUE_NOISE_PURE (types.h:124)
        // pattern.h:47-48
        pixel_index *= p.blue_noise_sequence_length;
        return uint3(sample + pixel_index, 0u, 0xffffffffu);
    }
    if (p.sampling_pattern == 3) { // SAMPLING_PATTERN_BLUE_NOISE_FIRST
        // pattern.h:56-60
        if (sample == 0u)
            return uint3(pixel_index, 0x0cd0519fu, 0xffffffffu);
        pixel_index *= p.blue_noise_sequence_length;
        return uint3((sample - 1u) + pixel_index, 0u, 0xffffffffu);
    }
    return uint3(0u, 0u, 0u); // kernel_assert(false) in Cycles.
}

// source: kernel/sample/pattern.h:66 (path_rng_1D). Only the SOBOL_BURLEY
// path is ported; the TABULATED_SOBOL branch (tabulated_sobol_sample_1D) is
// not covered.
float path_rng_1D(MathParams p, uint rng_pixel, uint sample, int dimension, bool neg)
{
    uint3 index = blue_noise_indexing(p, rng_pixel, sample);
    return sobol_burley_sample_1D(index.x, (uint)dimension, index.y, index.z, neg);
}

// source: kernel/sample/pattern.h:83 (path_rng_2D). SOBOL_BURLEY path only.
float2 path_rng_2D(MathParams p, uint rng_pixel, uint sample, int dimension, bool neg)
{
    uint3 index = blue_noise_indexing(p, rng_pixel, sample);
    return sobol_burley_sample_2D(index.x, (uint)dimension, index.y, index.z, neg);
}

// source: kernel/sample/pattern.h:100 (path_rng_3D). SOBOL_BURLEY path only.
float3 path_rng_3D(MathParams p, uint rng_pixel, uint sample, int dimension, bool neg)
{
    uint3 index = blue_noise_indexing(p, rng_pixel, sample);
    return sobol_burley_sample_3D(index.x, (uint)dimension, index.y, index.z, neg);
}

// source: kernel/sample/pattern.h:117 (path_rng_4D). SOBOL_BURLEY path only.
float4 path_rng_4D(MathParams p, uint rng_pixel, uint sample, int dimension, bool neg)
{
    uint3 index = blue_noise_indexing(p, rng_pixel, sample);
    return sobol_burley_sample_4D(index.x, (uint)dimension, index.y, index.z, neg);
}

// source: kernel/sample/pattern.h:134 (path_rng_pixel_init). For
// SOBOL_BURLEY (and TABULATED_SOBOL) this is hash_iqnt2d(x,y) ^ seed. The
// blue-noise fallback (morton2d + nested_uniform_scramble_base4) is ported but
// not exercised. `neg` (gate-only) perturbs the seed XOR to prove the negative
// test is detected.
uint path_rng_pixel_init(MathParams p, int sample, int x, int y, bool neg)
{
    uint pattern = (uint)p.sampling_pattern;
    if (pattern == 1u || pattern == 0u) { // TABULATED_SOBOL || SOBOL_BURLEY
        uint s = p.seed;
        if (neg)
            s += 1u; // NEGATIVE TEST: wrong seed XOR.
        return hash_iqnt2d((uint)x, (uint)y) ^ s; // pattern.h:143
    }
    // pattern.h:156 (blue-noise fallback; not exercised by the gate).
    return nested_uniform_scramble_base4(morton2d((uint)x, (uint)y), p.seed);
}

/* ---------------------------------------------------------------------------
 * Camera math (kernel/camera/camera.h). Perspective branch only, with the
 * clipping math. DOF / motion blur / panorama / ortho / custom / stereo /
 * ray-differentials / the FLT_MAX filter-table branch are NOT covered (see the
 * file header).
 * ------------------------------------------------------------------------- */

// Minimal float3 helpers (util/types_float3.h, util/transform.h semantics).
// source: util/transform.h:70 (transform_perspective)
float3 transform_perspective(float4 rowx, float4 rowy, float4 rowz, float4 roww, float3 a)
{
    float4 b = float4(a, 1.0f);
    float3 c = float3(dot(rowx, b), dot(rowy, b), dot(rowz, b));
    float w = dot(roww, b);
    return (w != 0.0f) ? c / w : float3(0.0f, 0.0f, 0.0f);
}

// source: util/transform.h:86 (transform_point, non-SSE path: homogeneous dot)
float3 transform_point(float4 rowx, float4 rowy, float4 rowz, float3 a)
{
    float4 a_ = float4(a, 1.0f);
    return float3(dot(a_, rowx), dot(a_, rowy), dot(a_, rowz));
}

// source: util/transform.h:157 (transform_direction, non-SSE path: w = 0)
float3 transform_direction(float4 rowx, float4 rowy, float4 rowz, float3 a)
{
    float4 a_ = float4(a, 0.0f);
    return float3(dot(a_, rowx), dot(a_, rowy), dot(a_, rowz));
}

// source: kernel/camera/camera.h:43 (camera_sample_perspective). Faithful for
// a PERSPECTIVE camera with have_perspective_motion==0, aperturesize==0,
// num_motion_steps==0 and interocular_offset==0 (the gate config). Outputs the
// clipped ray P/D/tmin/tmax (clipping math from camera.h:164-170). Ray
// differentials (dP/dD) are skipped: they are not part of the S1a comparison.
void camera_sample_perspective(MathParams p, float2 raster_xy, out float3 ray_P, out float3 ray_D, out float ray_tmin, out float ray_tmax)
{
    // Rebuild the matrix rows (scalars -> local float4; no cross-ABI layout).
    float4 rtc_x = float4(p.rtc_x_x, p.rtc_x_y, p.rtc_x_z, p.rtc_x_w);
    float4 rtc_y = float4(p.rtc_y_x, p.rtc_y_y, p.rtc_y_z, p.rtc_y_w);
    float4 rtc_z = float4(p.rtc_z_x, p.rtc_z_y, p.rtc_z_z, p.rtc_z_w);
    float4 rtc_w = float4(p.rtc_w_x, p.rtc_w_y, p.rtc_w_z, p.rtc_w_w);
    float4 ctw_x = float4(p.ctw_x_x, p.ctw_x_y, p.ctw_x_z, p.ctw_x_w);
    float4 ctw_y = float4(p.ctw_y_x, p.ctw_y_y, p.ctw_y_z, p.ctw_y_w);
    float4 ctw_z = float4(p.ctw_z_x, p.ctw_z_y, p.ctw_z_z, p.ctw_z_w);

    // camera.h:49-51
    float3 raster = float3(raster_xy, 0.0f);
    float3 Pcamera = transform_perspective(rtc_x, rtc_y, rtc_z, rtc_w, raster);

    // camera.h:53 have_perspective_motion branch: NOT covered (motion blur).

    // camera.h:71-72
    float3 P = float3(0.0f, 0.0f, 0.0f);
    float3 D = Pcamera;

    // camera.h:75-88 DOF branch: NOT covered (aperturesize == 0).

    // camera.h:91-98 motion-steps branch: NOT covered (num_motion_steps == 0).
    // camera.h:100-101
    P = transform_point(ctw_x, ctw_y, ctw_z, P);
    D = normalize(transform_direction(ctw_x, ctw_y, ctw_z, D));

    // camera.h:103-120 stereo branch: NOT covered (interocular_offset == 0).
    ray_P = P;
    ray_D = D;

    // camera.h:164-170 clipping.
    float z_inv = 1.0f / normalize(Pcamera).z;
    float nearclip = p.nearclip * z_inv;
    ray_P += nearclip * ray_D;
    ray_tmin = 0.0f;
    ray_tmax = p.cliplength * z_inv;
}

// source: kernel/camera/camera.h:449 (camera_sample). Faithful for the gate
// config: pixel_jitter.x != FLT_MAX (raster += -pixel_jitter, camera.h:466-468;
// the FLT_MAX filter-table branch at :461-464 is NOT covered), shuttertime ==
// -1 (ray->time = 0.5, camera.h:471-472; the shutter-table branch at :478-505 is
// NOT covered), and cam.type == CAMERA_PERSPECTIVE (camera.h:510-511). Other
// camera types return false in `ok`.
void camera_sample(MathParams p, int x, int y, float2 filter_uv, float time, float2 lens_uv,
                   out float3 ray_P, out float3 ray_D, out float ray_tmin, out float ray_tmax,
                   out float ray_time, out bool ok)
{
    float2 raster = float2((float)x, (float)y);
    ray_P = float3(0.0f, 0.0f, 0.0f);
    ray_D = float3(0.0f, 0.0f, 0.0f);
    ray_tmin = 0.0f;
    ray_tmax = 0.0f;
    ray_time = 0.0f;
    ok = false;

    // camera.h:461-468 pixel filter. Gate: pixel_jitter.x != FLT_MAX.
    if (p.pixel_jitter_x == asfloat(0x7F7FFFFFu) /* FLT_MAX */) {
        // Filter-table branch needs kernel_data.tables.lookup_table: NOT covered in S1a.
        ok = false;
        return;
    }
    raster += -float2(p.pixel_jitter_x, p.pixel_jitter_y);

    // camera.h:471-472 motion blur (time). Gate: shuttertime == -1.
    if (p.shuttertime == -1.0f) {
        ray_time = 0.5f;
    }
    else {
        // Shutter-table branch needs the shutter table: NOT covered in S1a.
        ok = false;
        return;
    }

    // camera.h:510-511 sample dispatch.
    if (p.cam_type == 0 /* CAMERA_PERSPECTIVE */) {
        camera_sample_perspective(p, raster, ray_P, ray_D, ray_tmin, ray_tmax);
        ok = true;
        return;
    }
    // Orthographic / panorama / custom: NOT covered.
    ok = false;
}
