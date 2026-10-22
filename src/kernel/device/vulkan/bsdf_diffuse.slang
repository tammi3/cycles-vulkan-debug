// SPDX-FileCopyrightText: 2011-2022 Blender Foundation
// SPDX-License-Identifier: Apache-2.0

/* Lambertian (diffuse) BSDF — M4 port of src/kernel/closure/bsdf_diffuse.h.
 *
 * This file is FUNCTIONS ONLY (no entry point), exactly like traverse.slang.
 * It is #included by a compilation unit that owns the compute entry point
 * (bsdf_probe.slang for the M4 gate; the real path_trace.slang later).
 *
 * Scope (AGENTS section 3A): only the LABEL_DIFFUSE closure path is in scope
 * for M4. This file therefore ports the two diffuse functions that the
 * surface integrator calls:
 *
 *   bsdf_diffuse_eval   (bsdf_diffuse.h:38-49)  f(wo) and its pdf
 *   bsdf_diffuse_sample (bsdf_diffuse.h:51-73)  cosine-weighted wo + pdf
 *
 * Every other closure type (translucent, microfacet, ...) is DEFERRED: they
 * are not referenced here at all, which is the Slang equivalent of returning
 * LABEL_NONE / a no-op stub.
 *
 * Faithful translation (AGENTS section 4): the arithmetic below is copied
 * line-for-line from the Cycles source cited above and from the helpers it
 * calls (mapping.h / projection.h / math_float3.h / math_base.h). No Vulkan-
 * specific "improvement" is made here — none of this is a measured hot spot
 * yet, and the new AGENTS rule permits divergence only when it demonstrably
 * speeds up old hardware, which we have not shown. */

/* -------------------------------------------------------------------------
 * Types and constants the diffuse functions need.
 * ---------------------------------------------------------------------- */

/* Spectrum is RGB float3 in this build, not a spectral block:
 *   types_spectrum.h:16  #define make_spectrum(f) make_float3(f)
 *   types_spectrum.h:20  #define zero_spectrum   zero_float3
 * So `make_spectrum(x)` is the grey RGB (x,x,x) and a zero spectrum is
 * float3(0). If this ever becomes a real spectral build, THIS is the single
 * place that has to change. */
float3 make_spectrum(float f)
{
    return float3(f, f, f);
}

float3 zero_spectrum()
{
    return float3(0.0f, 0.0f, 0.0f);
}

/* Closure labels (kernel/types.h:330-341). Only the two the diffuse path
 * returns are needed here; the rest of the enum is deferred with the other
 * closures. */
static const int LABEL_NONE     = 0;
static const int LABEL_TRANSMIT = 1;
static const int LABEL_REFLECT  = 2;
static const int LABEL_DIFFUSE  = 4;

/* 1/pi, pi/2, pi/4 — exact values from util/math_base.h:37,40,43. */
static const float M_1_PI_F = 0.3183098861837067f;
static const float M_PI_2_F = 1.5707963267948966f;
static const float M_PI_4_F = 0.7853981633974830f;

/* DiffuseBsdf = SHADER_CLOSURE_BASE (kernel/types.h:868-872):
 *     Spectrum weight; ClosureType type; float sample_weight; float3 N
 * Only `N` is read by bsdf_diffuse_eval/sample (bsdf_diffuse.h:44,60); the
 * other three fields are carried so the struct head stays identical to
 * ShaderClosure's head (types.h:883-884). That identity is what makes the
 * eventual `(DiffuseBsdf *)sc` cast from a real closure array a pure
 * reinterpret with no field fixups.
 *
 * NOTE: the ShaderClosure array itself lives in ShaderData and is NOT ported
 * yet. When the shader-evaluation stage is ported it will own that array and
 * hand a `DiffuseBsdf*` (i.e. the post-cast form) to these functions. Until
 * then this struct documents the layout the cast will rely on.
 *   // DEFERRED: ShaderClosure array / ShaderData */
struct DiffuseBsdf
{
    float3 weight;       // Spectrum = float3 (unused by eval/sample)
    uint   type;         // ClosureType (unused by eval/sample)
    float  sample_weight;// (unused by eval/sample)
    float3 N;            // shading normal — THE field these functions read
};

/* -------------------------------------------------------------------------
 * Math helpers, ported from the Cycles util headers. These use only Slang
 * built-ins (dot, cross, normalize, sqrt, max, cos, sin) — no invented names.
 * ---------------------------------------------------------------------- */

/* len_squared for float2 and float3 (util/math_float2.h:183, math_float3.h:307). */
float len_squared(float2 a)
{
    return dot(a, a);
}

float len_squared(float3 a)
{
    return dot(a, a);
}

/* safe_sqrtf — NaN-safe sqrt (util/math_base.h:538-541): sqrtf(max(f,0)). */
float safe_sqrtf(float f)
{
    return sqrt(max(f, 0.0f));
}

/* polar_to_cartesian (util/projection.h:30-33): (r cos phi, r sin phi). */
float2 polar_to_cartesian(float r, float phi)
{
    return float2(r * cos(phi), r * sin(phi));
}

/* make_orthonormals (util/math_float3.h:733-759). Builds a tangent `a` and
 * bitangent `b` from a normal N. The Cycles body starts with an `#if 0`
 * block (early-outs for axis-aligned N) that is COMPILED OUT at
 * math_float3.h:737-748, so we skip it and port only the live branch.
 *
 * The two cases pick a cross-product seed vector that is guaranteed not to
 * be parallel to N: (1,1,1) in the generic case, (-1,1,1) when N is close to
 * (1,1,1) (where (1,1,1) x N would degenerate). */
void make_orthonormals(float3 N, out float3 a, out float3 b)
{
    if (N.x != N.y || N.x != N.z)
    {
        a = float3(N.z - N.y, N.x - N.z, N.y - N.x);  // (1,1,1) x N
    }
    else
    {
        a = float3(N.z - N.y, N.x + N.z, -N.y - N.x); // (-1,1,1) x N
    }
    a = normalize(a);
    b = cross(N, a);
}

/* sample_uniform_disk (kernel/sample/mapping.h:21-42). Concentric mapping
 * from [0,1]^2 to the unit disk, preserving stratification better than a
 * naive polar map. `rand` is the two uniform samples. */
float2 sample_uniform_disk(float2 rand)
{
    float phi;
    float r;
    float a = 2.0f * rand.x - 1.0f;
    float b = 2.0f * rand.y - 1.0f;

    if (a == 0.0f && b == 0.0f)
    {
        return float2(0.0f, 0.0f);  // zero_float2()
    }

    if (a * a > b * b)
    {
        r = a;
        phi = M_PI_4_F * (b / a);
    }
    else
    {
        r = b;
        phi = M_PI_2_F - M_PI_4_F * (a / b);
    }

    return polar_to_cartesian(r, phi);
}

/* sample_cos_hemisphere (kernel/sample/mapping.h:71-84). Cosine-weighted
 * hemisphere sample around N: maps rand to the disk, lifts z with the
 * cosine, writes the direction `wo` and its pdf (cos(theta)/pi). */
void sample_cos_hemisphere(float3 N, float2 rand_in, out float3 wo, out float pdf)
{
    float2 rand = sample_uniform_disk(rand_in);
    float costheta = safe_sqrtf(1.0f - len_squared(rand));

    float3 T;
    float3 B;
    make_orthonormals(N, T, B);
    wo = rand.x * T + rand.y * B + costheta * N;
    pdf = costheta * M_1_PI_F;
}

/* -------------------------------------------------------------------------
 * The two M4 diffuse functions.
 * ---------------------------------------------------------------------- */

/* bsdf_diffuse_eval — source: kernel/closure/bsdf_diffuse.h:38-49.
 *
 * Returns the diffuse reflectance f(wo) (= cos(NO)/pi for unit albedo) and
 * writes its pdf. `wi` is unused in the Cycles signature (marked /*wi*\/ there)
 * and is kept here so the port mirrors the call shape; only N and wo matter. */
float3 bsdf_diffuse_eval(DiffuseBsdf bsdf, float3 wi, float3 wo, out float pdf)
{
    float3 N = bsdf.N;

    float cosNO = max(dot(N, wo), 0.0f) * M_1_PI_F;
    pdf = cosNO;
    return make_spectrum(cosNO);
}

/* bsdf_diffuse_sample — source: kernel/closure/bsdf_diffuse.h:51-73.
 *
 * Draws a cosine-weighted `wo` from `rand`, then rejects it to the geometric
 * hemisphere `Ng` (dot(Ng,wo)>0): if the sample points below the geometric
 * normal it is zeroed (pdf=0, eval=0). Returns the closure label
 * LABEL_REFLECT | LABEL_DIFFUSE (types.h:332-333 => 2|4 = 6). */
int bsdf_diffuse_sample(DiffuseBsdf bsdf, float3 Ng, float3 wi, float2 rand,
                        out float3 eval, out float3 wo, out float pdf)
{
    float3 N = bsdf.N;

    // distribution over the hemisphere
    sample_cos_hemisphere(N, rand, wo, pdf);

    if (dot(Ng, wo) > 0.0f)
    {
        eval = make_spectrum(pdf);
    }
    else
    {
        pdf = 0.0f;
        eval = zero_spectrum();
    }
    return LABEL_REFLECT | LABEL_DIFFUSE;
}
