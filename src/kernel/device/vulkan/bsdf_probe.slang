// SPDX-FileCopyrightText: 2011-2022 Blender Foundation
// SPDX-License-Identifier: Apache-2.0

/* bsdf_probe.slang : M4 diffuse-closure gate probe.
 *
 * One compute entry point compiled into bsdf_probe.spv with slangc (see the
 * host side, src/device/vulkan/kernels.cpp: vulkan_bsdf_probe()).
 *
 * Purpose: exercise the just-ported diffuse closure (bsdf_diffuse.slang) on
 * the GPU with a FIXED hit and FIXED directions, so the host can compare the
 * shader's arithmetic value-by-value against a CPU reference computed from
 * the same Cycles formulas. This is the "shade a single hit with a fixed
 * direction" gate — it does NOT traverse the BVH (traversal is already
 * verified by the M4c traverse probe); it only checks the BSDF math.
 *
 * Syntax follows the M4b ABI probe and the M4c traverse probe: push
 * constants carry a raw buffer device address as a typed pointer, no
 * descriptor sets. One thread. */

#include "bsdf_diffuse.slang"  // the port under test

/* Output layout (float words, little-endian, read back by the host):
 *   [0]      sentinel 0xB5DF0001 — proves the dispatch ran
 *   [1..3]   eval.rgb        from bsdf_diffuse_eval at wo_out
 *   [4]      pdf_eval        from bsdf_diffuse_eval
 *   [5..7]   wo.rgb          from bsdf_diffuse_sample (the drawn direction)
 *   [8]      pdf_sample      from bsdf_diffuse_sample
 *   [9..11]  eval_s.rgb      from bsdf_diffuse_sample
 *   [12]     label           return value of bsdf_diffuse_sample
 *   [13]     wo_len          |wo| (should be ~1.0: the sample is a direction)
 *   [14]     cosNO_dot       dot(N, wo) — for the host to sanity-check sign
 * The fixed inputs (N, Ng, wo, rand) are HARDCODED here and mirrored by the
 * host; changing them means changing both sides. */
static const uint BSDF_PROBE_SENTINEL = 0xB5DF0001u;

struct BsdfProbePush
{
    float* out_words;   // device pointer to >= 15 float words
};

[[vk::push_constant]]
ConstantBuffer<BsdfProbePush> bsdf_pc;

[shader("compute")]
[numthreads(1, 1, 1)]
void bsdf_probe_main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x != 0u)
        return;

    /* --- FIXED "hit": a flat floor facing +Z ------------------------------
     * N  (shading normal)  = +Z
     * Ng (geometric normal) = +Z  (smooth shading on a flat quad: N == Ng)
     * wo (outgoing dir)     = a tilted direction, so cos(NO) is neither
     *                        0 nor 1 and the max(dot,0) branch is exercised.
     * rand (sample pair)    = a mid-disk value that does NOT hit the
     *                        a==0 && b==0 or a*a==b*b tie branches. */
    float3 N  = float3(0.0f, 0.0f, 1.0f);
    float3 Ng = float3(0.0f, 0.0f, 1.0f);
    float3 wi = float3(0.0f, 0.0f, 1.0f);   // unused by diffuse, kept for shape
    float3 wo = normalize(float3(0.3f, -0.2f, 0.9f));
    float2 rand = float2(0.31f, 0.62f);

    DiffuseBsdf bsdf;
    bsdf.weight = float3(1.0f, 1.0f, 1.0f);
    bsdf.type = 0u;
    bsdf.sample_weight = 1.0f;
    bsdf.N = N;

    /* --- bsdf_diffuse_eval at the fixed wo ------------------------------- */
    float pdf_eval;
    float3 eval = bsdf_diffuse_eval(bsdf, wi, wo, pdf_eval);

    /* --- bsdf_diffuse_sample with the fixed rand ------------------------- */
    float3 eval_s;
    float3 wo_s;
    float pdf_s;
    int label = bsdf_diffuse_sample(bsdf, Ng, wi, rand, eval_s, wo_s, pdf_s);

    /* --- write results --------------------------------------------------- */
    float* o = bsdf_pc.out_words;
    o[0]  = asfloat(BSDF_PROBE_SENTINEL);
    o[1]  = eval.x;
    o[2]  = eval.y;
    o[3]  = eval.z;
    o[4]  = pdf_eval;
    o[5]  = wo_s.x;
    o[6]  = wo_s.y;
    o[7]  = wo_s.z;
    o[8]  = pdf_s;
    o[9]  = eval_s.x;
    o[10] = eval_s.y;
    o[11] = eval_s.z;
    o[12] = float(label);
    o[13] = length(wo_s);
    o[14] = dot(N, wo_s);
}
