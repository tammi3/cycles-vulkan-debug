// SPDX-FileCopyrightText: 2011-2022 Blender Foundation
// SPDX-License-Identifier: Apache-2.0

/* M4b ABI probe: proves that the Slang view of KernelParamsVulkan agrees
 * with the host C++ view, byte for byte, at the slots we care about.
 *
 * How it is run (see vulkan_abi_probe() in src/device/vulkan/kernels.cpp):
 *   - host creates a pipeline from this shader with a 24-byte push constant
 *     range and no descriptor sets at all;
 *   - push constant carries three raw device pointers: `params` (typed view
 *     of the KernelParamsVulkan buffer) and `params_raw` (the SAME address
 *     viewed as plain uint32 words), plus `out` (host-visible readback);
 *   - host dispatches ONE workgroup of ONE thread, waits, maps `out` and
 *     compares every value against the C++ side.
 *
 * What each comparison proves:
 *   - out[0]/out[1]: sizeof() as Slang computes it vs the host static_asserts
 *     -> catches any padding/size disagreement of the whole struct;
 *   - out[2]: first word of the KernelData blob through the typed view;
 *   - out[3..6] and out[7..10]: the bvh_nodes slot and the first
 *     integrator_state member read TWICE - once through the Slang struct
 *     layout (typed) and once as raw words at the byte offsets the host
 *     static_asserts (2560 / 2984). The raw bytes are exactly what the host
 *     uploaded with const_copy_to, so a typed-vs-raw MISMATCH means the
 *     Slang layout is wrong;
 *   - out[11]: sentinel proving the dispatch actually ran.
 *
 * Syntax: push-constant pointers and the entry-point form follow the M2c
 * spike (agentreference/bvh_traverse.slang:47,294-296), the only Slang code
 * known to compile and run on the RX 570. */

#include "vulkan_data.slang"

/* Byte offsets of the slots under test. These mirror the host static_asserts
 * in src/device/vulkan/kernels.cpp:
 *   offsetof(KernelParamsVulkan, bvh_nodes) == 2560
 *   offsetof(KernelParamsVulkan, integrator_state) == 2984 */
static const uint OFF_BVH_NODES = 2560u;
static const uint OFF_INTEGRATOR_STATE = 2984u;

/* Sentinel written to out[11]; the host checks it to prove the dispatch ran. */
static const uint PROBE_SENTINEL = 0xAB10E57Fu;

struct AbiProbeParams
{
    KernelParamsVulkan* params; // typed view of the params buffer (BDA)
    uint* params_raw;           // same address, raw uint32 words
    uint* out;                  // host-visible uint32[12] readback buffer
};                              // 24 bytes, pushed as one block

[[vk::push_constant]]
ConstantBuffer<AbiProbeParams> probe;

[shader("compute")]
[numthreads(1, 1, 1)]
void abi_probe_main(uint3 tid : SV_DispatchThreadID)
{
    /* One workgroup of one thread is dispatched; only thread 0 works. */
    if (tid.x != 0u)
        return;

    /* Slang-computed sizes (compared against host static_asserts). */
    probe.out[0] = uint(sizeof(KernelParamsVulkan));
    probe.out[1] = uint(sizeof(IntegratorStateGPU));

    /* First word of the KernelData blob, typed view. */
    probe.out[2] = probe.params->data.words[0];

    /* bvh_nodes slot: typed view vs raw words at OFF_BVH_NODES. */
    uint64_t typed_bvh = probe.params->bvh_nodes;
    probe.out[3] = uint(typed_bvh);
    probe.out[4] = uint(typed_bvh >> 32);
    probe.out[5] = probe.params_raw[OFF_BVH_NODES / 4];
    probe.out[6] = probe.params_raw[OFF_BVH_NODES / 4 + 1];

    /* First integrator_state member (path.render_pixel_index): typed view
     * vs raw words at OFF_INTEGRATOR_STATE. */
    uint64_t typed_ist = probe.params->integrator_state.path.render_pixel_index;
    probe.out[7] = uint(typed_ist);
    probe.out[8] = uint(typed_ist >> 32);
    probe.out[9] = probe.params_raw[OFF_INTEGRATOR_STATE / 4];
    probe.out[10] = probe.params_raw[OFF_INTEGRATOR_STATE / 4 + 1];

    /* Dispatch-completed sentinel. */
    probe.out[11] = PROBE_SENTINEL;
}
