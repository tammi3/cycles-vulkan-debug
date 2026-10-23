// SPDX-FileCopyrightText: 2011-2022 Blender Foundation
// SPDX-License-Identifier: Apache-2.0

/* M4e STEP 1b-0: dual-view KernelData-blob dump probe.
 *
 * Purpose: settle the M4b anomaly where the typed view
 * `params->data.words[356]` reportedly returned 0x80000803 while the raw
 * uint32 view at the SAME byte offset (1424) returned 0x00000000
 * (see the comment in traverse_probe.slang:104-115). This probe reads ALL
 * 640 words of the KernelData blob through BOTH typed/raw views (plus a
 * base-pointer variant) in ONE dispatch and writes them to a readback buffer
 * so the host can compare, index by index:
 *   view (a) typed mirror  : params->data.words[i]  (through KernelDataBlob)
 *   view (b) raw uint32    : params_raw[i]          (same device address)
 *   view (c) base-ptr      : (&params->data.words[0])[i]
 *
 * How it is run (see vulkan_kdata_probe() in src/device/vulkan/kernels.cpp):
 *   - host prefills the readback buffer with a sentinel (0xA5A5A5A5) so any
 *     word the shader did NOT overwrite is detectable;
 *   - host pushes three device pointers (params, params_raw = same address,
 *     out) and dispatches ONE workgroup of ONE thread;
 *   - host maps `out` and compares each view against the host bytes and
 *     against each other for every index, printing every disagreement.
 *
 * `out` layout (uint32 indices):
 *   [0    .. 639]   view (a) typed, loop-indexed   params->data.words[i]
 *   [640  .. 1279]  view (b) raw,   loop-indexed   params_raw[i]
 *   [1280]          canary constant (KDATA_CANARY)
 *   [1281]          view (a) typed, CONSTANT index params->data.words[356]
 *   [1282]          view (b) raw,   CONSTANT index params_raw[356]
 *   [1283 .. 1922]  view (c) base-ptr, loop-indexed (&data.words[0])[i]
 * Total 1923 words.
 *
 * View (c) isolates the ROOT CAUSE: it takes the typed array's base pointer
 * once (constant index 0, known-good) and then dynamic-indexes through that
 * plain pointer. If (c) matches host but (a) does not, the blob LAYOUT is
 * correct and the fault is the inline `params->data.words[i]` dynamic-index
 * codegen (two-level struct member + runtime index), not a stride/offset bug.
 *
 * Syntax follows the M4b ABI probe (abi_probe.slang), the only Slang code
 * known to compile and run on the RX 570. */

#include "vulkan_data.slang"

/* 640 words = 2560 bytes = sizeof(KernelData); mirrors
 * VULKAN_KERNELDATA_WORDS in vulkan_data.slang:37. */
static const uint KDATA_WORDS = 640u;
/* Word index of the historical anomaly (byte 1424). Re-read here through a
 * CONSTANT index so the exact original access pattern is reproduced. */
static const uint KDATA_ANOMALY_WORD = 356u;
/* Canary written to out[1280]; proves the dispatch reached the end. */
static const uint KDATA_CANARY = 0xCAFEF00Du;

struct KdataProbeParams
{
    KernelParamsVulkan* params; // typed view of the params buffer (BDA)
    uint* params_raw;           // same address, raw uint32 words
    uint* out;                  // host-visible uint32[1923] readback buffer
};                              // 24 bytes, pushed as one block

[[vk::push_constant]]
ConstantBuffer<KdataProbeParams> probe;

[shader("compute")]
[numthreads(1, 1, 1)]
void kdata_probe_main(uint3 tid : SV_DispatchThreadID)
{
    /* One workgroup of one thread is dispatched; only thread 0 works. */
    if (tid.x != 0u)
        return;

    /* View (a): the typed KernelDataBlob mirror, loop-indexed. If Slang laid
     * out `uint words[640]` with a stride/alignment that differs from a flat
     * uint32 array, these reads drift from the raw reads below. */
    for (uint i = 0u; i < KDATA_WORDS; i++)
        probe.out[i] = probe.params->data.words[i];

    /* View (b): the raw uint32 view of the SAME device address. This is what
     * the M4c traverse probe trusts (it matched the host dump byte-for-byte). */
    for (uint i = 0u; i < KDATA_WORDS; i++)
        probe.out[KDATA_WORDS + i] = probe.params_raw[i];

    /* Canary + the two CONSTANT-index reads that reproduce the original
     * `words[356]` access pattern exactly. */
    probe.out[2u * KDATA_WORDS] = KDATA_CANARY;
    probe.out[2u * KDATA_WORDS + 1u] = probe.params->data.words[KDATA_ANOMALY_WORD];
    probe.out[2u * KDATA_WORDS + 2u] = probe.params_raw[KDATA_ANOMALY_WORD];

    /* View (c): base-pointer + dynamic index. Take the typed blob's base
     * pointer ONCE (constant index 0 - the path proven correct above) and
     * dynamic-index through the plain pointer, exactly like view (b). Placed
     * right after the two constant slots (index 2*KDATA_WORDS + 3). */
    uint* blob_base = &probe.params->data.words[0];
    const uint c_base = 2u * KDATA_WORDS + 3u;
    for (uint i = 0u; i < KDATA_WORDS; i++)
        probe.out[c_base + i] = blob_base[i];
}
