// SPDX-FileCopyrightText: 2011-2022 Blender Foundation
// SPDX-License-Identifier: Apache-2.0

/* integrator_reset.slang - the PRODUCTION entry point of the
 * DEVICE_KERNEL_INTEGRATOR_RESET device kernel (M4f slice 5, dispatch
 * plumbing). This file exists so the SPVs of the slices 1-4 probes stay
 * untouched: it includes only what it needs and is compiled separately.
 *
 * Cycles source (verbatim contract): src/kernel/device/gpu/kernel.h:60-69
 *
 *   ccl_gpu_kernel(GPU_KERNEL_BLOCK_NUM_THREADS, GPU_KERNEL_MAX_REGISTERS)
 *       ccl_gpu_kernel_signature(integrator_reset, const int num_states)
 *   {
 *     const int state = ccl_gpu_global_id_x();
 *     if (state < num_states) {
 *       INTEGRATOR_STATE_WRITE(state, path, queued_kernel) = 0;
 *       INTEGRATOR_STATE_WRITE(state, shadow_path, queued_kernel) = 0;
 *     }
 *   }
 *
 * Mapping (AGENTS.md section 4, no storage class left to guess):
 *   - num_states -> push constant ResetPush.num_states. The host passes the
 *     enqueue() work_size, which PathTraceWorkGPU sets to max_num_paths_
 *     (source: src/integrator/path_trace_work_gpu.cpp:439-441).
 *   - INTEGRATOR_STATE_WRITE(..., queued_kernel) -> write_path_queued_kernel /
 *     write_shadow_path_queued_kernel (integrator_state.slang:334/:800), the
 *     sub-word accessors gated by the M4e state-access probe. Both slices
 *     are uint16-backed (state.h: queued_kernel is uint16_t), exactly as
 *     Cycles writes them on GPU.
 *   - kg: this kernel reads no kernel globals in Cycles either (kernel.h
 *     takes no kg-derived data), so none are passed.
 *   - state -> uint for the accessors; state >= 0 always (it comes from
 *     tid.x) and state < num_states is checked first.
 *
 * Launch config: [numthreads(64,1,1)], grid = ceil(num_states/64). 64 is
 * THIS backend's entry width (the same width every probe uses); Cycles'
 * CUDA value is GPU_KERNEL_BLOCK_NUM_THREADS = 256 (cuda/config.h:23) - a
 * per-device launch choice, not kernel logic: the `state < num_states`
 * guard makes the result identical for any block width.
 *
 * Host side: loaded from CYCLES_VULKAN_RESET_SPV, else ./integrator_reset.spv,
 * else src/kernel/device/vulkan/integrator_reset.spv (probe loading
 * precedent, kernels.cpp abi probe). slangc lowers the Slang entry to
 * SPIR-V "main" (kernels.cpp:451-456 documents that observation).
 * Compiled by hand - slangc is NOT part of the build (AGENTS.md). */

#include "integrator_state.slang"  // IntegratorStateGPU mirror + queued_kernel
                                   // write accessors (pulls in
                                   // vulkan_data.slang for KernelParamsVulkan)

/* Push block: device pointer of the params buffer + count (+ pad) = 16
 * bytes, well under the 128-byte spec minimum. The params buffer carries
 * the IntegratorStateGPU mirror at a fixed offset (params.h:43; the offset
 * is logged at device.cpp:341), and the mirror's members are the device
 * addresses of the real SoA slices uploaded by PathTraceWorkGPU. */
struct ResetPush
{
    KernelParamsVulkan* params;   // BDA of the fixed kernel params buffer
    int                  num_states;
    int                  pad0;    // keep the struct 8-byte aligned
};                                // 8 + 4 + 4 = 16 bytes (locked by
                                  // static_assert in device/vulkan/kernels.cpp)

[[vk::push_constant]]
ConstantBuffer<ResetPush> pc;

[shader("compute")]
[numthreads(64, 1, 1)]
void integrator_reset_main(uint3 tid : SV_DispatchThreadID)
{
    /* source: kernel.h:63 */
    const int state = (int)tid.x;

    /* source: kernel.h:65 - bounds guard (raw device pointers have no
     * bounds checking, AGENTS.md section 4). */
    if (state < pc.num_states) {
        /* source: kernel.h:66-67 - write 0 into both queued_kernel slices.
         * &params->integrator_state is a constant-offset member address: no
         * pointer arithmetic on a device pointer, no pointer==null compare
         * (film_queue.slang header constraint 3). */
        IntegratorStateGPU* ist = &pc.params->integrator_state;
        write_path_queued_kernel(ist, (uint)state, 0u);
        write_shadow_path_queued_kernel(ist, (uint)state, 0u);
    }
}
