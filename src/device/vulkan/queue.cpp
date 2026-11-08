/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 * SPDX-License-Identifier: Apache-2.0 */

#ifdef WITH_VULKAN

#include "device/vulkan/queue.h"
#include "device/vulkan/device_impl.h"
#include <algorithm>

CCL_NAMESPACE_BEGIN

/* M4b ABI probe hook, implemented in kernels.cpp. Runs at most once and is
 * a no-op unless CYCLES_VULKAN_ABI_PROBE=1 is set. */
bool vulkan_abi_probe(VulkanDevice *device);

/* M4c traverse probe hook, implemented in kernels.cpp. Runs at most once
 * and is a no-op unless CYCLES_VULKAN_TRAVERSE_PROBE=1 is set. */
bool vulkan_traverse_probe(VulkanDevice *device);

/* M4d diffuse-closure probe hook, implemented in kernels.cpp. Runs at most
 * once and is a no-op unless CYCLES_VULKAN_BSDF_PROBE=1 is set. */
bool vulkan_bsdf_probe(VulkanDevice *device);

/* M4e STEP 1a math probe hook, implemented in kernels.cpp. Runs at most once
 * and is a no-op unless CYCLES_VULKAN_MATH_PROBE=1 is set. */
bool vulkan_math_probe(VulkanDevice *device);

/* M4e STEP 1b-0 dual-view KernelData dump probe hook, implemented in
 * kernels.cpp. Runs at most once and is a no-op unless
 * CYCLES_VULKAN_KDATA_PROBE=1 is set. */
bool vulkan_kdata_probe(VulkanDevice *device);

/* M4e STEP 1b-1 minimal dynamic-index repro probe hook, implemented in
 * kernels.cpp. Runs at most once and is a no-op unless
 * CYCLES_VULKAN_DYNINDEX_REPRO=1 is set. */
bool vulkan_dynindex_repro(VulkanDevice *device);

/* M4e S1b-3 step 1 IntegratorStateGPU stride-table hook, implemented in
 * kernels.cpp. Runs at most once, is a no-op unless
 * CYCLES_VULKAN_STATE_STRIDE_TABLE=1, and never touches the GPU. */
bool vulkan_state_stride_table(VulkanDevice *device);

/* M4e S1b-3 step 2 Slang stride-check probe hook, implemented in kernels.cpp.
 * Runs at most once, no-op unless CYCLES_VULKAN_SLANG_STRIDE_PROBE=1. */
bool vulkan_stride_check(VulkanDevice *device);

/* M4e S1b-4 Stage A sub-word access probe hook, implemented in kernels.cpp.
 * Runs at most once, no-op unless CYCLES_VULKAN_SUBWORD_PROBE=1. */
bool vulkan_subword_check(VulkanDevice *device);

/* M4e S1b-6 state-accessor gate hook, implemented in kernels.cpp. Runs at
 * most once, no-op unless CYCLES_VULKAN_STATE_ACCESS_PROBE=1; runs the clean
 * mode plus the Task-4 fault modes 1..4 (see kernels.cpp for the full doc). */
bool vulkan_state_access_probe(VulkanDevice *device);

/* M4f SLICE 1 init-from-camera probe hook, implemented in kernels.cpp. Runs
 * at most once, no-op unless CYCLES_VULKAN_INIT_FROM_CAMERA_PROBE=1; compares
 * the ported integrate_camera_sample against the REAL Cycles function (see
 * kernels.cpp for the full doc). */
bool vulkan_init_from_camera_probe(VulkanDevice *device);

/* M4f SLICE 2 state-init probe hook, implemented in kernels.cpp. Runs at
 * most once, no-op unless CYCLES_VULKAN_STATE_INIT_PROBE=1; compares the
 * ported state-initialization cluster (path_state.slang) against the REAL
 * Cycles functions (see kernels.cpp for the full doc). */
bool vulkan_state_init_probe(VulkanDevice *device);

/* M4f SLICE 3 film+queue probe hook, implemented in kernels.cpp. Runs at
 * most once, no-op unless CYCLES_VULKAN_FILM_QUEUE_PROBE=1; compares the
 * ported film + queue layer (film_queue.slang) against the REAL Cycles
 * calls (see kernels.cpp for the full doc). */
bool vulkan_film_queue_probe(VulkanDevice *device);

/* M4f SLICE 4 init-from-camera BODY probe hook, implemented in kernels.cpp.
 * Runs at most once, no-op unless
 * CYCLES_VULKAN_INIT_FROM_CAMERA_BODY_PROBE=1; compares the ported
 * integrator_init_from_camera body (init_from_camera.slang) against the
 * REAL Cycles function (see kernels.cpp for the full doc). */
bool vulkan_init_from_camera_body_probe(VulkanDevice *device);

/* M4f SLICE 5 production dispatch, implemented in kernels.cpp. Tri-state:
 *   1  -> the kernel was dispatched (or there was nothing to do)
 *   0  -> not implemented yet (the M4 error below is kept for it)
 *  -1  -> failed; kernels.cpp has already called set_error(), so this file
 *         must NOT overwrite the device error and just returns false.
 * Scope is DEVICE_KERNEL_INTEGRATOR_RESET and
 * DEVICE_KERNEL_INTEGRATOR_INIT_FROM_CAMERA only (see kernels.cpp). */
int vulkan_queue_enqueue(VulkanDevice *device,
                         DeviceKernel k,
                         int n,
                         const DeviceKernelArguments &args);

VulkanQueue::VulkanQueue(VulkanDevice *d) : DeviceQueue(d), vulkan_device_(d) {}

int VulkanQueue::num_concurrent_states(const size_t s) const {
  return (int)std::min<size_t>(1 << 20, (512ull << 20) / s);
}

int VulkanQueue::num_concurrent_busy_states(const size_t s) const { 
  return num_concurrent_states(s) / 4; 
}

void VulkanQueue::init_execution() { 
  debug_init_execution(); 
}

void VulkanQueue::load_image_info() {
  // M4: Will be implemented when texture handling is added
}

bool VulkanQueue::enqueue(DeviceKernel k, const int n, const DeviceKernelArguments &args) {
  debug_enqueue_begin(k, n);
  /* M4b: optional one-shot ABI probe (env CYCLES_VULKAN_ABI_PROBE=1). The
   * first enqueue happens after the scene upload, so the probe sees what
   * Cycles actually put in the params buffer. No-op unless the env var is
   * set. Every probe hook below shares that semantics; the M4 stub error is
   * now emitted only for kernels OUTSIDE slice-5's scope (see the tail of
   * this function). */
  vulkan_abi_probe(vulkan_device_);
  /* M4c: optional one-shot traverse probe (env CYCLES_VULKAN_TRAVERSE_PROBE=1)
   * with the same no-op-unless-set semantics. */
  vulkan_traverse_probe(vulkan_device_);
  /* M4d: optional one-shot diffuse-closure probe (env CYCLES_VULKAN_BSDF_PROBE=1)
   * with the same no-op-unless-set semantics. */
  vulkan_bsdf_probe(vulkan_device_);
  /* M4e STEP 1a: optional one-shot math probe (env CYCLES_VULKAN_MATH_PROBE=1)
   * with the same no-op-unless-set semantics. */
  vulkan_math_probe(vulkan_device_);
  /* M4e STEP 1b-0: optional one-shot dual-view KernelData dump probe (env
   * CYCLES_VULKAN_KDATA_PROBE=1), same no-op-unless-set semantics. */
  vulkan_kdata_probe(vulkan_device_);
  /* M4e STEP 1b-1: optional one-shot minimal dynamic-index repro probe (env
   * CYCLES_VULKAN_DYNINDEX_REPRO=1), same no-op-unless-set semantics. */
  vulkan_dynindex_repro(vulkan_device_);
  /* M4e S1b-3 step 1: optional one-shot IntegratorStateGPU stride table (env
   * CYCLES_VULKAN_STATE_STRIDE_TABLE=1); host-only printf, no GPU work. */
  vulkan_state_stride_table(vulkan_device_);
  /* M4e S1b-3 step 2: optional one-shot Slang stride check (env
   * CYCLES_VULKAN_SLANG_STRIDE_PROBE=1), same no-op-unless-set semantics. */
  vulkan_stride_check(vulkan_device_);
  /* M4e S1b-4 Stage A: optional one-shot sub-word access probe (env
   * CYCLES_VULKAN_SUBWORD_PROBE=1), same no-op-unless-set semantics. */
  vulkan_subword_check(vulkan_device_);
  /* M4e S1b-6: optional one-shot integrator-state accessor gate (env
   * CYCLES_VULKAN_STATE_ACCESS_PROBE=1), same no-op-unless-set semantics. */
  vulkan_state_access_probe(vulkan_device_);
  /* M4f SLICE 1: optional one-shot init-from-camera probe (env
   * CYCLES_VULKAN_INIT_FROM_CAMERA_PROBE=1), same no-op-unless-set semantics. */
  vulkan_init_from_camera_probe(vulkan_device_);
  /* M4f SLICE 2: optional one-shot state-init probe (env
   * CYCLES_VULKAN_STATE_INIT_PROBE=1), same no-op-unless-set semantics. */
  vulkan_state_init_probe(vulkan_device_);
  /* M4f SLICE 3: optional one-shot film+queue probe (env
   * CYCLES_VULKAN_FILM_QUEUE_PROBE=1), same no-op-unless-set semantics. */
  vulkan_film_queue_probe(vulkan_device_);
  /* M4f SLICE 4: optional one-shot init-from-camera BODY probe (env
   * CYCLES_VULKAN_INIT_FROM_CAMERA_BODY_PROBE=1), same no-op-unless-set
   * semantics. */
  vulkan_init_from_camera_body_probe(vulkan_device_);
  /* M4f SLICE 5: the two production kernels the plumbing covers are actually
   * dispatched here (RESET + INIT_FROM_CAMERA). Everything else still falls
   * through to the M4 error, which is what stops a real session at the first
   * unimplemented kernel. */
  const int dispatched = vulkan_queue_enqueue(
      vulkan_device_, k, n, args);
  if (dispatched < 0) {
    /* vulkan_queue_enqueue() already set the device error; overwriting it
     * here would hide the real cause. */
    debug_enqueue_end();
    return false;
  }
  if (dispatched == 0) {
    /* Name the kernel: the slice-5 gate uses this line to prove the session
     * stopped at exactly the first unimplemented kernel. */
    vulkan_device_->set_error(
        string("Vulkan: kernel dispatch not implemented yet (M4): ") +
        device_kernel_as_string(k));
    debug_enqueue_end();
    return false;
  }
  debug_enqueue_end();
  return true;
}

bool VulkanQueue::synchronize() { 
  debug_synchronize();
  /* Report failure once the device is in error, exactly like the other GPU
   * backends (cuda/queue.cpp:143, hip/queue.cpp:145, oneapi/queue.cpp:133:
   * `return !(device->have_error())`). PathTraceWorkGPU::
   * update_queue_counter_and_cache() (path_trace_work_gpu.cpp:329-331) breaks
   * its render loop on a false return; without this the loop would keep
   * re-dispatching a kernel that can never succeed once slice 5 has advanced
   * the integrator past the kernels that ARE implemented. Probes and clean
   * runs never set an error, so their behaviour is unchanged. */
  return !vulkan_device_->have_error();
}

void VulkanQueue::zero_to_device(device_memory &m) { 
  vulkan_device_->mem_zero(m); 
}

void VulkanQueue::copy_to_device(device_memory &m) { 
  vulkan_device_->mem_copy_to(m); 
}

void VulkanQueue::copy_from_device(device_memory &m) { 
  vulkan_device_->mem_copy_from(m, 0, m.data_width, m.data_height == 0 ? 1 : m.data_height, m.data_elements); 
}

void *VulkanQueue::copy_from_device_synchronized(device_memory &m, vector<uint8_t> &s) {
  s.resize(m.memory_size());
  vulkan_device_->mem_copy_from(m, 0, m.data_width, m.data_height == 0 ? 1 : m.data_height, m.data_elements);
  memcpy(s.data(), m.host_pointer, s.size());
  return s.data();
}

CCL_NAMESPACE_END

#endif /* WITH_VULKAN */