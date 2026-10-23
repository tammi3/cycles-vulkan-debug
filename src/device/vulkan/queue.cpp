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

bool VulkanQueue::enqueue(DeviceKernel k, const int n, const DeviceKernelArguments &) {
  debug_enqueue_begin(k, n);
  /* M4b: optional one-shot ABI probe (env CYCLES_VULKAN_ABI_PROBE=1). The
   * first enqueue happens after the scene upload, so the probe sees what
   * Cycles actually put in the params buffer. No-op unless the env var is
   * set; the M4 stub error below is emitted either way. */
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
  vulkan_device_->set_error("Vulkan: kernel dispatch not implemented yet (M4)");
  debug_enqueue_end();
  return false;
}

bool VulkanQueue::synchronize() { 
  debug_synchronize();
  return true; 
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