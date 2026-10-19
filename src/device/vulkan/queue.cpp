/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 * SPDX-License-Identifier: Apache-2.0 */

#ifdef WITH_VULKAN

#include "device/vulkan/queue.h"
#include "device/vulkan/device_impl.h"
#include <algorithm>

CCL_NAMESPACE_BEGIN

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
  vulkan_device_->mem_copy_from(m, 0, m.data_width, m.data_height == 0 ? 1 : m.data_height, m.data_element_size()); 
}

void *VulkanQueue::copy_from_device_synchronized(device_memory &m, vector<uint8_t> &s) {
  s.resize(m.memory_size());
  vulkan_device_->mem_copy_from(m, 0, m.data_width, m.data_height == 0 ? 1 : m.data_height, m.data_element_size());
  memcpy(s.data(), m.host_pointer, s.size());
  return s.data();
}

CCL_NAMESPACE_END

#endif /* WITH_VULKAN */