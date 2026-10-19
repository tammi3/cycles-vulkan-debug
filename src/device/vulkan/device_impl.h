/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#ifdef WITH_VULKAN

#include "device/device.h"
#include "device/queue.h"
#include <vulkan/vulkan.h>

CCL_NAMESPACE_BEGIN

// Small struct to keep track of Vulkan allocations
struct VulkanAlloc {
  VkBuffer buffer;
  VkDeviceMemory memory;
};

class VulkanDevice : public GPUDevice {
 public:
  VulkanDevice(const DeviceInfo &info, Stats &stats, Profiler &profiler, bool headless);
  ~VulkanDevice() override;

  // Core Device overrides
  BVHLayoutMask get_bvh_layout_mask(const uint kernel_features) const override;
  void const_copy_to(const char *name, void *host, const size_t size) override;
  bool load_kernels(uint kernel_features) override;
  unique_ptr<DeviceQueue> gpu_queue_create() override;
  void build_bvh(BVH *bvh, Progress &progress, bool refit) override;

  // High-level memory overrides (we just call GPUDevice's generic_* helpers)
  void mem_alloc(device_memory &mem) override;
  void mem_copy_to(device_memory &mem) override;
  void mem_move_to_host(device_memory &mem) override;
  void mem_copy_from(device_memory &mem, const size_t y, size_t w, const size_t h, size_t elem) override;
  void mem_zero(device_memory &mem) override;
  void mem_free(device_memory &mem) override;

  // GPUDevice pure virtuals (The actual Vulkan API calls)
  bool alloc_device(void *&device_pointer, const size_t size) override;
  void free_device(void *device_pointer) override;
  void copy_host_to_device(void *device_pointer, void *host_pointer, const size_t size) override;
  void get_device_memory_info(size_t &total, size_t &free) override;

  // Shared memory overrides (Not implemented yet, just stubs to satisfy the compiler)
  bool shared_alloc(void *&shared_pointer, const size_t size) override;
  void shared_free(void *shared_pointer) override;
  void *shared_to_device_pointer(const void *shared_pointer) override;

 private:
  VkInstance instance;
  VkPhysicalDevice physical_device;
  VkDevice device;
  VkQueue queue;
  uint32_t queue_family_index;

  VkBuffer const_buffer = VK_NULL_HANDLE;
  VkDeviceMemory const_memory = VK_NULL_HANDLE;
  size_t const_size = 0;

  uint32_t find_queue_family(VkPhysicalDevice phys_dev);
  uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties);
};

CCL_NAMESPACE_END

#endif /* WITH_VULKAN */