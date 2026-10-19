/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#ifdef WITH_VULKAN

#include "device/device.h"
#include "device/queue.h"
#include <vulkan/vulkan.h>

CCL_NAMESPACE_BEGIN

class VulkanDevice : public Device {
 public:
  VulkanDevice(const DeviceInfo &info, Stats &stats, Profiler &profiler, bool headless);
  ~VulkanDevice() override;

  BVHLayoutMask get_bvh_layout_mask(const uint kernel_features) const override;
  void const_copy_to(const char *name, void *host, const size_t size) override;
  bool load_kernels(uint kernel_features) override;
  unique_ptr<DeviceQueue> gpu_queue_create() override;

  void mem_alloc(device_memory &mem) override;
  void mem_copy_to(device_memory &mem) override;
  void mem_move_to_host(device_memory &mem) override;
  void mem_copy_from(device_memory &mem, const size_t y, size_t w, const size_t h, size_t elem) override;
  void mem_zero(device_memory &mem) override;
  void mem_free(device_memory &mem) override;
  void build_bvh(BVH *bvh, Progress &progress, bool refit) override;

 private:
  // Vulkan handles
  VkInstance instance;
  VkPhysicalDevice physical_device;
  VkDevice device;
  VkQueue queue;
  uint32_t queue_family_index;

  // Helper to find a suitable queue family
  uint32_t find_queue_family(VkPhysicalDevice phys_dev);
};

CCL_NAMESPACE_END

#endif /* WITH_VULKAN */