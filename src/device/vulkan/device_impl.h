/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#ifdef WITH_VULKAN

#include "device/device.h"
#include "device/queue.h"
#include "kernel/device/vulkan/params.h"
#include <vulkan/vulkan.h>
#include <unordered_map>
#include <string>

CCL_NAMESPACE_BEGIN

struct VulkanAlloc {
  VkBuffer buffer;
  VkDeviceMemory memory;
  VkDeviceSize size;
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

  // High-level memory overrides
  void mem_alloc(device_memory &mem) override;
  void mem_copy_to(device_memory &mem) override;
  void mem_move_to_host(device_memory &mem) override;
  void mem_copy_from(device_memory &mem, const size_t y, size_t w, const size_t h, size_t elem) override;
  void mem_zero(device_memory &mem) override;
  void mem_free(device_memory &mem) override;

  // GPUDevice pure virtuals
  bool alloc_device(void *&device_pointer, const size_t size) override;
  void free_device(void *device_pointer) override;
  void copy_host_to_device(void *device_pointer, void *host_pointer, const size_t size) override;
  void get_device_memory_info(size_t &total, size_t &free) override;

  // Shared memory stubs
  bool shared_alloc(void *&shared_pointer, const size_t size) override;
  void shared_free(void *shared_pointer) override;
  void *shared_to_device_pointer(const void *shared_pointer) override;

  // Vulkan specific helpers
  VkDevice vk_device() const { return device; }
  VkPhysicalDevice vk_physical_device() const { return physical_device; }
  bool is_ready() const;
  uint32_t find_queue_family(VkPhysicalDevice phys_dev);
  uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties);
  void submit_and_wait(VkCommandBuffer cmd);

  /* Buffer Device Address of the fixed kernel params buffer, or 0 if it has
   * not been created yet (created lazily by const_copy_to). */
  uint64_t kernel_params_address() const;

  /* M4e STEP 1a: const access to the real KernelData (host shadow of the
   * "data" kernel global, filled by const_copy_to("data", ...)). The math
   * probe reads kernel_data.cam / kernel_data.integrator from here so the
   * camera matrices come from the real loaded scene, NOT the KernelParamsVulkan
   * blob (device.cpp:317 memcpy's const data into host_params_). Read-only. */
  const KernelData &host_kernel_data() const { return host_params_.data; }

 private:
  VkInstance instance;
  VkPhysicalDevice physical_device;
  VkDevice device;
  VkQueue queue;
  uint32_t queue_family_index;
  VkCommandPool cmd_pool;

  std::unordered_map<uint64_t, VulkanAlloc> address_to_alloc_map;
  // Snapshot of the last uploaded contents of each MEM_GLOBAL buffer, so we can
  // detect when the device_ptr values inside it changed and re-upload.
  std::unordered_map<const device_memory *, vector<uint8_t>> global_last_uploaded;

  /* Host shadow of the device kernel params buffer (fixed layout, see
   * kernel/device/vulkan/params.h). const_copy_to() patches one field here and
   * uploads only that byte range; the rest of the buffer stays zero. */
  KernelParamsVulkan host_params_;
  VkBuffer const_buffer;
  VkDeviceMemory const_memory;
  uint64_t const_device_address;
};

CCL_NAMESPACE_END

#endif /* WITH_VULKAN */