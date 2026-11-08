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

  /* M4f slice 5: const access to the host shadow of the whole fixed kernel
   * params buffer (KernelParamsVulkan, kernel/device/vulkan/params.h). The
   * production dispatch reads the BDA slots it needs from the SAME host bytes
   * const_copy_to uploaded: integrator_state (the SoA slice addresses set by
   * PathTraceWorkGPU) and the KERNEL_DATA_ARRAY pointers such as lookup_table
   * / sample_pattern_lut (device.cpp:532 writes each MEM_GLOBAL's device
   * pointer here by name). Read-only. */
  const KernelParamsVulkan &host_params() const { return host_params_; }

  /* M4f slice 5: copy of the last-uploaded HOST contents of a MEM_GLOBAL
   * kernel array (name = device_memory::global_name(), e.g. "lookup_table",
   * "sample_pattern_lut"). Returns an empty vector when no global with that
   * name exists or it was never uploaded. The slice-5 gate's CPU reference
   * reads the SAME bytes the GPU buffer holds (global_last_uploaded is the
   * snapshot mem_copy_to keeps, device.cpp:521-528). */
  vector<uint8_t> global_snapshot(const char *name) const;

  /* M4f slice 5: download `size` bytes of a device allocation - identified by
   * its Buffer Device Address, which is what device_ptr IS on this backend
   * (device.cpp:759-763) - starting `offset` bytes into it. Gate readback
   * helper: mirrors copy_host_to_device (device.cpp:778) with the
   * staging-download pattern of mem_copy_from (device.cpp:582-628);
   * address_to_alloc_map is private, so the lookup must live here. Returns
   * false with a LOG_ERROR when the device is not ready, the address is
   * unknown, or the request exceeds the allocation (fail loudly instead of
   * reading out of range). NOTE: the map keys are allocation BASE addresses
   * only, so `device_pointer` must be one - sub-ranges go through `offset`. */
  bool copy_device_to_host(const void *device_pointer,
                           const size_t offset,
                           void *host,
                           const size_t size);

  /* M4f slice 5: host -> device upload at an offset within an allocation.
   * Symmetric mirror of copy_device_to_host. */
  bool copy_host_to_device_at(void *device_pointer,
                              const size_t offset,
                              const void *host,
                              const size_t size);

  /* Allocation size of a device pointer registered in address_to_alloc_map (0 if unknown). */
  size_t allocation_size(const void *device_pointer) const;

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