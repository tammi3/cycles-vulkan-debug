/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#ifdef WITH_VULKAN

#include "device/device.h"
#include "device/queue.h"
#include "device/vulkan/vulkan_pipeline.h"
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
  VkCommandPool get_cmd_pool() const { return cmd_pool; }

  // Pipeline access for queue
  VulkanPipeline &pipeline() { return pipeline_; }
  const VulkanPipeline &pipeline() const { return pipeline_; }

  // Scene buffer accessors for kernel params
  uint64_t get_bvh_nodes_address() const { return get_buffer_address("bvh_nodes"); }
  uint64_t get_bvh_prim_offsets_address() const { return get_buffer_address("bvh_leaf_nodes"); }
  uint64_t get_bvh_prim_count_address() const { return 0; }  // Not used in BVH2 layout
  uint64_t get_tri_indices_address() const { return get_buffer_address("tri_vindex"); }
  uint64_t get_vert_co_address() const { return get_buffer_address("tri_verts"); }
  uint64_t get_object_transforms_address() const { return get_buffer_address("objects"); }
  uint32_t get_num_triangles() const { return num_triangles_; }
  void set_num_triangles(uint32_t n) { num_triangles_ = n; }

 private:
  uint64_t get_buffer_address(const std::string &name) const;

  VkInstance instance;
  VkPhysicalDevice physical_device;
  VkDevice device;
  VkQueue queue;
  uint32_t queue_family_index;
  VkCommandPool cmd_pool;

  std::unordered_map<uint64_t, VulkanAlloc> address_to_alloc_map;
  std::unordered_map<std::string, size_t> const_offsets;
  // Snapshot of the last uploaded contents of each MEM_GLOBAL buffer, so we can
  // detect when the device_ptr values inside it changed and re-upload.
  std::unordered_map<const device_memory *, vector<uint8_t>> global_last_uploaded;
  
  // Track device pointers for key scene buffers by name
  std::unordered_map<std::string, uint64_t> buffer_addresses_;
  
  size_t const_buffer_size;
  VkBuffer const_buffer;
  VkDeviceMemory const_memory;
  uint64_t const_device_address;
  uint32_t num_triangles_ = 0;

  // M4: Path tracing compute pipeline
  VulkanPipeline pipeline_;
};

CCL_NAMESPACE_END

#endif /* WITH_VULKAN */