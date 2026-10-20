/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 * SPDX-License-Identifier: Apache-2.0 */

#ifdef WITH_VULKAN

#include "device/vulkan/device_impl.h"
#include "device/vulkan/queue.h"
#include "util/log.h"
#include <vector>
#include <cstring>

CCL_NAMESPACE_BEGIN

uint32_t VulkanDevice::find_queue_family(VkPhysicalDevice phys_dev) {
  uint32_t queue_family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(phys_dev, &queue_family_count, nullptr);
  std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(phys_dev, &queue_family_count, queue_families.data());
  for (uint32_t i = 0; i < queue_family_count; i++) {
    if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) return i;
  }
  return 0;
}

uint32_t VulkanDevice::find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties) {
  VkPhysicalDeviceMemoryProperties mem_properties;
  vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_properties);
  for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
    if ((type_filter & (1 << i)) && (mem_properties.memoryTypes[i].propertyFlags & properties) == properties) {
      return i;
    }
  }
  return 0;
}

void VulkanDevice::submit_and_wait(VkCommandBuffer cmd) {
  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &cmd;
  
  VkFenceCreateInfo fence_info{};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  VkFence fence;
  vkCreateFence(device, &fence_info, nullptr, &fence);
  
  vkQueueSubmit(queue, 1, &submit_info, fence);
  vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
  
  vkDestroyFence(device, fence, nullptr);
  vkFreeCommandBuffers(device, cmd_pool, 1, &cmd);
}

VulkanDevice::VulkanDevice(const DeviceInfo &info, Stats &stats, Profiler &profiler, bool headless)
    : GPUDevice(info, stats, profiler, headless), 
      instance(VK_NULL_HANDLE), physical_device(VK_NULL_HANDLE),
      device(VK_NULL_HANDLE), queue(VK_NULL_HANDLE), queue_family_index(0),
      cmd_pool(VK_NULL_HANDLE), const_buffer_size(0), 
      const_buffer(VK_NULL_HANDLE), const_memory(VK_NULL_HANDLE), const_device_address(0)
{
  // 1. Create Instance
  VkApplicationInfo app_info{};
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pApplicationName = "Cycles Vulkan";
  app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.pEngineName = "Cycles";
  app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.apiVersion = VK_API_VERSION_1_2;

  VkInstanceCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.pApplicationInfo = &app_info;
  if (vkCreateInstance(&create_info, nullptr, &instance) != VK_SUCCESS) {
    LOG_FATAL << "Vulkan: Failed to create instance.";
  }

  // 2. Select Physical Device
  uint32_t device_count = 0;
  vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
  std::vector<VkPhysicalDevice> devices(device_count);
  vkEnumeratePhysicalDevices(instance, &device_count, devices.data());
  if (info.num < device_count) physical_device = devices[info.num];
  else LOG_FATAL << "Vulkan: Device index out of range.";

  queue_family_index = find_queue_family(physical_device);

  // 3. Create Logical Device with required features
  VkPhysicalDeviceBufferDeviceAddressFeatures buffer_address_features{};
  buffer_address_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
  buffer_address_features.bufferDeviceAddress = VK_TRUE;

  // shaderInt64 lives in the BASE VkPhysicalDeviceFeatures struct (core since Vulkan 1.0),
  // not in an extension struct. There is no VkPhysicalDeviceShaderInt64Features type.
  VkPhysicalDeviceFeatures base_features{};
  base_features.shaderInt64 = VK_TRUE;

  // Extension features are chained via the pNext pointer instead.
  float queue_priority = 1.0f;
  VkDeviceQueueCreateInfo queue_create_info{};
  queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queue_create_info.queueFamilyIndex = queue_family_index;
  queue_create_info.queueCount = 1;
  queue_create_info.pQueuePriorities = &queue_priority;

  // NOTE: we use designated initializers instead of assigning to a member like
  // `pFeatures`/`enabledFeatures`. The struct's feature-pointer member was
  // renamed across Vulkan header revisions, and aggregate init works with any
  // of them because it names the *enumerator*, not a field path.
  const VkDeviceCreateInfo dev_create_info{
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &buffer_address_features,        // Chained extension features (BDA)
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_create_info,
      .pEnabledFeatures = &base_features,       // Core features (shaderInt64)
  };

  if (vkCreateDevice(physical_device, &dev_create_info, nullptr, &device) != VK_SUCCESS) {
    LOG_FATAL << "Vulkan: Failed to create logical device.";
  }

  vkGetDeviceQueue(device, queue_family_index, 0, &queue);

  // 4. Create Command Pool for memory operations
  VkCommandPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool_info.queueFamilyIndex = queue_family_index;
  pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  vkCreateCommandPool(device, &pool_info, nullptr, &cmd_pool);

  init_host_memory();
}

VulkanDevice::~VulkanDevice() {
  vkDeviceWaitIdle(device);
  
  if (const_buffer != VK_NULL_HANDLE) vkDestroyBuffer(device, const_buffer, nullptr);
  if (const_memory != VK_NULL_HANDLE) vkFreeMemory(device, const_memory, nullptr);
  if (cmd_pool != VK_NULL_HANDLE) vkDestroyCommandPool(device, cmd_pool, nullptr);
  if (device != VK_NULL_HANDLE) vkDestroyDevice(device, nullptr);
  if (instance != VK_NULL_HANDLE) vkDestroyInstance(instance, nullptr);
}

// =========================================================================
// CORE DEVICE OVERRIDES
// =========================================================================
BVHLayoutMask VulkanDevice::get_bvh_layout_mask(const uint) const { return BVH_LAYOUT_BVH2; }

void VulkanDevice::const_copy_to(const char *name, void *host, const size_t size) {
  size_t offset = 0;
  auto it = const_offsets.find(name);
  if (it != const_offsets.end()) {
    offset = it->second;
  } else {
    offset = const_buffer_size;
    const_offsets[name] = offset;
    const_buffer_size += size;
  }

  // Allocate 64MB constant buffer on first use
  if (const_buffer == VK_NULL_HANDLE) {
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = 64 * 1024 * 1024; // 64MB
    buffer_info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | 
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT | 
                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateBuffer(device, &buffer_info, nullptr, &const_buffer);

    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(device, const_buffer, &mem_reqs);

    VkMemoryAllocateFlagsInfo alloc_flags{};
    alloc_flags.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    alloc_flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.pNext = &alloc_flags;
    alloc_info.allocationSize = mem_reqs.size;
    alloc_info.memoryTypeIndex = find_memory_type(mem_reqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkAllocateMemory(device, &alloc_info, nullptr, &const_memory);
    vkBindBufferMemory(device, const_buffer, const_memory, 0);

    VkBufferDeviceAddressInfo addr_info{};
    addr_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    addr_info.buffer = const_buffer;
    const_device_address = vkGetBufferDeviceAddress(device, &addr_info);
  }

  // Upload via staging buffer
  VkBuffer staging;
  VkDeviceMemory staging_mem;
  VkBufferCreateInfo st_info{};
  st_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  st_info.size = size;
  st_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  vkCreateBuffer(device, &st_info, nullptr, &staging);

  VkMemoryRequirements st_reqs;
  vkGetBufferMemoryRequirements(device, staging, &st_reqs);
  VkMemoryAllocateInfo st_alloc{};
  st_alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  st_alloc.allocationSize = st_reqs.size;
  st_alloc.memoryTypeIndex = find_memory_type(st_reqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  vkAllocateMemory(device, &st_alloc, nullptr, &staging_mem);
  vkBindBufferMemory(device, staging, staging_mem, 0);

  void* mapped;
  vkMapMemory(device, staging_mem, 0, size, 0, &mapped);
  memcpy(mapped, host, size);
  vkUnmapMemory(device, staging_mem);

  VkCommandBufferAllocateInfo cmd_alloc{};
  cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cmd_alloc.commandPool = cmd_pool;
  cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cmd_alloc.commandBufferCount = 1;
  VkCommandBuffer cmd;
  vkAllocateCommandBuffers(device, &cmd_alloc, &cmd);

  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  vkBeginCommandBuffer(cmd, &begin_info);

  VkBufferCopy copy_region{};
  copy_region.size = size;
  copy_region.dstOffset = offset;
  vkCmdCopyBuffer(cmd, staging, const_buffer, 1, &copy_region);

  vkEndCommandBuffer(cmd);
  submit_and_wait(cmd);

  vkDestroyBuffer(device, staging, nullptr);
  vkFreeMemory(device, staging_mem, nullptr);
}

bool VulkanDevice::load_kernels(uint) { return true; }
unique_ptr<DeviceQueue> VulkanDevice::gpu_queue_create() { return make_unique<VulkanQueue>(this); }
// NOTE: no build_bvh() override here. Device::build_bvh() in src/device/device.cpp
// already provides the generic CPU-side BVH2 build, which is exactly what we want
// until GPU-side BVH building exists. Adding an empty stub would silently disable it.

// =========================================================================
// HIGH-LEVEL MEMORY OVERRIDES
// =========================================================================
void VulkanDevice::mem_alloc(device_memory &mem) { generic_alloc(mem); }
void VulkanDevice::mem_copy_to(device_memory &mem) {
  if (mem.type == MEM_GLOBAL) {
    // A MEM_GLOBAL is a kernel-side global array: the buffer holds an array of
    // device_ptrs, and each element may point to another Vulkan buffer. Upload
    // the *current* pointer values every time they change (compare against the
    // last uploaded snapshot; do not rely on generic_copy_to's residency check,
    // which skips re-uploads when host data "looks" unchanged).
    const size_t size = mem.memory_size();
    if (!mem.device_pointer) {
      // GPUDevice::alloc_device() takes `void *&` (CUDA/HIP pass real pointers).
      // We store Buffer Device Address values in device_ptr (uint64_t), so go
      // through a local void* and copy the address back afterwards.
      void *device_pointer = nullptr;
      if (!alloc_device(device_pointer, size)) {
        LOG_ERROR << "Vulkan: failed to allocate MEM_GLOBAL '" << mem.name_
                  << "' (" << size << " bytes).";
        return;
      }
      /* `device_ptr` is a plain integer typedef (util/types_base.h: uint64_t),
       * so reinterpret_cast (which only works between pointer types) is illegal
       * here. A void* -> integer conversion needs static_cast instead. */
      mem.device_pointer = static_cast<device_ptr>(reinterpret_cast<uintptr_t>(device_pointer));
    }
    auto it = address_to_alloc_map.find((uint64_t)mem.device_pointer);
    vector<uint8_t> &last = global_last_uploaded[&mem];
    const bool contents_changed =
        (last.size() != size || memcmp(last.data(), mem.host_pointer, size) != 0);
    if (it != address_to_alloc_map.end() && it->second.size >= size && contents_changed) {
      void *device_pointer = reinterpret_cast<void *>(static_cast<uintptr_t>(mem.device_pointer));
      copy_host_to_device(device_pointer, mem.host_pointer, size);
      global_last_uploaded[&mem].resize(size);
      memcpy(global_last_uploaded[&mem].data(), mem.host_pointer, size);
    }
    // Tell the kernel where this global array lives in VRAM (device_ptr value
    // = BDA address), written into the constant buffer at the symbol's offset.
    const_copy_to(mem.global_name(), &mem.device_pointer, sizeof(device_ptr));
  } else {
    generic_copy_to(mem);
  }
}
void VulkanDevice::mem_move_to_host(device_memory &) {}
void VulkanDevice::mem_copy_from(device_memory &mem, const size_t y, size_t w, const size_t h, size_t elem) {
  // Cycles' tiling contract (see CUDADevice::mem_copy_from): the tile is a
  // CONTIGUOUS block of `h` rows of width `w`. The caller keeps host and device
  // buffers laid out identically for the current region, so a single linear
  // strided copy at offset = elem*y*w is correct; row-splitting would be WRONG.
  if (!mem.host_pointer) return;

  const size_t size_bytes = mem.memory_elements_size(elem * w) * h;
  const size_t offset_bytes = mem.memory_elements_size(elem * y * w);

  if (!mem.device_pointer) {
    // Device never got the data; zero the host tile so the image isn't garbage.
    memset(static_cast<uint8_t *>(mem.host_pointer) + offset_bytes, 0, size_bytes);
    return;
  }

  auto it = address_to_alloc_map.find((uint64_t)mem.device_pointer);
  if (it == address_to_alloc_map.end()) return;
  if (size_bytes == 0) return;

  VkBuffer staging;
  VkDeviceMemory staging_mem;
  VkBufferCreateInfo st_info{};
  st_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  st_info.size = size_bytes;
  st_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  vkCreateBuffer(device, &st_info, nullptr, &staging);

  VkMemoryRequirements st_reqs;
  vkGetBufferMemoryRequirements(device, staging, &st_reqs);

  VkMemoryAllocateInfo st_alloc{};
  st_alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  st_alloc.allocationSize = st_reqs.size;
  st_alloc.memoryTypeIndex = find_memory_type(st_reqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  vkAllocateMemory(device, &st_alloc, nullptr, &staging_mem);
  vkBindBufferMemory(device, staging, staging_mem, 0);

  VkCommandBufferAllocateInfo cmd_alloc{};
  cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cmd_alloc.commandPool = cmd_pool;
  cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cmd_alloc.commandBufferCount = 1;
  VkCommandBuffer cmd;
  vkAllocateCommandBuffers(device, &cmd_alloc, &cmd);

  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  vkBeginCommandBuffer(cmd, &begin_info);

  // Single linear copy of the tile region.
  VkBufferCopy copy_region{};
  copy_region.size = size_bytes;
  copy_region.srcOffset = offset_bytes;
  copy_region.dstOffset = 0;
  vkCmdCopyBuffer(cmd, it->second.buffer, staging, 1, &copy_region);

  vkEndCommandBuffer(cmd);
  submit_and_wait(cmd);

  void *mapped;
  vkMapMemory(device, staging_mem, 0, size_bytes, 0, &mapped);
  memcpy(static_cast<uint8_t *>(mem.host_pointer) + offset_bytes, mapped, size_bytes);
  vkUnmapMemory(device, staging_mem);

  vkDestroyBuffer(device, staging, nullptr);
  vkFreeMemory(device, staging_mem, nullptr);
}


void VulkanDevice::mem_zero(device_memory &mem) {
  // Safety check: allocate if Cycles hasn't allocated it yet
  if (!mem.device_pointer) {
    generic_alloc(mem);
  }
  
  auto it = address_to_alloc_map.find((uint64_t)mem.device_pointer);
  if (it != address_to_alloc_map.end()) {
    VkCommandBufferAllocateInfo cmd_alloc{};
    cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_alloc.commandPool = cmd_pool;
    cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_alloc.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device, &cmd_alloc, &cmd);

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmd, &begin_info);

    vkCmdFillBuffer(cmd, it->second.buffer, 0, VK_WHOLE_SIZE, 0);

    vkEndCommandBuffer(cmd);
    submit_and_wait(cmd);
  }
  
  // Keep the CPU host pointer in sync with the GPU
  if (mem.host_pointer) {
    memset(mem.host_pointer, 0, mem.memory_size());
  }
}
void VulkanDevice::mem_free(device_memory &mem) { generic_free(mem); }

// =========================================================================
// SHARED MEMORY OVERRIDES
// =========================================================================
bool VulkanDevice::shared_alloc(void *&shared_pointer, const size_t) { shared_pointer = nullptr; return false; }
void VulkanDevice::shared_free(void *) {}
void *VulkanDevice::shared_to_device_pointer(const void *) { return nullptr; }

// =========================================================================
// LOW-LEVEL VULKAN MEMORY
// =========================================================================
bool VulkanDevice::alloc_device(void *&device_pointer, const size_t size) {
  VkBuffer buffer;
  VkBufferCreateInfo buffer_info{};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.size = size;
  buffer_info.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | 
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | 
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT | 
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  if (vkCreateBuffer(device, &buffer_info, nullptr, &buffer) != VK_SUCCESS) return false;

  VkMemoryRequirements mem_reqs;
  vkGetBufferMemoryRequirements(device, buffer, &mem_reqs);

  VkMemoryAllocateFlagsInfo alloc_flags{};
  alloc_flags.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
  alloc_flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

  VkMemoryAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc_info.pNext = &alloc_flags;
  alloc_info.allocationSize = mem_reqs.size;
  alloc_info.memoryTypeIndex = find_memory_type(mem_reqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  VkDeviceMemory memory;
  if (vkAllocateMemory(device, &alloc_info, nullptr, &memory) != VK_SUCCESS) {
    vkDestroyBuffer(device, buffer, nullptr);
    return false;
  }
  vkBindBufferMemory(device, buffer, memory, 0);

  VkBufferDeviceAddressInfo addr_info{};
  addr_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
  addr_info.buffer = buffer;
  uint64_t device_address = vkGetBufferDeviceAddress(device, &addr_info);

  VulkanAlloc alloc{buffer, memory, size};
  address_to_alloc_map[device_address] = alloc;
  device_pointer = (void*)device_address;
  return true;
}

void VulkanDevice::free_device(void *device_pointer) {
  if (!device_pointer) return;
  uint64_t addr = (uint64_t)device_pointer;
  auto it = address_to_alloc_map.find(addr);
  if (it != address_to_alloc_map.end()) {
    vkDestroyBuffer(device, it->second.buffer, nullptr);
    vkFreeMemory(device, it->second.memory, nullptr);
    address_to_alloc_map.erase(it);
  }
}

void VulkanDevice::copy_host_to_device(void *device_pointer, void *host_pointer, const size_t size) {
  uint64_t addr = (uint64_t)device_pointer;
  auto it = address_to_alloc_map.find(addr);
  if (it == address_to_alloc_map.end()) return;

  VkBuffer staging;
  VkDeviceMemory staging_mem;
  VkBufferCreateInfo st_info{};
  st_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  st_info.size = size;
  st_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  vkCreateBuffer(device, &st_info, nullptr, &staging);

  VkMemoryRequirements st_reqs;
  vkGetBufferMemoryRequirements(device, staging, &st_reqs);
  VkMemoryAllocateInfo st_alloc{};
  st_alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  st_alloc.allocationSize = st_reqs.size;
  st_alloc.memoryTypeIndex = find_memory_type(st_reqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  vkAllocateMemory(device, &st_alloc, nullptr, &staging_mem);
  vkBindBufferMemory(device, staging, staging_mem, 0);

  void* mapped;
  vkMapMemory(device, staging_mem, 0, size, 0, &mapped);
  memcpy(mapped, host_pointer, size);
  vkUnmapMemory(device, staging_mem);

  VkCommandBufferAllocateInfo cmd_alloc{};
  cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cmd_alloc.commandPool = cmd_pool;
  cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cmd_alloc.commandBufferCount = 1;
  VkCommandBuffer cmd;
  vkAllocateCommandBuffers(device, &cmd_alloc, &cmd);

  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  vkBeginCommandBuffer(cmd, &begin_info);

  VkBufferCopy copy_region{};
  copy_region.size = size;
  vkCmdCopyBuffer(cmd, staging, it->second.buffer, 1, &copy_region);

  vkEndCommandBuffer(cmd);
  submit_and_wait(cmd);

  vkDestroyBuffer(device, staging, nullptr);
  vkFreeMemory(device, staging_mem, nullptr);
}

void VulkanDevice::get_device_memory_info(size_t &total, size_t &free) {
  VkPhysicalDeviceMemoryProperties mem_properties;
  vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_properties);
  total = 0;
  for (uint32_t i = 0; i < mem_properties.memoryHeapCount; i++) {
    if (mem_properties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
      total += mem_properties.memoryHeaps[i].size;
    }
  }
  free = total; 
}

CCL_NAMESPACE_END

#endif /* WITH_VULKAN */