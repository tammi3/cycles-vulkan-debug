/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 * SPDX-License-Identifier: Apache-2.0 */

#ifdef WITH_VULKAN

#include "device/vulkan/device_impl.h"
#include "util/log.h"
#include <vector>
#include <cstdio>
#include <cstring>

// Nuclear option debugging macros
#define VDBG_ENTER() printf("[DBG] %s ENTER\n", __func__); fflush(stdout);
#define VDBG_EXIT()  printf("[DBG] %s EXIT\n", __func__); fflush(stdout);

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

VulkanDevice::VulkanDevice(const DeviceInfo &info, Stats &stats, Profiler &profiler, bool headless)
    : GPUDevice(info, stats, profiler, headless), instance(VK_NULL_HANDLE), physical_device(VK_NULL_HANDLE),
      device(VK_NULL_HANDLE), queue(VK_NULL_HANDLE), queue_family_index(0)
{
  VDBG_ENTER();
  // ... (Constructor body is the same as before, just keeping it compact here) ...
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
  if (vkCreateInstance(&create_info, nullptr, &instance) != VK_SUCCESS) LOG_FATAL << "Failed to create instance.";

  uint32_t device_count = 0;
  vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
  std::vector<VkPhysicalDevice> devices(device_count);
  vkEnumeratePhysicalDevices(instance, &device_count, devices.data());
  if (info.num < device_count) physical_device = devices[info.num];
  else LOG_FATAL << "Device index out of range.";

  queue_family_index = find_queue_family(physical_device);

  float queue_priority = 1.0f;
  VkDeviceQueueCreateInfo queue_create_info{};
  queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queue_create_info.queueFamilyIndex = queue_family_index;
  queue_create_info.queueCount = 1;
  queue_create_info.pQueuePriorities = &queue_priority;

  VkDeviceCreateInfo dev_create_info{};
  dev_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  dev_create_info.queueCreateInfoCount = 1;
  dev_create_info.pQueueCreateInfos = &queue_create_info;
  if (vkCreateDevice(physical_device, &dev_create_info, nullptr, &device) != VK_SUCCESS) LOG_FATAL << "Failed to create logical device.";

  vkGetDeviceQueue(device, queue_family_index, 0, &queue);
  init_host_memory();
  VDBG_EXIT();
}

VulkanDevice::~VulkanDevice() {
  VDBG_ENTER();
  
  // Clean up constant memory
  if (const_buffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(device, const_buffer, nullptr);
  }
  if (const_memory != VK_NULL_HANDLE) {
    vkFreeMemory(device, const_memory, nullptr);
  }
  
  if (device != VK_NULL_HANDLE) {
    vkDeviceWaitIdle(device);
    vkDestroyDevice(device, nullptr);
  }
  if (instance != VK_NULL_HANDLE) {
    vkDestroyInstance(instance, nullptr);
  }
  VDBG_EXIT();
}

// =========================================================================
// CORE DEVICE OVERRIDES
// =========================================================================
BVHLayoutMask VulkanDevice::get_bvh_layout_mask(const uint /*kernel_features*/) const { VDBG_ENTER(); VDBG_EXIT(); return BVH_LAYOUT_BVH2; }

void VulkanDevice::const_copy_to(const char *name, void *host, const size_t size) {
  VDBG_ENTER();
  
  // If this is the first call, allocate the constant buffer
  if (const_buffer == VK_NULL_HANDLE) {
    // Allocate a buffer large enough for constant memory (Cycles uses ~64KB typically)
    // We'll allocate 1MB to be safe
    const_size = 1024 * 1024; 
    
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = const_size;
    buffer_info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    if (vkCreateBuffer(device, &buffer_info, nullptr, &const_buffer) != VK_SUCCESS) {
      LOG_FATAL << "Vulkan: Failed to create constant buffer.";
    }
    
    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(device, const_buffer, &mem_reqs);
    
    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_reqs.size;
    alloc_info.memoryTypeIndex = find_memory_type(mem_reqs.memoryTypeBits, 
                                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
                                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
    if (vkAllocateMemory(device, &alloc_info, nullptr, &const_memory) != VK_SUCCESS) {
      LOG_FATAL << "Vulkan: Failed to allocate constant memory.";
    }
    
    vkBindBufferMemory(device, const_buffer, const_memory, 0);
  }
  
  // Copy the data to the constant buffer
  // Cycles calls this multiple times with different 'name' values, each at a different offset
  // For now, we'll just copy to offset 0. A proper implementation would track offsets per name.
  void* mapped;
  vkMapMemory(device, const_memory, 0, size, 0, &mapped);
  memcpy(mapped, host, size);
  vkUnmapMemory(device, const_memory);
  
  VDBG_EXIT();
}

bool VulkanDevice::load_kernels(uint /*kernel_features*/) { VDBG_ENTER(); VDBG_EXIT(); return true; }

unique_ptr<DeviceQueue> VulkanDevice::gpu_queue_create() { VDBG_ENTER(); VDBG_EXIT(); return nullptr; }

void VulkanDevice::build_bvh(BVH *bvh, Progress &progress, bool refit) { VDBG_ENTER(); (void)bvh; (void)progress; (void)refit; VDBG_EXIT(); }

// =========================================================================
// HIGH-LEVEL MEMORY OVERRIDES
// =========================================================================
void VulkanDevice::mem_alloc(device_memory &mem) { VDBG_ENTER(); generic_alloc(mem); VDBG_EXIT(); }
void VulkanDevice::mem_copy_to(device_memory &mem) { VDBG_ENTER(); generic_copy_to(mem); VDBG_EXIT(); }
void VulkanDevice::mem_move_to_host(device_memory &mem) { VDBG_ENTER(); (void)mem; VDBG_EXIT(); }
void VulkanDevice::mem_copy_from(device_memory &mem, const size_t y, size_t w, const size_t h, size_t elem) { VDBG_ENTER(); (void)mem; (void)y; (void)w; (void)h; (void)elem; VDBG_EXIT(); }
void VulkanDevice::mem_zero(device_memory &mem) { VDBG_ENTER(); (void)mem; VDBG_EXIT(); }
void VulkanDevice::mem_free(device_memory &mem) { VDBG_ENTER(); generic_free(mem); VDBG_EXIT(); }

// =========================================================================
// SHARED MEMORY OVERRIDES
// =========================================================================
bool VulkanDevice::shared_alloc(void *&shared_pointer, const size_t size) { VDBG_ENTER(); (void)size; shared_pointer = nullptr; VDBG_EXIT(); return false; }
void VulkanDevice::shared_free(void *shared_pointer) { VDBG_ENTER(); (void)shared_pointer; VDBG_EXIT(); }
void *VulkanDevice::shared_to_device_pointer(const void *shared_pointer) { VDBG_ENTER(); (void)shared_pointer; VDBG_EXIT(); return nullptr; }

// =========================================================================
// LOW-LEVEL VULKAN MEMORY
// =========================================================================
bool VulkanDevice::alloc_device(void *&device_pointer, const size_t size) {
  VDBG_ENTER();
  VkBuffer buffer;
  VkBufferCreateInfo buffer_info{};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.size = size;
  buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  if (vkCreateBuffer(device, &buffer_info, nullptr, &buffer) != VK_SUCCESS) { VDBG_EXIT(); return false; }

  VkMemoryRequirements mem_reqs;
  vkGetBufferMemoryRequirements(device, buffer, &mem_reqs);

  VkMemoryAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc_info.allocationSize = mem_reqs.size;
  alloc_info.memoryTypeIndex = find_memory_type(mem_reqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  VkDeviceMemory memory;
  if (vkAllocateMemory(device, &alloc_info, nullptr, &memory) != VK_SUCCESS) {
    vkDestroyBuffer(device, buffer, nullptr);
    VDBG_EXIT();
    return false;
  }
  vkBindBufferMemory(device, buffer, memory, 0);

  VulkanAlloc* alloc = new VulkanAlloc{buffer, memory};
  device_pointer = alloc;
  VDBG_EXIT();
  return true;
}

void VulkanDevice::free_device(void *device_pointer) {
  VDBG_ENTER();
  if (!device_pointer) { VDBG_EXIT(); return; }
  VulkanAlloc* alloc = static_cast<VulkanAlloc*>(device_pointer);
  vkDestroyBuffer(device, alloc->buffer, nullptr);
  vkFreeMemory(device, alloc->memory, nullptr);
  delete alloc;
  VDBG_EXIT();
}

void VulkanDevice::copy_host_to_device(void *device_pointer, void *host_pointer, const size_t size) {
  VDBG_ENTER();
  VulkanAlloc* alloc = static_cast<VulkanAlloc*>(device_pointer);
  void* mapped;
  vkMapMemory(device, alloc->memory, 0, size, 0, &mapped);
  memcpy(mapped, host_pointer, size);
  vkUnmapMemory(device, alloc->memory);
  VDBG_EXIT();
}

void VulkanDevice::get_device_memory_info(size_t &total, size_t &free) {
  VDBG_ENTER();
  VkPhysicalDeviceMemoryProperties mem_properties;
  vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_properties);
  total = 0;
  for (uint32_t i = 0; i < mem_properties.memoryHeapCount; i++) {
    if (mem_properties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) total += mem_properties.memoryHeaps[i].size;
  }
  free = total; 
  VDBG_EXIT();
}

CCL_NAMESPACE_END

#endif /* WITH_VULKAN */