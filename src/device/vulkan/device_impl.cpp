/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 * SPDX-License-Identifier: Apache-2.0 */

#ifdef WITH_VULKAN

#include "device/vulkan/device_impl.h"
#include "util/log.h"
#include <vector>

CCL_NAMESPACE_BEGIN

uint32_t VulkanDevice::find_queue_family(VkPhysicalDevice phys_dev) {
  uint32_t queue_family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(phys_dev, &queue_family_count, nullptr);
  std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(phys_dev, &queue_family_count, queue_families.data());

  for (uint32_t i = 0; i < queue_family_count; i++) {
    // We need compute support. (Graphics is a bonus for interop later).
    if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
      return i;
    }
  }
  LOG_FATAL << "Vulkan: No compute queue family found.";
  return 0;
}

VulkanDevice::VulkanDevice(const DeviceInfo &info, Stats &stats, Profiler &profiler, bool headless)
    : Device(info, stats, profiler, headless), 
      instance(VK_NULL_HANDLE), 
      physical_device(VK_NULL_HANDLE),
      device(VK_NULL_HANDLE), 
      queue(VK_NULL_HANDLE), 
      queue_family_index(0)
{
  LOG_INFO << "Creating Vulkan device: " << info.description;

  // 1. Create Instance
  VkApplicationInfo app_info{};
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pApplicationName = "Cycles Vulkan";
  app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.pEngineName = "Cycles";
  app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.apiVersion = VK_API_VERSION_1_2; // RX 570 supports 1.2 easily

  VkInstanceCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.pApplicationInfo = &app_info;

  if (vkCreateInstance(&create_info, nullptr, &instance) != VK_SUCCESS) {
    LOG_FATAL << "Vulkan: Failed to create instance.";
  }

  // 2. Select Physical Device (match the index we found in device_vulkan_info)
  uint32_t device_count = 0;
  vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
  std::vector<VkPhysicalDevice> devices(device_count);
  vkEnumeratePhysicalDevices(instance, &device_count, devices.data());

  if (info.num < device_count) {
    physical_device = devices[info.num];
  } else {
    LOG_FATAL << "Vulkan: Requested device index " << info.num << " is out of range.";
  }

  // 3. Find Queue Family
  queue_family_index = find_queue_family(physical_device);

  // 4. Create Logical Device
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

  if (vkCreateDevice(physical_device, &dev_create_info, nullptr, &device) != VK_SUCCESS) {
    LOG_FATAL << "Vulkan: Failed to create logical device.";
  }

  vkGetDeviceQueue(device, queue_family_index, 0, &queue);
  LOG_INFO << "Vulkan: Device initialized successfully.";
}

VulkanDevice::~VulkanDevice() {
  LOG_INFO << "Destroying Vulkan device.";
  if (device != VK_NULL_HANDLE) {
    vkDestroyDevice(device, nullptr);
  }
  if (instance != VK_NULL_HANDLE) {
    vkDestroyInstance(instance, nullptr);
  }
}

BVHLayoutMask VulkanDevice::get_bvh_layout_mask(const uint /*kernel_features*/) const {
  return BVH_LAYOUT_BVH2;
}

void VulkanDevice::const_copy_to(const char *name, void *host, const size_t size) {
  (void)name; (void)host; (void)size;
}

bool VulkanDevice::load_kernels(uint /*kernel_features*/) {
  return true;
}

unique_ptr<DeviceQueue> VulkanDevice::gpu_queue_create() {
  return nullptr;
}

void VulkanDevice::mem_alloc(device_memory &mem) { (void)mem; }
void VulkanDevice::mem_copy_to(device_memory &mem) { (void)mem; }
void VulkanDevice::mem_move_to_host(device_memory &mem) { (void)mem; }
void VulkanDevice::mem_copy_from(device_memory &mem, const size_t y, size_t w, const size_t h, size_t elem) {
  (void)mem; (void)y; (void)w; (void)h; (void)elem;
}
void VulkanDevice::mem_zero(device_memory &mem) { (void)mem; }
void VulkanDevice::mem_free(device_memory &mem) { (void)mem; }

void VulkanDevice::build_bvh(BVH *bvh, Progress &progress, bool refit) {
  (void)bvh; (void)progress; (void)refit;
}

CCL_NAMESPACE_END

#endif /* WITH_VULKAN */