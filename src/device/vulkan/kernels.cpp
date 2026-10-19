/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 * SPDX-License-Identifier: Apache-2.0 */

/* Vulkan device entry points: enumeration, creation and capability reporting.
 * This mirrors the structure of the other backends (cuda/oneapi): a small
 * "public" translation unit that implements the functions declared in
 * device/vulkan/device.h, keeping the heavy implementation details inside
 * device_impl.h / device.cpp. */

#ifdef WITH_VULKAN

#include "device/vulkan/device.h"
#include "device/vulkan/device_impl.h"

#include "util/log.h"

CCL_NAMESPACE_BEGIN

namespace {

/* Cheap, instance-level probe. We deliberately do NOT cache the device list
 * here; Device::vulkan_devices() in device/device.cpp owns the cache and
 * calls device_vulkan_info() to populate it. */
bool vulkan_available()
{
  uint32_t count = 0;
  if (vkEnumerateInstanceLayerProperties(&count, nullptr) != VK_SUCCESS) {
    return false;
  }

  VkApplicationInfo app_info{};
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pApplicationName = "Cycles Vulkan probe";
  app_info.apiVersion = VK_API_VERSION_1_2;

  VkInstanceCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.pApplicationInfo = &app_info;

  VkInstance instance = VK_NULL_HANDLE;
  if (vkCreateInstance(&create_info, nullptr, &instance) != VK_SUCCESS) {
    return false;
  }

  uint32_t device_count = 0;
  const VkResult res = vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
  vkDestroyInstance(instance, nullptr);

  return (res == VK_SUCCESS && device_count > 0);
}

} /* namespace */

bool device_vulkan_init()
{
  static const bool available = vulkan_available();
  return available;
}

unique_ptr<Device> device_vulkan_create(const DeviceInfo &info,
                                        Stats &stats,
                                        Profiler &profiler,
                                        bool headless)
{
  return make_unique<VulkanDevice>(info, stats, profiler, headless);
}

void device_vulkan_info(vector<DeviceInfo> &devices)
{
  VkApplicationInfo app_info{};
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pApplicationName = "Cycles Vulkan info";
  app_info.apiVersion = VK_API_VERSION_1_2;

  VkInstanceCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.pApplicationInfo = &app_info;

  VkInstance instance = VK_NULL_HANDLE;
  if (vkCreateInstance(&create_info, nullptr, &instance) != VK_SUCCESS) {
    VLOG_ERROR << "Vulkan: failed to create instance during device enumeration.";
    return;
  }

  uint32_t device_count = 0;
  vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
  vector<VkPhysicalDevice> physical_devices(device_count);
  vkEnumeratePhysicalDevices(instance, &device_count, physical_devices.data());

  for (uint32_t i = 0; i < device_count; i++) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physical_devices[i], &props);

    DeviceInfo info;
    info.id = string("VK") + to_string(i);
    info.name = props.deviceName;
    info.type = DEVICE_VULKAN;
    info.num = i;
    info.has_automatic_memory_managed_by_kernels = true;

    /* Report only the device-local heap as usable memory; Cycles uses this
     * to decide what fits on the GPU. */
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(physical_devices[i], &mem_props);
    size_t device_local = 0;
    for (uint32_t h = 0; h < mem_props.memoryHeapCount; h++) {
      if (mem_props.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
        device_local += mem_props.memoryHeaps[h].size;
      }
    }
    info.total_memory = device_local;
    info.free_memory = device_local;

    devices.push_back(info);
  }

  vkDestroyInstance(instance, nullptr);
}

string device_vulkan_capabilities()
{
  string capabilities;

  VkApplicationInfo app_info{};
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pApplicationName = "Cycles Vulkan capabilities";
  app_info.apiVersion = VK_API_VERSION_1_2;

  VkInstanceCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.pApplicationInfo = &app_info;

  VkInstance instance = VK_NULL_HANDLE;
  if (vkCreateInstance(&create_info, nullptr, &instance) != VK_SUCCESS) {
    return capabilities;
  }

  uint32_t device_count = 0;
  vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
  vector<VkPhysicalDevice> physical_devices(device_count);
  vkEnumeratePhysicalDevices(instance, &device_count, physical_devices.data());

  for (uint32_t i = 0; i < device_count; i++) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physical_devices[i], &props);

    VkPhysicalDeviceFeatures features;
    vkGetPhysicalDeviceFeatures(physical_devices[i], &features);

    VkPhysicalDeviceBufferDeviceAddressFeatures bda_features{};
    bda_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &bda_features;
    vkGetPhysicalDeviceFeatures2(physical_devices[i], &features2);

    capabilities += string(props.deviceName) + ": API " +
                    to_string(VK_VERSION_MAJOR(props.apiVersion)) + "." +
                    to_string(VK_VERSION_MINOR(props.apiVersion)) + ", " +
                    "shaderInt64=" + (features.shaderInt64 ? "yes" : "no") + ", " +
                    "bufferDeviceAddress=" +
                    (bda_features.bufferDeviceAddress ? "yes" : "no") + "\n";
  }

  vkDestroyInstance(instance, nullptr);
  return capabilities;
}

CCL_NAMESPACE_END

#endif /* WITH_VULKAN */
