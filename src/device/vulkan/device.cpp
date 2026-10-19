/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 * SPDX-License-Identifier: Apache-2.0 */

#include "device/vulkan/device.h"

#include "device/device.h"
#include "util/log.h"

#ifdef WITH_VULKAN
#  include "device/vulkan/device_impl.h"
#  include <vulkan/vulkan.h>
#endif

CCL_NAMESPACE_BEGIN

bool device_vulkan_init()
{
#ifdef WITH_VULKAN
  /* The actual instance creation happens in device_vulkan_info. 
   * If we get here, the build system successfully linked the Vulkan SDK. */
  return true;
#else
  return false;
#endif
}

unique_ptr<Device> device_vulkan_create(const DeviceInfo &info,
                                        Stats &stats,
                                        Profiler &profiler,
                                        const bool headless)
{
#ifdef WITH_VULKAN
  return make_unique<VulkanDevice>(info, stats, profiler, headless);
#else
  (void)info;
  (void)stats;
  (void)profiler;
  (void)headless;
  LOG_FATAL << "Request to create Vulkan device without compiled-in support.";
  return nullptr;
#endif
}

void device_vulkan_info(vector<DeviceInfo> &devices)
{
#ifdef WITH_VULKAN
  /* 1. Create a temporary Vulkan Instance just to query devices */
  VkInstance instance;
  VkInstanceCreateInfo create_info = {};
  create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;

  if (vkCreateInstance(&create_info, nullptr, &instance) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: Failed to create Vulkan instance.";
    return;
  }

  /* 2. Enumerate physical devices (GPUs) */
  uint32_t device_count = 0;
  vkEnumeratePhysicalDevices(instance, &device_count, nullptr);

  if (device_count == 0) {
    LOG_WARNING << "Vulkan: No compatible GPUs found.";
    vkDestroyInstance(instance, nullptr);
    return;
  }

  std::vector<VkPhysicalDevice> physical_devices(device_count);
  vkEnumeratePhysicalDevices(instance, &device_count, physical_devices.data());

  /* 3. Populate the Cycles DeviceInfo list */
  for (uint32_t i = 0; i < device_count; i++) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physical_devices[i], &props);

    /* We only want actual discrete or integrated GPUs, not software rasterizers */
    if (props.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && 
        props.deviceType != VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
      continue;
    }

    DeviceInfo info;
    info.type = DEVICE_VULKAN;
    info.description = string(props.deviceName) + " (Vulkan)";
    info.num = i;
    info.id = string_printf("VULKAN_%s_%d", props.deviceName, i);
    info.has_gpu_queue = true;
    info.display_device = true; /* Assume it's connected to a display for now */
    
    devices.push_back(info);
    LOG_INFO << "Vulkan: Found device: " << props.deviceName;
  }

  /* 4. Clean up the temporary instance */
  vkDestroyInstance(instance, nullptr);
#else
  (void)devices;
#endif
}

string device_vulkan_capabilities()
{
#ifdef WITH_VULKAN
  return "Vulkan backend (experimental)";
#else
  return "";
#endif
}

CCL_NAMESPACE_END