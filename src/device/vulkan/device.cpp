/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 * SPDX-License-Identifier: Apache-2.0 */

#ifdef WITH_VULKAN

#include "device/vulkan/device_impl.h"
#include "device/vulkan/queue.h"
#include "kernel/device/vulkan/params.h"
#include "util/log.h"
#include <vector>
#include <algorithm>
#include <cstdlib>
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
      cmd_pool(VK_NULL_HANDLE), host_params_(),
      const_buffer(VK_NULL_HANDLE), const_memory(VK_NULL_HANDLE), const_device_address(0)
{
  /* M3 exit criterion is "fails gracefully", so nothing in this constructor may
   * call LOG_FATAL: Cycles' log implementation calls abort() on a FATAL message
   * (src/util/log.cpp), which would crash cycles.exe instead of reporting an
   * error the user/session can act on. */

  // 1. Create Instance
  VkApplicationInfo app_info{};
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pApplicationName = "Cycles Vulkan";
  app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.pEngineName = "Cycles";
  app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.apiVersion = VK_API_VERSION_1_2;

  /* On Windows the loader only exposes pNext-chained feature structs to
   * vkGetPhysicalDeviceFeatures2 if the VK_KHR_get_physical_device_properties2
   * instance extension is enabled (on Vulkan 1.1+ it is implicit in core, but
   * enabling it explicitly is harmless and keeps behaviour identical across
   * drivers). Without it, RADV reports bufferDeviceAddress=false even when the
   * driver supports it, and vkCreateDevice then fails with
   * VK_ERROR_FEATURE_NOT_PRESENT. */
  const char *instance_extensions[] = {"VK_KHR_get_physical_device_properties2"};
  bool has_props2_ext = false;
  {
    uint32_t ext_count = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &ext_count, nullptr);
    vector<VkExtensionProperties> extensions(ext_count);
    vkEnumerateInstanceExtensionProperties(nullptr, &ext_count, extensions.data());
    for (const auto &ext : extensions) {
      if (strcmp(ext.extensionName, instance_extensions[0]) == 0) {
        has_props2_ext = true;
        break;
      }
    }
  }

  VkInstanceCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.pApplicationInfo = &app_info;
  if (has_props2_ext) {
    create_info.enabledExtensionCount = 1;
    create_info.ppEnabledExtensionNames = instance_extensions;
  }
  if (vkCreateInstance(&create_info, nullptr, &instance) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: Failed to create instance.";
    return;
  }

  // 2. Select Physical Device
  uint32_t device_count = 0;
  vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
  std::vector<VkPhysicalDevice> devices(device_count);
  vkEnumeratePhysicalDevices(instance, &device_count, devices.data());
  if (info.num >= device_count) {
    LOG_ERROR << "Vulkan: Device index " << info.num << " out of range ("
               << device_count << " device(s) found).";
    return;
  }
  physical_device = devices[info.num];

  queue_family_index = find_queue_family(physical_device);

  // 3. Query support BEFORE requesting: requesting an unsupported feature from
  // vkCreateDevice is a hard failure; we want to degrade gracefully instead.
  VkPhysicalDeviceFeatures supported_features{};
  vkGetPhysicalDeviceFeatures(physical_device, &supported_features);

  VkPhysicalDeviceBufferDeviceAddressFeatures supported_bda_features{};
  supported_bda_features.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
  // shaderInt8 lives in VkPhysicalDeviceVulkan12Features (core 1.2, from
  // VK_KHR_8bit_storage / VK_KHR_shader_float16_int8); chain it after the BDA struct so
  // one vkGetPhysicalDeviceFeatures2 call reports BDA + shaderInt8 together.
  VkPhysicalDeviceVulkan12Features supported_v12_features{};
  supported_v12_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  VkPhysicalDeviceFeatures2 supported_features2{};
  supported_features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  supported_features2.pNext = &supported_bda_features;
  supported_bda_features.pNext = &supported_v12_features;
  vkGetPhysicalDeviceFeatures2(physical_device, &supported_features2);

  const bool have_shader_int64 = (supported_features.shaderInt64 == VK_TRUE);
  const bool have_bda = (supported_bda_features.bufferDeviceAddress == VK_TRUE);
  // Sub-word SoA fields (uint16_t bounce/queued_kernel/rng_offset, uint8_t
  // visibility/mnee -- S1b-4 Stage A) are written with TRUE narrow stores through
  // PhysicalStorageBuffer. Those declare SPIR-V Int16/Int8, which the validation layer
  // gates on shaderInt16 (base features) / shaderInt8 (Vulkan12), per the probe's
  // VUID-VkShaderModuleCreateInfo-pCode-08740. Enable them when the device reports them.
  const bool have_shader_int16 = (supported_features.shaderInt16 == VK_TRUE);
  const bool have_shader_int8 = (supported_v12_features.shaderInt8 == VK_TRUE);

  if (!have_shader_int64) {
    LOG_WARNING << "Vulkan: device lacks shaderInt64; kernels requiring 64-bit "
                 << "arithmetic will not be available.";
  }
  if (!have_shader_int16 || !have_shader_int8) {
    LOG_WARNING << "Vulkan: device lacks shaderInt16 and/or shaderInt8; sub-word "
                 << "(uint16/uint8) SoA fields will need the atomic fallback.";
  }
  if (!have_bda) {
    /* Our entire memory model stores Buffer Device Address values as device_ptr
     * (uint64_t) and dereferences them in shaders. Without BDA there is no
     * safe fallback path yet, so report and let Session surface the error. */
    LOG_ERROR << "Vulkan: device lacks bufferDeviceAddress support; "
               << "this backend requires it (see VkPhysicalDeviceBufferDeviceAddressFeatures).";
    return;
  }

  // bufferDeviceAddress was promoted into VkPhysicalDeviceVulkan12Features (core 1.2).
  // The spec forbids having BOTH VkPhysicalDeviceBufferDeviceAddressFeatures and
  // VkPhysicalDeviceVulkan12Features in the device-create pNext chain (VUID
  // -VkDeviceCreateInfo-pNext-02830: a feature set VK_TRUE in one struct and VK_FALSE in
  // the other would be ambiguous), so we enable bufferDeviceAddress + shaderInt8 in a
  // SINGLE Vulkan12Features struct. Only request features the device supports (queried
  // above); a Vulkan 1.2 device is required for this struct to be valid, which the
  // successful shaderInt8 query already implies.
  VkPhysicalDeviceVulkan12Features device_v12_features{};
  device_v12_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  device_v12_features.bufferDeviceAddress = VK_TRUE;
  device_v12_features.shaderInt8 = have_shader_int8 ? VK_TRUE : VK_FALSE;

  // shaderInt64 / shaderInt16 live in the BASE VkPhysicalDeviceFeatures struct (core
  // since Vulkan 1.0), not in an extension struct. There is no
  // VkPhysicalDeviceShaderInt64Features type. Only request what the device supports.
  VkPhysicalDeviceFeatures base_features{};
  base_features.shaderInt64 = have_shader_int64 ? VK_TRUE : VK_FALSE;
  base_features.shaderInt16 = have_shader_int16 ? VK_TRUE : VK_FALSE;

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
      .pNext = &device_v12_features,           // Chained extension features (BDA + shaderInt8)
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_create_info,
      .pEnabledFeatures = &base_features,       // Core features (shaderInt64)
  };

  if (vkCreateDevice(physical_device, &dev_create_info, nullptr, &device) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: Failed to create logical device for '"
               << info.description << "'.";
    return;
  }

  vkGetDeviceQueue(device, queue_family_index, 0, &queue);

  // 4. Create Command Pool for memory operations
  VkCommandPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool_info.queueFamilyIndex = queue_family_index;
  pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  if (vkCreateCommandPool(device, &pool_info, nullptr, &cmd_pool) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: Failed to create command pool.";
    return;
  }

  init_host_memory();
}

VulkanDevice::~VulkanDevice() {
  if (device != VK_NULL_HANDLE) vkDeviceWaitIdle(device);
  
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
  if (!is_ready()) {
    LOG_ERROR << "Vulkan: device not initialized; ignoring const_copy_to.";
    return;
  }

  /* Fixed layout: every kernel global has a compile-time offset in
   * KernelParamsVulkan (kernel/device/vulkan/params.h), mirroring Cycles'
   * KernelParamsHIP + offsetof lookup (src/device/hip/device_impl.cpp:630-649).
   * Call-order offsets (the old code) are invisible to the shader, so an
   * unknown name or a size that does not match the slot is a hard error. */
  size_t offset = 0;
  size_t expected_size = 0;
  if (!kernel_params_offset(name, &offset, &expected_size)) {
    const string message = string("Vulkan: unknown kernel global '") + name +
                           "' passed to const_copy_to.";
    LOG_ERROR << message;
    set_error(message);
    return;
  }
  if (size != expected_size) {
    const string message = string("Vulkan: kernel global '") + name + "' has size " +
                           std::to_string(size) + ", expected " +
                           std::to_string(expected_size) + ".";
    LOG_ERROR << message;
    set_error(message);
    return;
  }

  // Allocate the kernel params buffer (sizeof(KernelParamsVulkan)) on first use
  if (const_buffer == VK_NULL_HANDLE) {
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = sizeof(KernelParamsVulkan);
    buffer_info.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer = VK_NULL_HANDLE;
    if (vkCreateBuffer(device, &buffer_info, nullptr, &buffer) != VK_SUCCESS) {
      const string message = "Vulkan: failed to create kernel params buffer.";
      LOG_ERROR << message;
      set_error(message);
      return;
    }

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

    VkDeviceMemory memory = VK_NULL_HANDLE;
    if (vkAllocateMemory(device, &alloc_info, nullptr, &memory) != VK_SUCCESS) {
      vkDestroyBuffer(device, buffer, nullptr);
      const string message = "Vulkan: failed to allocate kernel params memory.";
      LOG_ERROR << message;
      set_error(message);
      return;
    }
    if (vkBindBufferMemory(device, buffer, memory, 0) != VK_SUCCESS) {
      vkFreeMemory(device, memory, nullptr);
      vkDestroyBuffer(device, buffer, nullptr);
      const string message = "Vulkan: failed to bind kernel params memory.";
      LOG_ERROR << message;
      set_error(message);
      return;
    }
    const_buffer = buffer;
    const_memory = memory;

    VkBufferDeviceAddressInfo addr_info{};
    addr_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    addr_info.buffer = const_buffer;
    const_device_address = vkGetBufferDeviceAddress(device, &addr_info);
    if (const_device_address == 0) {
      const string message = "Vulkan: failed to get kernel params device address.";
      LOG_ERROR << message;
      set_error(message);
      return;
    }

    LOG_INFO << "Vulkan: KernelParamsVulkan " << sizeof(KernelParamsVulkan)
             << " bytes; offset(data)=" << offsetof(KernelParamsVulkan, data)
             << " offset(bvh_nodes)=" << offsetof(KernelParamsVulkan, bvh_nodes)
             << " offset(integrator_state)=" << offsetof(KernelParamsVulkan, integrator_state);
  }

  memcpy(reinterpret_cast<uint8_t *>(&host_params_) + offset, host, size);

  /* VkBufferCopy::size must be a multiple of nonCoherentAtomSize (always <= 4
   * on desktop drivers, 4 in practice). Round up; staging is sized copy_size so
   * the over-copy stays inside both buffers. */
  const size_t copy_size = ((size + 3) / 4) * 4;
  VkBuffer staging = VK_NULL_HANDLE;
  VkDeviceMemory staging_mem = VK_NULL_HANDLE;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  const auto staging_fail = [&](const char *step) {
    if (cmd != VK_NULL_HANDLE) vkFreeCommandBuffers(device, cmd_pool, 1, &cmd);
    if (staging != VK_NULL_HANDLE) vkDestroyBuffer(device, staging, nullptr);
    if (staging_mem != VK_NULL_HANDLE) vkFreeMemory(device, staging_mem, nullptr);
    const string message = string("Vulkan: const_copy_to upload failed (") + step + ").";
    LOG_ERROR << message;
    set_error(message);
  };

  // Upload only the patched byte range via staging buffer
  VkBufferCreateInfo st_info{};
  st_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  st_info.size = copy_size;
  st_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  if (vkCreateBuffer(device, &st_info, nullptr, &staging) != VK_SUCCESS) {
    staging_fail("vkCreateBuffer");
    return;
  }

  VkMemoryRequirements st_reqs;
  vkGetBufferMemoryRequirements(device, staging, &st_reqs);
  VkMemoryAllocateInfo st_alloc{};
  st_alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  st_alloc.allocationSize = st_reqs.size;
  st_alloc.memoryTypeIndex = find_memory_type(st_reqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  if (vkAllocateMemory(device, &st_alloc, nullptr, &staging_mem) != VK_SUCCESS) {
    staging_fail("vkAllocateMemory");
    return;
  }
  if (vkBindBufferMemory(device, staging, staging_mem, 0) != VK_SUCCESS) {
    staging_fail("vkBindBufferMemory");
    return;
  }

  /* The round-up can expose up to 3 bytes the caller did not provide; zero the
   * staging tail so padding that reaches the params buffer is not garbage. */
  void *mapped = nullptr;
  if (vkMapMemory(device, staging_mem, 0, copy_size, 0, &mapped) != VK_SUCCESS) {
    staging_fail("vkMapMemory");
    return;
  }
  memset(mapped, 0, copy_size);
  memcpy(mapped, host, size);
  vkUnmapMemory(device, staging_mem);

  VkCommandBufferAllocateInfo cmd_alloc{};
  cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cmd_alloc.commandPool = cmd_pool;
  cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cmd_alloc.commandBufferCount = 1;
  if (vkAllocateCommandBuffers(device, &cmd_alloc, &cmd) != VK_SUCCESS) {
    staging_fail("vkAllocateCommandBuffers");
    return;
  }

  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  if (vkBeginCommandBuffer(cmd, &begin_info) != VK_SUCCESS) {
    staging_fail("vkBeginCommandBuffer");
    return;
  }

  VkBufferCopy copy_region{};
  copy_region.size = copy_size;
  copy_region.dstOffset = offset;
  vkCmdCopyBuffer(cmd, staging, const_buffer, 1, &copy_region);

  if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
    staging_fail("vkEndCommandBuffer");
    return;
  }
  submit_and_wait(cmd);

  vkDestroyBuffer(device, staging, nullptr);
  vkFreeMemory(device, staging_mem, nullptr);
}

uint64_t VulkanDevice::kernel_params_address() const { return const_device_address; }

bool VulkanDevice::is_ready() const
{
  /* A constructor may have bailed out early (LOG_FATAL was replaced by graceful
   * LOG_ERROR+return so M3 never abort()s). Guard every entry point on the
   * logical device actually existing. */
  return (device != VK_NULL_HANDLE) && (queue != VK_NULL_HANDLE);
}

bool VulkanDevice::load_kernels(uint)
{
  if (!is_ready()) {
    set_error("Vulkan: device initialization failed; see log for details.");
    return false;
  }
  if (getenv("CYCLES_VULKAN_ALLOW_NO_KERNELS") != nullptr) {
    /* M4a-1 test hook. Without it this function returns false + set_error(),
     * and Scene::device_update() bails out at src/scene/scene.cpp:244
     * (device->have_error()) *before* any device array is uploaded, so
     * const_copy_to()/mem_copy_to() would never run. Returning true WITHOUT
     * set_error() lets the scene upload proceed; the first
     * VulkanQueue::enqueue() then reports the M4 error and the session stops
     * cleanly (src/device/vulkan/queue.cpp:30-35).
     * Default behaviour (variable unset) is unchanged. */
    LOG_WARNING << "Vulkan: CYCLES_VULKAN_ALLOW_NO_KERNELS=1 set -- continuing "
                 << "without compiled kernels; dispatch will fail (M4).";
    return true;
  }
  /* M3 exit criterion: fail gracefully. Cycles has no C++->SPIR-V pipeline
   * yet (the "kernel compilation problem", our M4 work). Returning false
   * makes Session abort cleanly before any task is enqueued, instead of
   * letting a render run into the VulkanQueue::enqueue() stub mid-frame. */
  const string message =
      "Vulkan: no kernels compiled yet -- device is enumerated and memory-routed "
      "only (Milestone 3). Rendering on Vulkan will be implemented in Milestone 4.";
  LOG_WARNING << message;
  set_error(message); /* Scene::load_kernels() surfaces error_message() to the UI/CLI. */
  return false;
}
unique_ptr<DeviceQueue> VulkanDevice::gpu_queue_create() { return make_unique<VulkanQueue>(this); }
// NOTE: no build_bvh() override here. Device::build_bvh() in src/device/device.cpp
// already provides the generic CPU-side BVH2 build, which is exactly what we want
// until GPU-side BVH building exists. Adding an empty stub would silently disable it.

// =========================================================================
// HIGH-LEVEL MEMORY OVERRIDES
// =========================================================================
void VulkanDevice::mem_alloc(device_memory &mem) {
  if (!is_ready()) {
    LOG_ERROR << "Vulkan: device not initialized; ignoring mem_alloc.";
    return;
  }
  generic_alloc(mem);
}
void VulkanDevice::mem_copy_to(device_memory &mem) {
  if (!is_ready()) {
    LOG_ERROR << "Vulkan: device not initialized; ignoring mem_copy_to.";
    return;
  }
  if (mem.type == MEM_GLOBAL) {
    // A MEM_GLOBAL is a kernel-side global array: the buffer holds an array of
    // device_ptrs, and each element may point to another Vulkan buffer. Upload
    // the *current* pointer values every time they change (compare against the
    // last uploaded snapshot; do not rely on generic_copy_to's residency check,
    // which skips re-uploads when host data "looks" unchanged).
    const size_t size = mem.memory_size();
    if (!mem.device_pointer) {
      // Allocate through generic_alloc() rather than calling alloc_device()
      // directly. generic_alloc() does everything the raw call did PLUS the
      // bookkeeping this buffer MUST have to be freeable later:
      //   - it stores the pointer into mem.device_pointer (device.cpp:848),
      //     handling the Buffer-Device-Address round-trip (alloc_device()
      //     takes `void *&` because CUDA/HIP pass real pointers; we pass the
      //     BDA value through as an integer);
      //   - it sets mem.device_size and updates stats (device.cpp:849-850);
      //   - it INSERTS the allocation into device_mem_map (device.cpp:858),
      //     which GPUDevice::generic_free() relies on to erase the entry at
      //     free time. The old direct alloc_device() call skipped the insert,
      //     so the later free missed in find() and erase(end()) freed the
      //     map's header node -> heap corruption -> 0xC0000374 at exit
      //     (full root-cause report: lastsession.txt).
      // generic_alloc() reports OOM via set_error(); the LOG_ERROR below adds
      // the buffer name for context, and we skip the upload like before.
      if (generic_alloc(mem) == nullptr) {
        LOG_ERROR << "Vulkan: failed to allocate MEM_GLOBAL '" << mem.name_
                  << "' (" << size << " bytes).";
        return;
      }
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
  if (!is_ready()) {
    LOG_ERROR << "Vulkan: device not initialized; ignoring mem_copy_from.";
    return;
  }
  // Cycles' tiling contract (see CUDADevice::mem_copy_from): the tile is a
  // CONTIGUOUS block of `h` rows; each device row is (w + padding) elements
  // wide because the film buffer carries one extra boundary element per row.
  if (!mem.host_pointer) return;

  /* Tile geometry: derive the padded row stride from the buffer layout instead
   * of trusting the caller's width alone:
   *   total_elements = data_size * data_elements
   *   elements_per_row = w + padding
   * This also keeps working when elem != data_elements (partial-element reads). */
  const size_t total_elements = mem.data_size * mem.data_elements;
  const size_t elements_per_row = (w > 0) ? (total_elements / (w + 1)) : elem;
  const size_t padding = (elements_per_row > elem) ? (elements_per_row - elem) : 0;
  const size_t row_stride_elements = elem + padding;

  const size_t size_bytes = mem.memory_elements_size(row_stride_elements) * h;
  const size_t offset_bytes = mem.memory_elements_size(y * row_stride_elements);

  /* Clamp to the real buffer extent; never read past what was allocated. */
  const size_t max_bytes = mem.memory_size();
  if (offset_bytes >= max_bytes) return;
  const size_t clamped_bytes = std::min(size_bytes, max_bytes - offset_bytes);

  if (!mem.device_pointer) {
    // Device never got the data; zero the host tile so the image isn't garbage.
    memset(static_cast<uint8_t *>(mem.host_pointer) + offset_bytes, 0, clamped_bytes);
    return;
  }

  auto it = address_to_alloc_map.find((uint64_t)mem.device_pointer);
  if (it == address_to_alloc_map.end()) return;
  if (clamped_bytes == 0) return;

  /* Never copy beyond the actual allocation on the device. */
  const size_t device_room = (offset_bytes < it->second.size) ?
                             (it->second.size - offset_bytes) : 0;
  const size_t copy_bytes = std::min(clamped_bytes, device_room);
  if (copy_bytes == 0) return;

  VkBuffer staging;
  VkDeviceMemory staging_mem;
  VkBufferCreateInfo st_info{};
  st_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  st_info.size = copy_bytes;
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

  // Single linear copy of the tile region (size rounded to nonCoherentAtomSize).
  VkBufferCopy copy_region{};
  copy_region.size = ((copy_bytes + 3) / 4) * 4;
  copy_region.srcOffset = offset_bytes;
  copy_region.dstOffset = 0;
  vkCmdCopyBuffer(cmd, it->second.buffer, staging, 1, &copy_region);

  vkEndCommandBuffer(cmd);
  submit_and_wait(cmd);

  void *mapped;
  vkMapMemory(device, staging_mem, 0, copy_bytes, 0, &mapped);
  memcpy(static_cast<uint8_t *>(mem.host_pointer) + offset_bytes, mapped, copy_bytes);
  vkUnmapMemory(device, staging_mem);

  vkDestroyBuffer(device, staging, nullptr);
  vkFreeMemory(device, staging_mem, nullptr);
}


void VulkanDevice::mem_zero(device_memory &mem) {
  if (!is_ready()) {
    LOG_ERROR << "Vulkan: device not initialized; ignoring mem_zero.";
    return;
  }
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

    /* vkCmdFillBuffer requires a 4-byte-aligned fill size and fills uint32
     * units; buffer sizes from alloc_device are not guaranteed aligned. Copy
     * zeros from a small host-visible staging buffer instead (always valid). */
    VkBuffer zero_staging;
    VkDeviceMemory zero_mem;
    VkBufferCreateInfo zs_info{};
    zs_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    zs_info.size = it->second.size;
    zs_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    if (vkCreateBuffer(device, &zs_info, nullptr, &zero_staging) == VK_SUCCESS) {
      VkMemoryRequirements zs_reqs;
      vkGetBufferMemoryRequirements(device, zero_staging, &zs_reqs);
      VkMemoryAllocateInfo zs_alloc{};
      zs_alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
      zs_alloc.allocationSize = zs_reqs.size;
      zs_alloc.memoryTypeIndex = find_memory_type(
          zs_reqs.memoryTypeBits,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
      if (vkAllocateMemory(device, &zs_alloc, nullptr, &zero_mem) == VK_SUCCESS) {
        vkBindBufferMemory(device, zero_staging, zero_mem, 0);
        void *mapped_zero = nullptr;
        vkMapMemory(device, zero_mem, 0, zs_reqs.size, 0, &mapped_zero);
        memset(mapped_zero, 0, zs_reqs.size);
        vkUnmapMemory(device, zero_mem);

        VkBufferCopy full{};
        full.srcOffset = 0;
        full.dstOffset = 0;
        full.size = ((it->second.size + 3) / 4) * 4;
        vkCmdCopyBuffer(cmd, zero_staging, it->second.buffer, 1, &full);

        vkEndCommandBuffer(cmd);
        submit_and_wait(cmd);
        vkFreeMemory(device, zero_mem, nullptr);
      }
      vkDestroyBuffer(device, zero_staging, nullptr);
    }
    else {
      vkEndCommandBuffer(cmd);
      vkFreeCommandBuffers(device, cmd_pool, 1, &cmd);
    }
  }
  
  // Keep the CPU host pointer in sync with the GPU
  if (mem.host_pointer) {
    memset(mem.host_pointer, 0, mem.memory_size());
  }
}
void VulkanDevice::mem_free(device_memory &mem) {
  if (!is_ready()) {
    LOG_ERROR << "Vulkan: device not initialized; ignoring mem_free.";
    return;
  } generic_free(mem); }

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
  if (!is_ready()) {
    LOG_ERROR << "Vulkan: device not initialized; cannot allocate.";
    return false;
  }
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
  copy_region.size = ((size + 3) / 4) * 4; /* nonCoherentAtomSize multiple */
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