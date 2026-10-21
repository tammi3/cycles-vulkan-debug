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

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

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

  /* On Windows the loader only exposes pNext-chained feature structs to
   * vkGetPhysicalDeviceFeatures2 when this instance extension is enabled. */
  const char *instance_extensions[] = {"VK_KHR_get_physical_device_properties2"};

  VkInstanceCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.pApplicationInfo = &app_info;
  create_info.enabledExtensionCount = 1;
  create_info.ppEnabledExtensionNames = instance_extensions;

  VkInstance instance = VK_NULL_HANDLE;
  if (vkCreateInstance(&create_info, nullptr, &instance) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: failed to create instance during device enumeration.";
    return;
  }

  uint32_t device_count = 0;
  vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
  vector<VkPhysicalDevice> physical_devices(device_count);
  vkEnumeratePhysicalDevices(instance, &device_count, physical_devices.data());

  for (uint32_t i = 0; i < device_count; i++) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physical_devices[i], &props);

    /* NOTE: DeviceInfo has no `name`, `total_memory`, `free_memory` or
     * `has_automatic_memory_managed_by_kernels` members in this version of
     * Cycles (see src/device/device.h). Follow the CUDA backend convention:
     *   - description: human-readable name shown by --list-devices
     *   - id:          stable identifier persisted in user preferences
     * Memory reporting is not part of DeviceInfo here; the VulkanDevice
     * queries VkPhysicalDeviceMemoryProperties itself at allocation time. */
    DeviceInfo info;
    info.type = DEVICE_VULKAN;
    info.description = string(props.deviceName);
    info.num = i;
    info.id = string_printf("VULKAN_%s_%.8x_%.8x",
                            props.deviceName,
                            props.vendorID,
                            props.deviceID);

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

/* =========================================================================
 * M4b ABI probe (optional, env CYCLES_VULKAN_ABI_PROBE=1)
 * =========================================================================
 * Proves that the Slang kernel's view of KernelParamsVulkan
 * (src/kernel/device/vulkan/vulkan_data.slang) matches the host C++ view.
 * A one-thread dispatch of abi_probe.slang reads the params buffer through
 * the Slang struct layout AND through raw byte offsets, writes both into a
 * host-visible buffer, and this code compares the results value by value.
 *
 * It is called from VulkanQueue::enqueue (queue.cpp): by the first enqueue
 * the scene upload has finished, so the buffer holds what Cycles actually
 * uploaded. Without the env variable it does nothing at all, and the M4a-1
 * behaviour is unchanged.
 *
 * Note on "expected" values: the params buffer cannot be copied to the host
 * (it has no TRANSFER_SRC usage, and device_impl.h is frozen), so the probe
 * compares the shader's typed view against the same bytes read through raw
 * offsets - those raw bytes were placed there by const_copy_to using the
 * host's offsetof() values, which makes this an A/B test of the two layouts. */

/* Layout constants mirrored by src/kernel/device/vulkan/vulkan_data.slang.
 * If any of these fails to compile, Cycles' layout changed: update the
 * .slang mirror before anything else. */
static_assert(offsetof(KernelParamsVulkan, data) == 0, "params.data offset");
static_assert(sizeof(KernelData) == 2560, "KernelData size (Slang: opaque 640-word blob)");
static_assert(offsetof(KernelParamsVulkan, bvh_nodes) == 2560, "params.bvh_nodes offset");
static_assert(offsetof(KernelParamsVulkan, integrator_state) == 2984,
              "params.integrator_state offset");
static_assert(sizeof(IntegratorStateGPU) == 2128,
              "IntegratorStateGPU size (Slang mirror has an explicit tail pad)");
static_assert(sizeof(KernelParamsVulkan) == 5120,
              "KernelParamsVulkan size (Slang mirror has an explicit tail pad)");

namespace {

/* Number of uint32 words abi_probe.slang writes to the readback buffer. */
constexpr uint32_t ABI_PROBE_OUT_WORDS = 12;
/* AbiProbeParams = three 64-bit device pointers (abi_probe.slang). */
constexpr uint32_t ABI_PROBE_PUSH_BYTES = 24;
/* Sentinel written by the shader; proves the dispatch actually ran. */
constexpr uint32_t ABI_PROBE_SENTINEL = 0xAB10E57Fu;

bool probe_read_whole_file(const char *path, std::vector<char> &out)
{
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  f.seekg(0, std::ios::end);
  const size_t n = (size_t)f.tellg();
  f.seekg(0, std::ios::beg);
  out.resize(n);
  if (n && !f.read(out.data(), n)) return false;
  return true;
}

std::string probe_hex64(const uint64_t v)
{
  return string_printf("%016llx", (unsigned long long)v);
}

/* RAII cleanup for every Vulkan handle the probe creates, so any error path
 * (there are many early returns below) still destroys what was created and
 * the validation layer reports nothing. */
struct AbiProbeResources {
  VkDevice device = VK_NULL_HANDLE;
  VkShaderModule module = VK_NULL_HANDLE;
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;
  VkCommandPool pool = VK_NULL_HANDLE;
  VkBuffer out_buffer = VK_NULL_HANDLE;
  VkDeviceMemory out_memory = VK_NULL_HANDLE;

  ~AbiProbeResources()
  {
    if (device == VK_NULL_HANDLE) return;
    if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
    if (layout) vkDestroyPipelineLayout(device, layout, nullptr);
    if (module) vkDestroyShaderModule(device, module, nullptr);
    if (pool) vkDestroyCommandPool(device, pool, nullptr);
    if (out_buffer) vkDestroyBuffer(device, out_buffer, nullptr);
    if (out_memory) vkFreeMemory(device, out_memory, nullptr);
  }
};

} /* namespace */

/* One-shot ABI probe. Declared (as an extern) in queue.cpp, returns true if
 * the probe passed or was intentionally skipped, false on any failure. The
 * caller keeps its own behaviour either way. */
bool vulkan_abi_probe(VulkanDevice *dev)
{
  /* Run at most once per process, and only when the env hook is set - the
   * default path stays identical to M4a-1. */
  static bool ran = false;
  if (ran) return true;
  ran = true;
  if (getenv("CYCLES_VULKAN_ABI_PROBE") == nullptr) return true;

  LOG_INFO << "Vulkan: ABI probe starting (CYCLES_VULKAN_ABI_PROBE=1).";

  AbiProbeResources res;
  const VkDevice vk_dev = dev->vk_device();
  res.device = vk_dev;

  /* 1. Device address of the KernelParamsVulkan buffer (const_copy_to
   *    created it lazily during the scene upload). */
  const uint64_t params_addr = dev->kernel_params_address();
  if (params_addr == 0) {
    LOG_ERROR << "Vulkan: ABI probe: kernel params buffer does not exist; probe skipped.";
    return false;
  }

  /* 2. SPIR-V: compiled by hand with slangc (the build system has no Slang
   *    rule yet); located via env override or repo-relative defaults. */
  std::vector<char> spirv;
  const char *env_path = getenv("CYCLES_VULKAN_ABI_PROBE_SPV");
  const char *candidates[3] = {env_path,
                               "abi_probe.spv",
                               "src/kernel/device/vulkan/abi_probe.spv"};
  const char *found = nullptr;
  for (const char *c : candidates) {
    if (c && probe_read_whole_file(c, spirv)) {
      found = c;
      break;
    }
  }
  if (!found) {
    LOG_ERROR << "Vulkan: ABI probe: cannot find abi_probe.spv "
                 "(compile it with slangc or set CYCLES_VULKAN_ABI_PROBE_SPV).";
    return false;
  }
  if (spirv.size() % 4 != 0) {
    LOG_ERROR << "Vulkan: ABI probe: '" << found << "' is not a valid SPIR-V module.";
    return false;
  }

  /* 3. Push-constant limit. The Vulkan spec guarantees at least 128 bytes;
   *    we use ABI_PROBE_PUSH_BYTES of them. */
  VkPhysicalDeviceProperties props{};
  vkGetPhysicalDeviceProperties(dev->vk_physical_device(), &props);
  LOG_INFO << "Vulkan: ABI probe: spirv='" << found << "' (" << spirv.size()
           << " bytes), maxPushConstantsSize=" << props.limits.maxPushConstantsSize;
  if (props.limits.maxPushConstantsSize < ABI_PROBE_PUSH_BYTES) {
    LOG_ERROR << "Vulkan: ABI probe: maxPushConstantsSize too small.";
    return false;
  }

  /* 4. Shader module. (The entry point name is handled at step 6.) */
  VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smi.codeSize = spirv.size();
  smi.pCode = reinterpret_cast<const uint32_t *>(spirv.data());
  if (vkCreateShaderModule(vk_dev, &smi, nullptr, &res.module) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: ABI probe: vkCreateShaderModule failed.";
    return false;
  }

  /* 5. Pipeline layout: NO descriptor sets (everything travels through push
   *    constants + buffer device addresses), one push-constant range. */
  VkPushConstantRange pcr{};
  pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pcr.offset = 0;
  pcr.size = ABI_PROBE_PUSH_BYTES;
  VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pli.pushConstantRangeCount = 1;
  pli.pPushConstantRanges = &pcr;
  if (vkCreatePipelineLayout(vk_dev, &pli, nullptr, &res.layout) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: ABI probe: pipeline layout creation failed.";
    return false;
  }

  /* 6. Compute pipeline. NOTE: slangc emits the entry point as SPIR-V "main"
   *    regardless of the Slang function name (observed via the validation
   *    layer, VUID-VkPipelineShaderStageCreateInfo-pName-00707: "the only
   *    entry point found was main"; the AMD driver crashed outright on the
   *    wrong name). slangc's -fvk-use-entrypoint-name would keep the Slang
   *    name instead; we use "main" and need no extra flag. */
  VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  cpi.stage.module = res.module;
  cpi.stage.pName = "main";
  cpi.layout = res.layout;
  if (vkCreateComputePipelines(vk_dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &res.pipeline) !=
      VK_SUCCESS) {
    LOG_ERROR << "Vulkan: ABI probe: compute pipeline creation failed "
                 "(bad entry point name?).";
    return false;
  }

  /* 7. Readback buffer. Host-visible + coherent so the host can read it
   *    right after the fence signals; STORAGE | SHADER_DEVICE_ADDRESS
   *    because the shader writes through a raw device pointer (same usage
   *    pair the M2c spike used, spikes/common/vk_util.h:270,278). */
  VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bci.size = ABI_PROBE_OUT_WORDS * sizeof(uint32_t);
  bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  if (vkCreateBuffer(vk_dev, &bci, nullptr, &res.out_buffer) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: ABI probe: readback buffer creation failed.";
    return false;
  }
  VkMemoryRequirements reqs;
  vkGetBufferMemoryRequirements(vk_dev, res.out_buffer, &reqs);
  VkMemoryAllocateFlagsInfo flags_info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
  flags_info.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.pNext = &flags_info;
  ai.allocationSize = reqs.size;
  ai.memoryTypeIndex = dev->find_memory_type(
      reqs.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  if (vkAllocateMemory(vk_dev, &ai, nullptr, &res.out_memory) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: ABI probe: readback memory allocation failed.";
    return false;
  }
  if (vkBindBufferMemory(vk_dev, res.out_buffer, res.out_memory, 0) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: ABI probe: readback buffer bind failed.";
    return false;
  }
  VkBufferDeviceAddressInfo bai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
  bai.buffer = res.out_buffer;
  const uint64_t out_addr = vkGetBufferDeviceAddress(vk_dev, &bai);

  /* 8. Record and submit. Our own command pool on the same family and queue
   *    index the device itself uses (device.cpp:124,191: find_queue_family
   *    + vkGetDeviceQueue(..., 0, ...)). We do NOT use submit_and_wait()
   *    because it frees the command buffer from the device's own private
   *    command pool (device.cpp:54). */
  const uint32_t family = dev->find_queue_family(dev->vk_physical_device());
  VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  cpci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
  cpci.queueFamilyIndex = family;
  if (vkCreateCommandPool(vk_dev, &cpci, nullptr, &res.pool) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: ABI probe: command pool creation failed.";
    return false;
  }
  VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbai.commandPool = res.pool;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  if (vkAllocateCommandBuffers(vk_dev, &cbai, &cmd) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: ABI probe: command buffer allocation failed.";
    return false;
  }
  VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  if (vkBeginCommandBuffer(cmd, &begin_info) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: ABI probe: vkBeginCommandBuffer failed.";
    return false;
  }
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, res.pipeline);
  const uint64_t push[3] = {params_addr, params_addr, out_addr};
  vkCmdPushConstants(cmd, res.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);
  vkCmdDispatch(cmd, 1, 1, 1); /* one workgroup; the shader runs one thread */
  if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: ABI probe: vkEndCommandBuffer failed.";
    return false;
  }
  VkQueue queue = VK_NULL_HANDLE;
  vkGetDeviceQueue(vk_dev, family, 0, &queue);
  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkFence fence = VK_NULL_HANDLE;
  if (vkCreateFence(vk_dev, &fci, nullptr, &fence) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: ABI probe: fence creation failed.";
    return false;
  }
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  bool submitted = false;
  if (vkQueueSubmit(queue, 1, &si, fence) == VK_SUCCESS) {
    submitted = vkWaitForFences(vk_dev, 1, &fence, VK_TRUE, 10000000000ull) == VK_SUCCESS;
  }
  vkDestroyFence(vk_dev, fence, nullptr);
  if (!submitted) {
    LOG_ERROR << "Vulkan: ABI probe: queue submit/wait failed.";
    return false;
  }

  /* 9. Read back and compare. */
  void *mapped = nullptr;
  if (vkMapMemory(vk_dev, res.out_memory, 0, bci.size, 0, &mapped) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: ABI probe: mapping readback buffer failed.";
    return false;
  }
  uint32_t out[ABI_PROBE_OUT_WORDS];
  memcpy(out, mapped, sizeof(out));
  vkUnmapMemory(vk_dev, res.out_memory);

  const uint64_t gpu_bvh = ((uint64_t)out[4] << 32) | out[3];
  const uint64_t raw_bvh = ((uint64_t)out[6] << 32) | out[5];
  const uint64_t gpu_ist = ((uint64_t)out[8] << 32) | out[7];
  const uint64_t raw_ist = ((uint64_t)out[10] << 32) | out[9];

  int mismatches = 0;
  auto check = [&mismatches](const char *label, const uint64_t gpu, const uint64_t host) {
    LOG_INFO << "Vulkan: ABI probe: " << label << ": GPU=0x" << probe_hex64(gpu) << " HOST=0x"
             << probe_hex64(host) << (gpu == host ? " MATCH" : " MISMATCH");
    if (gpu != host) mismatches++;
  };

  /* Slang sizeof vs host static_asserted sizeof. */
  check("sizeof(KernelParamsVulkan)", out[0], sizeof(KernelParamsVulkan));
  check("sizeof(IntegratorStateGPU)", out[1], sizeof(IntegratorStateGPU));
  /* bvh_nodes: typed view vs raw words at the host-asserted offset (those
   * raw words are what const_copy_to uploaded for this slot). */
  check("bvh_nodes typed vs raw@2560", gpu_bvh, raw_bvh);
  /* integrator_state.path.render_pixel_index: same idea at offset 2984. */
  check("integrator_state[0] typed vs raw@2984", gpu_ist, raw_ist);
  /* First KernelData word: value as the shader sees it. No host-side
   * reference exists (host_params_ is private and the params buffer has no
   * TRANSFER_SRC usage to copy it back), so it is logged, not compared. */
  LOG_INFO << "Vulkan: ABI probe: KernelData.data[0]=0x" << probe_hex64(out[2])
           << " (value as seen by shader, no host reference)";
  check("dispatch sentinel", out[11], ABI_PROBE_SENTINEL);

  if (mismatches == 0) {
    LOG_INFO << "Vulkan: ABI probe: ALL VALUES MATCH.";
  }
  else {
    LOG_ERROR << "Vulkan: ABI probe: " << mismatches
              << " MISMATCH value(s); the Slang ABI does not match KernelParamsVulkan.";
  }
  return mismatches == 0;
}

CCL_NAMESPACE_END

#endif /* WITH_VULKAN */
