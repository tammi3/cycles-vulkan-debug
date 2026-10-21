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
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
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

/* =========================================================================
 * M4c traverse probe (optional, env CYCLES_VULKAN_TRAVERSE_PROBE=1)
 * =========================================================================
 * Traces 64x64 pinhole rays on the GPU through the ported closest-hit
 * traversal (src/kernel/device/vulkan/traverse.slang) and compares every
 * pixel against a host-side brute-force reference built from a
 * cycles_bvh_dump sidecar run of the same scene. Runs three fault
 * injections (negative tests) that MUST be detected, per the M4c gate.
 *
 * Data flow, and why it is shaped like this (M4c decisions Q1-Q3):
 *   - host_params_ and address_to_alloc_map are PRIVATE (device_impl.h:65-81)
 *     and frozen, so the host probe can never read the scene arrays'
 *     device addresses or contents directly. The SHADER can (params buffer
 *     BDA via the public kernel_params_address()), so the prepare kernel
 *     copies the nine pointer slots + the blob root into a probe header
 *     buffer the host CAN see;
 *   - the reference data comes from a sidecar run of cycles_bvh_dump
 *     (src/app/cycles_bvh_dump.cpp) - it exercises the upload path too.
 *     Because the dump is a SEPARATE Cycles run, the prepare kernel
 *     checksums the live bvh_nodes + tri_verts: on mismatch the probe
 *     reports INCONCLUSIVE ("dump differs from live scene"), not FAIL;
 *   - counts for the shader's bounds checks (AGENTS.md section 4) come
 *     from the dump meta file (Q2) - probe only; production kernels will
 *     not carry counts, exactly like HIP;
 *   - root is read from the KernelData blob at the word index the host
 *     computes with offsetof(KernelData, bvh) (Q3): this mirrors how
 *     production kernels read kernel_data.bvh.root (traversal.h:42) and
 *     additionally tests the blob upload against the dump's kernel_bvh.
 *
 * Called from VulkanQueue::enqueue (queue.cpp) right after the M4b ABI
 * probe; a no-op unless the env variable is set. */

/* Layout locks shared with the shaders and the frozen Cycles headers. */
static_assert(offsetof(KernelObject, position_offset) == 168,
              "KernelObject.position_offset must be word 42 (traverse.slang)");
static_assert(sizeof(KernelObject) == 256,
              "KernelObject must be 64 words (traverse.slang OBJECT_WORDS)");
static_assert(sizeof(Intersection) == 24,
              "Intersection must be 24 bytes (types.h:614-619)");
/* KernelData.bvh is the KernelBVH struct and root is its FIRST member
 * (src/kernel/data_template.h:51-52), so the blob word index is simply
 * the member offset divided by 4. */
static_assert(offsetof(KernelData, bvh) % 4 == 0,
              "KernelData.bvh must be word aligned");

namespace {

/* ---- probe constants --------------------------------------------------- */
constexpr uint32_t TP_WIDTH = 64;   /* gate: 64x64 rays */
constexpr uint32_t TP_HEIGHT = 64;
constexpr uint32_t TP_PIXELS = TP_WIDTH * TP_HEIGHT;
constexpr uint32_t TP_CANARY_RAYS = 16;    /* m2c pattern, m2c_main.cpp:97-99 */
constexpr uint32_t TP_CANARY_WORD = 0xCAFEBABEu;
constexpr uint32_t TP_UNWRITTEN_WORD = 0xDEADBEEFu;
constexpr uint32_t TP_PREPARE_SENTINEL = 0xB10C5EEDu; /* traverse_probe.slang */
constexpr uint32_t TP_PREPARE_OUT_WORDS = 645; /* 5 header words + whole blob */
constexpr uint32_t TP_PREPARE_PUSH_BYTES = 56; /* PreparePush (6 ptrs + word) */
constexpr uint32_t TP_PROBE_PUSH_BYTES = 24;   /* ProbePush (3 ptrs) */
constexpr uint32_t TP_HEADER_BYTES = 192;      /* sizeof(TpHeader)=168, rounded */
constexpr uint32_t TP_OBJECT_WORDS = 64;       /* traverse.slang OBJECT_WORDS */
constexpr uint32_t TP_POSITION_OFFSET_WORD = 42;
constexpr float TP_TMIN = 1e-4f;    /* raytri_cpu.h:21 */
constexpr float TP_NOHIT_T = 1e30f; /* raytri_cpu.h:22 */
constexpr double TP_EDGE_TOL = 1e-5;    /* m2c_main.cpp:108 */
constexpr double TP_PLANE_HALF = 0.8;   /* Q4: image plane spans [-0.8, 0.8] */

/* Host mirror of traverse.slang Intersection / types.h:614-619. */
struct TpIntersection {
  float t, u, v;
  int32_t prim, object, type;
};
static_assert(sizeof(TpIntersection) == 24, "TpIntersection layout");

/* Host mirror of traverse.slang TraverseHeader: scalar members only, so
 * C++ and Slang cannot disagree on padding. Every offset is asserted. */
struct TpHeader {
  uint64_t bvh_nodes;        //   0
  uint64_t bvh_leaf_nodes;   //   8
  uint64_t prim_index;       //  16
  uint64_t prim_visibility;  //  24
  uint64_t prim_object;      //  32
  uint64_t tri_vindex;       //  40
  uint64_t tri_verts;        //  48
  uint64_t objects;          //  56
  uint64_t object_node;      //  64
  int32_t root;              //  72
  uint32_t node_count;       //  76
  uint32_t leaf_count;       //  80
  uint32_t prim_count;       //  84
  uint32_t tri_count;        //  88
  uint32_t vert_count;       //  92
  uint32_t object_count;     //  96
  uint32_t object_node_count;// 100
  uint32_t width;            // 104
  uint32_t height;           // 108
  uint32_t visibility;       // 112
  uint32_t test_mode;        // 116
  float cam[12];             // 120..167  o(3) l(3) u(3) v(3)
};
static_assert(offsetof(TpHeader, root) == 72, "TpHeader layout");
static_assert(offsetof(TpHeader, node_count) == 76, "TpHeader layout");
static_assert(offsetof(TpHeader, object_node_count) == 100, "TpHeader layout");
static_assert(offsetof(TpHeader, width) == 104, "TpHeader layout");
static_assert(offsetof(TpHeader, test_mode) == 116, "TpHeader layout");
static_assert(offsetof(TpHeader, cam) == 120, "TpHeader layout");
static_assert(sizeof(TpHeader) == 168, "TpHeader layout");

/* Push-constant host mirrors. */
struct TpPreparePush {
  uint64_t hdr, hdr_raw, hdr_root, params, out, params_raw;
  uint32_t blob_root_word, pad;
};
static_assert(sizeof(TpPreparePush) == TP_PREPARE_PUSH_BYTES, "push layout");
struct TpProbePush {
  uint64_t hdr, out, status;
};
static_assert(sizeof(TpProbePush) == TP_PROBE_PUSH_BYTES, "push layout");

/* The fixed gate camera (Q4): origin (0,-6,3), looking at the origin,
 * image plane spanning x,y in [-0.8, 0.8] at depth 1 (~77.3 degrees FOV).
 * The exact numbers are printed at run time. */
struct TpCam {
  float o[3], l[3], u[3], v[3];
};

/* ---- dump loading (cycles_bvh_dump format, cycles_bvh_dump.cpp:58-92) -- */
struct TpDump {
  std::vector<uint8_t> bvh_nodes, bvh_leaf_nodes, prim_index;
  std::vector<uint8_t> prim_visibility, prim_object, object_node;
  std::vector<uint8_t> objects, tri_verts, tri_vindex, kernel_bvh;
  uint32_t node_count = 0, leaf_count = 0, prim_count = 0;
  uint32_t tri_count = 0, vert_count = 0, object_count = 0;
  uint32_t object_node_count = 0;
  int32_t root = 0; /* dump's kernel_bvh.root, cross-checked vs the blob */
};

static bool tp_load_file(const std::string &path, std::vector<uint8_t> &out)
{
  std::ifstream f(path, std::ios::binary);
  if (!f)
    return false;
  f.seekg(0, std::ios::end);
  const std::streamoff n = f.tellg();
  f.seekg(0, std::ios::beg);
  out.resize((size_t)(n < 0 ? 0 : n));
  return out.empty() || (bool)f.read((char *)out.data(), (std::streamsize)out.size());
}

/* meta.txt lines are "<name> <total_bytes> <element_bytes> <element_count>"
 * (cycles_bvh_dump.cpp:66). */
static bool tp_load_dump(const std::string &dir, TpDump &d, std::string &err)
{
  std::ifstream meta(dir + "/meta.txt");
  if (!meta) {
    err = "cannot open " + dir + "/meta.txt";
    return false;
  }
  std::string line;
  while (std::getline(meta, line)) {
    char name[128];
    unsigned long long bytes = 0, elem = 0, count = 0;
    if (sscanf(line.c_str(), "%127s %llu %llu %llu", name, &bytes, &elem, &count) != 4)
      continue;
    std::vector<uint8_t> *slot = nullptr;
    size_t expect_elem = 0; /* 0 = any */
    if (!strcmp(name, "bvh_nodes"))
      slot = &d.bvh_nodes, expect_elem = 16; /* float4 */
    else if (!strcmp(name, "bvh_leaf_nodes"))
      slot = &d.bvh_leaf_nodes, expect_elem = 16;
    else if (!strcmp(name, "prim_index"))
      slot = &d.prim_index, expect_elem = 4;
    else if (!strcmp(name, "prim_visibility"))
      slot = &d.prim_visibility, expect_elem = 4;
    else if (!strcmp(name, "prim_object"))
      slot = &d.prim_object, expect_elem = 4;
    else if (!strcmp(name, "object_node"))
      slot = &d.object_node, expect_elem = 4;
    else if (!strcmp(name, "objects"))
      slot = &d.objects, expect_elem = 256; /* KernelObject, locked above */
    else if (!strcmp(name, "tri_verts"))
      slot = &d.tri_verts, expect_elem = 12; /* packed_float3 */
    else if (!strcmp(name, "tri_vindex"))
      slot = &d.tri_vindex, expect_elem = 12; /* packed_uint3 */
    else if (!strcmp(name, "kernel_bvh"))
      slot = &d.kernel_bvh, expect_elem = 0;
    else
      continue;
    if (expect_elem && elem != expect_elem) {
      err = std::string(name) + ": element size " + std::to_string(elem) +
            " does not match the layout the shader expects";
      return false;
    }
    if (!tp_load_file(dir + "/" + name + ".bin", *slot)) {
      err = "cannot read " + dir + "/" + name + ".bin";
      return false;
    }
    if (slot->size() != (size_t)bytes) {
      err = std::string(name) + ": file size does not match meta.txt";
      return false;
    }
  }
  d.node_count = (uint32_t)(d.bvh_nodes.size() / 16);
  d.leaf_count = (uint32_t)(d.bvh_leaf_nodes.size() / 16);
  d.prim_count = (uint32_t)(d.prim_index.size() / 4);
  d.tri_count = (uint32_t)(d.tri_vindex.size() / 12);
  d.vert_count = (uint32_t)(d.tri_verts.size() / 12);
  d.object_count = (uint32_t)(d.objects.size() / 256);
  d.object_node_count = (uint32_t)(d.object_node.size() / 4);
  if (d.node_count == 0 || d.leaf_count == 0 || d.object_count == 0 ||
      d.kernel_bvh.size() < 4) {
    err = "dump is missing required arrays";
    return false;
  }
  /* kernel_bvh: KernelBVH's first member is root (data_template.h:51-52). */
  memcpy(&d.root, d.kernel_bvh.data(), sizeof(d.root));
  return true;
}

/* ---- checksum (must match fnv_word in traverse_probe.slang) ----------- */
static uint32_t tp_fnv1a(const uint8_t *p, size_t n)
{
  uint32_t h = 2166136261u;
  for (size_t i = 0; i < n; i++) {
    h ^= p[i];
    h *= 16777619u;
  }
  return h;
}

/* ---- reference triangles (m2c_main.cpp:280-308) -----------------------
 * For every object, transform its triangles to WORLD space with tfm
 * (words 0-11; gate: tfm, NOT itfm) and label them (object, prim). */
struct TpWorldTri {
  float v[9];
  uint32_t obj, prim;
};

static bool tp_build_world_tris(
    const TpDump &d, std::vector<TpWorldTri> &out, std::string &err)
{
  out.clear();
  out.reserve((size_t)d.object_count * d.tri_count);
  for (uint32_t obj = 0; obj < d.object_count; obj++) {
    const uint8_t *od = &d.objects[(size_t)obj * TP_OBJECT_WORDS * 4];
    float tfm[12];
    for (int i = 0; i < 12; i++)
      memcpy(&tfm[i], od + (size_t)i * 4, 4);
    int32_t off = 0;
    memcpy(&off, od + (size_t)TP_POSITION_OFFSET_WORD * 4, 4);
    for (uint32_t prim = 0; prim < d.tri_count; prim++) {
      TpWorldTri wt;
      wt.obj = obj;
      wt.prim = prim;
      for (int k = 0; k < 3; k++) {
        uint32_t vi = 0;
        memcpy(&vi, &d.tri_vindex[(size_t)prim * 12 + (size_t)k * 4], 4);
        const uint32_t v_idx = vi + (uint32_t)off;
        if (v_idx >= d.vert_count) {
          err = "triangle " + std::to_string(prim) + " uses vertex " +
                std::to_string(v_idx) + " but there are only " +
                std::to_string(d.vert_count) + " vertices";
          return false;
        }
        float vx, vy, vz;
        memcpy(&vx, &d.tri_verts[(size_t)v_idx * 12 + 0], 4);
        memcpy(&vy, &d.tri_verts[(size_t)v_idx * 12 + 4], 4);
        memcpy(&vz, &d.tri_verts[(size_t)v_idx * 12 + 8], 4);
        /* transform_point with Transform = {float4 x,y,z} (transform.h:52):
         * rows are tfm[0..3], tfm[4..7], tfm[8..11], .w is translation. */
        wt.v[k * 3 + 0] = vx * tfm[0] + vy * tfm[1] + vz * tfm[2] + tfm[3];
        wt.v[k * 3 + 1] = vx * tfm[4] + vy * tfm[5] + vz * tfm[6] + tfm[7];
        wt.v[k * 3 + 2] = vx * tfm[8] + vy * tfm[9] + vz * tfm[10] + tfm[11];
      }
      out.push_back(wt);
    }
  }
  return true;
}

/* ---- brute-force trace (raytri_cpu.h:95-124, Moller-Trumbore) --------- */
struct TpHit {
  float t;
  int32_t prim, object;
};

static TpHit tp_trace_cpu(const std::vector<TpWorldTri> &tris,
                          const float o[3], const float d[3])
{
  TpHit best{TP_NOHIT_T, -1, -1};
  for (size_t i = 0; i < tris.size(); i++) {
    const float *t = tris[i].v;
    float e1[3], e2[3], p[3], tv[3], q[3];
    for (int a = 0; a < 3; a++) {
      e1[a] = t[3 + a] - t[0 + a];
      e2[a] = t[6 + a] - t[0 + a];
      tv[a] = o[a] - t[0 + a];
    }
    /* p = d x e2 (raytri_cpu.h:106). */
    p[0] = d[1] * e2[2] - d[2] * e2[1];
    p[1] = d[2] * e2[0] - d[0] * e2[2];
    p[2] = d[0] * e2[1] - d[1] * e2[0];
    const float det = e1[0] * p[0] + e1[1] * p[1] + e1[2] * p[2];
    if (std::fabs(det) < 1e-8f)
      continue;
    const float inv = 1.0f / det;
    const float u = (tv[0] * p[0] + tv[1] * p[1] + tv[2] * p[2]) * inv;
    if (u < 0.0f || u > 1.0f)
      continue;
    q[0] = tv[1] * e1[2] - tv[2] * e1[1];
    q[1] = tv[2] * e1[0] - tv[0] * e1[2];
    q[2] = tv[0] * e1[1] - tv[1] * e1[0];
    const float v = (d[0] * q[0] + d[1] * q[1] + d[2] * q[2]) * inv;
    if (v < 0.0f || u + v > 1.0f)
      continue;
    const float th = (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]) * inv;
    if (th > TP_TMIN && th < best.t) { /* strict '<': lower index wins ties */
      best.t = th;
      /* Report Cycles' numbering, not our enumeration index: the mesh-
       * local triangle index (tri_vindex slot) and the object index, the
       * same pair the traversal reports (traverse.slang reads
       * prim_index[prim_addr] and current_object). */
      best.prim = (int32_t)tris[i].prim;
      best.object = (int32_t)tris[i].obj;
    }
  }
  return best;
}

/* ---- camera (Q4) ------------------------------------------------------- */
static void tp_build_camera(TpCam &cam)
{
  const double o[3] = {0.0, -6.0, 3.0};
  const double target[3] = {0.0, 0.0, 0.0};
  const double up[3] = {0.0, 1.0, 0.0};
  double f[3] = {target[0] - o[0], target[1] - o[1], target[2] - o[2]};
  const double flen = std::sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]);
  for (int a = 0; a < 3; a++)
    f[a] /= flen;
  double r[3] = {f[1] * up[2] - f[2] * up[1],
                 f[2] * up[0] - f[0] * up[2],
                 f[0] * up[1] - f[1] * up[0]};
  const double rlen = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
  for (int a = 0; a < 3; a++)
    r[a] /= rlen;
  const double ux[3] = {r[1] * f[2] - r[2] * f[1],
                        r[2] * f[0] - r[0] * f[2],
                        r[0] * f[1] - r[1] * f[0]};
  for (int a = 0; a < 3; a++) {
    cam.o[a] = (float)o[a];
    /* Image plane at depth 1 spanning [-TP_PLANE_HALF, +TP_PLANE_HALF]:
     * lower-left corner l, full-width u, full-height v (the M2c camera
     * convention, raytri_cpu.h:40-47). raytri_cpu.h writes l = o - w -
     * ... where w points FROM the object TOWARD the camera (raytri_cpu.h:61,
     * :77); our f is the forward direction (target - o), i.e. f = -w, so
     * the same formula is l = o + f - ... . (Getting this sign wrong puts
     * the plane one unit BEHIND the camera: every ray then points away
     * from the scene and both sides report all-miss.) */
    cam.l[a] = (float)(o[a] + f[a] - TP_PLANE_HALF * r[a] - TP_PLANE_HALF * ux[a]);
    cam.u[a] = (float)(2.0 * TP_PLANE_HALF * r[a]);
    cam.v[a] = (float)(2.0 * TP_PLANE_HALF * ux[a]);
  }
}

/* Direction of the ray through pixel (x, y): identical formula to the
 * shader (traverse_probe.slang, probe kernel). Double precision here; the
 * ~1 ULP difference vs the GPU's float math is inside the t tolerance. */
static void tp_pixel_dir(const TpCam &cam, uint32_t x, uint32_t y, float d[3])
{
  const double s = ((double)x + 0.5) / (double)TP_WIDTH;
  const double t = 1.0 - ((double)y + 0.5) / (double)TP_HEIGHT;
  double dir[3];
  for (int a = 0; a < 3; a++)
    dir[a] = (double)cam.l[a] + s * (double)cam.u[a] + t * (double)cam.v[a] -
             (double)cam.o[a];
  const double len = std::sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
  for (int a = 0; a < 3; a++)
    d[a] = (float)(dir[a] / len);
}

/* ---- edge-grazing probe (m2c_main.cpp:115-145, double precision) ------ */
struct TpEdgeProbe {
  double margin = 1e30;
  int tri = -1;
};

static TpEdgeProbe tp_probe_edge(const std::vector<TpWorldTri> &tris,
                                 const float *o, const float *d)
{
  TpEdgeProbe best;
  for (size_t i = 0; i < tris.size(); i++) {
    const float *t = tris[i].v;
    double e1[3], e2[3], tv[3], dd[3];
    for (int a = 0; a < 3; a++) {
      e1[a] = (double)t[3 + a] - t[a];
      e2[a] = (double)t[6 + a] - t[a];
      tv[a] = (double)o[a] - t[a];
      dd[a] = d[a];
    }
    auto cross = [](const double *x, const double *y, double *r) {
      r[0] = x[1] * y[2] - x[2] * y[1];
      r[1] = x[2] * y[0] - x[0] * y[2];
      r[2] = x[0] * y[1] - x[1] * y[0];
    };
    auto dot = [](const double *x, const double *y) {
      return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
    };
    double p[3], q[3];
    cross(dd, e2, p);
    const double det = dot(e1, p);
    if (std::fabs(det) < 1e-300)
      continue;
    const double u = dot(tv, p) / det;
    cross(tv, e1, q);
    const double v = dot(dd, q) / det;
    const double tt = dot(e2, q) / det;
    if (tt <= 0.0)
      continue;
    const double w = 1.0 - u - v;
    const double mn = std::min(u, std::min(v, w));
    if (mn < -TP_EDGE_TOL)
      continue;
    if (mn < best.margin) {
      best.margin = std::fabs(mn);
      best.tri = (int)i;
    }
  }
  return best;
}

/* ---- comparison (m2c_main.cpp:158-226) -------------------------------- */
struct TpCmp {
  size_t same = 0, miss = 0, ties = 0, grazing = 0, bad = 0, errors = 0;
  std::vector<size_t> examples;
};

static TpCmp tp_compare(const std::vector<TpHit> &ref,
                        const std::vector<TpIntersection> &gpu,
                        const std::vector<int32_t> &status,
                        const std::vector<TpWorldTri> &tris, const TpCam &cam,
                        bool details)
{
  TpCmp c;
  for (size_t i = 0; i < ref.size(); i++) {
    const TpHit &r = ref[i];
    const TpIntersection &g = gpu[i];
    if (status[i] != 0) { /* any shader error id is a failure (Q7) */
      c.errors++;
      if (c.examples.size() < 5)
        c.examples.push_back(i);
      continue;
    }
    bool disagree = false;
    const bool r_miss = (r.prim < 0 && r.object < 0);
    const bool g_miss = (g.prim < 0 && g.object < 0);
    if (r_miss && g_miss) {
      c.miss++;
    }
    else if (!r_miss && !g_miss && r.prim == g.prim && r.object == g.object) {
      /* Same (object, prim): t must agree within 1e-4 relative. */
      if (std::fabs(g.t - r.t) <= 1e-4f * std::fmax(1.0f, r.t))
        c.same++;
      else
        disagree = true;
    }
    else if (!r_miss && !g_miss &&
             std::fabs(g.t - r.t) <= 1e-3f * std::fmax(1.0f, r.t)) {
      c.ties++; /* different prim, same distance: an edge tie, acceptable */
    }
    else {
      disagree = true;
    }
    if (!disagree)
      continue;
    float d[3];
    tp_pixel_dir(cam, (uint32_t)(i % TP_WIDTH), (uint32_t)(i / TP_WIDTH), d);
    const TpEdgeProbe e = tp_probe_edge(tris, cam.o, d);
    if (e.margin <= TP_EDGE_TOL)
      c.grazing++;
    else
      c.bad++;
    /* Diagnostic detail for grazing/bad pixels: is the GPU's answer a
     * plausible alternative hit (adjacent face through an edge, where the
     * watertight Pluecker test and Moller-Trumbore legitimately pick
     * different faces) or an unrelated point in space? */
    if (details && c.grazing + c.bad <= 20) {
      LOG_INFO << string_printf(
          "Vulkan: traverse probe:   %s pixel (%u,%u): ref t=%.6g (obj %d, prim %d) "
          "gpu t=%.6g (obj %d, prim %d) edge margin %.3g",
          e.margin <= TP_EDGE_TOL ? "grazing" : "BAD",
          (unsigned)(i % TP_WIDTH), (unsigned)(i / TP_WIDTH), (double)r.t,
          r.object, r.prim, (double)g.t, g.object, g.prim, e.margin);
    }
    if (c.examples.size() < 5)
      c.examples.push_back(i);
  }
  return c;
}

/* Gate tolerance (m2c_main.cpp:223-226): no hard mismatches, no shader
 * errors, and at most max(8, n/50000) edge-grazing disagreements. */
static bool tp_cmp_ok(const TpCmp &c, size_t n)
{
  const size_t grazing_limit = std::max<size_t>(8, n / 50000);
  return c.bad == 0 && c.errors == 0 && c.grazing <= grazing_limit;
}

static void tp_print_cmp(const char *title, const TpCmp &c, size_t n)
{
  LOG_INFO << "Vulkan: traverse probe: " << title << " (" << n << " pixels)";
  LOG_INFO << "Vulkan: traverse probe:   both missed:                   " << c.miss;
  LOG_INFO << "Vulkan: traverse probe:   same (object,prim), same t:    " << c.same;
  LOG_INFO << "Vulkan: traverse probe:   edge ties (acceptable):        " << c.ties;
  LOG_INFO << "Vulkan: traverse probe:   edge-grazing disagreements:    " << c.grazing;
  LOG_INFO << "Vulkan: traverse probe:   MISMATCHES (not near an edge): " << c.bad;
  LOG_INFO << "Vulkan: traverse probe:   shader error codes:            " << c.errors;
  for (size_t i : c.examples) {
    LOG_INFO << "Vulkan: traverse probe:   example pixel (" << (i % TP_WIDTH)
             << "," << (i / TP_WIDTH) << ")";
  }
}

/* ---- Vulkan resources (RAII, pattern: AbiProbeResources above) -------- */
struct TpResources {
  VkDevice device = VK_NULL_HANDLE;
  VkShaderModule mod_prepare = VK_NULL_HANDLE;
  VkShaderModule mod_probe = VK_NULL_HANDLE;
  VkPipelineLayout layout_prepare = VK_NULL_HANDLE;
  VkPipelineLayout layout_probe = VK_NULL_HANDLE;
  VkPipeline pipe_prepare = VK_NULL_HANDLE;
  VkPipeline pipe_probe = VK_NULL_HANDLE;
  VkCommandPool pool = VK_NULL_HANDLE;
  VkBuffer hdr_buf = VK_NULL_HANDLE, out_buf = VK_NULL_HANDLE;
  VkBuffer status_buf = VK_NULL_HANDLE, cksum_buf = VK_NULL_HANDLE;
  VkDeviceMemory hdr_mem = VK_NULL_HANDLE, out_mem = VK_NULL_HANDLE;
  VkDeviceMemory status_mem = VK_NULL_HANDLE, cksum_mem = VK_NULL_HANDLE;
  void *hdr_map = nullptr, *out_map = nullptr;
  void *status_map = nullptr, *cksum_map = nullptr;
  uint64_t hdr_addr = 0, out_addr = 0, status_addr = 0, cksum_addr = 0;

  ~TpResources()
  {
    if (device == VK_NULL_HANDLE)
      return;
    if (hdr_map) vkUnmapMemory(device, hdr_mem);
    if (out_map) vkUnmapMemory(device, out_mem);
    if (status_map) vkUnmapMemory(device, status_mem);
    if (cksum_map) vkUnmapMemory(device, cksum_mem);
    if (hdr_buf) vkDestroyBuffer(device, hdr_buf, nullptr);
    if (out_buf) vkDestroyBuffer(device, out_buf, nullptr);
    if (status_buf) vkDestroyBuffer(device, status_buf, nullptr);
    if (cksum_buf) vkDestroyBuffer(device, cksum_buf, nullptr);
    if (hdr_mem) vkFreeMemory(device, hdr_mem, nullptr);
    if (out_mem) vkFreeMemory(device, out_mem, nullptr);
    if (status_mem) vkFreeMemory(device, status_mem, nullptr);
    if (cksum_mem) vkFreeMemory(device, cksum_mem, nullptr);
    if (pipe_prepare) vkDestroyPipeline(device, pipe_prepare, nullptr);
    if (pipe_probe) vkDestroyPipeline(device, pipe_probe, nullptr);
    if (layout_prepare) vkDestroyPipelineLayout(device, layout_prepare, nullptr);
    if (layout_probe) vkDestroyPipelineLayout(device, layout_probe, nullptr);
    if (mod_prepare) vkDestroyShaderModule(device, mod_prepare, nullptr);
    if (mod_probe) vkDestroyShaderModule(device, mod_probe, nullptr);
    if (pool) vkDestroyCommandPool(device, pool, nullptr);
  }
};

/* Host-visible + coherent buffer with a device address (same usage pair as
 * the ABI probe readback, kernels.cpp:369-371 and the M2c spike). */
static bool tp_create_mapped_buffer(VulkanDevice *dev,
                                    VkDeviceSize bytes,
                                    VkBuffer &buf,
                                    VkDeviceMemory &mem,
                                    void **mapped,
                                    uint64_t *addr)
{
  const VkDevice vk_dev = dev->vk_device();
  VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bci.size = bytes;
  bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
              VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  if (vkCreateBuffer(vk_dev, &bci, nullptr, &buf) != VK_SUCCESS)
    return false;
  VkMemoryRequirements reqs;
  vkGetBufferMemoryRequirements(vk_dev, buf, &reqs);
  VkMemoryAllocateFlagsInfo flags_info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
  flags_info.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.pNext = &flags_info;
  ai.allocationSize = reqs.size;
  ai.memoryTypeIndex = dev->find_memory_type(
      reqs.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  if (ai.memoryTypeIndex == UINT32_MAX)
    return false;
  if (vkAllocateMemory(vk_dev, &ai, nullptr, &mem) != VK_SUCCESS)
    return false;
  if (vkBindBufferMemory(vk_dev, buf, mem, 0) != VK_SUCCESS)
    return false;
  if (vkMapMemory(vk_dev, mem, 0, bytes, 0, mapped) != VK_SUCCESS)
    return false;
  VkBufferDeviceAddressInfo bai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
  bai.buffer = buf;
  *addr = vkGetBufferDeviceAddress(vk_dev, &bai);
  return *addr != 0;
}

/* Record + submit + wait one dispatch, from the probe's own command pool
 * (same pattern as the ABI probe, kernels.cpp:399-454). The caller passes
 * extra pipeline barriers as a callback so the prepare->probe header
 * dependency gets a real memory dependency (Vulkan guarantees no
 * cross-dispatch visibility on its own). */
template<typename Extra> static bool tp_dispatch(VulkanDevice *dev,
                                                 VkCommandPool pool,
                                                 VkPipeline pipe,
                                                 VkPipelineLayout layout,
                                                 uint32_t push_bytes,
                                                 const void *push,
                                                 uint32_t gx,
                                                 uint32_t gy,
                                                 Extra &&extra)
{
  const VkDevice vk_dev = dev->vk_device();
  VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbai.commandPool = pool;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  if (vkAllocateCommandBuffers(vk_dev, &cbai, &cmd) != VK_SUCCESS)
    return false;
  VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  if (vkBeginCommandBuffer(cmd, &begin_info) != VK_SUCCESS)
    return false;
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
  vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes, push);
  vkCmdDispatch(cmd, gx, gy, 1);
  extra(cmd); /* barriers, if any */
  if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
    return false;
  const uint32_t family = dev->find_queue_family(dev->vk_physical_device());
  VkQueue queue = VK_NULL_HANDLE;
  vkGetDeviceQueue(vk_dev, family, 0, &queue);
  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkFence fence = VK_NULL_HANDLE;
  if (vkCreateFence(vk_dev, &fci, nullptr, &fence) != VK_SUCCESS)
    return false;
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  bool ok = vkQueueSubmit(queue, 1, &si, fence) == VK_SUCCESS &&
            vkWaitForFences(vk_dev, 1, &fence, VK_TRUE, 10000000000ull) ==
                VK_SUCCESS;
  vkDestroyFence(vk_dev, fence, nullptr);
  vkFreeCommandBuffers(vk_dev, pool, 1, &cmd);
  return ok;
}

} /* namespace */

bool vulkan_traverse_probe(VulkanDevice *dev)
{
  /* Run at most once per process, only when the env hook is set. */
  static bool ran = false;
  if (ran)
    return true;
  ran = true;
  if (getenv("CYCLES_VULKAN_TRAVERSE_PROBE") == nullptr)
    return true;

  LOG_INFO << "Vulkan: M4c traverse probe starting (CYCLES_VULKAN_TRAVERSE_PROBE=1).";
  TpResources res;
  res.device = dev->vk_device();
  const VkDevice vk_dev = res.device;
  bool infra_ok = true;
  auto fail = [&infra_ok](const char *what) {
    LOG_ERROR << "Vulkan: traverse probe: " << what;
    infra_ok = false;
    return false;
  };

  /* 1. Params buffer device address (public accessor; the same one the
   *    ABI probe uses, kernels.cpp:282). */
  const uint64_t params_addr = dev->kernel_params_address();
  if (params_addr == 0)
    return fail("kernel params buffer does not exist");

  /* 2. Reference data: sidecar dump (Q1). */
  const char *dump_env = getenv("CYCLES_VULKAN_TRAVERSE_DUMP_DIR");
  const std::string dump_dir = dump_env ? dump_env : "traverse_dump";
  TpDump dump;
  std::string err;
  if (!tp_load_dump(dump_dir, dump, err))
    return fail(("cannot load dump '" + dump_dir + "': " + err +
                 " (run cycles_bvh_dump <scene.xml> <dir> first)")
                    .c_str());
  LOG_INFO << "Vulkan: traverse probe: dump '" << dump_dir << "': " << dump.node_count
           << " node float4s, " << dump.leaf_count << " leaves, " << dump.prim_count
           << " prims, " << dump.tri_count << " triangles, " << dump.vert_count
           << " vertices, " << dump.object_count << " objects";

  /* Checksums of the dump's arrays; the prepare kernel computes the same
   * FNV-1a over the LIVE arrays (Q1 guard). */
  const uint32_t fnv_nodes = tp_fnv1a(dump.bvh_nodes.data(), dump.node_count * 16);
  const uint32_t fnv_verts = tp_fnv1a(dump.tri_verts.data(), dump.vert_count * 12);

  /* 3. Reference world triangles + camera + CPU brute force. */
  std::vector<TpWorldTri> world_tris;
  if (!tp_build_world_tris(dump, world_tris, err))
    return fail(err.c_str());
  TpCam cam;
  tp_build_camera(cam);
  const float fov_deg = (float)(2.0 * std::atan(TP_PLANE_HALF) * 180.0 / 3.14159265358979323846);
  LOG_INFO << string_printf(
      "Vulkan: traverse probe: camera origin (%.9g, %.9g, %.9g), plane half-extent %.9g at depth 1, FOV %.4f deg",
      cam.o[0], cam.o[1], cam.o[2], TP_PLANE_HALF, fov_deg);
  LOG_INFO << string_printf(
      "Vulkan: traverse probe:   plane lower-left (%.9g, %.9g, %.9g), width vec (%.9g, %.9g, %.9g), height vec (%.9g, %.9g, %.9g)",
      cam.l[0], cam.l[1], cam.l[2], cam.u[0], cam.u[1], cam.u[2], cam.v[0],
      cam.v[1], cam.v[2]);

  std::vector<TpHit> ref(TP_PIXELS);
  for (uint32_t y = 0; y < TP_HEIGHT; y++) {
    for (uint32_t x = 0; x < TP_WIDTH; x++) {
      float d[3];
      tp_pixel_dir(cam, x, y, d);
      ref[(size_t)y * TP_WIDTH + x] = tp_trace_cpu(world_tris, cam.o, d);
    }
  }

  /* 4. SPIR-V modules (compiled by hand with slangc, like the ABI probe;
   *    env override or repo-relative defaults). */
  auto load_spv = [&](const char *env_name, const char *a, const char *b,
                      std::vector<char> &out, const char **found) -> bool {
    const char *env_path = getenv(env_name);
    const char *candidates[3] = {env_path, a, b};
    for (const char *c : candidates) {
      if (c && probe_read_whole_file(c, out)) {
        *found = c;
        return true;
      }
    }
    return false;
  };
  std::vector<char> spv_prepare, spv_probe;
  const char *found = nullptr;
  if (!load_spv("CYCLES_VULKAN_TRAVERSE_PREPARE_SPV", "traverse_prepare.spv",
                "src/kernel/device/vulkan/traverse_prepare.spv", spv_prepare,
                &found))
    return fail("cannot find traverse_prepare.spv (compile traverse_probe.slang "
                "with slangc or set CYCLES_VULKAN_TRAVERSE_PREPARE_SPV)");
  const char *found_probe = nullptr;
  if (!load_spv("CYCLES_VULKAN_TRAVERSE_PROBE_SPV", "traverse_probe.spv",
                "src/kernel/device/vulkan/traverse_probe.spv", spv_probe,
                &found_probe))
    return fail("cannot find traverse_probe.spv");

  /* 5. Modules, layouts (one push-constant range each, no descriptors -
   *    everything travels as device addresses) and pipelines. The SPIR-V
   *    entry point is "main": slangc renames it (kernels.cpp:346-351). */
  auto make_pipeline = [&](const std::vector<char> &spirv, uint32_t push_bytes,
                           VkShaderModule &mod, VkPipelineLayout &layout,
                           VkPipeline &pipe) -> bool {
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = spirv.size();
    smi.pCode = reinterpret_cast<const uint32_t *>(spirv.data());
    if (vkCreateShaderModule(vk_dev, &smi, nullptr, &mod) != VK_SUCCESS)
      return false;
    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = push_bytes;
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(vk_dev, &pli, nullptr, &layout) != VK_SUCCESS)
      return false;
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = mod;
    cpi.stage.pName = "main";
    cpi.layout = layout;
    return vkCreateComputePipelines(vk_dev, VK_NULL_HANDLE, 1, &cpi, nullptr,
                                    &pipe) == VK_SUCCESS;
  };
  if (!make_pipeline(spv_prepare, TP_PREPARE_PUSH_BYTES, res.mod_prepare,
                     res.layout_prepare, res.pipe_prepare) ||
      !make_pipeline(spv_probe, TP_PROBE_PUSH_BYTES, res.mod_probe,
                     res.layout_probe, res.pipe_probe))
    return fail("pipeline creation failed (bad entry point name?)");

  /* Command pool on the family/queue index the device itself uses. */
  {
    const uint32_t family = dev->find_queue_family(dev->vk_physical_device());
    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    cpci.queueFamilyIndex = family;
    if (vkCreateCommandPool(vk_dev, &cpci, nullptr, &res.pool) != VK_SUCCESS)
      return fail("command pool creation failed");
  }

  /* 6. Buffers: header (host-filled counts/camera/flags + shader-filled
   *    pointers), Intersection output with a canary zone, status words,
   *    prepare checksum block. */
  const VkDeviceSize out_bytes =
      (VkDeviceSize)(TP_PIXELS + TP_CANARY_RAYS) * sizeof(TpIntersection);
  const VkDeviceSize status_bytes =
      (VkDeviceSize)(TP_PIXELS + TP_CANARY_RAYS) * sizeof(int32_t);
  if (!tp_create_mapped_buffer(dev, TP_HEADER_BYTES, res.hdr_buf, res.hdr_mem,
                               &res.hdr_map, &res.hdr_addr) ||
      !tp_create_mapped_buffer(dev, out_bytes, res.out_buf, res.out_mem,
                               &res.out_map, &res.out_addr) ||
      !tp_create_mapped_buffer(dev, status_bytes, res.status_buf, res.status_mem,
                               &res.status_map, &res.status_addr) ||
      !tp_create_mapped_buffer(dev, TP_PREPARE_OUT_WORDS * 4, res.cksum_buf,
                               res.cksum_mem, &res.cksum_map, &res.cksum_addr))
    return fail("buffer creation failed");

  /* 7. Fill the header (everything except the pointer slots + root, which
   *    the prepare kernel reads from the LIVE params buffer). */
  TpHeader hdr{};
  hdr.node_count = dump.node_count;
  hdr.leaf_count = dump.leaf_count;
  hdr.prim_count = dump.prim_count;
  hdr.tri_count = dump.tri_count;
  hdr.vert_count = dump.vert_count;
  hdr.object_count = dump.object_count;
  hdr.object_node_count = dump.object_node_count;
  hdr.width = TP_WIDTH;
  hdr.height = TP_HEIGHT;
  hdr.visibility = 0xFFFFFFFFu; /* Q5 */
  hdr.test_mode = 0;
  for (int i = 0; i < 3; i++) {
    hdr.cam[0 + i] = cam.o[i];
    hdr.cam[3 + i] = cam.l[i];
    hdr.cam[6 + i] = cam.u[i];
    hdr.cam[9 + i] = cam.v[i];
  }
  memcpy(res.hdr_map, &hdr, sizeof(hdr));

  /* 8. Prepare dispatch: shader reads the nine slot addresses + the blob
   *    root (Q3) from the live params, checksums the live arrays. One
   *    command buffer with a compute->compute memory barrier afterwards,
   *    so the ray pass sees the header writes (Vulkan gives no implicit
   *    cross-dispatch visibility). */
  {
    TpPreparePush pp{};
    pp.hdr = res.hdr_addr;
    pp.hdr_raw = res.hdr_addr; /* bytes 0-71 are exactly the 9 pointer slots */
    pp.hdr_root = res.hdr_addr + 72;
    pp.params = params_addr;
    pp.out = res.cksum_addr;
    pp.params_raw = params_addr;
    pp.blob_root_word = (uint32_t)(offsetof(KernelData, bvh) / 4);
    auto with_header_barrier = [&](VkCommandBuffer cmd) {
      VkBufferMemoryBarrier bmb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
      bmb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      bmb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      bmb.buffer = res.hdr_buf;
      bmb.offset = 0;
      bmb.size = VK_WHOLE_SIZE;
      vkCmdPipelineBarrier(cmd,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           0, 0, nullptr, 1, &bmb, 0, nullptr);
    };
    if (!tp_dispatch(dev, res.pool, res.pipe_prepare, res.layout_prepare,
                     TP_PREPARE_PUSH_BYTES, &pp, 1, 1, with_header_barrier))
      return fail("prepare dispatch failed");
  }

  /* 9. Read the checksum block and cross-check against the dump. */
  uint32_t ck[TP_PREPARE_OUT_WORDS];
  memcpy(ck, res.cksum_map, sizeof(ck));
  const int32_t live_root = (int32_t)ck[3];
  if (ck[4] != TP_PREPARE_SENTINEL)
    return fail("prepare sentinel missing: the prepare dispatch did not run");
  if (ck[0] != sizeof(TpHeader))
    return fail("Slang sizeof(TraverseHeader) does not match the host mirror");

  /* Debugging aid: dump the whole live KernelData blob and locate the
   * KernelBVH block {root, have_motion, have_curves, have_points,
   * have_volumes, bvh_layout==1, use_bvh_steps, curve_subdivisions}
   * (src/kernel/data_template.h:51-60) empirically. */
  {
    LOG_INFO << string_printf(
        "Vulkan: traverse probe: offsetof(KernelData, bvh) = %u bytes (word %u); "
        "blob word there = 0x%08x",
        (unsigned)offsetof(KernelData, bvh),
        (unsigned)(offsetof(KernelData, bvh) / 4), (uint32_t)live_root);
    if (FILE *f = fopen("m4c_live_kerneldata.bin", "wb")) {
      fwrite(&ck[5], 4, 640, f);
      fclose(f);
      LOG_INFO << "Vulkan: traverse probe: live KernelData blob written to "
                  "m4c_live_kerneldata.bin (640 words)";
    }
    for (uint32_t w = 0; w + 8 <= 640; w++) {
      const uint32_t *b = &ck[5 + w];
      const bool small = (b[1] | b[2] | b[3] | b[4] | b[6] | b[7]) <= 1u;
      if (small && b[5] == 1u)
        LOG_INFO << string_printf(
            "Vulkan: traverse probe: KernelBVH signature candidate at blob word %u: "
            "{%d, %d, %d, %d, %d, %d, %d, %d}",
            w, (int32_t)b[0], (int32_t)b[1], (int32_t)b[2], (int32_t)b[3],
            (int32_t)b[4], (int32_t)b[5], (int32_t)b[6], (int32_t)b[7]);
    }
  }
  LOG_INFO << string_printf(
      "Vulkan: traverse probe: root: live blob=0x%08x dump=0x%08x | "
      "bvh_nodes checksum: live=0x%08x dump=0x%08x | tri_verts: live=0x%08x dump=0x%08x",
      (uint32_t)live_root, (uint32_t)dump.root, ck[1], fnv_nodes, ck[2], fnv_verts);
  if (ck[1] != fnv_nodes || ck[2] != fnv_verts || live_root != dump.root) {
    LOG_WARNING << "Vulkan: traverse probe: INCONCLUSIVE - dump differs from live "
                   "scene (the BVH build of a separate Cycles run is not "
                   "byte-identical); re-run cycles_bvh_dump and try again.";
    return true; /* not a code failure - the two runs disagree (Q1) */
  }

  /* 10. One ray-pass run: fill out/status with the unwritten marker +
   *    canaries, dispatch, verify, copy out. */
  auto run_probe = [&](uint32_t visibility, uint32_t test_mode, TpCmp &cmp,
                       size_t &unwritten, size_t &err_pixels,
                       bool &canaries_ok, bool details) -> bool {
    uint32_t *outw = (uint32_t *)res.out_map;
    const size_t real_words = (size_t)TP_PIXELS * 6; /* Intersection = 6 words */
    const size_t canary_words = (size_t)TP_CANARY_RAYS * 6;
    for (size_t i = 0; i < real_words; i++)
      outw[i] = TP_UNWRITTEN_WORD;
    for (size_t i = 0; i < canary_words; i++)
      outw[real_words + i] = TP_CANARY_WORD;
    int32_t *statw = (int32_t *)res.status_map;
    for (size_t i = 0; i < TP_PIXELS; i++)
      statw[i] = (int32_t)TP_UNWRITTEN_WORD;

    /* Visibility and test_mode live in the header the host owns. */
    ((TpHeader *)res.hdr_map)->visibility = visibility;
    ((TpHeader *)res.hdr_map)->test_mode = test_mode;

    TpProbePush pp{};
    pp.hdr = res.hdr_addr;
    pp.out = res.out_addr;
    pp.status = res.status_addr;
    auto no_barrier = [](VkCommandBuffer cmd) { (void)cmd; };
    if (!tp_dispatch(dev, res.pool, res.pipe_probe, res.layout_probe,
                     TP_PROBE_PUSH_BYTES, &pp, (TP_WIDTH + 7) / 8,
                     (TP_HEIGHT + 7) / 8, no_barrier))
      return false;

    canaries_ok = true;
    for (size_t i = 0; i < canary_words; i++)
      if (outw[real_words + i] != TP_CANARY_WORD)
        canaries_ok = false;

    std::vector<TpIntersection> gpu(TP_PIXELS);
    std::vector<int32_t> status(TP_PIXELS);
    memcpy(gpu.data(), res.out_map, (size_t)TP_PIXELS * sizeof(TpIntersection));
    memcpy(status.data(), res.status_map, (size_t)TP_PIXELS * sizeof(int32_t));
    unwritten = 0;
    err_pixels = 0;
    for (size_t i = 0; i < TP_PIXELS; i++) {
      const uint32_t tbits = ((const uint32_t *)&gpu[i])[0];
      if (tbits == TP_UNWRITTEN_WORD && ((const uint32_t *)&gpu[i])[4] ==
                                            TP_UNWRITTEN_WORD)
        unwritten++;
      if (status[i] != 0)
        err_pixels++;
    }
    cmp = tp_compare(ref, gpu, status, world_tris, cam, details);
    return true;
  };

  /* 11. Main run: visibility 0xFFFFFFFF, no faults (Q5). */
  TpCmp main_cmp;
  size_t unwritten = 0, err_pixels = 0;
  bool canaries_ok = false;
  if (!run_probe(0xFFFFFFFFu, 0, main_cmp, unwritten, err_pixels, canaries_ok,
                 /*details=*/true))
    return fail("ray dispatch failed");
  tp_print_cmp("reference (brute force, tfm world-space) vs GPU (ported traversal)",
               main_cmp, TP_PIXELS);
  LOG_INFO << "Vulkan: traverse probe: canaries: "
           << (canaries_ok ? "INTACT" : "OVERWRITTEN !!!")
           << ", pixels never written: " << unwritten
           << ", shader error pixels: " << err_pixels;

  /* Per-cube hit counts (gate requirement): GPU hits grouped by object,
   * labelled with each object's world-space x centre. */
  {
    std::vector<size_t> hits((size_t)dump.object_count, 0);
    const TpIntersection *gpu = (const TpIntersection *)res.out_map;
    const int32_t *status = (const int32_t *)res.status_map;
    for (size_t i = 0; i < TP_PIXELS; i++) {
      if (status[i] == 0 && gpu[i].prim >= 0 && gpu[i].object >= 0 &&
          (uint32_t)gpu[i].object < dump.object_count)
        hits[(size_t)gpu[i].object]++;
    }
    for (uint32_t obj = 0; obj < dump.object_count; obj++) {
      float cx = 0.0f;
      size_t n = 0;
      for (const TpWorldTri &wt : world_tris) {
        if (wt.obj != obj)
          continue;
        for (int k = 0; k < 3; k++, n++)
          cx += wt.v[k * 3 + 0];
      }
      cx = (n ? cx / (float)n : 0.0f);
      LOG_INFO << string_printf(
          "Vulkan: traverse probe: object %u (world x centre %+.3f): %zu hit pixels",
          obj, cx, hits[obj]);
    }
  }

  /* 12. Negative tests (gate): each fault MUST be detected, i.e. the
   *    comparison against the un-faulted reference must FAIL. */
  auto run_negative = [&](const char *label, uint32_t visibility,
                          uint32_t test_mode) -> bool {
    TpCmp c;
    size_t uw = 0, ep = 0;
    bool ck = false;
    if (!run_probe(visibility, test_mode, c, uw, ep, ck, /*details=*/false)) {
      LOG_ERROR << "Vulkan: traverse probe: NEGATIVE TEST " << label
                << ": dispatch failed";
      return false;
    }
    const bool detected = !tp_cmp_ok(c, TP_PIXELS);
    LOG_INFO << string_printf(
        "Vulkan: traverse probe: NEGATIVE TEST %-28s: %s (bad=%zu errors=%zu grazing=%zu)",
        label, detected ? "DETECTED (gate failure, as expected)" : "NOT DETECTED !!!",
        c.bad, c.errors, c.grazing);
    return detected;
  };
  const bool neg1 = run_negative("1 skip instance pop", 0xFFFFFFFFu, 1u /*TEST_MODE_SKIP_POP*/);
  const bool neg2 = run_negative("2 corrupt object_node", 0xFFFFFFFFu, 2u /*TEST_MODE_CORRUPT_OBJECT_NODE*/);
  const bool neg3 = run_negative("3 visibility mask 0", 0u, 0u);

  /* 13. Verdict. */
  const bool main_ok = tp_cmp_ok(main_cmp, TP_PIXELS) && canaries_ok &&
                       unwritten == 0 && err_pixels == 0;
  const bool pass = main_ok && neg1 && neg2 && neg3 && infra_ok;
  LOG_INFO << "Vulkan: traverse probe: "
           << (pass ? "PASS - main run matches the reference, all three faults detected"
                    : "FAIL - see the numbers above");
  return pass;
}

CCL_NAMESPACE_END

#endif /* WITH_VULKAN */
