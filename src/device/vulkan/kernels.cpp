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

/* M4e Part 1: the REAL Cycles Lambertian closure, compiled on the host. On
 * this target ccl_device is `static inline` and ccl_private/ccl_global are
 * empty (util/defines.h:25,63,67), so bsdf_diffuse_eval/_sample are directly
 * callable. kernel/types.h is already in this TU via device_impl.h -> params.h,
 * so this include adds only the closure functions (bsdf_diffuse.h pulls
 * kernel/closure/alloc.h and kernel/sample/mapping.h). Used as a second,
 * independent GPU reference below (vulkan_bsdf_probe) so a formula misreading
 * shared by my Slang port AND the hand re-implementation is still caught. */
#include "kernel/closure/bsdf_diffuse.h"

/* M4e STEP 1a: the REAL Cycles sampler / camera headers, compiled on the host,
 * used as the GPU reference for the math probe (vulkan_math_probe). On this
 * target ccl_device is `static inline` and ccl_private/ccl_global are empty, so
 * path_rng_pixel_init / path_rng_1D..4D / camera_sample are directly callable.
 * kernel/globals.h (when !__KERNEL_GPU__) pulls kernel/device/cpu/globals.h,
 * giving us KernelGlobalsCPU / ThreadKernelGlobalsCPU / KernelGlobals and the
 * `kernel_data` (= kg->data) macro; pattern.h gives the path_rng_* family;
 * camera.h gives camera_sample. */
#include "kernel/globals.h"
#include "kernel/sample/pattern.h"
#include "kernel/camera/camera.h"

#include "util/log.h"

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdint>
/* M4e S1b-5: <cstddef> for offsetof(), <type_traits> for std::remove_reference_t, both
 * used by the IntegratorStateGPU array-member layout static_asserts below. */
#include <cstddef>
#include <type_traits>

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

/* M4e S1b-5 Task 1: lock the ARRAY-MEMBER layout of IntegratorStateGPU.
 *
 * This build is NON-packed on both host and Slang: __INTEGRATOR_GPU_PACKED_STATE__ is
 * defined only for Apple ARM64 (types.h:626-628), which this Windows/MSVC target is not.
 * In non-packed __KERNEL_GPU__ mode, state.h:174 maps KERNEL_STRUCT_ARRAY_MEMBER to
 * KERNEL_STRUCT_MEMBER (ccl_global type *name) and state.h:178-180 maps
 * KERNEL_STRUCT_END_ARRAY to "} name[gpu_size];". So every array member is an ARRAY OF
 * STRUCTS OF POINTERS (AoS), NOT a struct of arrays (SoA):
 *
 *   volume_stack[32]        = { int   *object; int  *shader; }        stride 16
 *   shadow_isect[4]         = { float *t; float*u; float*v;
 *                               int   *prim; int *object; int *type; } stride 48
 *   shadow_volume_stack[32] = { int   *object; int  *shader; }        stride 16
 *   (sizes: MAX_VOLUME_STACK_SIZE=32 types.h:847; INTEGRATOR_SHADOW_ISECT_SIZE_GPU=4
 *    types.h:59)
 *
 * The host allocator confirms it: path_trace_work_gpu.cpp:174-186 writes
 * integrator_state_gpu_.parent_struct[array_index].name, i.e. each [i].field is its own
 * SoA array allocation (its own device pointer), and .device_pointer is memcpy'd into the
 * element. The ELEMENT is a struct of pointers; the DATA behind each pointer is SoA.
 *
 * These asserts lock (a) each element's stride and (b) each member's offset inside the
 * element, so a Cycles layout change breaks the build instead of silently corrupting the
 * Slang mirror. */
using VolumeStackElem = std::remove_reference_t<decltype(
    ((IntegratorStateGPU *)nullptr)->volume_stack[0])>;
using ShadowIsectElem = std::remove_reference_t<decltype(
    ((IntegratorStateGPU *)nullptr)->shadow_isect[0])>;
using ShadowVolumeStackElem = std::remove_reference_t<decltype(
    ((IntegratorStateGPU *)nullptr)->shadow_volume_stack[0])>;

/* volume_stack: base + per-element AoS offsets. */
static_assert(offsetof(IntegratorStateGPU, volume_stack) == 376, "volume_stack base offset");
static_assert(sizeof(VolumeStackElem) == 16, "volume_stack element stride (AoS)");
static_assert(offsetof(VolumeStackElem, object) == 0, "volume_stack[].object offset");
static_assert(offsetof(VolumeStackElem, shader) == 8, "volume_stack[].shader offset");

/* shadow_isect: base + per-element AoS offsets (6 pointers -> 48 B). */
static_assert(offsetof(IntegratorStateGPU, shadow_isect) == 1240, "shadow_isect base offset");
static_assert(sizeof(ShadowIsectElem) == 48, "shadow_isect element stride (AoS)");
static_assert(offsetof(ShadowIsectElem, t) == 0, "shadow_isect[].t offset");
static_assert(offsetof(ShadowIsectElem, u) == 8, "shadow_isect[].u offset");
static_assert(offsetof(ShadowIsectElem, v) == 16, "shadow_isect[].v offset");
static_assert(offsetof(ShadowIsectElem, prim) == 24, "shadow_isect[].prim offset");
static_assert(offsetof(ShadowIsectElem, object) == 32, "shadow_isect[].object offset");
static_assert(offsetof(ShadowIsectElem, type) == 40, "shadow_isect[].type offset");

/* shadow_volume_stack: same shape as volume_stack. */
static_assert(offsetof(IntegratorStateGPU, shadow_volume_stack) == 1432,
              "shadow_volume_stack base offset");
static_assert(sizeof(ShadowVolumeStackElem) == 16, "shadow_volume_stack element stride (AoS)");
static_assert(offsetof(ShadowVolumeStackElem, object) == 0, "shadow_volume_stack[].object offset");
static_assert(offsetof(ShadowVolumeStackElem, shader) == 8, "shadow_volume_stack[].shader offset");

namespace {

/* Number of uint32 words abi_probe.slang writes to the readback buffer.
 * M4e S1b-5: 12 -> 24. The extra 12 words (out[12..23]) carry three typed-vs-raw
 * pairs for the ARRAY-MEMBER AoS decomposition (volume_stack[1].shader,
 * shadow_isect[1].object, shadow_volume_stack[1].shader). Each pair is lo/hi of the
 * typed 64-bit pointer read through the Slang mirror, then lo/hi of the raw words at
 * the host static_assert's expected byte offset. A flat-SoA mirror would place the
 * typed read at a different byte offset than the raw read and the pair would MISMATCH. */
constexpr uint32_t ABI_PROBE_OUT_WORDS = 24;
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
  /* M4e S1b-5 Task 2: array-member AoS decomposition. out[12..23] are three
   * typed-vs-raw pairs (lo,hi each). The typed read goes through the Slang
   * IntegratorStateGPU mirror at a NON-zero slot index; the raw read is at the
   * byte offset the host static_asserts expect (AoS: base + stride*i + inner).
   * A flat-SoA mirror would land the typed read at a different offset and
   * MISMATCH. See the OFF_* constants in abi_probe.slang. */
  check("volume_stack[1].shader typed vs raw@3384",
        ((uint64_t)out[13] << 32) | out[12],
        ((uint64_t)out[15] << 32) | out[14]);
  check("shadow_isect[1].object typed vs raw@4304",
        ((uint64_t)out[17] << 32) | out[16],
        ((uint64_t)out[19] << 32) | out[18]);
  check("shadow_volume_stack[1].shader typed vs raw@4440",
        ((uint64_t)out[21] << 32) | out[20],
        ((uint64_t)out[23] << 32) | out[22]);
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
 * M4e S1b-6 Task 3: integrator-state ACCESSOR GATE
 * (optional, env CYCLES_VULKAN_STATE_ACCESS_PROBE=1; runs at most once.
 *  CYCLES_VULKAN_STATE_ACCESS_MODE restricts the FAULT modes that run after
 *  the mandatory clean mode-0 baseline: unset = run 1..4, "0" = clean only,
 *  "1".."4" = clean + that single fault mode.)
 * =========================================================================
 *
 * WHAT IT PROVES, byte for byte, on the GPU:
 *  - every read_/write_ accessor in integrator_state.slang addresses the exact
 *    SoA byte offset Cycles uses. Cycles allocates ONE array of max_num_paths
 *    elements per KERNEL_STRUCT_MEMBER and per array slot
 *    (path_trace_work_gpu.cpp:143-201); this gate allocates GATE_STATE_COUNT
 *    elements per slice (stand-in for max_num_paths) and fills a REAL
 *    IntegratorStateGPU - layout locked by the static_asserts above - with the
 *    slice device addresses, exactly the way the Cycles allocator memcpy's its
 *    device pointers in (path_trace_work_gpu.cpp:153-155).
 *  - packed3 members stride 12 (float3*), AoS slots stride 16/48/16.
 *  - sub-word (uint16/uint8) members pack two/four per 32-bit word without
 *    clobbering the neighbouring state (adjacent states run in DIFFERENT
 *    threads of one workgroup), and members/slots no accessor touches keep
 *    their 0xEE init pattern.
 *  - out[s] equals the host's recomputation of the shader's readback
 *    accumulator (mode 0), i.e. every accessor's READ returns what was written.
 *
 * TASK-4 NEGATIVES: modes 1..4 inject one fault each (see the probe's header).
 * Per AGENTS.md 11 a negative counts as DETECTED only if the fault CHANGES THE
 * OUTPUT relative to the clean run of the same configuration, so mode 0 runs
 * first, its bytes are snapshotted, and each fault mode is compared against
 * that snapshot (payload bytes AND readback words).
 *
 * Composes with the M4b ABI probe: that probe proved the Slang mirror at
 * KernelParamsVulkan+2984 is byte-identical to IntegratorStateGPU, so
 * uploading the struct here exercises the same reading path the kernels will
 * use. Neither probe writes the live params buffer. */
namespace {

/* Gate sizing: matches numthreads(8,1,1) in state_access_probe.slang. */
constexpr uint32_t GATE_STATES = 8;
/* PushGate = { IntegratorStateGPU* ist; uint* out; uint* throughput_words;
 * uint test_mode; } with Slang scalar layout: 8 + 8 + 8 + 4, padded to 8-byte
 * alignment -> 32. M4e S1b-7: throughput_words (a uint* word view of the
 * path.throughput slice) was added so negative mode 1 can inject its
 * stride-16 fault WITHOUT a pointer OpBitcast — SwiftShader's JIT crashes
 * (0xC00000FD) on ANY PhysicalStorageBuffer pointer bitcast, even though the
 * SPIR-V is spec-valid; see state_access_probe.slang mode 1. */
constexpr uint32_t GATE_PUSH_BYTES = 32;
constexpr uint32_t GATE_OUT_WORDS = GATE_STATES;
constexpr uint32_t GATE_CANARY_BYTES = 16; /* guard after every slice */
constexpr uint32_t GATE_CANARY_BYTE = 0x5C;
constexpr uint32_t GATE_INIT_BYTE = 0xEE;  /* fill for slices/bytes never written */
constexpr uint32_t GATE_MAX_REPORT = 8;    /* mismatch detail lines logged per mode */

/* Exact host mirror of gbits()/gbits3() in state_access_probe.slang. */
uint32_t gate_gbits(const uint32_t id, const uint32_t i, const uint32_t s)
{
  return 0x40000000u + (i << 20) + (id << 8) + s;
}

/* What the accessors should have stored in a slice (their write formula). */
enum GateExpect {
  GATE_UNTOUCHED = 0, /* no accessor writes here: must keep the 0xEE pattern */
  GATE_U32,           /* uint, or float BITS: gbits(id,slot,s) */
  GATE_U16,           /* sub-word: (gbits & 0xFFFF) */
  GATE_U8,            /* sub-word: (gbits & 0xFF) */
  GATE_I32_NEG,       /* -int(gbits(id,slot,s)): the int accessors' value */
  GATE_F32X3,         /* packed3: gbits(id), gbits(id+64), gbits(id+128) */
};

/* One SoA slice of the gate buffer: GATE_STATES elements + a canary guard.
 * `mirror` is the byte offset of the POINTER for this member inside
 * IntegratorStateGPU (the struct's own layout, not the payload's). */
struct GateSlice {
  std::string name;   /* log label */
  size_t      off;    /* payload byte offset inside the SoA buffer */
  size_t      mirror; /* pointer byte offset inside IntegratorStateGPU */
  uint32_t    elem;   /* element size in bytes */
  int         expect; /* GateExpect */
  uint32_t    id;     /* gbits id (first component for GATE_F32X3) */
  uint32_t    slot;   /* gbits `i` argument: the AoS slot, 0 for scalars */
};

/* RAII for every Vulkan handle the gate creates, so any early return still
 * destroys what was created and the validation layer reports nothing. */
struct GateResources {
  VkDevice device = VK_NULL_HANDLE;
  VkShaderModule module = VK_NULL_HANDLE;
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;
  VkCommandPool pool = VK_NULL_HANDLE;
  VkBuffer soa_buffer = VK_NULL_HANDLE;
  VkDeviceMemory soa_memory = VK_NULL_HANDLE;
  VkBuffer ist_buffer = VK_NULL_HANDLE;
  VkDeviceMemory ist_memory = VK_NULL_HANDLE;
  VkBuffer out_buffer = VK_NULL_HANDLE;
  VkDeviceMemory out_memory = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;

  ~GateResources()
  {
    if (device == VK_NULL_HANDLE) return;
    if (fence) vkDestroyFence(device, fence, nullptr);
    if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
    if (layout) vkDestroyPipelineLayout(device, layout, nullptr);
    if (module) vkDestroyShaderModule(device, module, nullptr);
    if (pool) vkDestroyCommandPool(device, pool, nullptr);
    if (soa_buffer) vkDestroyBuffer(device, soa_buffer, nullptr);
    if (soa_memory) vkFreeMemory(device, soa_memory, nullptr);
    if (ist_buffer) vkDestroyBuffer(device, ist_buffer, nullptr);
    if (ist_memory) vkFreeMemory(device, ist_memory, nullptr);
    if (out_buffer) vkDestroyBuffer(device, out_buffer, nullptr);
    if (out_memory) vkFreeMemory(device, out_memory, nullptr);
  }
};

} /* namespace */

bool vulkan_state_access_probe(VulkanDevice *dev)
{
  /* Run at most once per process, and only when the env hook is set. */
  static bool ran = false;
  if (ran) return true;
  ran = true;
  const char *mode_env = getenv("CYCLES_VULKAN_STATE_ACCESS_MODE");
  if (getenv("CYCLES_VULKAN_STATE_ACCESS_PROBE") == nullptr) return true;

  /* Which modes run: 0 (clean baseline) always; faults unless restricted. */
  int only_mode = -1; /* -1 = all */
  if (mode_env) {
    only_mode = atoi(mode_env);
    if (only_mode < 0 || only_mode > 4) {
      LOG_ERROR << "Vulkan: state access gate: invalid "
                   "CYCLES_VULKAN_STATE_ACCESS_MODE='" << mode_env << "' (0..4).";
      return false;
    }
  }
  std::vector<int> modes;
  modes.push_back(0);
  if (only_mode >= 1 && only_mode <= 4) {
    modes.push_back(only_mode);
  }
  else if (only_mode < 0) {
    modes.push_back(1);
    modes.push_back(2);
    modes.push_back(3);
    modes.push_back(4);
  }

  LOG_INFO << "Vulkan: state access gate starting "
              "(CYCLES_VULKAN_STATE_ACCESS_PROBE=1, " << modes.size() << " mode(s)).";

  /* Opt-in step trace (CYCLES_VULKAN_GATE_TRACE=1). LOG_WARNING goes to the
   * UNBUFFERED stderr branch of log.cpp:93-101 and fflushes stdout first, so
   * when a device crashes mid-run (e.g. SwiftShader 0xC00000FD) the trace
   * still shows the exact step that was in flight — plain LOG_INFO output is
   * block-buffered and lost on a crash. */
  const bool trace = getenv("CYCLES_VULKAN_GATE_TRACE") != nullptr;
  auto step = [&](const char *what) {
    if (trace)
      LOG_WARNING << "Vulkan: state access gate: trace: " << what;
  };

  /* 1. SPIR-V: compiled by hand with slangc (no build-system Slang rule). */
  std::vector<char> spirv;
  const char *env_path = getenv("CYCLES_VULKAN_STATE_ACCESS_PROBE_SPV");
  const char *candidates[3] = {env_path,
                               "state_access_probe.spv",
                               "src/kernel/device/vulkan/state_access_probe.spv"};
  const char *found = nullptr;
  for (const char *c : candidates) {
    if (c && probe_read_whole_file(c, spirv)) {
      found = c;
      break;
    }
  }
  if (!found) {
    LOG_ERROR << "Vulkan: state access gate: cannot find state_access_probe.spv "
                 "(compile it with slangc or set CYCLES_VULKAN_STATE_ACCESS_PROBE_SPV).";
    return false;
  }
  if (spirv.size() % 4 != 0) {
    LOG_ERROR << "Vulkan: state access gate: '" << found << "' is not a valid SPIR-V module.";
    return false;
  }
  step("spirv loaded");

  /* 2. Slice table: one slice per Cycles SoA member the gate reaches.
   *    Cycles sizes every member to max_num_paths elements and allocates EVERY
   *    array slot (path_trace_work_gpu.cpp:143-201), so the gate does too -
   *    fault mode 3 writes volume_stack slots 2..16 and must find them there. */
  std::vector<GateSlice> slices;
  IntegratorStateGPU ist{}; /* zero-init: unreachable members stay null */

#define GATE_OFF(member) (offsetof(IntegratorStateGPU, member))

  auto member = [&](const std::string &name, const size_t mirror, const uint32_t elem,
                    const int kind, const uint32_t id, const uint32_t slot) -> size_t {
    GateSlice sl;
    sl.name = name;
    sl.off = 0;
    sl.mirror = mirror;
    sl.elem = elem;
    sl.expect = kind;
    sl.id = id;
    sl.slot = slot;
    slices.push_back(std::move(sl));
    return slices.size() - 1;
  };

  /* --- path (state_template.h:7-74), gbits ids 1..17 as in the shader --- */
  member("path.render_pixel_index", GATE_OFF(path.render_pixel_index), 4, GATE_U32, 1, 0);
  member("path.sample", GATE_OFF(path.sample), 4, GATE_U32, 2, 0);
  member("path.bounce", GATE_OFF(path.bounce), 2, GATE_U16, 3, 0);
  member("path.transparent_bounce", GATE_OFF(path.transparent_bounce), 2, GATE_U16, 4, 0);
  member("path.queued_kernel", GATE_OFF(path.queued_kernel), 2, GATE_U16, 5, 0);
  member("path.rng_offset", GATE_OFF(path.rng_offset), 2, GATE_U16, 6, 0);
  member("path.visibility", GATE_OFF(path.visibility), 1, GATE_U8, 7, 0);
  member("path.flag", GATE_OFF(path.flag), 4, GATE_U32, 8, 0);
  member("path.mnee", GATE_OFF(path.mnee), 1, GATE_U8, 9, 0);
  /* float members are compared as their BITS (GATE_U32): same bytes. */
  member("path.mis_ray_pdf", GATE_OFF(path.mis_ray_pdf), 4, GATE_U32, 10, 0);
  member("path.min_ray_pdf", GATE_OFF(path.min_ray_pdf), 4, GATE_U32, 11, 0);
  member("path.continuation_probability",
         GATE_OFF(path.continuation_probability), 4, GATE_U32, 12, 0);
  member("path.optical_depth", GATE_OFF(path.optical_depth), 4, GATE_U32, 13, 0);
  member("path.shader_sort_key", GATE_OFF(path.shader_sort_key), 4, GATE_U32, 14, 0);
  member("path.throughput", GATE_OFF(path.throughput), 12, GATE_F32X3, 15, 0);
  member("path.mis_origin_n", GATE_OFF(path.mis_origin_n), 12, GATE_F32X3, 16, 0);
  member("path.pass_diffuse_weight",
         GATE_OFF(path.pass_diffuse_weight), 12, GATE_F32X3, 17, 0);

  /* --- ray (state_template.h:78-87), ids 18..24 --- */
  member("ray.P", GATE_OFF(ray.P), 12, GATE_F32X3, 18, 0);
  member("ray.D", GATE_OFF(ray.D), 12, GATE_F32X3, 19, 0);
  member("ray.dP", GATE_OFF(ray.dP), 4, GATE_U32, 20, 0);
  member("ray.dD", GATE_OFF(ray.dD), 4, GATE_U32, 21, 0);
  member("ray.tmin", GATE_OFF(ray.tmin), 4, GATE_U32, 22, 0);
  member("ray.tmax", GATE_OFF(ray.tmax), 4, GATE_U32, 23, 0);
  member("ray.time", GATE_OFF(ray.time), 4, GATE_U32, 24, 0);

  /* --- isect (state_template.h:92-99), ids 25..28 --- */
  member("isect.t", GATE_OFF(isect.t), 4, GATE_U32, 25, 0);
  member("isect.prim", GATE_OFF(isect.prim), 4, GATE_I32_NEG, 26, 0);
  member("isect.object", GATE_OFF(isect.object), 4, GATE_I32_NEG, 27, 0);
  member("isect.type", GATE_OFF(isect.type), 4, GATE_I32_NEG, 28, 0);

  /* --- volume_stack[32] AoS (state_template.h:112-117; stride 16, object@0,
   *     shader@8 per the static_asserts above). All 32 slots allocated. --- */
  const size_t vs0 = slices.size();
  for (uint32_t i = 0; i < 32; i++) {
    const size_t base = GATE_OFF(volume_stack) + i * sizeof(VolumeStackElem);
    member(string_printf("volume_stack[%u].object", i),
           base + offsetof(VolumeStackElem, object), 4, GATE_UNTOUCHED, 0, 0);
    member(string_printf("volume_stack[%u].shader", i),
           base + offsetof(VolumeStackElem, shader), 4, GATE_UNTOUCHED, 0, 0);
  }
  /* The probe writes object@1, shader@2 and BOTH members of slot 31. */
  slices[vs0 + 1 * 2 + 0].expect = GATE_I32_NEG; /* object[1]  = -gbits(29,1,s) */
  slices[vs0 + 1 * 2 + 0].id = 29;
  slices[vs0 + 1 * 2 + 0].slot = 1;
  slices[vs0 + 2 * 2 + 1].expect = GATE_I32_NEG; /* shader[2]  = -gbits(30,2,s) */
  slices[vs0 + 2 * 2 + 1].id = 30;
  slices[vs0 + 2 * 2 + 1].slot = 2;
  slices[vs0 + 31 * 2 + 0].expect = GATE_I32_NEG; /* object[31] = -gbits(29,31,s) */
  slices[vs0 + 31 * 2 + 0].id = 29;
  slices[vs0 + 31 * 2 + 0].slot = 31;
  slices[vs0 + 31 * 2 + 1].expect = GATE_I32_NEG; /* shader[31] = -gbits(30,31,s) */
  slices[vs0 + 31 * 2 + 1].id = 30;
  slices[vs0 + 31 * 2 + 1].slot = 31;

  /* --- shadow_path (shadow_state_template.h), ids 31..35 --- */
  member("shadow_path.render_pixel_index", GATE_OFF(shadow_path.render_pixel_index),
         4, GATE_U32, 31, 0);
  member("shadow_path.queued_kernel", GATE_OFF(shadow_path.queued_kernel),
         2, GATE_U16, 32, 0);
  member("shadow_path.flag", GATE_OFF(shadow_path.flag), 4, GATE_U32, 33, 0);
  member("shadow_path.lightgroup", GATE_OFF(shadow_path.lightgroup), 1, GATE_U8, 34, 0);
  member("shadow_path.throughput", GATE_OFF(shadow_path.throughput),
         12, GATE_F32X3, 35, 0);

  /* --- shadow_ray (shadow_state_template.h:78-88), ids 36..38 --- */
  member("shadow_ray.P", GATE_OFF(shadow_ray.P), 12, GATE_F32X3, 36, 0);
  member("shadow_ray.D", GATE_OFF(shadow_ray.D), 12, GATE_F32X3, 37, 0);
  member("shadow_ray.tmin", GATE_OFF(shadow_ray.tmin), 4, GATE_U32, 38, 0);

  /* --- shadow_isect[4] AoS (stride 48, six 4-byte members per slot).
   *     The probe writes t[3] and object[2]; all six x four are allocated
   *     because Cycles allocates them all. --- */
  const size_t si0 = slices.size();
  for (uint32_t i = 0; i < 4; i++) {
    const size_t base = GATE_OFF(shadow_isect) + i * sizeof(ShadowIsectElem);
    member(string_printf("shadow_isect[%u].t", i),
           base + offsetof(ShadowIsectElem, t), 4, GATE_UNTOUCHED, 0, 0);
    member(string_printf("shadow_isect[%u].u", i),
           base + offsetof(ShadowIsectElem, u), 4, GATE_UNTOUCHED, 0, 0);
    member(string_printf("shadow_isect[%u].v", i),
           base + offsetof(ShadowIsectElem, v), 4, GATE_UNTOUCHED, 0, 0);
    member(string_printf("shadow_isect[%u].prim", i),
           base + offsetof(ShadowIsectElem, prim), 4, GATE_UNTOUCHED, 0, 0);
    member(string_printf("shadow_isect[%u].object", i),
           base + offsetof(ShadowIsectElem, object), 4, GATE_UNTOUCHED, 0, 0);
    member(string_printf("shadow_isect[%u].type", i),
           base + offsetof(ShadowIsectElem, type), 4, GATE_UNTOUCHED, 0, 0);
  }
  slices[si0 + 3 * 6 + 0].expect = GATE_U32;   /* t[3]      = gbits(39,3,s) */
  slices[si0 + 3 * 6 + 0].id = 39;
  slices[si0 + 3 * 6 + 0].slot = 3;
  slices[si0 + 2 * 6 + 4].expect = GATE_I32_NEG; /* object[2] = -gbits(40,2,s) */
  slices[si0 + 2 * 6 + 4].id = 40;
  slices[si0 + 2 * 6 + 4].slot = 2;

  /* --- shadow_volume_stack[32] AoS (same shape as volume_stack) --- */
  const size_t svs0 = slices.size();
  for (uint32_t i = 0; i < 32; i++) {
    const size_t base = GATE_OFF(shadow_volume_stack) + i * sizeof(ShadowVolumeStackElem);
    member(string_printf("shadow_volume_stack[%u].object", i),
           base + offsetof(ShadowVolumeStackElem, object), 4, GATE_UNTOUCHED, 0, 0);
    member(string_printf("shadow_volume_stack[%u].shader", i),
           base + offsetof(ShadowVolumeStackElem, shader), 4, GATE_UNTOUCHED, 0, 0);
  }
  slices[svs0 + 1 * 2 + 0].expect = GATE_I32_NEG; /* object[1] = -gbits(41,1,s) */
  slices[svs0 + 1 * 2 + 0].id = 41;
  slices[svs0 + 1 * 2 + 0].slot = 1;
  slices[svs0 + 3 * 2 + 1].expect = GATE_I32_NEG; /* shader[3] = -gbits(42,3,s) */
  slices[svs0 + 3 * 2 + 1].id = 42;
  slices[svs0 + 3 * 2 + 1].slot = 3;

#undef GATE_OFF

  /* Payload offsets: every slice = GATE_STATES elements + canary guard. */
  size_t soa_size = 0;
  for (auto &sl : slices) {
    sl.off = soa_size;
    soa_size += (size_t)sl.elem * GATE_STATES + GATE_CANARY_BYTES;
  }

  GateResources res;
  const VkDevice vk_dev = dev->vk_device();
  res.device = vk_dev;

  /* 3. Buffers: host-visible + coherent so the host can init and read back
   *    without a transfer; STORAGE | SHADER_DEVICE_ADDRESS because the shader
   *    reaches them through raw device pointers (same pair the ABI probe uses). */
  auto make_buffer = [&](const size_t size, VkBuffer &buf, VkDeviceMemory &mem,
                         uint64_t &addr, const char *what) -> bool {
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = size;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(vk_dev, &bci, nullptr, &buf) != VK_SUCCESS) {
      LOG_ERROR << "Vulkan: state access gate: " << what << " buffer creation failed.";
      return false;
    }
    VkMemoryRequirements reqs;
    vkGetBufferMemoryRequirements(vk_dev, buf, &reqs);
    VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.pNext = &flags;
    ai.allocationSize = reqs.size;
    ai.memoryTypeIndex = dev->find_memory_type(
        reqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (vkAllocateMemory(vk_dev, &ai, nullptr, &mem) != VK_SUCCESS) {
      LOG_ERROR << "Vulkan: state access gate: " << what << " memory allocation failed.";
      return false;
    }
    if (vkBindBufferMemory(vk_dev, buf, mem, 0) != VK_SUCCESS) {
      LOG_ERROR << "Vulkan: state access gate: " << what << " bind failed.";
      return false;
    }
    VkBufferDeviceAddressInfo bai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    bai.buffer = buf;
    addr = vkGetBufferDeviceAddress(vk_dev, &bai);
    /* No VkResult here: vkGetBufferDeviceAddress signals failure by returning 0. */
    if (addr == 0) {
      LOG_ERROR << "Vulkan: state access gate: " << what
                << " buffer device address query failed (returned 0).";
      return false;
    }
    return true;
  };

  uint64_t soa_addr = 0, ist_addr = 0, out_addr = 0;
  if (!make_buffer(soa_size, res.soa_buffer, res.soa_memory, soa_addr, "SoA")) return false;
  step("buffers allocated");
  if (!make_buffer(sizeof(IntegratorStateGPU), res.ist_buffer, res.ist_memory, ist_addr,
                   "mirror"))
    return false;
  if (!make_buffer(GATE_OUT_WORDS * sizeof(uint32_t), res.out_buffer, res.out_memory,
                   out_addr, "readback"))
    return false;

  /* 4. Fill the mirror: one device address per SoA member (the same memcpy
   *    shape Cycles uses, path_trace_work_gpu.cpp:153-155), then upload it. */
  for (const auto &sl : slices) {
    const uint64_t addr = soa_addr + sl.off;
    memcpy(reinterpret_cast<char *>(&ist) + sl.mirror, &addr, sizeof(addr));
  }
  /* M4e S1b-7: device address of the path.throughput slice, pushed as a uint*
   * word view so negative mode 1 can inject its stride-16 fault with plain
   * uint stores. The shader cannot form that view itself (float3* -> uint*
   * needs a pointer OpBitcast, which crashes SwiftShader's JIT at pipeline
   * creation); the byte pattern it must produce is documented in
   * state_access_probe.slang at mode 1. */
  uint64_t throughput_words_addr = 0;
  for (const auto &sl : slices) {
    if (sl.name == "path.throughput") {
      throughput_words_addr = soa_addr + sl.off;
      break;
    }
  }
  if (throughput_words_addr == 0) {
    LOG_ERROR << "Vulkan: state access gate: path.throughput slice not found.";
    return false;
  }
  {
    void *mapped = nullptr;
    if (vkMapMemory(vk_dev, res.ist_memory, 0, sizeof(IntegratorStateGPU), 0, &mapped) !=
        VK_SUCCESS) {
      LOG_ERROR << "Vulkan: state access gate: mapping mirror buffer failed.";
      return false;
    }
    memcpy(mapped, &ist, sizeof(ist));
    vkUnmapMemory(vk_dev, res.ist_memory);
  }
  step("mirror uploaded");

  /* 5. Shader module + pipeline (entry point is SPIR-V "main"; see the ABI
   *    probe's note at its step 6). No descriptor sets: push constants + BDA. */
  VkPhysicalDeviceProperties props{};
  vkGetPhysicalDeviceProperties(dev->vk_physical_device(), &props);
  if (props.limits.maxPushConstantsSize < GATE_PUSH_BYTES) {
    LOG_ERROR << "Vulkan: state access gate: maxPushConstantsSize too small.";
    return false;
  }
  VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smi.codeSize = spirv.size();
  smi.pCode = reinterpret_cast<const uint32_t *>(spirv.data());
  if (vkCreateShaderModule(vk_dev, &smi, nullptr, &res.module) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: state access gate: vkCreateShaderModule failed.";
    return false;
  }
  step("shader module created");
  VkPushConstantRange pcr{};
  pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pcr.offset = 0;
  pcr.size = GATE_PUSH_BYTES;
  VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pli.pushConstantRangeCount = 1;
  pli.pPushConstantRanges = &pcr;
  if (vkCreatePipelineLayout(vk_dev, &pli, nullptr, &res.layout) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: state access gate: pipeline layout creation failed.";
    return false;
  }
  step("pipeline layout created");
  VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  cpi.stage.module = res.module;
  cpi.stage.pName = "main";
  cpi.layout = res.layout;
  if (vkCreateComputePipelines(vk_dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &res.pipeline) !=
      VK_SUCCESS) {
    LOG_ERROR << "Vulkan: state access gate: compute pipeline creation failed "
                 "(bad entry point name?).";
    return false;
  }
  step("compute pipeline created");

  /* 6. Command pool/buffer + fence, on the device's own family/queue. */
  const uint32_t family = dev->find_queue_family(dev->vk_physical_device());
  VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  cpci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
  cpci.queueFamilyIndex = family;
  if (vkCreateCommandPool(vk_dev, &cpci, nullptr, &res.pool) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: state access gate: command pool creation failed.";
    return false;
  }
  VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbai.commandPool = res.pool;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  if (vkAllocateCommandBuffers(vk_dev, &cbai, &cmd) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: state access gate: command buffer allocation failed.";
    return false;
  }
  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  if (vkCreateFence(vk_dev, &fci, nullptr, &res.fence) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: state access gate: fence creation failed.";
    return false;
  }
  VkQueue queue = VK_NULL_HANDLE;
  vkGetDeviceQueue(vk_dev, family, 0, &queue);
  step("command pool/buffer/fence ready");

  /* 7. Helpers: payload init / readback / byte formatting. */
  auto soa_init = [&]() {
    void *p = nullptr;
    if (vkMapMemory(vk_dev, res.soa_memory, 0, soa_size, 0, &p) != VK_SUCCESS) return false;
    uint8_t *b = static_cast<uint8_t *>(p);
    memset(b, GATE_INIT_BYTE, soa_size); /* payload: "never written" pattern */
    for (const auto &sl : slices) /* canary guard after each payload */
      memset(b + sl.off + (size_t)sl.elem * GATE_STATES, GATE_CANARY_BYTE,
             GATE_CANARY_BYTES);
    vkUnmapMemory(vk_dev, res.soa_memory);
    return true;
  };
  auto soa_read = [&](std::vector<uint8_t> &dst) {
    void *p = nullptr;
    dst.resize(soa_size);
    if (vkMapMemory(vk_dev, res.soa_memory, 0, soa_size, 0, &p) != VK_SUCCESS) return false;
    memcpy(dst.data(), p, soa_size);
    vkUnmapMemory(vk_dev, res.soa_memory);
    return true;
  };
  auto out_read = [&](uint32_t *dst) {
    void *p = nullptr;
    if (vkMapMemory(vk_dev, res.out_memory, 0, GATE_OUT_WORDS * sizeof(uint32_t), 0, &p) !=
        VK_SUCCESS)
      return false;
    memcpy(dst, p, GATE_OUT_WORDS * sizeof(uint32_t));
    vkUnmapMemory(vk_dev, res.out_memory);
    return true;
  };
  auto hex_bytes = [](const uint8_t *b, const size_t n) {
    std::string s;
    char buf[4];
    for (size_t i = 0; i < n; i++) {
      snprintf(buf, sizeof(buf), "%02x", b[i]);
      s += buf;
    }
    return s;
  };

  /* Expected bytes of one slice for state s: mirror of the shader's write. */
  auto expected_bytes = [](const GateSlice &sl, const uint32_t s, uint8_t *out) {
    switch (sl.expect) {
      case GATE_U32: {
        const uint32_t v = gate_gbits(sl.id, sl.slot, s);
        memcpy(out, &v, 4);
        break;
      }
      case GATE_U16: {
        const uint16_t v = (uint16_t)(gate_gbits(sl.id, sl.slot, s) & 0xFFFFu);
        memcpy(out, &v, 2);
        break;
      }
      case GATE_U8: {
        const uint8_t v = (uint8_t)(gate_gbits(sl.id, sl.slot, s) & 0xFFu);
        memcpy(out, &v, 1);
        break;
      }
      case GATE_I32_NEG: {
        const int32_t v = -(int32_t)gate_gbits(sl.id, sl.slot, s);
        memcpy(out, &v, 4);
        break;
      }
      case GATE_F32X3: {
        const uint32_t v0 = gate_gbits(sl.id, sl.slot, s);
        const uint32_t v1 = gate_gbits(sl.id + 64u, sl.slot, s);
        const uint32_t v2 = gate_gbits(sl.id + 128u, sl.slot, s);
        memcpy(out, &v0, 4);
        memcpy(out + 4, &v1, 4);
        memcpy(out + 8, &v2, 4);
        break;
      }
      default:
        memset(out, GATE_INIT_BYTE, sl.elem); /* untouched: init pattern */
        break;
    }
  };

  /* Expected out[s] for mode 0: term-for-term mirror of the shader's `acc`
   * (state_access_probe.slang), i.e. every accessor's READ returns exactly
   * what the preceding WRITE stored. */
  auto expected_acc = [](const uint32_t s) {
    const auto G = [s](const uint32_t id, const uint32_t i) {
      return gate_gbits(id, i, s);
    };
    const auto NEG = [s](const uint32_t id, const uint32_t i) {
      return (uint32_t)(-(int32_t)gate_gbits(id, i, s));
    };
    uint32_t acc = 0;
    acc ^= G(1, 0);
    acc ^= G(2, 0);
    acc ^= G(3, 0) & 0xFFFFu;
    acc ^= G(4, 0) & 0xFFFFu;
    acc ^= G(5, 0) & 0xFFFFu;
    acc ^= G(6, 0) & 0xFFFFu;
    acc ^= G(7, 0) & 0xFFu;
    acc ^= G(8, 0);
    acc ^= G(9, 0) & 0xFFu;
    acc ^= G(10, 0);
    acc ^= G(11, 0);
    acc ^= G(12, 0);
    acc ^= G(13, 0);
    acc ^= G(14, 0);
    acc ^= G(15, 0);
    acc ^= G(16, 0);
    acc ^= G(17, 0);
    acc ^= G(18, 0);
    acc ^= G(19, 0);
    acc ^= G(20, 0);
    acc ^= G(21, 0);
    acc ^= G(22, 0);
    acc ^= G(23, 0);
    acc ^= G(24, 0);
    acc ^= G(25, 0);
    acc ^= NEG(26, 0);
    acc ^= NEG(27, 0);
    acc ^= NEG(28, 0);
    acc ^= NEG(29, 1);
    acc ^= NEG(30, 2);
    acc ^= NEG(29, 31);
    acc ^= G(31, 0);
    acc ^= G(32, 0) & 0xFFFFu;
    acc ^= G(33, 0);
    acc ^= G(34, 0) & 0xFFu;
    acc ^= G(35, 0);
    acc ^= G(36, 0);
    acc ^= G(37, 0);
    acc ^= G(38, 0);
    acc ^= G(39, 3);
    acc ^= NEG(40, 2);
    acc ^= NEG(41, 1);
    acc ^= NEG(42, 3);
    return acc;
  };

  LOG_INFO << "Vulkan: state access gate: " << slices.size() << " SoA slice(s), "
           << soa_size << " bytes (payload + canaries), "
           << sizeof(IntegratorStateGPU) << "-byte mirror at 0x" << probe_hex64(ist_addr);

  /* 8. Mode loop: init payload, dispatch, read back, verify. */
  bool mode0_ok = false;
  bool faults_ok = true;
  std::vector<uint8_t> clean; /* mode-0 payload snapshot (AGENTS.md 11 baseline) */
  uint32_t clean_out[GATE_OUT_WORDS] = {0};

  for (const int mode : modes) {
    if (!soa_init()) {
      LOG_ERROR << "Vulkan: state access gate: payload init failed (mode " << mode << ").";
      return false;
    }

    if (vkResetCommandPool(vk_dev, res.pool, 0) != VK_SUCCESS) {
      LOG_ERROR << "Vulkan: state access gate: vkResetCommandPool failed.";
      return false;
    }
    VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(cmd, &begin_info) != VK_SUCCESS) {
      LOG_ERROR << "Vulkan: state access gate: vkBeginCommandBuffer failed.";
      return false;
    }
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, res.pipeline);
    const uint64_t push[4] = {ist_addr, out_addr, throughput_words_addr,
                              (uint64_t)mode};
    /* Layout: {u64 ist, u64 out, u64 throughput_words, u32 mode, u32 pad};
     * mode occupies bytes 24..27. (M4e S1b-7: third pointer = SwiftShader-safe
     * mode-1 fault; GATE_PUSH_BYTES = 32.) */
    vkCmdPushConstants(cmd, res.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);
    vkCmdDispatch(cmd, 1, 1, 1); /* one workgroup of 8 threads -> states 0..7 */
    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
      LOG_ERROR << "Vulkan: state access gate: vkEndCommandBuffer failed.";
      return false;
    }
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    /* The fence is reused across modes: after the previous wait it is
     * SIGNALED, and submitting a signaled fence violates
     * VUID-vkQueueSubmit-fence-00063 (and makes the next wait return before
     * the work is done). Reset it before every submit. */
    if (vkResetFences(vk_dev, 1, &res.fence) != VK_SUCCESS) {
      LOG_ERROR << "Vulkan: state access gate: vkResetFences failed.";
      return false;
    }
    bool submitted = false;
    if (trace) {
      char m[64];
      snprintf(m, sizeof(m), "submitting mode %u", mode);
      step(m);
    }
    if (vkQueueSubmit(queue, 1, &si, res.fence) == VK_SUCCESS) {
      submitted = vkWaitForFences(vk_dev, 1, &res.fence, VK_TRUE, 10000000000ull) ==
                  VK_SUCCESS;
    }
    if (!submitted) {
      LOG_ERROR << "Vulkan: state access gate: queue submit/wait failed (mode " << mode
                << ").";
      return false;
    }
    if (trace) {
      char m[64];
      snprintf(m, sizeof(m), "fence signaled mode %u", mode);
      step(m);
    }

    std::vector<uint8_t> got;
    uint32_t out[GATE_OUT_WORDS] = {0};
    if (!soa_read(got) || !out_read(out)) {
      LOG_ERROR << "Vulkan: state access gate: readback failed (mode " << mode << ").";
      return false;
    }
    if (trace) {
      char m[64];
      snprintf(m, sizeof(m), "readback done mode %u", mode);
      step(m);
    }

    /* --- verify payload: expected bytes, canaries, and (for faults) the
     *     difference against the clean run, which is what DETECTED means --- */
    uint32_t mismatches = 0, bad_canaries = 0, logged = 0;
    for (const auto &sl : slices) {
      const uint8_t *payload = got.data() + sl.off;
      const uint8_t *canary = payload + (size_t)sl.elem * GATE_STATES;
      for (uint32_t b = 0; b < GATE_CANARY_BYTES; b++) {
        if (canary[b] != GATE_CANARY_BYTE) {
          bad_canaries++;
          if (logged < GATE_MAX_REPORT) {
            LOG_ERROR << "Vulkan: state access gate: mode " << mode << " CANARY clobbered in "
                      << sl.name;
            logged++;
          }
          break;
        }
      }
      for (uint32_t s = 0; s < GATE_STATES; s++) {
        uint8_t exp[12];
        expected_bytes(sl, s, exp);
        const uint8_t *p = payload + (size_t)s * sl.elem;
        if (memcmp(exp, p, sl.elem) != 0) {
          mismatches++;
          if (logged < GATE_MAX_REPORT) {
            LOG_ERROR << "Vulkan: state access gate: mode " << mode << " MISMATCH " << sl.name
                      << "[state " << s << "] expected " << hex_bytes(exp, sl.elem) << " got "
                      << hex_bytes(p, sl.elem);
            logged++;
          }
        }
      }
    }

    uint32_t diff_bytes = 0, diff_out = 0;
    if (mode == 0) {
      bool acc_ok = true;
      for (uint32_t s = 0; s < GATE_STATES; s++) {
        if (out[s] != expected_acc(s)) {
          acc_ok = false;
          LOG_ERROR << "Vulkan: state access gate: mode 0 READBACK out[" << s
                    << "] expected 0x" << probe_hex64(expected_acc(s)) << " got 0x"
                    << probe_hex64(out[s]);
        }
      }
      mode0_ok = (mismatches == 0) && (bad_canaries == 0) && acc_ok;
      LOG_INFO << "Vulkan: state access gate: mode 0 (clean): payload mismatches="
               << mismatches << ", bad canaries=" << bad_canaries << ", readback acc "
               << (acc_ok ? "OK" : "WRONG") << " => " << (mode0_ok ? "PASS" : "FAIL");
      clean = got;
      memcpy(clean_out, out, sizeof(clean_out));
    }
    else {
      for (size_t i = 0; i < got.size(); i++)
        if (got[i] != clean[i]) diff_bytes++;
      for (uint32_t s = 0; s < GATE_STATES; s++)
        if (out[s] != clean_out[s]) diff_out++;
      const bool detected = (diff_bytes > 0) || (diff_out > 0);
      if (!detected) faults_ok = false;
      LOG_INFO << "Vulkan: state access gate: mode " << mode
               << " (fault): payload mismatches=" << mismatches
               << ", bad canaries=" << bad_canaries
               << ", bytes differing from clean run=" << diff_bytes
               << ", out words differing=" << diff_out << " => "
               << (detected ? "DETECTED" : "NOT DETECTED");
    }
  }

  const bool ok = mode0_ok && faults_ok;
  if (ok) {
    LOG_INFO << "Vulkan: state access gate: ALL CHECKS PASSED (mode 0 clean PASS, every "
                "requested fault DETECTED).";
  }
  else {
    LOG_ERROR << "Vulkan: state access gate: FAILED (mode 0 "
              << (mode0_ok ? "PASS" : "FAIL") << ", faults "
              << (faults_ok ? "all DETECTED" : "at least one NOT DETECTED")
              << "). See the lines above; details are per-mode.";
  }
  return ok;
}

/* =========================================================================
 * M4d diffuse-closure probe (optional, env CYCLES_VULKAN_BSDF_PROBE=1)
 * =========================================================================
 * Exercises the just-ported Lambertian closure (bsdf_diffuse.slang) on the
 * GPU with a FIXED hit and FIXED directions, and compares the shader's
 * arithmetic value-by-value against a CPU reference computed from the same
 * Cycles formulas. This is the "shade a single hit with a fixed direction"
 * gate: it checks the BSDF math in isolation. It does NOT traverse the BVH
 * (traversal is already verified by the M4c traverse probe) and does NOT
 * change any rendering logic - it is a pure function check.
 *
 * The CPU reference reimplements the SAME formulas the shader ports (mapping.h
 * / projection.h / math_float3.h / bsdf_diffuse.h), so GPU and CPU are the
 * same algorithm class: any divergence is a real porting bug, not a tolerance
 * question. Fixed inputs are mirrored exactly from bsdf_probe.slang. */

namespace {

/* Number of float words bsdf_probe.slang writes (see its layout comment). */
constexpr uint32_t BSDF_PROBE_OUT_WORDS = 15;
/* BsdfProbePush = one 64-bit device pointer. */
constexpr uint32_t BSDF_PROBE_PUSH_BYTES = 8;

struct BsdfProbeResources {
  VkDevice device = VK_NULL_HANDLE;
  VkShaderModule module = VK_NULL_HANDLE;
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;
  VkCommandPool pool = VK_NULL_HANDLE;
  VkBuffer out_buffer = VK_NULL_HANDLE;
  VkDeviceMemory out_memory = VK_NULL_HANDLE;

  ~BsdfProbeResources()
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

/* ---- CPU reference: same formulas as bsdf_diffuse.slang ---------------- */
static const float kM1PI = 0.3183098861837067f;
static const float kMPI2 = 1.5707963267948966f;
static const float kMPI4 = 0.7853981633974830f;

struct V3 { float x, y, z; };
static V3 v3(float x, float y, float z) { return {x, y, z}; }
static V3 vadd(V3 a, V3 b) { return v3(a.x + b.x, a.y + b.y, a.z + b.z); }
static V3 vsub(V3 a, V3 b) { return v3(a.x - b.x, a.y - b.y, a.z - b.z); }
static V3 vscale(V3 a, float s) { return v3(a.x * s, a.y * s, a.z * s); }
static float vdot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 vcross(V3 a, V3 b)
{
  return v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
static V3 vnorm(V3 a)
{
  float l = std::sqrt(vdot(a, a));
  return vscale(a, 1.0f / l);
}

/* make_orthonormals (math_float3.h:733-759); the #if 0 early-outs are out. */
static void cpu_make_orthonormals(V3 N, V3 &a, V3 &b)
{
  if (N.x != N.y || N.x != N.z)
    a = v3(N.z - N.y, N.x - N.z, N.y - N.x);
  else
    a = v3(N.z - N.y, N.x + N.z, -N.y - N.x);
  a = vnorm(a);
  b = vcross(N, a);
}

/* sample_uniform_disk (mapping.h:21-42). */
static void cpu_sample_uniform_disk(float rx, float ry, float &ox, float &oy)
{
  float a = 2.0f * rx - 1.0f;
  float b = 2.0f * ry - 1.0f;
  if (a == 0.0f && b == 0.0f) { ox = 0.0f; oy = 0.0f; return; }
  float r, phi;
  if (a * a > b * b) { r = a; phi = kMPI4 * (b / a); }
  else { r = b; phi = kMPI2 - kMPI4 * (a / b); }
  ox = r * std::cos(phi);
  oy = r * std::sin(phi);
}

/* sample_cos_hemisphere (mapping.h:71-84). */
static void cpu_sample_cos_hemisphere(V3 N, float rx, float ry, V3 &wo, float &pdf)
{
  float dx, dy;
  cpu_sample_uniform_disk(rx, ry, dx, dy);
  float len2 = dx * dx + dy * dy;
  float costheta = std::sqrt(std::max(1.0f - len2, 0.0f));
  V3 T, B;
  cpu_make_orthonormals(N, T, B);
  wo = vadd(vadd(vscale(T, dx), vscale(B, dy)), vscale(N, costheta));
  pdf = costheta * kM1PI;
}

} /* namespace */

bool vulkan_bsdf_probe(VulkanDevice *dev)
{
  static bool ran = false;
  if (ran) return true;
  ran = true;
  if (getenv("CYCLES_VULKAN_BSDF_PROBE") == nullptr) return true;

  LOG_INFO << "Vulkan: BSDF probe starting (CYCLES_VULKAN_BSDF_PROBE=1).";

  BsdfProbeResources res;
  const VkDevice vk_dev = dev->vk_device();
  res.device = vk_dev;

  /* 1. SPIR-V located via env override or repo-relative default. */
  std::vector<char> spirv;
  const char *env_path = getenv("CYCLES_VULKAN_BSDF_PROBE_SPV");
  const char *candidates[2] = {env_path, "bsdf_probe.spv"};
  const char *found = nullptr;
  for (const char *c : candidates) {
    if (c && probe_read_whole_file(c, spirv)) { found = c; break; }
  }
  if (!found) {
    LOG_ERROR << "Vulkan: BSDF probe: cannot find bsdf_probe.spv "
                 "(compile it with slangc or set CYCLES_VULKAN_BSDF_PROBE_SPV).";
    return false;
  }
  if (spirv.size() % 4 != 0) {
    LOG_ERROR << "Vulkan: BSDF probe: '" << found << "' is not a valid SPIR-V module.";
    return false;
  }
  LOG_INFO << "Vulkan: BSDF probe: spirv='" << found << "' (" << spirv.size() << " bytes).";

  /* 2. Shader module. */
  VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smi.codeSize = spirv.size();
  smi.pCode = reinterpret_cast<const uint32_t *>(spirv.data());
  if (vkCreateShaderModule(vk_dev, &smi, nullptr, &res.module) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: BSDF probe: vkCreateShaderModule failed.";
    return false;
  }

  /* 3. Pipeline layout: no descriptor sets; one push-constant range (the
   *    output buffer device address). */
  VkPushConstantRange pcr{};
  pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pcr.offset = 0;
  pcr.size = BSDF_PROBE_PUSH_BYTES;
  VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pli.pushConstantRangeCount = 1;
  pli.pPushConstantRanges = &pcr;
  if (vkCreatePipelineLayout(vk_dev, &pli, nullptr, &res.layout) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: BSDF probe: pipeline layout creation failed.";
    return false;
  }

  /* 4. Compute pipeline. slangc emits SPIR-V entry point "main" regardless of
   *    the Slang function name (observed in the M4b ABI probe; see its note). */
  VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  cpi.stage.module = res.module;
  cpi.stage.pName = "main";
  cpi.layout = res.layout;
  if (vkCreateComputePipelines(vk_dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &res.pipeline) !=
      VK_SUCCESS) {
    LOG_ERROR << "Vulkan: BSDF probe: compute pipeline creation failed "
                 "(bad entry point name?).";
    return false;
  }

  /* 5. Readback buffer: host-visible + coherent, STORAGE | SHADER_DEVICE_ADDRESS
   *    (the shader writes through a raw device pointer). */
  VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bci.size = BSDF_PROBE_OUT_WORDS * sizeof(float);
  bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  if (vkCreateBuffer(vk_dev, &bci, nullptr, &res.out_buffer) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: BSDF probe: readback buffer creation failed.";
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
    LOG_ERROR << "Vulkan: BSDF probe: readback memory allocation failed.";
    return false;
  }
  if (vkBindBufferMemory(vk_dev, res.out_buffer, res.out_memory, 0) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: BSDF probe: readback memory bind failed.";
    return false;
  }
  VkBufferDeviceAddressInfo bai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
  bai.buffer = res.out_buffer;
  const uint64_t out_addr = vkGetBufferDeviceAddress(vk_dev, &bai);

  /* 6. Record and submit: our own command pool on the device's queue family. */
  const uint32_t family = dev->find_queue_family(dev->vk_physical_device());
  VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  cpci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
  cpci.queueFamilyIndex = family;
  if (vkCreateCommandPool(vk_dev, &cpci, nullptr, &res.pool) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: BSDF probe: command pool creation failed.";
    return false;
  }
  VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbai.commandPool = res.pool;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  if (vkAllocateCommandBuffers(vk_dev, &cbai, &cmd) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: BSDF probe: command buffer allocation failed.";
    return false;
  }
  VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd, &begin_info);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, res.pipeline);
  const uint64_t push_addr = out_addr;
  vkCmdPushConstants(cmd, res.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_addr),
                     &push_addr);
  vkCmdDispatch(cmd, 1, 1, 1); /* one workgroup; the shader runs one thread */
  vkEndCommandBuffer(cmd);

  VkQueue queue = VK_NULL_HANDLE;
  vkGetDeviceQueue(vk_dev, family, 0, &queue);
  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkFence fence = VK_NULL_HANDLE;
  if (vkCreateFence(vk_dev, &fci, nullptr, &fence) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: BSDF probe: fence creation failed.";
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
    LOG_ERROR << "Vulkan: BSDF probe: queue submit/wait failed.";
    return false;
  }

  /* 7. Read back and compare against the CPU reference. The buffer holds
   *    float words; the sentinel at [0] is a bit pattern, so reinterpret it. */
  void *mapped = nullptr;
  if (vkMapMemory(vk_dev, res.out_memory, 0, bci.size, 0, &mapped) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: BSDF probe: mapping readback buffer failed.";
    return false;
  }
  float out[BSDF_PROBE_OUT_WORDS];
  memcpy(out, mapped, sizeof(out));
  vkUnmapMemory(vk_dev, res.out_memory);

  uint32_t sent;
  memcpy(&sent, &out[0], 4);
  if (sent != 0xB5DF0001u) {
    LOG_ERROR << "Vulkan: BSDF probe: sentinel missing (dispatch did not run).";
    return false;
  }

  uint32_t mismatches = 0;
  auto check = [&mismatches](const char *name, float got, float want, float tol) {
    bool ok = std::fabs(got - want) <= tol;
    if (!ok) mismatches++;
    LOG_INFO << "Vulkan: BSDF probe: " << (ok ? "MATCH   " : "MISMATCH") << " " << name
             << " gpu=" << got << " cpu=" << want;
  };

  /* CPU reference from the SAME fixed inputs as bsdf_probe.slang. */
  const V3 N = v3(0.0f, 0.0f, 1.0f);
  const V3 wo_fixed = vnorm(v3(0.3f, -0.2f, 0.9f));
  const float r0 = 0.31f, r1 = 0.62f;

  /* bsdf_diffuse_eval: cosNO = max(dot(N,wo),0)/pi. */
  float cosNO = std::max(vdot(N, wo_fixed), 0.0f) * kM1PI;
  check("eval.r", out[1], cosNO, 1e-6f);
  check("eval.g", out[2], cosNO, 1e-6f);
  check("eval.b", out[3], cosNO, 1e-6f);
  check("pdf_eval", out[4], cosNO, 1e-6f);

  /* bsdf_diffuse_sample. */
  V3 wo_s; float pdf_s;
  cpu_sample_cos_hemisphere(N, r0, r1, wo_s, pdf_s);
  float eval_s = pdf_s; /* dot(Ng,wo)>0 for N=Ng=+Z and an upper-hemi sample */
  if (!(vdot(N, wo_s) > 0.0f)) { pdf_s = 0.0f; eval_s = 0.0f; }
  check("wo.x", out[5], wo_s.x, 1e-5f);
  check("wo.y", out[6], wo_s.y, 1e-5f);
  check("wo.z", out[7], wo_s.z, 1e-5f);
  check("pdf_sample", out[8], pdf_s, 1e-6f);
  check("eval_s.r", out[9], eval_s, 1e-6f);
  check("label", out[12], 6.0f /* LABEL_REFLECT|LABEL_DIFFUSE */, 0.0f);

  /* ---- SECOND reference: the REAL Cycles closure, compiled on the host ----
   * The checks above compare the GPU against a hand reimplementation of the
   * formulas. If I misread a formula, BOTH the Slang port and that
   * reimplementation would be wrong the same way and still agree. To catch
   * that, call the actual Cycles functions (kernel/closure/bsdf_diffuse.h,
   * included at the top of this file) on the same fixed inputs and compare the
   * GPU output against THEM. This is independent of my reading of the source:
   * it is Cycles' own arithmetic. */
  {
    /* Build a ShaderClosure with only N set: bsdf_diffuse_eval/_sample read
     * nothing but bsdf->N (bsdf_diffuse.h:44,60). On host ccl_private is
     * empty, so these are plain pointers. */
    ShaderClosure sc = {};
    sc.N = make_float3(0.0f, 0.0f, 1.0f);

    const float3 wi = make_float3(0.0f, 0.0f, 1.0f); /* unused by diffuse */
    const float3 wo_cy = normalize(make_float3(0.3f, -0.2f, 0.9f));
    const float2 rand_cy = make_float2(0.31f, 0.62f);
    const float3 Ng_cy = make_float3(0.0f, 0.0f, 1.0f);

    /* bsdf_diffuse_eval (real Cycles). */
    float pdf_cy_eval = 0.0f;
    const float3 eval_cy = bsdf_diffuse_eval(&sc, wi, wo_cy, &pdf_cy_eval);
    check("cyc_eval.r", out[1], GET_SPECTRUM_CHANNEL(eval_cy, 0), 1e-6f);
    check("cyc_eval.g", out[2], GET_SPECTRUM_CHANNEL(eval_cy, 1), 1e-6f);
    check("cyc_eval.b", out[3], GET_SPECTRUM_CHANNEL(eval_cy, 2), 1e-6f);
    check("cyc_pdf_eval", out[4], pdf_cy_eval, 1e-6f);

    /* bsdf_diffuse_sample (real Cycles). */
    float3 wo_cy_s = make_float3(0.0f, 0.0f, 0.0f);
    float3 eval_cy_s = make_float3(0.0f, 0.0f, 0.0f);
    float pdf_cy_s = 0.0f;
    const int label_cy = bsdf_diffuse_sample(
        &sc, Ng_cy, wi, rand_cy, &eval_cy_s, &wo_cy_s, &pdf_cy_s);
    check("cyc_wo.x", out[5], wo_cy_s.x, 1e-6f);
    check("cyc_wo.y", out[6], wo_cy_s.y, 1e-6f);
    check("cyc_wo.z", out[7], wo_cy_s.z, 1e-6f);
    check("cyc_pdf_sample", out[8], pdf_cy_s, 1e-6f);
    check("cyc_eval_s.r", out[9], GET_SPECTRUM_CHANNEL(eval_cy_s, 0), 1e-6f);
    check("cyc_label", out[12], (float)label_cy, 0.0f);
  }

  if (mismatches == 0) {
    LOG_INFO << "Vulkan: BSDF probe: ALL VALUES MATCH.";
  }
  else {
    LOG_ERROR << "Vulkan: BSDF probe: " << mismatches
              << " MISMATCH value(s); the ported diffuse closure disagrees with the CPU "
                 "reference.";
  }
  return mismatches == 0;
}

/* =========================================================================
 * M4e STEP 1a math probe (optional, env CYCLES_VULKAN_MATH_PROBE=1)
 * =========================================================================
 * Exercises the just-ported sampler / hash / camera MATH CORE (math_core.slang
 * via math_probe.slang) on the GPU with MANY threads, and compares every output
 * against the REAL Cycles headers compiled on the host (kernel/sample/pattern.h
 * path_rng_*, kernel/camera/camera.h camera_sample, util/hash.h hash_iqnt2d,
 * kernel/sample/sobol_burley.h). This is the "sampler + camera math" gate in
 * isolation: no integrator state, no queue, no traversal. The camera matrices
 * and sampling config come from the REAL kernel data of the loaded scene
 * (dev->host_kernel_data()), NOT the KernelParamsVulkan blob.
 *
 * Coverage: pixels over a 64x48 tile, samples {0,1,5,15,255,1023}, seeds
 * {0,1,large} (integrator.seed varied across clean passes), dimensions
 * low..high. Integers (hash_iqnt2d, path_rng_pixel_init) compared BIT-EXACT;
 * floats (sobol / path_rng / camera P,D,tmin,tmax,time) compared with
 * |a-b| <= 1e-5*max(1,|b|) (gate = MAXIMUM normalized error; 99.9th percentile
 * and mean printed for information only). Sobol floats are additionally
 * bit-compared (informational) since uint_to_float_excl is a single FMA-free
 * multiply, so a bit-identical float implies a bit-identical sobol integer.
 *
 * Three fault injections (negative tests) MUST be detected:
 *   neg_mode 1 -> perturb the first dimension-1 sobol table read,
 *   neg_mode 2 -> wrong seed XOR in path_rng_pixel_init,
 *   neg_mode 3 -> transposed rastertocamera.
 * The clean pass (neg_mode 0) must MATCH for all three seeds. */

namespace {

/* Case kinds (must match math_probe.slang). */
enum MathProbeKind : uint32_t {
  MATHK_HASH = 0,
  MATHK_PIXEL_INIT = 1,
  MATHK_SOBOL1D = 2,
  MATHK_SOBOL2D = 3,
  MATHK_SOBOL3D = 4,
  MATHK_SOBOL4D = 5,
  MATHK_RNG1D = 6,
  MATHK_RNG2D = 7,
  MATHK_RNG3D = 8,
  MATHK_RNG4D = 9,
  MATHK_CAMERA = 10,
  MATHK_COUNT = 11
};

/* Host mirrors of the Slang probe structs. All members are 4-byte scalars so
 * the layouts are identical in Slang and C++ (no alignment padding to disagree
 * on). Locked with static_asserts below. */
struct MathProbeCase {
  uint32_t kind;
  int32_t sample;
  uint32_t x, y;
  uint32_t dim, seed, mask, index;
  float fx, fy, fz, fw;
};
static_assert(sizeof(MathProbeCase) == 48, "ProbeCase layout must match Slang");

struct MathProbeResult {
  uint32_t u0, has_u, nfloat, pad;
  float f[12];
};
static_assert(sizeof(MathProbeResult) == 64, "ProbeResult layout must match Slang");

struct MathProbeParams {
  int32_t cam_type, have_perspective_motion;
  float aperturesize;
  int32_t num_motion_steps;
  float interocular_offset, shuttertime;
  int32_t rolling_shutter_type;
  float nearclip, cliplength;
  float pixel_jitter_x, pixel_jitter_y;
  float ctw_x_x, ctw_x_y, ctw_x_z, ctw_x_w;
  float ctw_y_x, ctw_y_y, ctw_y_z, ctw_y_w;
  float ctw_z_x, ctw_z_y, ctw_z_z, ctw_z_w;
  float rtc_x_x, rtc_x_y, rtc_x_z, rtc_x_w;
  float rtc_y_x, rtc_y_y, rtc_y_z, rtc_y_w;
  float rtc_z_x, rtc_z_y, rtc_z_z, rtc_z_w;
  float rtc_w_x, rtc_w_y, rtc_w_z, rtc_w_w;
  int32_t sampling_pattern;
  uint32_t seed, sobol_index_mask, blue_noise_sequence_length;
  int32_t neg_mode;
};
static_assert(sizeof(MathProbeParams) == 176, "MathParams layout must match Slang");

/* Push block: 3 device pointers + count = 32 bytes (matches Slang Push). */
struct MathProbePush {
  VkDeviceAddress cases;
  VkDeviceAddress out;
  VkDeviceAddress params;
  uint32_t num_cases;
};
static_assert(sizeof(MathProbePush) == 32, "Push layout must match Slang");

constexpr uint32_t MATH_PROBE_PUSH_BYTES = 32;
constexpr uint32_t MATH_PROBE_CANARY_WORDS = 4;
constexpr uint32_t MATH_PROBE_CANARY = 0xCAFEBABEu;

/* RAII for every Vulkan handle the probe creates (same pattern as the M4d
 * probe), so early-return error paths still destroy everything. */
struct MathProbeResources {
  VkDevice device = VK_NULL_HANDLE;
  VkShaderModule module = VK_NULL_HANDLE;
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;
  VkCommandPool pool = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;
  VkBuffer cases_buffer = VK_NULL_HANDLE, out_buffer = VK_NULL_HANDLE,
           params_buffer = VK_NULL_HANDLE;
  VkDeviceMemory cases_memory = VK_NULL_HANDLE, out_memory = VK_NULL_HANDLE,
                  params_memory = VK_NULL_HANDLE;

  ~MathProbeResources()
  {
    if (device == VK_NULL_HANDLE) return;
    if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
    if (layout) vkDestroyPipelineLayout(device, layout, nullptr);
    if (module) vkDestroyShaderModule(device, module, nullptr);
    if (pool) vkDestroyCommandPool(device, pool, nullptr);
    if (fence) vkDestroyFence(device, fence, nullptr);
    if (cases_buffer) vkDestroyBuffer(device, cases_buffer, nullptr);
    if (out_buffer) vkDestroyBuffer(device, out_buffer, nullptr);
    if (params_buffer) vkDestroyBuffer(device, params_buffer, nullptr);
    if (cases_memory) vkFreeMemory(device, cases_memory, nullptr);
    if (out_memory) vkFreeMemory(device, out_memory, nullptr);
    if (params_memory) vkFreeMemory(device, params_memory, nullptr);
  }
};

} /* namespace */

bool vulkan_math_probe(VulkanDevice *dev)
{
  static bool ran = false;
  if (ran) return true;
  ran = true;
  if (getenv("CYCLES_VULKAN_MATH_PROBE") == nullptr) return true;

  LOG_INFO << "Vulkan: Math probe starting (CYCLES_VULKAN_MATH_PROBE=1).";

  MathProbeResources res;
  const VkDevice vk_dev = dev->vk_device();
  res.device = vk_dev;

  /* ---- 1. SPIR-V (env override or repo-relative default) ---- */
  std::vector<char> spirv;
  const char *env_path = getenv("CYCLES_VULKAN_MATH_PROBE_SPV");
  const char *candidates[2] = {env_path, "math_probe.spv"};
  const char *found = nullptr;
  for (const char *c : candidates) {
    if (c && probe_read_whole_file(c, spirv)) { found = c; break; }
  }
  if (!found) {
    LOG_ERROR << "Vulkan: Math probe: cannot find math_probe.spv "
                 "(compile it with slangc or set CYCLES_VULKAN_MATH_PROBE_SPV).";
    return false;
  }
  if (spirv.size() % 4 != 0) {
    LOG_ERROR << "Vulkan: Math probe: '" << found << "' is not a valid SPIR-V module.";
    return false;
  }
  LOG_INFO << "Vulkan: Math probe: spirv='" << found << "' (" << spirv.size() << " bytes).";

  /* ---- 2. Shader module ---- */
  VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smi.codeSize = spirv.size();
  smi.pCode = reinterpret_cast<const uint32_t *>(spirv.data());
  if (vkCreateShaderModule(vk_dev, &smi, nullptr, &res.module) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: Math probe: vkCreateShaderModule failed.";
    return false;
  }

  /* ---- 3. Pipeline layout: no descriptor sets, one push-constant range ---- */
  VkPushConstantRange pcr{};
  pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pcr.offset = 0;
  pcr.size = MATH_PROBE_PUSH_BYTES;
  VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pli.pushConstantRangeCount = 1;
  pli.pPushConstantRanges = &pcr;
  if (vkCreatePipelineLayout(vk_dev, &pli, nullptr, &res.layout) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: Math probe: pipeline layout creation failed.";
    return false;
  }

  /* ---- 4. Compute pipeline (entry name "main"; slangc renames) ---- */
  VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  cpi.stage.module = res.module;
  cpi.stage.pName = "main";
  cpi.layout = res.layout;
  if (vkCreateComputePipelines(vk_dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &res.pipeline) !=
      VK_SUCCESS) {
    LOG_ERROR << "Vulkan: Math probe: compute pipeline creation failed "
                 "(bad entry point name?).";
    return false;
  }

  /* ---- 5. Build MathParams from the REAL kernel data of the scene ---- */
  const KernelData &kd = dev->host_kernel_data();
  MathProbeParams mp{};
  mp.cam_type = kd.cam.type;
  mp.have_perspective_motion = kd.cam.have_perspective_motion;
  mp.aperturesize = kd.cam.aperturesize;
  mp.num_motion_steps = kd.cam.num_motion_steps;
  mp.interocular_offset = kd.cam.interocular_offset;
  mp.shuttertime = kd.cam.shuttertime;
  mp.rolling_shutter_type = kd.cam.rolling_shutter_type;
  mp.nearclip = kd.cam.nearclip;
  mp.cliplength = kd.cam.cliplength;
  /* NOTE: pixel_jitter is OVERRIDDEN to a non-FLT_MAX value so camera_sample
   * takes the no-table branch (raster += -pixel_jitter). The FLT_MAX
   * (Blackman-Harris filter-table) branch needs kernel_data.tables.lookup_table,
   * which is NOT passed to the shader in S1a; see the report. Both the GPU and
   * the host reference use this SAME overridden value, so the gate is still a
   * like-for-like check of camera_sample. */
  mp.pixel_jitter_x = 0.25f;
  mp.pixel_jitter_y = -0.5f;
  mp.ctw_x_x = kd.cam.cameratoworld.x.x; mp.ctw_x_y = kd.cam.cameratoworld.x.y;
  mp.ctw_x_z = kd.cam.cameratoworld.x.z; mp.ctw_x_w = kd.cam.cameratoworld.x.w;
  mp.ctw_y_x = kd.cam.cameratoworld.y.x; mp.ctw_y_y = kd.cam.cameratoworld.y.y;
  mp.ctw_y_z = kd.cam.cameratoworld.y.z; mp.ctw_y_w = kd.cam.cameratoworld.y.w;
  mp.ctw_z_x = kd.cam.cameratoworld.z.x; mp.ctw_z_y = kd.cam.cameratoworld.z.y;
  mp.ctw_z_z = kd.cam.cameratoworld.z.z; mp.ctw_z_w = kd.cam.cameratoworld.z.w;
  mp.rtc_x_x = kd.cam.rastertocamera.x.x; mp.rtc_x_y = kd.cam.rastertocamera.x.y;
  mp.rtc_x_z = kd.cam.rastertocamera.x.z; mp.rtc_x_w = kd.cam.rastertocamera.x.w;
  mp.rtc_y_x = kd.cam.rastertocamera.y.x; mp.rtc_y_y = kd.cam.rastertocamera.y.y;
  mp.rtc_y_z = kd.cam.rastertocamera.y.z; mp.rtc_y_w = kd.cam.rastertocamera.y.w;
  mp.rtc_z_x = kd.cam.rastertocamera.z.x; mp.rtc_z_y = kd.cam.rastertocamera.z.y;
  mp.rtc_z_z = kd.cam.rastertocamera.z.z; mp.rtc_z_w = kd.cam.rastertocamera.z.w;
  mp.rtc_w_x = kd.cam.rastertocamera.w.x; mp.rtc_w_y = kd.cam.rastertocamera.w.y;
  mp.rtc_w_z = kd.cam.rastertocamera.w.z; mp.rtc_w_w = kd.cam.rastertocamera.w.w;
  /* sampling_pattern is OVERRIDDEN to SOBOL_BURLEY. instance2.xml's real
   * sampling_pattern is TABULATED_SOBOL (=1, confirmed by the probe log), which
   * path_rng implements via tabulated_sobol_sample_* reading
   * kernel_data.tables.sobol_rng_sample_order — a table NOT passed to the
   * shader in S1a and explicitly out of the S1a scope (the task scopes the
   * SOBOL_BURLEY branch of path_rng). Overriding to SOBOL_BURLEY in BOTH the
   * GPU params and the host reference exercises the in-scope SOBOL_BURLEY
   * branch of path_rng, like-for-like. The TABULATED_SOBOL branch is NOT
   * covered (see the report). */
  mp.sampling_pattern = SAMPLING_PATTERN_SOBOL_BURLEY;
  mp.sobol_index_mask = (uint32_t)kd.integrator.sobol_index_mask;
  mp.blue_noise_sequence_length = (uint32_t)kd.integrator.blue_noise_sequence_length;
  mp.seed = 0; /* set per pass */

  LOG_INFO << "Vulkan: Math probe: cam.type=" << mp.cam_type
           << " shuttertime=" << mp.shuttertime << " aperturesize=" << mp.aperturesize
           << " nearclip=" << mp.nearclip << " cliplength=" << mp.cliplength
           << " sampling_pattern=" << mp.sampling_pattern
           << " sobol_index_mask=" << (int)mp.sobol_index_mask
           << " have_persp_motion=" << mp.have_perspective_motion
           << " num_motion_steps=" << mp.num_motion_steps
           << " interocular=" << mp.interocular_offset;

  if (mp.cam_type != 0 /* CAMERA_PERSPECTIVE */ || mp.shuttertime != -1.0f ||
      mp.aperturesize != 0.0f || mp.have_perspective_motion != 0 ||
      mp.num_motion_steps != 0 || mp.interocular_offset != 0.0f) {
    LOG_WARNING << "Vulkan: Math probe: scene camera is outside the S1a covered config "
                   "(perspective, aperture 0, shuttertime -1, no motion/stereo); the "
                   "camera cases will report as unsupported and the gate is not met.";
  }

  /* ---- 6. Build the host reference globals (REAL Cycles functions) ---- */
  /* KernelGlobalsCPU holds KernelData + the kernel_array data pointers. We fill
   * `data` from the real kernel data (with the same pixel_jitter override) and
   * build a ThreadKernelGlobalsCPU from it. lookup_table is left empty (unused:
   * pixel_jitter != FLT_MAX skips the filter-table read; shuttertime == -1
   * skips the shutter-table read). OSL globals are null (safe: OSLThreadData
   * returns early on null, kernel/osl/globals.cpp:17). Path guiding is off. */
  KernelGlobalsCPU kg_cpu;
  kg_cpu.data = kd;
  kg_cpu.data.integrator.pixel_jitter = make_float2(mp.pixel_jitter_x, mp.pixel_jitter_y);
  /* Match the GPU override: SOBOL_BURLEY (the scene's real pattern is
   * TABULATED_SOBOL, whose table is not passed to the shader in S1a; see the
   * MathParams comment). Both sides must use the same pattern for a
   * like-for-like check of the in-scope SOBOL_BURLEY path_rng branch. */
  kg_cpu.data.integrator.sampling_pattern = SAMPLING_PATTERN_SOBOL_BURLEY;
  Profiler math_probe_profiler;
  ThreadKernelGlobalsCPU tkg(kg_cpu, nullptr, math_probe_profiler, 0);
  KernelGlobals kg = &tkg;

  /* ---- 7. Build the case list ---- */
  std::vector<MathProbeCase> cases;
  const uint32_t samples[6] = {0, 1, 5, 15, 255, 1023};
  const uint32_t dims[8] = {0, 1, 2, 3, 8, 32, 200, 1023};
  const uint32_t sobol_seeds[3] = {0u, 1u, 0xDEADBEEFu};
  const uint32_t sobol_masks[2] = {0xFFFFFFFFu, (uint32_t)kd.integrator.sobol_index_mask};
  const uint32_t sobol_indices[8] = {0u, 1u, 2u, 5u, 255u, 1023u, 0xFFFFu, 0x12345u};
  const uint32_t sobol_dimsets[4] = {0u, 1u, 5u, 31u};

  auto add = [&cases](uint32_t kind, int32_t sample, uint32_t x, uint32_t y, uint32_t dim,
                      uint32_t seed, uint32_t mask, uint32_t index, float fx, float fy,
                      float fz) {
    MathProbeCase c{};
    c.kind = kind; c.sample = sample; c.x = x; c.y = y;
    c.dim = dim; c.seed = seed; c.mask = mask; c.index = index;
    c.fx = fx; c.fy = fy; c.fz = fz; c.fw = 0.0f;
    cases.push_back(c);
  };

  /* Pixel grid over the 64x48 tile (stride 2 -> 32x24 = 768 pixels). */
  std::vector<std::pair<int, int>> pix;
  for (int y = 0; y < 48; y += 2)
    for (int x = 0; x < 64; x += 2)
      pix.emplace_back(x, y);

  /* hash_iqnt2d: one case per pixel (bit-exact integer). */
  for (auto &p : pix)
    add(MATHK_HASH, 0, (uint32_t)p.first, (uint32_t)p.second, 0, 0, 0, 0, 0, 0, 0);
  /* path_rng_pixel_init: one case per pixel (bit-exact integer; seed axis is
   * covered by the 3 clean passes varying integrator.seed). */
  for (auto &p : pix)
    add(MATHK_PIXEL_INIT, 0, (uint32_t)p.first, (uint32_t)p.second, 0, 0, 0, 0, 0, 0, 0);
  /* camera_sample: one case per pixel (ray P/D/tmin/tmax/time). filter_uv/time
   * are ignored in the covered config but passed for signature fidelity. */
  for (auto &p : pix)
    add(MATHK_CAMERA, 0, (uint32_t)p.first, (uint32_t)p.second, 0, 0, 0, 0, 0.5f, 0.5f, 0.5f);
  /* sobol_burley_sample_1D/2D/3D/4D: cross(index, dim_set, seed, mask). */
  for (uint32_t s = 1; s <= 4; s++)
    for (uint32_t ii = 0; ii < 8; ii++)
      for (uint32_t dd = 0; dd < 4; dd++)
        for (uint32_t ss = 0; ss < 3; ss++)
          for (uint32_t mm = 0; mm < 2; mm++)
            add(MATHK_SOBOL1D + (s - 1), 0, 0, 0, sobol_dimsets[dd], sobol_seeds[ss],
                sobol_masks[mm], sobol_indices[ii], 0, 0, 0);
  /* path_rng_1D/2D/3D/4D: cross(a few pixels, sample, dim). rng_pixel is
   * derived inside the shader/reference from (x,y,seed). */
  const uint32_t rng_pix[6] = {0, 1, 2, 32, 33, 767};
  for (uint32_t s = 6; s <= 9; s++)
    for (uint32_t pi = 0; pi < 6; pi++) {
      uint32_t xy = rng_pix[pi];
      uint32_t px = (xy % 32) * 2, py = (xy / 32) * 2;
      for (uint32_t si = 0; si < 6; si++)
        for (uint32_t dd = 0; dd < 8; dd++)
          add(MATHK_RNG1D + (s - 6), (int32_t)samples[si], px, py, dims[dd], 0, 0, 0, 0, 0, 0);
    }

  const uint32_t num_cases = (uint32_t)cases.size();
  LOG_INFO << "Vulkan: Math probe: " << num_cases << " cases, "
           << (num_cases + 63) / 64 << " workgroups of 64 threads.";

  /* ---- 8. Allocate the three host-visible + device-address buffers ---- */
  auto make_buffer = [&](VkDeviceSize size, VkBuffer &buf, VkDeviceMemory &mem,
                         uint64_t &addr) -> bool {
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = size;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(vk_dev, &bci, nullptr, &buf) != VK_SUCCESS) {
      LOG_ERROR << "Vulkan: Math probe: buffer creation failed.";
      return false;
    }
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
    if (ai.memoryTypeIndex == UINT32_MAX) {
      LOG_ERROR << "Vulkan: Math probe: no host-visible+coherent memory type.";
      return false;
    }
    if (vkAllocateMemory(vk_dev, &ai, nullptr, &mem) != VK_SUCCESS) {
      LOG_ERROR << "Vulkan: Math probe: memory allocation failed.";
      return false;
    }
    if (vkBindBufferMemory(vk_dev, buf, mem, 0) != VK_SUCCESS) {
      LOG_ERROR << "Vulkan: Math probe: memory bind failed.";
      return false;
    }
    VkBufferDeviceAddressInfo bai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    bai.buffer = buf;
    addr = vkGetBufferDeviceAddress(vk_dev, &bai);
    return true;
  };

  uint64_t cases_addr = 0, out_addr = 0, params_addr = 0;
  const VkDeviceSize cases_bytes = (VkDeviceSize)num_cases * sizeof(MathProbeCase);
  const VkDeviceSize out_bytes =
      (VkDeviceSize)num_cases * sizeof(MathProbeResult) +
      (VkDeviceSize)MATH_PROBE_CANARY_WORDS * sizeof(uint32_t);
  const VkDeviceSize params_bytes = sizeof(MathProbeParams);
  if (!make_buffer(cases_bytes, res.cases_buffer, res.cases_memory, cases_addr) ||
      !make_buffer(out_bytes, res.out_buffer, res.out_memory, out_addr) ||
      !make_buffer(params_bytes, res.params_buffer, res.params_memory, params_addr)) {
    return false;
  }

  /* ---- 9. Upload the case list (write once) ---- */
  {
    void *m = nullptr;
    if (vkMapMemory(vk_dev, res.cases_memory, 0, cases_bytes, 0, &m) != VK_SUCCESS) {
      LOG_ERROR << "Vulkan: Math probe: mapping cases buffer failed.";
      return false;
    }
    memcpy(m, cases.data(), (size_t)cases_bytes);
    vkUnmapMemory(vk_dev, res.cases_memory);
  }
  /* Pre-fill the output canary region so we can detect an overrun. */
  {
    void *m = nullptr;
    if (vkMapMemory(vk_dev, res.out_memory, 0, out_bytes, 0, &m) != VK_SUCCESS) {
      LOG_ERROR << "Vulkan: Math probe: mapping out buffer (canary) failed.";
      return false;
    }
    uint32_t *canary = reinterpret_cast<uint32_t *>(static_cast<uint8_t *>(m) +
                                                    (size_t)num_cases * sizeof(MathProbeResult));
    for (uint32_t i = 0; i < MATH_PROBE_CANARY_WORDS; i++)
      canary[i] = MATH_PROBE_CANARY;
    vkUnmapMemory(vk_dev, res.out_memory);
  }

  /* ---- 10. Command pool + buffer ---- */
  const uint32_t family = dev->find_queue_family(dev->vk_physical_device());
  VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  cpci.queueFamilyIndex = family;
  if (vkCreateCommandPool(vk_dev, &cpci, nullptr, &res.pool) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: Math probe: command pool creation failed.";
    return false;
  }
  VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbai.commandPool = res.pool;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  if (vkAllocateCommandBuffers(vk_dev, &cbai, &cmd) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: Math probe: command buffer allocation failed.";
    return false;
  }
  VkQueue queue = VK_NULL_HANDLE;
  vkGetDeviceQueue(vk_dev, family, 0, &queue);
  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  if (vkCreateFence(vk_dev, &fci, nullptr, &res.fence) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: Math probe: fence creation failed.";
    return false;
  }

  /* ---- 11. Per-pass runner: update seed/neg_mode, dispatch, read back ---- */
  std::vector<MathProbeResult> results(num_cases);
  auto run_pass = [&](uint32_t seed, int neg_mode) -> bool {
    /* Update params (seed + fault switch) on the GPU buffer. */
    mp.seed = seed;
    mp.neg_mode = neg_mode;
    {
      void *m = nullptr;
      if (vkMapMemory(vk_dev, res.params_memory, 0, params_bytes, 0, &m) != VK_SUCCESS) {
        LOG_ERROR << "Vulkan: Math probe: mapping params buffer failed.";
        return false;
      }
      memcpy(m, &mp, sizeof(mp));
      vkUnmapMemory(vk_dev, res.params_memory);
    }
    /* Keep the host reference in sync (same integrator.seed). */
    tkg.data.integrator.seed = (int)seed;

    /* Record + submit. */
    VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkResetCommandBuffer(cmd, 0);
    vkBeginCommandBuffer(cmd, &begin_info);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, res.pipeline);
    MathProbePush push{cases_addr, out_addr, params_addr, num_cases};
    vkCmdPushConstants(cmd, res.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(cmd, (num_cases + 63) / 64, 1, 1);
    vkEndCommandBuffer(cmd);

    vkResetFences(vk_dev, 1, &res.fence);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    if (vkQueueSubmit(queue, 1, &si, res.fence) != VK_SUCCESS) {
      LOG_ERROR << "Vulkan: Math probe: queue submit failed.";
      return false;
    }
    if (vkWaitForFences(vk_dev, 1, &res.fence, VK_TRUE, 10000000000ull) != VK_SUCCESS) {
      LOG_ERROR << "Vulkan: Math probe: fence wait failed.";
      return false;
    }

    /* Read back. */
    void *m = nullptr;
    if (vkMapMemory(vk_dev, res.out_memory, 0, out_bytes, 0, &m) != VK_SUCCESS) {
      LOG_ERROR << "Vulkan: Math probe: mapping out buffer failed.";
      return false;
    }
    memcpy(results.data(), m, (size_t)num_cases * sizeof(MathProbeResult));
    /* Verify the canary region is intact (no overrun). */
    const uint32_t *canary = reinterpret_cast<const uint32_t *>(
        static_cast<const uint8_t *>(m) + (size_t)num_cases * sizeof(MathProbeResult));
    bool canary_ok = true;
    for (uint32_t i = 0; i < MATH_PROBE_CANARY_WORDS; i++)
      if (canary[i] != MATH_PROBE_CANARY) canary_ok = false;
    vkUnmapMemory(vk_dev, res.out_memory);
    if (!canary_ok) {
      LOG_ERROR << "Vulkan: Math probe: OUTPUT CANARY CORRUPTED (shader overran the "
                   "results buffer).";
      return false;
    }
    return true;
  };

  /* ---- 12. Compare one pass against the REAL Cycles reference ---- */
  struct PassStats {
    uint64_t int_cases = 0, int_mismatch = 0;
    uint64_t float_comps = 0, float_mismatch = 0;
    uint64_t sobol_bitexact = 0, sobol_floats = 0;
    double max_norm_err = 0.0, sum_norm_err = 0.0;
    std::vector<double> norm_errs;
    uint64_t by_kind_mismatch[MATHK_COUNT] = {0};
    bool bad_output_shape = false;
  };
  auto compare_pass = [&](PassStats &st) {
    st.norm_errs.reserve(num_cases * 2);
    for (uint32_t i = 0; i < num_cases; i++) {
      const MathProbeCase &c = cases[i];
      const MathProbeResult &r = results[i];
      auto bad_shape = [&](const char *what) {
        if (!st.bad_output_shape) {
          LOG_ERROR << "Vulkan: Math probe: case " << i << " (" << what
                    << ") has invalid has_u/nfloat; dispatch did not run correctly.";
        }
        st.bad_output_shape = true;
      };

      if (c.kind == MATHK_HASH) {
        const uint32_t ref = hash_iqnt2d(c.x, c.y);
        if (r.has_u != 1) { bad_shape("hash"); continue; }
        st.int_cases++;
        if (r.u0 != ref) { st.int_mismatch++; st.by_kind_mismatch[c.kind]++; }
      }
      else if (c.kind == MATHK_PIXEL_INIT) {
        const uint32_t ref = path_rng_pixel_init(kg, c.sample, (int)c.x, (int)c.y);
        if (r.has_u != 1) { bad_shape("pixel_init"); continue; }
        st.int_cases++;
        if (r.u0 != ref) { st.int_mismatch++; st.by_kind_mismatch[c.kind]++; }
      }
      else if (c.kind >= MATHK_SOBOL1D && c.kind <= MATHK_SOBOL4D) {
        float ref[4];
        uint32_t n = 0;
        if (c.kind == MATHK_SOBOL1D) {
          ref[0] = sobol_burley_sample_1D(c.index, c.dim, c.seed, c.mask); n = 1;
        }
        else if (c.kind == MATHK_SOBOL2D) {
          const float2 v = sobol_burley_sample_2D(c.index, c.dim, c.seed, c.mask);
          ref[0] = v.x; ref[1] = v.y; n = 2;
        }
        else if (c.kind == MATHK_SOBOL3D) {
          const float3 v = sobol_burley_sample_3D(c.index, c.dim, c.seed, c.mask);
          ref[0] = v.x; ref[1] = v.y; ref[2] = v.z; n = 3;
        }
        else {
          const float4 v = sobol_burley_sample_4D(c.index, c.dim, c.seed, c.mask);
          ref[0] = v.x; ref[1] = v.y; ref[2] = v.z; ref[3] = v.w; n = 4;
        }
        if (r.nfloat != n) { bad_shape("sobol"); continue; }
        bool any = false;
        for (uint32_t k = 0; k < n; k++) {
          const double denom = std::max(1.0, (double)std::fabs(ref[k]));
          const double err = std::fabs((double)r.f[k] - (double)ref[k]) / denom;
          st.float_comps++;
          st.sum_norm_err += err;
          if (err > st.max_norm_err) st.max_norm_err = err;
          st.norm_errs.push_back(err);
          if (err > 1e-5) { st.float_mismatch++; any = true; }
          /* Informational: exact-bit (implies sobol integer bit-exact). */
          uint32_t gb, rb;
          memcpy(&gb, &r.f[k], 4); memcpy(&rb, &ref[k], 4);
          st.sobol_floats++;
          if (gb == rb) st.sobol_bitexact++;
        }
        if (any) st.by_kind_mismatch[c.kind]++;
      }
      else if (c.kind >= MATHK_RNG1D && c.kind <= MATHK_RNG4D) {
        const uint32_t rng_pixel = path_rng_pixel_init(kg, c.sample, (int)c.x, (int)c.y);
        float ref[4];
        uint32_t n = 0;
        if (c.kind == MATHK_RNG1D) {
          ref[0] = path_rng_1D(kg, rng_pixel, (uint32_t)c.sample, (int)c.dim); n = 1;
        }
        else if (c.kind == MATHK_RNG2D) {
          const float2 v = path_rng_2D(kg, rng_pixel, c.sample, (int)c.dim);
          ref[0] = v.x; ref[1] = v.y; n = 2;
        }
        else if (c.kind == MATHK_RNG3D) {
          const float3 v = path_rng_3D(kg, rng_pixel, c.sample, (int)c.dim);
          ref[0] = v.x; ref[1] = v.y; ref[2] = v.z; n = 3;
        }
        else {
          const float4 v = path_rng_4D(kg, rng_pixel, c.sample, (int)c.dim);
          ref[0] = v.x; ref[1] = v.y; ref[2] = v.z; ref[3] = v.w; n = 4;
        }
        if (r.nfloat != n) { bad_shape("rng"); continue; }
        bool any = false;
        for (uint32_t k = 0; k < n; k++) {
          const double denom = std::max(1.0, (double)std::fabs(ref[k]));
          const double err = std::fabs((double)r.f[k] - (double)ref[k]) / denom;
          st.float_comps++;
          st.sum_norm_err += err;
          if (err > st.max_norm_err) st.max_norm_err = err;
          st.norm_errs.push_back(err);
          if (err > 1e-5) { st.float_mismatch++; any = true; }
        }
        if (any) st.by_kind_mismatch[c.kind]++;
      }
      else if (c.kind == MATHK_CAMERA) {
        Ray ray;
        memset(&ray, 0, sizeof(ray));
        int cache_miss = 0;
        const float2 filter_uv = make_float2(c.fx, c.fy);
        const float2 lens_uv = make_float2(0.0f, 0.0f);
        camera_sample(kg, (int)c.x, (int)c.y, filter_uv, c.fz, lens_uv, &ray, cache_miss);
        if (r.nfloat != 9) { bad_shape("camera"); continue; }
        const float ref[9] = {ray.P.x, ray.P.y, ray.P.z,
                              ray.D.x, ray.D.y, ray.D.z,
                              ray.tmin, ray.tmax, ray.time};
        bool any = false;
        for (int k = 0; k < 9; k++) {
          const double denom = std::max(1.0, (double)std::fabs(ref[k]));
          const double err = std::fabs((double)r.f[k] - (double)ref[k]) / denom;
          st.float_comps++;
          st.sum_norm_err += err;
          if (err > st.max_norm_err) st.max_norm_err = err;
          st.norm_errs.push_back(err);
          if (err > 1e-5) { st.float_mismatch++; any = true; }
        }
        if (any) st.by_kind_mismatch[c.kind]++;
      }
    }
    std::sort(st.norm_errs.begin(), st.norm_errs.end());
  };

  auto log_stats = [&](const char *label, uint32_t seed, const PassStats &st) {
    const double mean = st.norm_errs.empty() ?
                            0.0 :
                            st.sum_norm_err / (double)st.norm_errs.size();
    const double p999 = st.norm_errs.empty() ?
                            0.0 :
                            st.norm_errs[(size_t)((double)(st.norm_errs.size() - 1) * 0.999)];
    LOG_INFO << "Vulkan: Math probe: [" << label << " seed=" << seed << "] int "
             << st.int_mismatch << "/" << st.int_cases << " mismatch; float "
             << st.float_mismatch << "/" << st.float_comps << " over tol; max_norm_err="
             << st.max_norm_err << " mean=" << mean << " p99.9=" << p999
             << " sobol_bitexact=" << st.sobol_bitexact << "/" << st.sobol_floats
             << (st.bad_output_shape ? " [BAD OUTPUT SHAPE]" : "");
  };

  /* ---- 13. Run: clean for seeds {0,1,large}; negatives at seed 0 ---- */
  const uint32_t clean_seeds[3] = {0u, 1u, 0xDEADBEEFu};
  bool clean_ok = true;
  for (int s = 0; s < 3; s++) {
    if (!run_pass(clean_seeds[s], 0)) return false;
    PassStats st;
    compare_pass(st);
    log_stats("CLEAN", clean_seeds[s], st);
    if (st.int_mismatch != 0 || st.float_mismatch != 0 || st.bad_output_shape) clean_ok = false;
  }

  bool neg_detected[4] = {true, false, false, false};
  const char *neg_name[4] = {"", "sobol-table", "seed-xor", "transpose-rtc"};
  for (int n = 1; n <= 3; n++) {
    if (!run_pass(0u, n)) return false;
    PassStats st;
    compare_pass(st);
    log_stats(neg_name[n], 0, st);
    /* Detection is per targeted family. */
    if (n == 1) { /* sobol table: any sobol/rng float mismatch */
      uint64_t m = 0;
      for (int k = MATHK_SOBOL1D; k <= MATHK_RNG4D; k++) m += st.by_kind_mismatch[k];
      neg_detected[1] = (m > 0);
    }
    else if (n == 2) { /* seed xor: any pixel_init int mismatch */
      neg_detected[2] = (st.by_kind_mismatch[MATHK_PIXEL_INIT] > 0);
    }
    else { /* transpose: any camera float mismatch */
      neg_detected[3] = (st.by_kind_mismatch[MATHK_CAMERA] > 0);
    }
  }

  /* ---- 14. Verdict ---- */
  bool all_ok = clean_ok;
  for (int n = 1; n <= 3; n++) {
    LOG_INFO << "Vulkan: Math probe: negative '" << neg_name[n] << "' "
             << (neg_detected[n] ? "DETECTED" : "NOT DETECTED");
    if (!neg_detected[n]) all_ok = false;
  }
  if (clean_ok)
    LOG_INFO << "Vulkan: Math probe: CLEAN PASS (all seeds bit/tolerance-exact).";
  else
    LOG_ERROR << "Vulkan: Math probe: CLEAN PASS FAILED (GPU disagrees with the real "
                 "Cycles reference).";
  LOG_INFO << "Vulkan: Math probe: " << (all_ok ? "GATE MET." : "GATE NOT MET.");
  return all_ok;
}

/* =========================================================================
 * M4e STEP 1b-0: dual-view KernelData dump probe (optional, env
 * CYCLES_VULKAN_KDATA_PROBE=1)
 * =========================================================================
 * Settles the M4b anomaly: the typed view `params->data.words[356]`
 * reportedly returned 0x80000803 while the raw uint32 view at the SAME byte
 * offset (1424) returned 0x00000000 (traverse_probe.slang:104-115). This
 * dispatches kdata_probe.slang, which reads ALL 640 words of the KernelData
 * blob through BOTH views in one pass, then compares each view against the
 * host bytes (host_kernel_data(), the same blob const_copy_to uploaded via
 * device.cpp:317) and against each other, index by index. Every disagreement
 * is printed with the owning KernelData field (offsetof from the real
 * data_template macros) and the surrounding fields.
 *
 * The readback buffer is prefilled with 0xA5A5A5A5 (NOT 0x80000803) so the
 * probe can settle whether 0x80000803 was ever a probe-written sentinel: if
 * the typed view returns 0x80000803 it cannot have come from this prefill. */

namespace {

/* Readback layout must match kdata_probe.slang: [0..639] typed loop,
 * [640..1279] raw loop, [1280] canary, [1281] typed words[356],
 * [1282] raw params_raw[356], [1283..1922] base-ptr loop. */
constexpr uint32_t KDATA_WORDS = 640;
constexpr uint32_t KDATA_ANOMALY_WORD = 356;
constexpr uint32_t KDATA_OUT_WORDS = 3 * KDATA_WORDS + 3; /* 1923 */
/* Offset of view (c) in the readback: right after canary + 2 constant slots. */
constexpr uint32_t KDATA_VIEW_C_BASE = 2 * KDATA_WORDS + 3; /* 1283 */
/* KdataProbeParams = three 64-bit device pointers (kdata_probe.slang). */
constexpr uint32_t KDATA_PUSH_BYTES = 24;
/* Prefill sentinel: a readback word still equal to this was NOT written by
 * the shader. Deliberately NOT 0x80000803 (the historical anomaly value). */
constexpr uint32_t KDATA_PREFILL_SENTINEL = 0xA5A5A5A5u;
/* Canary the shader writes at out[1280]; must match kdata_probe.slang. */
constexpr uint32_t KDATA_CANARY = 0xCAFEF00Du;

std::string kdata_hex32(const uint32_t v) { return string_printf("%08x", v); }

struct KdataProbeResources {
  VkDevice device = VK_NULL_HANDLE;
  VkShaderModule module = VK_NULL_HANDLE;
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;
  VkCommandPool pool = VK_NULL_HANDLE;
  VkBuffer out_buffer = VK_NULL_HANDLE;
  VkDeviceMemory out_memory = VK_NULL_HANDLE;

  ~KdataProbeResources()
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

/* Field-attribution table: top-level KernelData members with their REAL
 * offsets (offsetof). Sizes are implied by consecutive offsets; the last
 * field runs to sizeof(KernelData). Built from the actual member names so a
 * mismatching byte offset maps to the Cycles field that owns it. */
struct KdataField {
  size_t offset;
  const char *name;
};
static const KdataField kdata_field_table[] = {
    {offsetof(KernelData, kernel_features), "kernel_features"},
    {offsetof(KernelData, max_closures), "max_closures"},
    {offsetof(KernelData, max_shaders), "max_shaders"},
    {offsetof(KernelData, volume_stack_size), "volume_stack_size"},
    {offsetof(KernelData, cam), "cam"},
    {offsetof(KernelData, bake), "bake"},
    {offsetof(KernelData, tables), "tables"},
    {offsetof(KernelData, light_link_sets), "light_link_sets"},
    {offsetof(KernelData, scene_time), "scene_time"},
    {offsetof(KernelData, background), "background"},
    {offsetof(KernelData, bvh), "bvh"},
    {offsetof(KernelData, film), "film"},
    {offsetof(KernelData, integrator), "integrator"},
    {offsetof(KernelData, image), "image"},
    {offsetof(KernelData, svm_usage), "svm_usage"},
    {offsetof(KernelData, device_bvh), "device_bvh"},
};
static const size_t KDATA_FIELD_COUNT = sizeof(kdata_field_table) /
                                        sizeof(kdata_field_table[0]);

/* KernelBVH sub-fields (data_template.h:51-60), for drill-down when the
 * owning top-level field is `bvh` (the historical anomaly lives here). */
struct BvhField {
  size_t offset;
  const char *name;
};
static const BvhField bvh_field_table[] = {
    {offsetof(KernelBVH, root), "root"},
    {offsetof(KernelBVH, have_motion), "have_motion"},
    {offsetof(KernelBVH, have_curves), "have_curves"},
    {offsetof(KernelBVH, have_points), "have_points"},
    {offsetof(KernelBVH, have_volumes), "have_volumes"},
    {offsetof(KernelBVH, bvh_layout), "bvh_layout"},
    {offsetof(KernelBVH, use_bvh_steps), "use_bvh_steps"},
    {offsetof(KernelBVH, curve_subdivisions), "curve_subdivisions"},
};
static const size_t BVH_FIELD_COUNT = sizeof(bvh_field_table) / sizeof(bvh_field_table[0]);

/* Print the owning KernelData field for a byte offset, plus the fields on
 * either side (surrounding context). Drill into KernelBVH when the offset
 * lands inside `bvh`. */
void kdata_attrib(const size_t byte_off)
{
  size_t idx = 0;
  for (size_t i = 0; i < KDATA_FIELD_COUNT; i++) {
    if (kdata_field_table[i].offset <= byte_off) idx = i;
  }
  const size_t field_off = kdata_field_table[idx].offset;
  const size_t field_end = (idx + 1 < KDATA_FIELD_COUNT) ?
                               kdata_field_table[idx + 1].offset :
                               sizeof(KernelData);
  if (idx > 0)
    LOG_INFO << "Vulkan: KData probe:   prev field: " << kdata_field_table[idx - 1].name
             << " @ " << kdata_field_table[idx - 1].offset;
  LOG_INFO << "Vulkan: KData probe:   OWNING FIELD: " << kdata_field_table[idx].name
           << " @ offset " << field_off << " size " << (field_end - field_off) << " (+"
           << (byte_off - field_off) << " into field)";
  if (idx + 1 < KDATA_FIELD_COUNT)
    LOG_INFO << "Vulkan: KData probe:   next field: " << kdata_field_table[idx + 1].name
             << " @ " << kdata_field_table[idx + 1].offset;
  if (strcmp(kdata_field_table[idx].name, "bvh") == 0) {
    const size_t rel = byte_off - field_off;
    for (size_t i = 0; i < BVH_FIELD_COUNT; i++) {
      if (bvh_field_table[i].offset <= rel) {
        LOG_INFO << "Vulkan: KData probe:     -> KernelBVH." << bvh_field_table[i].name
                 << " (+" << (rel - bvh_field_table[i].offset) << " into it)";
        break;
      }
    }
  }
}

} /* namespace */

bool vulkan_kdata_probe(VulkanDevice *dev)
{
  static bool ran = false;
  if (ran) return true;
  ran = true;
  if (getenv("CYCLES_VULKAN_KDATA_PROBE") == nullptr) return true;

  LOG_INFO << "Vulkan: KData probe starting (CYCLES_VULKAN_KDATA_PROBE=1).";

  KdataProbeResources res;
  const VkDevice vk_dev = dev->vk_device();
  res.device = vk_dev;

  /* 1. Device address of the KernelParamsVulkan buffer (const_copy_to created
   *    it during the scene upload; this probe runs from enqueue, after that). */
  const uint64_t params_addr = dev->kernel_params_address();
  if (params_addr == 0) {
    LOG_ERROR << "Vulkan: KData probe: kernel params buffer does not exist; probe skipped.";
    return false;
  }

  /* 2. SPIR-V (env override or repo-relative default). */
  std::vector<char> spirv;
  const char *env_path = getenv("CYCLES_VULKAN_KDATA_PROBE_SPV");
  const char *candidates[2] = {env_path, "kdata_probe.spv"};
  const char *found = nullptr;
  for (const char *c : candidates) {
    if (c && probe_read_whole_file(c, spirv)) {
      found = c;
      break;
    }
  }
  if (!found) {
    LOG_ERROR << "Vulkan: KData probe: cannot find kdata_probe.spv "
                 "(compile it with slangc or set CYCLES_VULKAN_KDATA_PROBE_SPV).";
    return false;
  }
  if (spirv.size() % 4 != 0) {
    LOG_ERROR << "Vulkan: KData probe: '" << found << "' is not a valid SPIR-V module.";
    return false;
  }
  LOG_INFO << "Vulkan: KData probe: spirv='" << found << "' (" << spirv.size() << " bytes).";

  /* 3. Shader module. */
  VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smi.codeSize = spirv.size();
  smi.pCode = reinterpret_cast<const uint32_t *>(spirv.data());
  if (vkCreateShaderModule(vk_dev, &smi, nullptr, &res.module) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: KData probe: vkCreateShaderModule failed.";
    return false;
  }

  /* 4. Pipeline layout: no descriptor sets, one 24-byte push-constant range. */
  VkPushConstantRange pcr{};
  pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pcr.offset = 0;
  pcr.size = KDATA_PUSH_BYTES;
  VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pli.pushConstantRangeCount = 1;
  pli.pPushConstantRanges = &pcr;
  if (vkCreatePipelineLayout(vk_dev, &pli, nullptr, &res.layout) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: KData probe: pipeline layout creation failed.";
    return false;
  }

  /* 5. Compute pipeline (entry name "main"; slangc renames). */
  VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  cpi.stage.module = res.module;
  cpi.stage.pName = "main";
  cpi.layout = res.layout;
  if (vkCreateComputePipelines(vk_dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &res.pipeline) !=
      VK_SUCCESS) {
    LOG_ERROR << "Vulkan: KData probe: compute pipeline creation failed "
                 "(bad entry point name?).";
    return false;
  }

  /* 6. Readback buffer, host-visible + coherent so the host can prefill the
   *    sentinel and read back after the fence. STORAGE | SHADER_DEVICE_ADDRESS
   *    because the shader writes through a raw device pointer. */
  VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bci.size = KDATA_OUT_WORDS * sizeof(uint32_t);
  bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  if (vkCreateBuffer(vk_dev, &bci, nullptr, &res.out_buffer) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: KData probe: readback buffer creation failed.";
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
    LOG_ERROR << "Vulkan: KData probe: readback memory allocation failed.";
    return false;
  }
  if (vkBindBufferMemory(vk_dev, res.out_buffer, res.out_memory, 0) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: KData probe: readback memory bind failed.";
    return false;
  }
  VkBufferDeviceAddressInfo bai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
  bai.buffer = res.out_buffer;
  const uint64_t out_addr = vkGetBufferDeviceAddress(vk_dev, &bai);

  /* 7. Prefill the readback buffer with the sentinel so unwritten words are
   *    detectable. Host-coherent memory: the host write is made visible to the
   *    device by the queue submission, so no explicit barrier is needed. */
  {
    void *m = nullptr;
    if (vkMapMemory(vk_dev, res.out_memory, 0, bci.size, 0, &m) != VK_SUCCESS) {
      LOG_ERROR << "Vulkan: KData probe: mapping readback buffer for prefill failed.";
      return false;
    }
    uint32_t *words = static_cast<uint32_t *>(m);
    for (uint32_t i = 0; i < KDATA_OUT_WORDS; i++) words[i] = KDATA_PREFILL_SENTINEL;
    vkUnmapMemory(vk_dev, res.out_memory);
  }

  /* 8. Record + submit one workgroup of one thread. */
  const uint32_t family = dev->find_queue_family(dev->vk_physical_device());
  VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  cpci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
  cpci.queueFamilyIndex = family;
  if (vkCreateCommandPool(vk_dev, &cpci, nullptr, &res.pool) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: KData probe: command pool creation failed.";
    return false;
  }
  VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbai.commandPool = res.pool;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  if (vkAllocateCommandBuffers(vk_dev, &cbai, &cmd) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: KData probe: command buffer allocation failed.";
    return false;
  }
  VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  if (vkBeginCommandBuffer(cmd, &begin_info) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: KData probe: vkBeginCommandBuffer failed.";
    return false;
  }
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, res.pipeline);
  const uint64_t push[3] = {params_addr, params_addr, out_addr};
  vkCmdPushConstants(cmd, res.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);
  vkCmdDispatch(cmd, 1, 1, 1);
  if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: KData probe: vkEndCommandBuffer failed.";
    return false;
  }
  VkQueue queue = VK_NULL_HANDLE;
  vkGetDeviceQueue(vk_dev, family, 0, &queue);
  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkFence fence = VK_NULL_HANDLE;
  if (vkCreateFence(vk_dev, &fci, nullptr, &fence) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: KData probe: fence creation failed.";
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
    LOG_ERROR << "Vulkan: KData probe: queue submit/wait failed.";
    return false;
  }

  /* 9. Read back. */
  uint32_t out[KDATA_OUT_WORDS];
  {
    void *m = nullptr;
    if (vkMapMemory(vk_dev, res.out_memory, 0, bci.size, 0, &m) != VK_SUCCESS) {
      LOG_ERROR << "Vulkan: KData probe: mapping readback buffer failed.";
      return false;
    }
    memcpy(out, m, sizeof(out));
    vkUnmapMemory(vk_dev, res.out_memory);
  }

  /* 10. Host reference: the SAME KernelData blob const_copy_to uploaded into
   *     the params buffer (host_params_.data, mirrored 1:1 by device.cpp:317).
   *     Reinterpret as a flat uint32 array of 640 words. */
  const KernelData &kd = dev->host_kernel_data();
  static_assert(sizeof(KernelData) == 2560, "KernelData must be 640 uint32 words");
  const uint32_t *host_words = reinterpret_cast<const uint32_t *>(&kd);

  /* 11. Verify the canary and detect any word the shader did not overwrite.
   *     Check the two loop regions (views a,b and view c), skipping the
   *     canary and the two constant slots in between. */
  const bool canary_ok = (out[2 * KDATA_WORDS] == KDATA_CANARY);
  uint32_t unwritten = 0;
  for (uint32_t i = 0; i < 2 * KDATA_WORDS; i++) {
    if (out[i] == KDATA_PREFILL_SENTINEL) unwritten++;
  }
  for (uint32_t i = 0; i < KDATA_WORDS; i++) {
    if (out[KDATA_VIEW_C_BASE + i] == KDATA_PREFILL_SENTINEL) unwritten++;
  }
  if (!canary_ok)
    LOG_ERROR << "Vulkan: KData probe: CANARY MISMATCH (out[" << (2 * KDATA_WORDS)
              << "]=0x" << kdata_hex32(out[2 * KDATA_WORDS]) << ", expected 0x"
              << kdata_hex32(KDATA_CANARY) << ").";
  if (unwritten)
    LOG_ERROR << "Vulkan: KData probe: " << unwritten
              << " word(s) still hold the prefill sentinel (shader did not write them).";

  /* 12. Compare each view against the host bytes and against each other.
   *     Print every disagreement with full attribution. View (c) is the
   *     root-cause discriminator: base-pointer + dynamic index. */
  uint32_t mism_a_host = 0, mism_b_host = 0, mism_a_b = 0, mism_c_host = 0;
  for (uint32_t i = 0; i < KDATA_WORDS; i++) {
    const uint32_t a = out[i];               /* typed view        */
    const uint32_t b = out[KDATA_WORDS + i]; /* raw view          */
    const uint32_t c = out[KDATA_VIEW_C_BASE + i]; /* base-ptr view */
    const uint32_t host = host_words[i];
    const bool a_bad = (a != host);
    const bool b_bad = (b != host);
    const bool c_bad = (c != host);
    const bool ab_bad = (a != b);
    if (a_bad) mism_a_host++;
    if (b_bad) mism_b_host++;
    if (c_bad) mism_c_host++;
    if (ab_bad) mism_a_b++;
    if (a_bad || b_bad || c_bad || ab_bad) {
      const size_t byte_off = (size_t)i * 4;
      LOG_ERROR << "Vulkan: KData probe: MISMATCH word " << i << " (byte " << byte_off
                << "): typed=0x" << kdata_hex32(a) << " raw=0x" << kdata_hex32(b)
                << " baseptr=0x" << kdata_hex32(c) << " host=0x" << kdata_hex32(host) << "  [typed"
                << (a_bad ? "!=host" : "==host") << " raw" << (b_bad ? "!=host" : "==host")
                << " baseptr" << (c_bad ? "!=host" : "==host") << (ab_bad ? " typed!=raw" : "")
                << "]";
      kdata_attrib(byte_off);
    }
  }

  /* 13. The historical anomaly read, through CONSTANT index356. */
  {
    const uint32_t a_const = out[2 * KDATA_WORDS + 1];
    const uint32_t b_const = out[2 * KDATA_WORDS + 2];
    const uint32_t host356 = host_words[KDATA_ANOMALY_WORD];
    LOG_INFO << "Vulkan: KData probe: anomaly word " << KDATA_ANOMALY_WORD << " (byte "
             << (KDATA_ANOMALY_WORD * 4) << ") CONSTANT-index: typed=0x" << kdata_hex32(a_const)
             << " raw=0x" << kdata_hex32(b_const) << " host=0x" << kdata_hex32(host356)
             << (a_const == b_const && a_const == host356 ? " AGREE" : " DISAGREE");
    /* Where is KernelBVH / bvh.root really? Print for the record. */
    LOG_INFO << "Vulkan: KData probe: offsetof(KernelData,bvh)=" << offsetof(KernelData, bvh)
             << " offsetof(KernelBVH,root)=" << offsetof(KernelBVH, root) << " => bvh.root at byte "
             << (offsetof(KernelData, bvh) + offsetof(KernelBVH, root)) << " (word "
             << ((offsetof(KernelData, bvh) + offsetof(KernelBVH, root)) / 4) << ")";
  }

  /* 14. Verdict. The root-cause discriminator is view (c): if baseptr==host
   *     for all words but typed!=host, the blob LAYOUT is correct and only
   *     the inline `params->data.words[i]` dynamic-index codegen is broken. */
  const bool ok = canary_ok && (unwritten == 0) && (mism_a_host == 0) && (mism_b_host == 0) &&
                  (mism_a_b == 0) && (mism_c_host == 0);
  const bool layout_ok = (mism_b_host == 0) && (mism_c_host == 0);
  LOG_INFO << "Vulkan: KData probe: typed-vs-host=" << mism_a_host
           << " raw-vs-host=" << mism_b_host << " baseptr-vs-host=" << mism_c_host
           << " typed-vs-raw=" << mism_a_b << " unwritten=" << unwritten
           << (canary_ok ? "" : " CANARY-BAD");
  if (layout_ok && mism_a_host != 0)
    LOG_ERROR << "Vulkan: KData probe: ROOT CAUSE = the KernelData blob LAYOUT is correct "
                 "(raw + base-ptr dynamic index match host for all 640 words), but the inline "
                 "`params->data.words[i]` DYNAMIC index is miscompiled by this Slang/SPIR-V "
                 "toolchain (returns a constant for i>=1). Constant index works. Fix: access the "
                 "blob via a base pointer or raw view, never inline dynamic-index data.words[].";
  if (ok)
    LOG_INFO << "Vulkan: KData probe: NO MISMATCH. All three views agree with the host bytes for "
                "all 640 words (anomaly NOT reproduced).";
  else
    LOG_ERROR << "Vulkan: KData probe: MISMATCH PRESENT (see above).";
  return ok;
}

/* =========================================================================
 * M4e S1b-3 step 1: IntegratorStateGPU STRIDE TABLE (host, no GPU; optional,
 * env CYCLES_VULKAN_STATE_STRIDE_TABLE=1).
 *
 * Expands state_template.h + shadow_state_template.h with the SAME macros the
 * real SoA allocator uses (path_trace_work_gpu.cpp:143-201), but PRINTS one
 * row per member instead of allocating. The allocator's element size for a
 * member is sizeof(type): it builds `device_only_memory<type>` and calls
 * `alloc_to_device(max_num_paths_)` (path_trace_work_gpu.cpp:151-152 for
 * KERNEL_STRUCT_MEMBER, 180-181 for KERNEL_STRUCT_ARRAY_MEMBER), so the SoA
 * element stride a Slang accessor MUST use equals sizeof(type).
 *
 * Two hazard classes are flagged, because the natural Slang type for those
 * members has a DIFFERENT size than the allocator element (a silent stride
 * corruption if the accessor uses the natural type):
 *   PACKED3  sizeof==12 : packed_float3 / PackedSpectrum (types_float3.h:198
 *            static_assert 12; types_spectrum.h:14 PackedSpectrum=packed_float3).
 *            NOT 16-byte Slang float3. Read via a float* base + 3*i (see step 2).
 *   SUB4     sizeof<4   : uint16_t (2) / uint8_t (1). Not 4-byte uint/int; use
 *            an exact-width view (a uint16 or uint8 pointer, or raw bytes),
 *            never uint.
 * Array members (volume_stack, shadow_isect, shadow_volume_stack) are
 * ARRAY-OF-STRUCTS in the allocator (a `for (array_index...)` loop emits a
 * separate device_only_memory per slot), so `vs[ai].object` is at ai*16 + 0,
 * NOT ai*8. The base pointer must therefore be to the whole slot struct.
 *
 * This function never touches the GPU: it only sizeof()/prints, so it is safe
 * to call on any device (or before one exists). ========================================================================= */

bool vulkan_state_stride_table(VulkanDevice *dev)
{
  (void)dev;
  static bool ran = false;
  if (ran) return true;
  ran = true;
  if (getenv("CYCLES_VULKAN_STATE_STRIDE_TABLE") == nullptr) return true;

  fprintf(stderr,
          "\n=== M4e S1b-3 IntegratorStateGPU STRIDE TABLE ===\n"
          "allocator element size == sizeof(type)  [device_only_memory<type>.alloc_to_device("
          "max_num_paths_), path_trace_work_gpu.cpp:151-152 / 180-181]\n"
          "flags: PACKED3 = packed_float3/PackedSpectrum (12B; Slang float3 is 16B -> use float* "
          "base+3*i);  SUB4 = uint16_t/uint8_t (<4B; use exact-width view, not uint*)\n\n");

  /* Same macro skeleton as path_trace_work_gpu.cpp:143-196, but printing a
   * table row per member. `feature` is unused so KernelFeatureRequest(...) is
   * discarded by the preprocessor (need not be defined here). The shadow
   * template's struct names already carry the "shadow_" prefix, so no extra
   * label is required. */
#define KERNEL_STRUCT_BEGIN(name) \
  fprintf(stderr, "STRUCT %s\n", #name); \
  for (int _ai = 0;; _ai++) {
#define STRIDE_FLAG(type) (sizeof(type) == 12 ? "PACKED3" : (sizeof(type) < 4 ? "SUB4" : "ok"))
#define KERNEL_STRUCT_MEMBER(parent_struct, type, name, feature) \
  fprintf(stderr, "  %s.%-26s type=%-16s elem_size=%-3zu %s\n", \
          #parent_struct, #name, #type, sizeof(type), STRIDE_FLAG(type));
#define KERNEL_STRUCT_MEMBER_PACKED KERNEL_STRUCT_MEMBER
#define KERNEL_STRUCT_BEGIN_PACKED(parent_struct, feature) KERNEL_STRUCT_BEGIN(parent_struct)
#define KERNEL_STRUCT_ARRAY_MEMBER(parent_struct, type, name, feature) \
  fprintf(stderr, "  %s[%d].%-22s type=%-16s elem_size=%-3zu %s\n", \
          #parent_struct, _ai, #name, #type, sizeof(type), STRIDE_FLAG(type));
#define KERNEL_STRUCT_END(name) \
  (void)_ai; \
  break; \
  }
#define KERNEL_STRUCT_END_ARRAY(name, cpu_size, gpu_size) \
  if (_ai >= (int)(gpu_size)-1) break; \
  }
#define KERNEL_STRUCT_VOLUME_STACK_SIZE MAX_VOLUME_STACK_SIZE

#include "kernel/integrator/state_template.h"
#include "kernel/integrator/shadow_state_template.h"

#undef KERNEL_STRUCT_BEGIN
#undef KERNEL_STRUCT_BEGIN_PACKED
#undef KERNEL_STRUCT_MEMBER
#undef KERNEL_STRUCT_MEMBER_PACKED
#undef KERNEL_STRUCT_ARRAY_MEMBER
#undef KERNEL_STRUCT_END
#undef KERNEL_STRUCT_END_ARRAY
#undef KERNEL_STRUCT_VOLUME_STACK_SIZE
#undef STRIDE_FLAG

  fprintf(stderr, "=== end stride table ===\n\n");
  return true;
}

/* =========================================================================
 * M4e S1b-3 step 2: GPU SLANG STRIDE CHECK (optional, env
 * CYCLES_VULKAN_SLANG_STRIDE_PROBE=1 / ..._SPV).
 *
 * Dispatches stride_check.slang: 6 typed pointers (one per 64-byte region of a
 * 512-byte buffer) are pushed; 3 threads write element k = tid.x (a NON-constant
 * index) via each accessor pattern; the host reads the raw bytes back and
 * reports where each value landed, i.e. the actual element STRIDE for that type
 * on this compiler+driver. Compared against the step-1 table (sizeof(type)).
 *
 * The two questions it answers:
 *   (1) what stride does a Slang float3* use? (region r0) -- SPIR-V ArrayStride says
 *       12; runtime confirms whether the driver honors 12 (usable for packed_float3)
 *       or aligns to 16 (would silently corrupt packed data).
 *   (2) how to read/write a 12-byte packed_float3: via a float* base + 3*i (region
 *       r1), which only depends on float stride 4 and is therefore robust either way.
 * Run on BOTH the RX 570 and SwiftShader (DEVICE RULE). ========================================================================= */

namespace {

constexpr uint32_t STRIDE_BUF_BYTES = 512;
constexpr uint32_t STRIDE_CANARY = 0xDEADBEEFu;
/* 6 typed region pointers, 64 bytes apart (matches stride_check.slang). */
constexpr uint32_t STRIDE_PUSH_BYTES = 6 * 8; /* 48 */
struct StridePushHost {
  uint64_t r0, r1, r2, r3, r4, r5;
};

struct StrideResources {
  VkDevice device = VK_NULL_HANDLE;
  VkShaderModule module = VK_NULL_HANDLE;
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;
  VkCommandPool pool = VK_NULL_HANDLE;
  VkBuffer buf = VK_NULL_HANDLE;
  VkDeviceMemory mem = VK_NULL_HANDLE;

  ~StrideResources()
  {
    if (device == VK_NULL_HANDLE) return;
    if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
    if (layout) vkDestroyPipelineLayout(device, layout, nullptr);
    if (module) vkDestroyShaderModule(device, module, nullptr);
    if (pool) vkDestroyCommandPool(device, pool, nullptr);
    if (buf) vkDestroyBuffer(device, buf, nullptr);
    if (mem) vkFreeMemory(device, mem, nullptr);
  }
};

} /* namespace */

bool vulkan_stride_check(VulkanDevice *dev)
{
  static bool ran = false;
  if (ran) return true;
  ran = true;
  if (getenv("CYCLES_VULKAN_SLANG_STRIDE_PROBE") == nullptr) return true;

  LOG_INFO << "Vulkan: Slang stride check starting (CYCLES_VULKAN_SLANG_STRIDE_PROBE=1).";

  StrideResources res;
  const VkDevice vk_dev = dev->vk_device();
  res.device = vk_dev;

  /* 1. SPIR-V. */
  std::vector<char> spirv;
  const char *env_path = getenv("CYCLES_VULKAN_SLANG_STRIDE_PROBE_SPV");
  const char *candidates[2] = {env_path, "stride_check.spv"};
  const char *found = nullptr;
  for (const char *c : candidates) {
    if (c && probe_read_whole_file(c, spirv)) {
      found = c;
      break;
    }
  }
  if (!found) {
    LOG_ERROR << "Vulkan: Slang stride check: cannot find stride_check.spv "
                 "(compile with slangc or set CYCLES_VULKAN_SLANG_STRIDE_PROBE_SPV).";
    return false;
  }
  if (spirv.size() % 4 != 0) {
    LOG_ERROR << "Vulkan: Slang stride check: '" << found << "' not a valid SPIR-V module.";
    return false;
  }
  LOG_INFO << "Vulkan: Slang stride check: spirv='" << found << "' (" << spirv.size() << " bytes).";

  /* 2. Module. */
  VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smi.codeSize = spirv.size();
  smi.pCode = reinterpret_cast<const uint32_t *>(spirv.data());
  if (vkCreateShaderModule(vk_dev, &smi, nullptr, &res.module) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: Slang stride check: vkCreateShaderModule failed.";
    return false;
  }

  /* 3. Pipeline layout: one 48-byte push range. */
  VkPushConstantRange pcr{};
  pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pcr.offset = 0;
  pcr.size = STRIDE_PUSH_BYTES;
  VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pli.pushConstantRangeCount = 1;
  pli.pPushConstantRanges = &pcr;
  if (vkCreatePipelineLayout(vk_dev, &pli, nullptr, &res.layout) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: Slang stride check: pipeline layout creation failed.";
    return false;
  }

  /* 4. Compute pipeline (entry "main"). */
  VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  cpi.stage.module = res.module;
  cpi.stage.pName = "main";
  cpi.layout = res.layout;
  if (vkCreateComputePipelines(vk_dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &res.pipeline) !=
      VK_SUCCESS) {
    LOG_ERROR << "Vulkan: Slang stride check: compute pipeline creation failed.";
    return false;
  }

  /* 5. One 512-byte host-visible+coherent+device-address buffer. */
  auto make_bda_buffer = [&](VkDeviceSize size, VkBuffer &buf, VkDeviceMemory &mem) -> bool {
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = size;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(vk_dev, &bci, nullptr, &buf) != VK_SUCCESS) {
      LOG_ERROR << "Vulkan: Slang stride check: buffer creation failed.";
      return false;
    }
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
    if (vkAllocateMemory(vk_dev, &ai, nullptr, &mem) != VK_SUCCESS) {
      LOG_ERROR << "Vulkan: Slang stride check: memory allocation failed.";
      return false;
    }
    if (vkBindBufferMemory(vk_dev, buf, mem, 0) != VK_SUCCESS) {
      LOG_ERROR << "Vulkan: Slang stride check: memory bind failed.";
      return false;
    }
    return true;
  };
  if (!make_bda_buffer(STRIDE_BUF_BYTES, res.buf, res.mem)) return false;

  /* Prefill with the canary so unwritten slots are detectable. */
  {
    void *m = nullptr;
    if (vkMapMemory(vk_dev, res.mem, 0, STRIDE_BUF_BYTES, 0, &m) != VK_SUCCESS) {
      LOG_ERROR << "Vulkan: Slang stride check: mapping buffer failed.";
      return false;
    }
    uint32_t *w = static_cast<uint32_t *>(m);
    for (uint32_t i = 0; i < STRIDE_BUF_BYTES / 4; i++) w[i] = STRIDE_CANARY;
    vkUnmapMemory(vk_dev, res.mem);
  }

  VkBufferDeviceAddressInfo bai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
  bai.buffer = res.buf;
  const uint64_t base = vkGetBufferDeviceAddress(vk_dev, &bai);

  StridePushHost push{};
  push.r0 = base + 0;
  push.r1 = base + 64;
  push.r2 = base + 128;
  push.r3 = base + 192;
  push.r4 = base + 256;
  push.r5 = base + 320;

  /* 6. Record + submit one workgroup of 3 threads (numthreads(3,1,1)). */
  const uint32_t family = dev->find_queue_family(dev->vk_physical_device());
  VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  cpci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
  cpci.queueFamilyIndex = family;
  if (vkCreateCommandPool(vk_dev, &cpci, nullptr, &res.pool) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: Slang stride check: command pool creation failed.";
    return false;
  }
  VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbai.commandPool = res.pool;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  if (vkAllocateCommandBuffers(vk_dev, &cbai, &cmd) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: Slang stride check: command buffer allocation failed.";
    return false;
  }
  VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  if (vkBeginCommandBuffer(cmd, &begin_info) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: Slang stride check: vkBeginCommandBuffer failed.";
    return false;
  }
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, res.pipeline);
  vkCmdPushConstants(cmd, res.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, STRIDE_PUSH_BYTES, &push);
  vkCmdDispatch(cmd, 1, 1, 1);
  if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: Slang stride check: vkEndCommandBuffer failed.";
    return false;
  }

  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkFence fence_real = VK_NULL_HANDLE;
  if (vkCreateFence(vk_dev, &fci, nullptr, &fence_real) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: Slang stride check: fence creation failed.";
    return false;
  }
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  VkQueue queue = VK_NULL_HANDLE;
  vkGetDeviceQueue(vk_dev, family, 0, &queue);
  if (vkQueueSubmit(queue, 1, &si, fence_real) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: Slang stride check: queue submit failed.";
    vkDestroyFence(vk_dev, fence_real, nullptr);
    return false;
  }
  if (vkWaitForFences(vk_dev, 1, &fence_real, VK_TRUE, 5000000000ull) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: Slang stride check: fence wait failed (shader hung?).";
    vkDestroyFence(vk_dev, fence_real, nullptr);
    return false;
  }
  vkDestroyFence(vk_dev, fence_real, nullptr);

  /* 7. Read back and measure strides. */
  std::vector<uint8_t> raw(STRIDE_BUF_BYTES);
  {
    void *m = nullptr;
    if (vkMapMemory(vk_dev, res.mem, 0, STRIDE_BUF_BYTES, 0, &m) != VK_SUCCESS) {
      LOG_ERROR << "Vulkan: Slang stride check: readback map failed.";
      return false;
    }
    memcpy(raw.data(), m, STRIDE_BUF_BYTES);
    vkUnmapMemory(vk_dev, res.mem);
  }

  auto rf = [&](int off) -> float {
    float f;
    memcpy(&f, raw.data() + off, 4);
    return f;
  };
  auto ru32 = [&](int off) -> uint32_t {
    uint32_t v;
    memcpy(&v, raw.data() + off, 4);
    return v;
  };
  auto ri32 = [&](int off) -> int32_t {
    int32_t v;
    memcpy(&v, raw.data() + off, 4);
    return v;
  };
  /* Find where a given float value landed inside [start,end). */
  auto find_f = [&](int start, int end, float want) -> int {
    for (int o = start; o < end; o += 4) {
      if (rf(o) == want) return o;
    }
    return -1;
  };

  bool all_ok = true;
  auto expect = [&](const char *name, bool ok) {
    if (!ok) all_ok = false;
    fprintf(stderr, "  %-28s %s\n", name, ok ? "OK" : "MISMATCH");
  };

  fprintf(stderr, "\n=== M4e S1b-3 SLANG STRIDE CHECK (RX570 vs SwiftShader reported per run) ===\n");

  /* R0: float3* stride (measured, not assumed). */
  int o0 = find_f(0, 64, 1000.0f);
  int o1 = find_f(0, 64, 1001.0f);
  int o2 = find_f(0, 64, 1002.0f);
  int stride_f3 = (o0 >= 0 && o1 >= 0) ? (o1 - o0) : -1;
  fprintf(stderr,
          "  float3* stride (region r0): off(0)=%d off(1)=%d off(2)=%d => stride=%d "
          "(SPIR-V ArrayStride=12; 12 matches packed_float3, 16 would NOT)\n",
          o0, o1, o2, stride_f3);
  /* The packed_float3 accessor must land elements 12 bytes apart. */
  expect("float3* stride == 12 (packed_float3-safe)", stride_f3 == 12);

  /* R1: packed3 via float* base + 3*i -> element k at 64 + k*12. */
  bool r1_ok = true;
  for (int k = 0; k < 3; k++) {
    int eb = 64 + k * 12;
    if (rf(eb) != 100.0f + k || rf(eb + 4) != 200.0f + k || rf(eb + 8) != 300.0f + k)
      r1_ok = false;
  }
  expect("packed3 via float*+3i @12B (k=0,1,2)", r1_ok);

  /* R2: float* -> 128 + k*4. */
  bool r2_ok = true;
  for (int k = 0; k < 3; k++)
    if (rf(128 + k * 4) != 10.0f + k) r2_ok = false;
  expect("float* stride == 4", r2_ok);

  /* R3: uint* -> 192 + k*4. */
  bool r3_ok = true;
  for (int k = 0; k < 3; k++)
    if (ru32(192 + k * 4) != 7000u + k) r3_ok = false;
  expect("uint* stride == 4", r3_ok);

  /* R4: int* -> 256 + k*4. */
  bool r4_ok = true;
  for (int k = 0; k < 3; k++)
    if (ri32(256 + k * 4) != -100 - k) r4_ok = false;
  expect("int* stride == 4", r4_ok);

  /* R5: float4* -> 320 + k*stride (measure). */
  int f4o0 = find_f(320, 384, 50.0f);
  int f4o1 = find_f(320, 384, 51.0f);
  int stride_f4 = (f4o0 >= 320 && f4o1 >= 320) ? (f4o1 - f4o0) : -1;
  fprintf(stderr, "  float4* stride (region r5): off(0)=%d off(1)=%d => stride=%d (expect 16)\n",
          f4o0, f4o1, stride_f4);
  expect("float4* stride == 16", stride_f4 == 16);

  /* Canary: a spot in region 0 past the writes (byte 60) should be untouched. */
  bool canary_ok = (ru32(60) == STRIDE_CANARY);
  expect("canary at byte 60 intact", canary_ok);

  fprintf(stderr, "=== Slang stride check: %s ===\n\n", all_ok ? "ALL OK" : "MISMATCHES PRESENT");
  return all_ok;
}

/* =========================================================================
 * M4e S1b-4 STAGE A: SUB-WORD ACCESS probe (optional, env
 * CYCLES_VULKAN_SUBWORD_PROBE=1 / ..._SPV_WORD / ..._SPV_NARROW).
 *
 * Decides how Cycles' sub-word SoA fields (uint16_t bounce/queued_kernel/rng_offset,
 * uint8_t visibility/mnee, bool guiding -- element size 2 or 1, per the S1b-3 stride
 * table) should be WRITTEN. Adjacent elements share one 32-bit word, so a store to
 * element t must not clobber its neighbour. Six variants, each a full 256-thread x
 * 64-iteration dispatch on a freshly canaried 4 KB buffer, then byte-exact verify:
 *
 *   module W (subword_word.slang, uint32 + OpAtomicAnd/Or only, NO Int16/Int8):
 *     mode 0  atomic16  -- atomic AND/OR-on-word fallback, uint16 elements
 *     mode 1  atomic8   -- atomic AND/OR-on-word fallback, uint8 elements
 *     mode 2  RMW-neg16 -- NEGATIVE: non-atomic read-modify-write, uint16
 *     mode 3  RMW-neg8  -- NEGATIVE: non-atomic read-modify-write, uint8
 *   module N (subword_narrow.slang, Int16 + Int8):
 *     mode 0  true16    -- true uint16 store (needs feature; captures the VUID)
 *     mode 1  true8     -- true uint8 store
 *
 * Verification: for non-negative variants every element must equal its LAST-iteration
 * value (no lost/clobbered values) and every canary byte outside the region must be
 * 0xCD (no out-of-bounds). For the NEGATIVE variants the race is EXPECTED to lose
 * updates, so DETECTED means lost>0 (and canaries intact, i.e. corruption is confined
 * to lost updates, not OOB). If a negative shows lost=0 the race simply did not
 * manifest on that device -- reported honestly as NOT-DETECTED.
 *
 * Also prints the device-REPORTED 8/16-bit features (item 1) and, for module N, the
 * run script greps the validation VUID to cite which feature gates Int16/Int8 through
 * PhysicalStorageBuffer. Run on BOTH the RX 570 and SwiftShader (DEVICE RULE). */

namespace {

constexpr uint32_t SUB_BUF_BYTES = 4096;
constexpr uint8_t SUB_CANARY = 0xCDu;   // prefill byte; unwritten locations stay 0xCD
constexpr uint32_t SUB_N = 256;         // N threads, element t = tid.x
constexpr uint32_t SUB_M = 64;          // iterations (MW == MN == 64)
constexpr uint32_t SUB_REGION_A = 0;    // uint16 elements at byte 0  (256 * 2 = 512 B)
constexpr uint32_t SUB_REGION_B = 1024; // uint8 elements at byte 1024 (256 * 1 = 256 B)
constexpr uint32_t SUB_PUSH_BYTES = 24; // PushW/PushN std430: ptr@0, ptr@8, mode@16

// Host mirrors of the shaders' push structs (std430, offsets confirmed from the SPIR-V).
struct WordPushHost {
  uint64_t wordA, wordB;
  uint32_t mode, pad;
};
struct NarrowPushHost {
  uint64_t p16, p8;
  uint32_t mode, pad;
};

struct SubPipe {
  VkShaderModule module = VK_NULL_HANDLE;
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;
};
struct SubResources {
  VkDevice device = VK_NULL_HANDLE;
  SubPipe word, narrow;
  VkCommandPool pool = VK_NULL_HANDLE;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  VkBuffer buf = VK_NULL_HANDLE;
  VkDeviceMemory mem = VK_NULL_HANDLE;

  ~SubResources()
  {
    if (device == VK_NULL_HANDLE) return;
    for (SubPipe *p : {&word, &narrow}) {
      if (p->pipeline) vkDestroyPipeline(device, p->pipeline, nullptr);
      if (p->layout) vkDestroyPipelineLayout(device, p->layout, nullptr);
      if (p->module) vkDestroyShaderModule(device, p->module, nullptr);
    }
    if (pool) vkDestroyCommandPool(device, pool, nullptr);
    if (buf) vkDestroyBuffer(device, buf, nullptr);
    if (mem) vkFreeMemory(device, mem, nullptr);
  }
};

} /* namespace */

// Mirrors of the shaders' value functions. The LAST iteration (SUB_M-1) is what a
// correct store leaves behind; sub_v16/sub_v8 are exactly subword_{word,narrow}.slang.
static uint32_t sub_v16(uint32_t t, uint32_t iter)
{
  return (iter * 37u + t * 3u + 11u) & 0xFFFFu;
}
static uint32_t sub_v8(uint32_t t, uint32_t iter)
{
  return (iter * 13u + t * 5u + 7u) & 0xFFu;
}

bool vulkan_subword_check(VulkanDevice *dev)
{
  static bool ran = false;
  if (ran) return true;
  ran = true;
  if (getenv("CYCLES_VULKAN_SUBWORD_PROBE") == nullptr) return true;

  LOG_INFO << "Vulkan: sub-word access probe starting (CYCLES_VULKAN_SUBWORD_PROBE=1).";

  const VkDevice vk_dev = dev->vk_device();
  const VkPhysicalDevice phys = dev->vk_physical_device();
  SubResources res;
  res.device = vk_dev;

  /* ---- 0. Device-REPORTED 8/16-bit features (Stage A item 1). ---- */
  /* have_narrow gates the narrow (Int16/Int8) module: SwiftShader reports NONE of
   * these and hard-crashes (0xC00000FD, same trap as Int64) when asked to JIT a
   * module that declares OpCapability Int16/Int8, so we SKIP it there and run only
   * the word-based (uint32 atomic / RMW) variants, which need no narrow capability. */
  bool have_narrow = false;
  {
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    VkPhysicalDeviceVulkan11Features f11{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceVulkan12Features f12{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    f2.pNext = &f11;
    f11.pNext = &f12;
    vkGetPhysicalDeviceFeatures2(phys, &f2);
    have_narrow = (f2.features.shaderInt16 != VK_FALSE) && (f12.shaderInt8 != VK_FALSE);
    fprintf(stderr,
            "\n=== M4e S1b-4 STAGE A: SUB-WORD ACCESS (device-REPORTED features) ===\n"
            "  shaderInt16              = %u  (SPIR-V Int16=22; base VkPhysicalDeviceFeatures,"
            " core 1.0)\n"
            "  storageBuffer16BitAccess = %u  (16BitStorage->Vulkan11; VK_KHR_16bit_storage)\n"
            "  shaderInt8               = %u  (SPIR-V Int8=39; 8BitStorage->Vulkan12;"
            " VK_KHR_shader_float16_int8)\n"
            "  storageBuffer8BitAccess  = %u  (8BitStorage->Vulkan12; VK_KHR_8bit_storage)\n"
            "  (device.cpp currently enables NONE of these; only shaderInt64 +"
            " bufferDeviceAddress)\n"
            "  narrow (Int16/Int8) module: %s\n",
            f2.features.shaderInt16,
            f11.storageBuffer16BitAccess,
            f12.shaderInt8,
            f12.storageBuffer8BitAccess,
            have_narrow ? "SUPPORTED -> will test true16/true8" :
                           "NOT supported -> SKIP true16/true8 (RX570-only)");
  }

  /* ---- 1. Load both .spv and build one pipeline each. ---- */
  auto make_pipe = [&](const char *env_path,
                       const char *fallback,
                       const char *name,
                       SubPipe &out) -> bool {
    std::vector<char> spirv;
    const char *envp = getenv(env_path);
    const char *cands[2] = {envp, fallback};
    const char *found = nullptr;
    for (const char *c : cands) {
      if (c && probe_read_whole_file(c, spirv)) {
        found = c;
        break;
      }
    }
    if (!found) {
      LOG_ERROR << "Vulkan: subword probe (" << name << "): cannot find " << fallback
                << " (compile with slangc or set " << env_path << ").";
      return false;
    }
    if (spirv.size() % 4 != 0) {
      LOG_ERROR << "Vulkan: subword probe (" << name << "): bad SPIR-V (" << found << ").";
      return false;
    }
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = spirv.size();
    smi.pCode = reinterpret_cast<const uint32_t *>(spirv.data());
    if (vkCreateShaderModule(vk_dev, &smi, nullptr, &out.module) != VK_SUCCESS) {
      LOG_ERROR << "Vulkan: subword probe (" << name << "): vkCreateShaderModule failed.";
      return false;
    }
    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = SUB_PUSH_BYTES;
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(vk_dev, &pli, nullptr, &out.layout) != VK_SUCCESS) {
      LOG_ERROR << "Vulkan: subword probe (" << name << "): pipeline layout failed.";
      return false;
    }
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = out.module;
    cpi.stage.pName = "main";
    cpi.layout = out.layout;
    if (vkCreateComputePipelines(vk_dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &out.pipeline) !=
        VK_SUCCESS) {
      LOG_ERROR << "Vulkan: subword probe (" << name << "): compute pipeline failed.";
      return false;
    }
    return true;
  };
  if (!make_pipe("CYCLES_VULKAN_SUBWORD_PROBE_SPV_WORD",
                 "subword_word.spv",
                 "word",
                 res.word))
    return false;
  /* Only build the narrow (Int16/Int8) pipeline if the device reports the features;
   * otherwise (SwiftShader) it would crash, so skip it and mark true16/true8 RX570-only. */
  if (have_narrow && !make_pipe("CYCLES_VULKAN_SUBWORD_PROBE_SPV_NARROW",
                                "subword_narrow.spv",
                                "narrow",
                                res.narrow))
    return false;

  /* ---- 2. One 4 KB host-visible BDA buffer. ---- */
  {
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = SUB_BUF_BYTES;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(vk_dev, &bci, nullptr, &res.buf) != VK_SUCCESS) {
      LOG_ERROR << "Vulkan: subword probe: buffer creation failed.";
      return false;
    }
    VkMemoryRequirements reqs;
    vkGetBufferMemoryRequirements(vk_dev, res.buf, &reqs);
    VkMemoryAllocateFlagsInfo flags_info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    flags_info.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.pNext = &flags_info;
    ai.allocationSize = reqs.size;
    ai.memoryTypeIndex = dev->find_memory_type(
        reqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (vkAllocateMemory(vk_dev, &ai, nullptr, &res.mem) != VK_SUCCESS) {
      LOG_ERROR << "Vulkan: subword probe: memory allocation failed.";
      return false;
    }
    if (vkBindBufferMemory(vk_dev, res.buf, res.mem, 0) != VK_SUCCESS) {
      LOG_ERROR << "Vulkan: subword probe: memory bind failed.";
      return false;
    }
  }
  VkBufferDeviceAddressInfo bai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
  bai.buffer = res.buf;
  const uint64_t base = vkGetBufferDeviceAddress(vk_dev, &bai);

  /* ---- 3. Command pool + one reusable command buffer + queue. ---- */
  const uint32_t family = dev->find_queue_family(dev->vk_physical_device());
  {
    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    /* RESET_COMMAND_BUFFER_BIT is REQUIRED for vkResetCommandBuffer (VUID
     * -vkResetCommandBuffer-commandBuffer-00046); TRANSIENT because each command
     * buffer is short-lived and re-recorded per variant. */
    cpci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
                 VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = family;
    if (vkCreateCommandPool(vk_dev, &cpci, nullptr, &res.pool) != VK_SUCCESS) {
      LOG_ERROR << "Vulkan: subword probe: command pool failed.";
      return false;
    }
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = res.pool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(vk_dev, &cbai, &res.cmd) != VK_SUCCESS) {
      LOG_ERROR << "Vulkan: subword probe: command buffer failed.";
      return false;
    }
  }
  VkQueue queue = VK_NULL_HANDLE;
  vkGetDeviceQueue(vk_dev, family, 0, &queue);

  /* ---- 4. Helpers: prefill canary / dispatch / readback. ---- */
  auto prefill = [&]() -> bool {
    void *m = nullptr;
    if (vkMapMemory(vk_dev, res.mem, 0, SUB_BUF_BYTES, 0, &m) != VK_SUCCESS) return false;
    memset(m, SUB_CANARY, SUB_BUF_BYTES);
    vkUnmapMemory(vk_dev, res.mem);
    return true;
  };
  auto readback = [&](std::vector<uint8_t> &raw) -> bool {
    void *m = nullptr;
    if (vkMapMemory(vk_dev, res.mem, 0, SUB_BUF_BYTES, 0, &m) != VK_SUCCESS) return false;
    memcpy(raw.data(), m, SUB_BUF_BYTES);
    vkUnmapMemory(vk_dev, res.mem);
    return true;
  };
  auto run = [&](VkPipeline pipe, VkPipelineLayout layout, const void *push) -> bool {
    if (vkResetCommandBuffer(res.cmd, 0) != VK_SUCCESS) return false;
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(res.cmd, &bi) != VK_SUCCESS) return false;
    vkCmdBindPipeline(res.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdPushConstants(
        res.cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, SUB_PUSH_BYTES, push);
    vkCmdDispatch(res.cmd, 1, 1, 1);
    if (vkEndCommandBuffer(res.cmd) != VK_SUCCESS) return false;
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence = VK_NULL_HANDLE;
    if (vkCreateFence(vk_dev, &fci, nullptr, &fence) != VK_SUCCESS) return false;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &res.cmd;
    bool ok = vkQueueSubmit(queue, 1, &si, fence) == VK_SUCCESS;
    if (ok)
      ok = vkWaitForFences(vk_dev, 1, &fence, VK_TRUE, 10000000000ull) == VK_SUCCESS;
    vkDestroyFence(vk_dev, fence, nullptr);
    return ok;
  };

  /* ---- 5. Run each variant, verify, report. ---- */
  struct Var {
    const char *name;
    bool narrow;  // false = module W (word/atomic/RMW), true = module N (narrow store)
    uint32_t mode;
    bool u16;     // true = uint16 elements (region A), false = uint8 elements (region B)
    bool neg;     // true = NEGATIVE (expects lost updates)
  };
  const Var vars[] = {
      {"atomic16 (AND/OR fallback)", false, 0, true, false},
      {"atomic8  (AND/OR fallback)", false, 1, false, false},
      {"RMW-neg16 (non-atomic NEG)", false, 2, true, true},
      {"RMW-neg8  (non-atomic NEG)", false, 3, false, true},
      {"true16 (narrow store)", true, 0, true, false},
      {"true8   (narrow store)", true, 1, false, false},
  };

  bool all_ok = true;
  for (const Var &v : vars) {
    /* Narrow variants need the Int16/Int8 module, which the device may not support
     * (SwiftShader). Skip them there rather than crash; mark clearly RX570-only. */
    if (v.narrow && !have_narrow) {
      fprintf(stderr, "  %-26s %-12s (device lacks Int16/Int8; RX570-only)\n", v.name, "SKIPPED");
      continue;
    }
    if (!prefill()) {
      LOG_ERROR << "Vulkan: subword probe: prefill failed.";
      return false;
    }
    if (!v.narrow) {
      WordPushHost push{};
      push.wordA = base + SUB_REGION_A;
      push.wordB = base + SUB_REGION_B;
      push.mode = v.mode;
      if (!run(res.word.pipeline, res.word.layout, &push)) {
        LOG_ERROR << "Vulkan: subword probe: dispatch failed (" << v.name << ").";
        return false;
      }
    }
    else {
      NarrowPushHost push{};
      push.p16 = base + SUB_REGION_A;
      push.p8 = base + SUB_REGION_B;
      push.mode = v.mode;
      if (!run(res.narrow.pipeline, res.narrow.layout, &push)) {
        LOG_ERROR << "Vulkan: subword probe: dispatch failed (" << v.name << ").";
        return false;
      }
    }

    std::vector<uint8_t> raw(SUB_BUF_BYTES);
    if (!readback(raw)) {
      LOG_ERROR << "Vulkan: subword probe: readback failed.";
      return false;
    }

    /* Verify: elements == LAST-iteration value; canaries outside the region == 0xCD. */
    uint32_t lost = 0;
    bool canary_ok = true;
    if (v.u16) {
      for (uint32_t t = 0; t < SUB_N; t++) {
        uint16_t got;
        memcpy(&got, raw.data() + SUB_REGION_A + t * 2, 2);
        if (got != (uint16_t)sub_v16(t, SUB_M - 1)) lost++;
      }
      for (uint32_t b = SUB_REGION_A + SUB_N * 2; b < SUB_BUF_BYTES; b++)
        if (raw[b] != SUB_CANARY) canary_ok = false;
    }
    else {
      for (uint32_t t = 0; t < SUB_N; t++) {
        if (raw[SUB_REGION_B + t] != (uint8_t)sub_v8(t, SUB_M - 1)) lost++;
      }
      for (uint32_t b = 0; b < SUB_REGION_B; b++)
        if (raw[b] != SUB_CANARY) canary_ok = false;
      for (uint32_t b = SUB_REGION_B + SUB_N; b < SUB_BUF_BYTES; b++)
        if (raw[b] != SUB_CANARY) canary_ok = false;
    }

    bool verdict;
    const char *tag;
    if (v.neg) {
      verdict = (lost > 0) && canary_ok;  // DETECTED: lost updates present, no OOB
      tag = verdict ? "DETECTED" : "NOT-DETECTED";
    }
    else {
      verdict = (lost == 0) && canary_ok;  // PASS: no lost values, no OOB
      tag = verdict ? "PASS" : "FAIL";
    }
    if (!verdict) all_ok = false;
    fprintf(stderr,
            "  %-26s %-12s lost=%-3u canary=%s\n",
            v.name,
            tag,
            lost,
            canary_ok ? "ok" : "CORRUPT");
  }

  fprintf(stderr,
          "=== sub-word access probe: %s ===\n\n",
          all_ok ? "all variants behaved as expected"
                 : "UNEXPECTED result (see above; a NOT-DETECTED negative = race did not"
                   " manifest on this device)");
  return all_ok;
}

/* =========================================================================
 * M4e S1b-1 minimal dynamic-index repro probe (optional, env
 * CYCLES_VULKAN_DYNINDEX_REPRO=1)
 * =========================================================================
 * Dispatches dynindex_repro.slang, the SMALLEST shader that reproduces the
 * Slang miscompile kdata_probe root-caused (M4e S1b-0): an inline DYNAMIC
 * index on a fixed-array member behind a device pointer (`p->data.words[i]`)
 * reads back WRONG elements (the index is not applied correctly). The
 * shader reads the same 8-word pattern {10..17} three ways plus a
 * constant-index control:
 *   out[0..7]   (a) p->data.words[i]        inline dynamic  -> WRONG (all 10)
 *   out[8..15]  (c) (&p->data.words[0])[i]  base-ptr dynamic -> correct
 *   out[16..23] (b) raw[i]                  raw dynamic     -> correct
 *   out[24]     p->data.words[3]            const index     -> correct (13)
 *   out[25]     raw[3]                      const index     -> correct (13)
 * The host allocates a 32-byte Params buffer holding {10..17}, pushes its
 * device address as BOTH the typed `p` and the raw view, dispatches one
 * thread, and compares each region against the known pattern. No
 * KernelData, no scene: isolates the codegen fault to a 2-struct shader. */

namespace {

constexpr uint32_t DYNI_WORDS = 8;
/* out layout: a(8) + c(8) + b(8) + const-a + const-b = 26 words. */
constexpr uint32_t DYNI_OUT_WORDS = 3 * DYNI_WORDS + 2; /* 26 */
constexpr uint32_t DYNI_PARAMS_WORDS = DYNI_WORDS;      /* 8  */
constexpr uint32_t DYNI_PUSH_BYTES = 24;                /* ReproPush = 3 pointers */
constexpr uint32_t DYNI_PREFILL = 0xA5A5A5A5u;          /* unwritten sentinel */

struct DyniResources {
  VkDevice device = VK_NULL_HANDLE;
  VkShaderModule module = VK_NULL_HANDLE;
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;
  VkCommandPool pool = VK_NULL_HANDLE;
  VkBuffer params_buffer = VK_NULL_HANDLE;
  VkDeviceMemory params_memory = VK_NULL_HANDLE;
  VkBuffer out_buffer = VK_NULL_HANDLE;
  VkDeviceMemory out_memory = VK_NULL_HANDLE;

  ~DyniResources()
  {
    if (device == VK_NULL_HANDLE) return;
    if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
    if (layout) vkDestroyPipelineLayout(device, layout, nullptr);
    if (module) vkDestroyShaderModule(device, module, nullptr);
    if (pool) vkDestroyCommandPool(device, pool, nullptr);
    if (params_buffer) vkDestroyBuffer(device, params_buffer, nullptr);
    if (params_memory) vkFreeMemory(device, params_memory, nullptr);
    if (out_buffer) vkDestroyBuffer(device, out_buffer, nullptr);
    if (out_memory) vkFreeMemory(device, out_memory, nullptr);
  }
};

} /* namespace */

bool vulkan_dynindex_repro(VulkanDevice *dev)
{
  static bool ran = false;
  if (ran) return true;
  ran = true;
  if (getenv("CYCLES_VULKAN_DYNINDEX_REPRO") == nullptr) return true;

  LOG_INFO << "Vulkan: DynIndex repro starting (CYCLES_VULKAN_DYNINDEX_REPRO=1).";

  DyniResources res;
  const VkDevice vk_dev = dev->vk_device();
  res.device = vk_dev;

  /* 1. SPIR-V (env override or repo-relative default). */
  std::vector<char> spirv;
  const char *env_path = getenv("CYCLES_VULKAN_DYNINDEX_REPRO_SPV");
  const char *candidates[2] = {env_path, "dynindex_repro.spv"};
  const char *found = nullptr;
  for (const char *c : candidates) {
    if (c && probe_read_whole_file(c, spirv)) {
      found = c;
      break;
    }
  }
  if (!found) {
    LOG_ERROR << "Vulkan: DynIndex repro: cannot find dynindex_repro.spv "
                 "(compile it with slangc or set CYCLES_VULKAN_DYNINDEX_REPRO_SPV).";
    return false;
  }
  if (spirv.size() % 4 != 0) {
    LOG_ERROR << "Vulkan: DynIndex repro: '" << found << "' is not a valid SPIR-V module.";
    return false;
  }
  LOG_INFO << "Vulkan: DynIndex repro: spirv='" << found << "' (" << spirv.size() << " bytes).";

  /* 2. Shader module. */
  VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smi.codeSize = spirv.size();
  smi.pCode = reinterpret_cast<const uint32_t *>(spirv.data());
  if (vkCreateShaderModule(vk_dev, &smi, nullptr, &res.module) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: DynIndex repro: vkCreateShaderModule failed.";
    return false;
  }

  /* 3. Pipeline layout: no descriptor sets, one 24-byte push range. */
  VkPushConstantRange pcr{};
  pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pcr.offset = 0;
  pcr.size = DYNI_PUSH_BYTES;
  VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pli.pushConstantRangeCount = 1;
  pli.pPushConstantRanges = &pcr;
  if (vkCreatePipelineLayout(vk_dev, &pli, nullptr, &res.layout) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: DynIndex repro: pipeline layout creation failed.";
    return false;
  }

  /* 4. Compute pipeline (entry "main"; slangc renames). */
  VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  cpi.stage.module = res.module;
  cpi.stage.pName = "main";
  cpi.layout = res.layout;
  if (vkCreateComputePipelines(vk_dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &res.pipeline) !=
      VK_SUCCESS) {
    LOG_ERROR << "Vulkan: DynIndex repro: compute pipeline creation failed.";
    return false;
  }

  /* 5. Two host-visible + coherent + device-address buffers: Params (the
   *    8-word pattern the shader reads) and out (the readback). A small
   *    lambda keeps the identical allocation boilerplate in one place. */
  auto make_bda_buffer = [&](VkDeviceSize size, VkBuffer &buf, VkDeviceMemory &mem) -> bool {
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = size;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(vk_dev, &bci, nullptr, &buf) != VK_SUCCESS) {
      LOG_ERROR << "Vulkan: DynIndex repro: buffer creation failed.";
      return false;
    }
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
    if (vkAllocateMemory(vk_dev, &ai, nullptr, &mem) != VK_SUCCESS) {
      LOG_ERROR << "Vulkan: DynIndex repro: memory allocation failed.";
      return false;
    }
    if (vkBindBufferMemory(vk_dev, buf, mem, 0) != VK_SUCCESS) {
      LOG_ERROR << "Vulkan: DynIndex repro: memory bind failed.";
      return false;
    }
    return true;
  };

  if (!make_bda_buffer(DYNI_PARAMS_WORDS * sizeof(uint32_t), res.params_buffer,
                       res.params_memory))
    return false;
  if (!make_bda_buffer(DYNI_OUT_WORDS * sizeof(uint32_t), res.out_buffer, res.out_memory))
    return false;

  /* Fill the params pattern {10..17} and prefill the out buffer with the
   * unwritten sentinel. Host-coherent memory: the queue submit makes the
   * host writes visible to the device, so no explicit barrier is needed. */
  {
    void *m = nullptr;
    if (vkMapMemory(vk_dev, res.params_memory, 0, DYNI_PARAMS_WORDS * 4, 0, &m) != VK_SUCCESS) {
      LOG_ERROR << "Vulkan: DynIndex repro: mapping params buffer failed.";
      return false;
    }
    uint32_t *w = static_cast<uint32_t *>(m);
    for (uint32_t i = 0; i < DYNI_PARAMS_WORDS; i++) w[i] = 10u + i;
    vkUnmapMemory(vk_dev, res.params_memory);
  }
  {
    void *m = nullptr;
    if (vkMapMemory(vk_dev, res.out_memory, 0, DYNI_OUT_WORDS * 4, 0, &m) != VK_SUCCESS) {
      LOG_ERROR << "Vulkan: DynIndex repro: mapping out buffer failed.";
      return false;
    }
    uint32_t *w = static_cast<uint32_t *>(m);
    for (uint32_t i = 0; i < DYNI_OUT_WORDS; i++) w[i] = DYNI_PREFILL;
    vkUnmapMemory(vk_dev, res.out_memory);
  }

  VkBufferDeviceAddressInfo bai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
  bai.buffer = res.params_buffer;
  const uint64_t params_addr = vkGetBufferDeviceAddress(vk_dev, &bai);
  bai.buffer = res.out_buffer;
  const uint64_t out_addr = vkGetBufferDeviceAddress(vk_dev, &bai);

  /* 6. Record + submit one workgroup of one thread. */
  const uint32_t family = dev->find_queue_family(dev->vk_physical_device());
  VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  cpci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
  cpci.queueFamilyIndex = family;
  if (vkCreateCommandPool(vk_dev, &cpci, nullptr, &res.pool) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: DynIndex repro: command pool creation failed.";
    return false;
  }
  VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbai.commandPool = res.pool;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  if (vkAllocateCommandBuffers(vk_dev, &cbai, &cmd) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: DynIndex repro: command buffer allocation failed.";
    return false;
  }
  VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  if (vkBeginCommandBuffer(cmd, &begin_info) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: DynIndex repro: vkBeginCommandBuffer failed.";
    return false;
  }
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, res.pipeline);
  /* p = typed params pointer, raw = SAME address as a flat uint32 view,
   * out = readback. The two params forms point at identical bytes. */
  const uint64_t push[3] = {params_addr, params_addr, out_addr};
  vkCmdPushConstants(cmd, res.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);
  vkCmdDispatch(cmd, 1, 1, 1);
  if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: DynIndex repro: vkEndCommandBuffer failed.";
    return false;
  }
  VkQueue queue = VK_NULL_HANDLE;
  vkGetDeviceQueue(vk_dev, family, 0, &queue);
  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkFence fence = VK_NULL_HANDLE;
  if (vkCreateFence(vk_dev, &fci, nullptr, &fence) != VK_SUCCESS) {
    LOG_ERROR << "Vulkan: DynIndex repro: fence creation failed.";
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
    LOG_ERROR << "Vulkan: DynIndex repro: queue submit/wait failed.";
    return false;
  }

  /* 7. Read back. */
  uint32_t out[DYNI_OUT_WORDS];
  {
    void *m = nullptr;
    if (vkMapMemory(vk_dev, res.out_memory, 0, DYNI_OUT_WORDS * 4, 0, &m) != VK_SUCCESS) {
      LOG_ERROR << "Vulkan: DynIndex repro: mapping readback buffer failed.";
      return false;
    }
    memcpy(out, m, sizeof(out));
    vkUnmapMemory(vk_dev, res.out_memory);
  }

  /* 8. Compare each region against the known pattern {10..17}. */
  uint32_t a_bad = 0, c_bad = 0, b_bad = 0, unwritten = 0;
  for (uint32_t i = 0; i < DYNI_WORDS; i++) {
    const uint32_t expect = 10u + i;
    if (out[i] == DYNI_PREFILL) unwritten++;
    if (out[DYNI_WORDS + i] == DYNI_PREFILL) unwritten++;
    if (out[2 * DYNI_WORDS + i] == DYNI_PREFILL) unwritten++;
    if (out[i] != expect) a_bad++;
    if (out[DYNI_WORDS + i] != expect) c_bad++;
    if (out[2 * DYNI_WORDS + i] != expect) b_bad++;
  }
  const uint32_t const_a = out[2 * DYNI_WORDS + DYNI_WORDS];      /* out[24] */
  const uint32_t const_b = out[2 * DYNI_WORDS + DYNI_WORDS + 1];  /* out[25] */
  LOG_INFO << string_printf(
      "Vulkan: DynIndex repro: (a) inline-dynamic wrong=%u/%u  (c) base-ptr wrong=%u/%u  "
      "(b) raw wrong=%u/%u  const-index a=%u b=%u (expect 13)  unwritten=%u",
      a_bad, DYNI_WORDS, c_bad, DYNI_WORDS, b_bad, DYNI_WORDS, const_a, const_b, unwritten);

  /* 9. Verdict.
   *    layout_ok : the base-ptr and raw dynamic views are correct -> the
   *                buffer contents + layout are fine.
   *    repro     : the inline-dynamic view reads back WRONG values while
   *                base-ptr + raw are correct and constant index works.
   *                That is the exact bug: only the inline DYNAMIC index on
   *                the fixed-array member behind a device pointer is broken.
   *                (It is NOT simply words[0] for every i -- the distortion
   *                is context-dependent; over 8 words it came back ~i>>2.) */
  const bool layout_ok = (c_bad == 0) && (b_bad == 0);
  const bool const_ok = (const_a == 13u) && (const_b == 13u);
  const bool repro = (a_bad > 0u) && layout_ok && const_ok;

  if (repro)
    LOG_ERROR << "Vulkan: DynIndex repro: BUG REPRODUCED - inline `p->data.words[i]` dynamic "
                 "index read back WRONG values (wrong=" << a_bad << "/" << DYNI_WORDS
              << "), while base-ptr and raw dynamic views and the constant index are all correct. "
                 "Isolates the Slang miscompile with no KernelData (see the out(a) dump on stderr).";
  else if (layout_ok && a_bad == 0)
    LOG_INFO << "Vulkan: DynIndex repro: BUG NOT REPRODUCED - the inline-dynamic view is correct "
                "here (toolchain may differ from the one that showed the fault).";
  else
    LOG_ERROR << "Vulkan: DynIndex repro: UNEXPECTED - see the counts above.";
  /* Mirror the verdict to unbuffered stderr: the Cycles progress bar can
   * overwrite a same-timestamp LOG line on the console, and this verdict is
   * the probe's whole point (AGENTS.md debug trap: use stderr for probes). */
  fprintf(stderr,
          "[dynindex-repro] inline-dynamic wrong=%u/%u base-ptr wrong=%u/%u raw wrong=%u/%u "
          "const a=%u b=%u -> %s\n",
          a_bad, DYNI_WORDS, c_bad, DYNI_WORDS, b_bad, DYNI_WORDS, const_a, const_b,
          repro ? "BUG REPRODUCED" : (layout_ok && a_bad == 0 ? "not reproduced" : "UNEXPECTED"));
  /* Dump the actual inline-dynamic words so the exact fault value is on the
   * record (is it words[0], or something else?). */
  fprintf(stderr, "[dynindex-repro] out(a)= ");
  for (uint32_t i = 0; i < DYNI_WORDS; i++)
    fprintf(stderr, "%u ", out[i]);
  fprintf(stderr, " | out(c)= ");
  for (uint32_t i = 0; i < DYNI_WORDS; i++)
    fprintf(stderr, "%u ", out[DYNI_WORDS + i]);
  fprintf(stderr, " | out(b)= ");
  for (uint32_t i = 0; i < DYNI_WORDS; i++)
    fprintf(stderr, "%u ", out[2 * DYNI_WORDS + i]);
  fprintf(stderr, "\n");

  return repro;
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
/* D3 (user decision): ONE extra INFORMATIONAL run, same origin but the camera
 * yawed 7 degrees about the vertical axis, at 128x128. The output/status
 * buffers are sized for this larger run once and reused by the 64x64 gate
 * run, so the second pass needs no second allocation. This run is reported
 * but never enters the verdict. */
constexpr uint32_t TP_MAX_WIDTH = 128;
constexpr uint32_t TP_MAX_HEIGHT = 128;
constexpr uint32_t TP_MAX_PIXELS = TP_MAX_WIDTH * TP_MAX_HEIGHT;
constexpr double TP_D3_YAW_DEG = 7.0;      /* degrees, about world +Y */
constexpr uint32_t TP_CANARY_RAYS = 16;    /* m2c pattern, m2c_main.cpp:97-99 */
constexpr uint32_t TP_CANARY_WORD = 0xCAFEBABEu;
constexpr uint32_t TP_UNWRITTEN_WORD = 0xDEADBEEFu;
constexpr uint32_t TP_PREPARE_SENTINEL = 0xB10C5EEDu; /* traverse_probe.slang */
constexpr uint32_t TP_PREPARE_OUT_WORDS = 645; /* 5 header words + whole blob */
constexpr uint32_t TP_PREPARE_PUSH_BYTES = 56; /* PreparePush (6 ptrs + word) */
constexpr uint32_t TP_PROBE_PUSH_BYTES = 32;   /* ProbePush (4 ptrs; +visits run 9) */
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
  uint64_t hdr, out, status, visits; /* visits: run 9 instrumentation */
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
/* `type` was added for Q6: both CPU references report the primitive type the
 * same way the shader does (PRIMITIVE_TRIANGLE on a hit, PRIMITIVE_NONE on a
 * miss; src/kernel/types.h:686-687). */
struct TpHit {
  float t;
  int32_t prim, object, type;
};

static TpHit tp_trace_cpu(const std::vector<TpWorldTri> &tris,
                          const float o[3], const float d[3])
{
  TpHit best{TP_NOHIT_T, -1, -1, PRIMITIVE_NONE};
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
      /* Same value the Cycles triangle test writes on a hit:
       * src/kernel/geom/triangle_intersect.h:109. */
      best.type = PRIMITIVE_TRIANGLE;
    }
  }
  return best;
}

/* ---- D1 reference: the SAME watertight Pluecker test the GPU uses ------ */
/* Ported statement for statement from agentreference/bvh_traverse.slang:93-117
 * (the M2c spike - the only Slang in this project known to compile and run on
 * the RX 570). AGENTS.md 11: the PASS/FAIL reference of a traversal gate must
 * be the same algorithm class as the port, so the verdict compares the GPU
 * against THIS function; the Moller-Trumbore brute force above is kept as an
 * informational line only (user decisions D1 and D2).
 *
 * `t` comes out identical to the spike's because the u/v bookkeeping below
 * never feeds back into it - it is kept so the port stays 1:1. */
static bool tp_plucker_test(const float P[3], const float D[3], float tmin,
                            float tmax, const float tri[9], float *ot)
{
  float v0[3], v1[3], v2[3], e0[3], e1[3], e2[3];
  for (int a = 0; a < 3; a++) { /* spike:95 */
    v0[a] = tri[0 + a] - P[a];
    v1[a] = tri[3 + a] - P[a];
    v2[a] = tri[6 + a] - P[a];
  }
  for (int a = 0; a < 3; a++) { /* spike:96 */
    e0[a] = v2[a] - v0[a];
    e1[a] = v0[a] - v1[a];
    e2[a] = v1[a] - v2[a];
  }
  /* Helper ops, written as lambdas exactly like tp_probe_edge above. */
  auto cross = [](const float *x, const float *y, float *r) {
    r[0] = x[1] * y[2] - x[2] * y[1];
    r[1] = x[2] * y[0] - x[0] * y[2];
    r[2] = x[0] * y[1] - x[1] * y[0];
  };
  auto dot = [](const float *x, const float *y) {
    return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
  };
  auto add = [](const float *x, const float *y, float *r) {
    for (int a = 0; a < 3; a++)
      r[a] = x[a] + y[a];
  };

  /* Side tests U, V, W (spike:97-104): reject when the ray misses all three
   * "sides" of the triangle by more than one float ulp of UVW. */
  float tmp[3], U, V, W;
  {
    float c0[3];
    add(v2, v0, c0);
    cross(e0, c0, tmp);
    U = dot(tmp, D);
  }
  {
    float c1[3];
    add(v0, v1, c1);
    cross(e1, c1, tmp);
    V = dot(tmp, D);
  }
  {
    float c2[3];
    add(v1, v2, c2);
    cross(e2, c2, tmp);
    W = dot(tmp, D);
  }
  const float UVW = U + V + W;
  const float eps = 1.1920929e-07f * std::fabs(UVW); /* spike:101 */
  const float minUVW = std::fmin(U, std::fmin(V, W));
  const float maxUVW = std::fmax(U, std::fmax(V, W));
  if (!(minUVW >= -eps || maxUVW <= eps))
    return false; /* spike:104 */

  /* Plane hit (spike:105-111). */
  float Ng1[3], Ng[3];
  cross(e1, e0, Ng1);
  for (int a = 0; a < 3; a++)
    Ng[a] = Ng1[a] + Ng1[a];
  const float den = dot(Ng, D);
  if (den == 0.0f)
    return false;
  const float t = dot(v0, Ng) / den;
  if (!(t >= tmin && t <= tmax))
    return false;

  /* Barycentrics (spike:112-114) - computed for a faithful 1:1 port, unused
   * by the gate, which only compares t/prim/object/type. */
  const float rcp_uvw = (std::fabs(UVW) < 1e-18f) ? 0.0f : 1.0f / UVW;
  const float ou = std::fmin(U * rcp_uvw, 1.0f);
  const float ov = std::fmin(V * rcp_uvw, 1.0f);
  (void)ou;
  (void)ov;

  *ot = t;
  return true;
}

/* Brute force over every world-space triangle using tp_plucker_test: the
 * reference of record for the M4c verdict (D1). Closest hit wins, ties go to
 * the lower enumeration index (strict '<'), same rule as tp_trace_cpu. */
static TpHit tp_trace_cpu_plucker(const std::vector<TpWorldTri> &tris,
                                  const float o[3], const float d[3])
{
  TpHit best{TP_NOHIT_T, -1, -1, PRIMITIVE_NONE};
  for (size_t i = 0; i < tris.size(); i++) {
    float t;
    if (tp_plucker_test(o, d, TP_TMIN, best.t, tris[i].v, &t)) {
      best.t = t;
      best.prim = (int32_t)tris[i].prim;
      best.object = (int32_t)tris[i].obj;
      best.type = PRIMITIVE_TRIANGLE; /* types.h:687 */
    }
  }
  return best;
}

/* ---- camera (Q4) ------------------------------------------------------- */
/* `yaw_deg` is 0 for the 64x64 gate run. For the D3 informational run the
 * view direction is rotated about the vertical axis while the origin stays
 * (0, -6, 3), which is what "camera rotated N degrees about the vertical
 * axis" means for a pinhole camera: the image plane swings, the eye does
 * not. Rotation about +Y (right-handed, y up): x' = x*cos + z*sin,
 * z' = -x*sin + z*cos. */
static void tp_build_camera(TpCam &cam, double yaw_deg)
{
  const double o[3] = {0.0, -6.0, 3.0};
  const double target[3] = {0.0, 0.0, 0.0};
  const double up[3] = {0.0, 1.0, 0.0};
  double f[3] = {target[0] - o[0], target[1] - o[1], target[2] - o[2]};
  const double flen = std::sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]);
  for (int a = 0; a < 3; a++)
    f[a] /= flen;
  if (yaw_deg != 0.0) {
    const double rad = yaw_deg * 3.14159265358979323846 / 180.0;
    const double cy = std::cos(rad), sy = std::sin(rad);
    const double fx = f[0] * cy + f[2] * sy;
    const double fz = -f[0] * sy + f[2] * cy;
    f[0] = fx;
    f[2] = fz;
    /* f is still unit length: a rotation preserves the norm. */
  }
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
static void tp_pixel_dir(const TpCam &cam, uint32_t x, uint32_t y, uint32_t w,
                         uint32_t h, float d[3])
{
  const double s = ((double)x + 0.5) / (double)w;
  const double t = 1.0 - ((double)y + 0.5) / (double)h;
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
/* Decision 2: a tie is a pixel where the reference and the GPU report the
 * same distance but a different (object, prim) pair. The breakdown counts
 * ties per (ref object, gpu object, same prim?) triple, so the report can
 * NAME the pair that produces them instead of just counting them. Object
 * indices are a scene detail, so the array is deliberately small: index 0..15
 * each side, and anything wider is counted in `tie_out_of_range` rather than
 * being read out of bounds (AGENTS 4: raw indices are never trusted). */
constexpr uint32_t TP_TIE_OBJS = 16;
struct TpCmp {
  size_t same = 0, miss = 0, ties = 0, grazing = 0, bad = 0, errors = 0;
  /* Q6: primitive-type disagreements, split by row kind so the report can
   * say which of the two assertions failed. */
  size_t type_bad_hit = 0, type_bad_miss = 0;
  /* Decision 2 tie breakdown. tie_same[i][j] = ties where the reference said
   * object i and the GPU said object j with the SAME prim index;
   * tie_diff[i][j] = same but a DIFFERENT prim index. */
  size_t tie_same[TP_TIE_OBJS][TP_TIE_OBJS] = {};
  size_t tie_diff[TP_TIE_OBJS][TP_TIE_OBJS] = {};
  size_t tie_out_of_range = 0;
  std::vector<size_t> examples;
};

static TpCmp tp_compare(const std::vector<TpHit> &ref,
                        const std::vector<TpIntersection> &gpu,
                        const std::vector<int32_t> &status,
                        const std::vector<TpWorldTri> &tris, const TpCam &cam,
                        bool details, uint32_t w, uint32_t h)
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
    /* `agree_row` = hit/miss AND (object, prim) already agree, so the type
     * assertion below is checked exactly on the rows Q6 talks about and a
     * grazing/bad row is never counted twice. */
    bool agree_row = false;
    const bool r_miss = (r.prim < 0 && r.object < 0);
    const bool g_miss = (g.prim < 0 && g.object < 0);
    if (r_miss && g_miss) {
      c.miss++;
      agree_row = true;
    }
    else if (!r_miss && !g_miss && r.prim == g.prim && r.object == g.object) {
      /* Same (object, prim): t must agree within 1e-4 relative. */
      if (std::fabs(g.t - r.t) <= 1e-4f * std::fmax(1.0f, r.t)) {
        c.same++;
        agree_row = true;
      }
      else
        disagree = true;
    }
    else if (!r_miss && !g_miss &&
             std::fabs(g.t - r.t) <= 1e-3f * std::fmax(1.0f, r.t)) {
      /* different prim, same distance: an edge tie, acceptable.
       * Decision 2: also tally WHICH pair disagreed, so the report can test
       * the hypothesis "the ties are the two coincident objects". */
      c.ties++;
      if (r.object >= 0 && g.object >= 0 && r.object < (int32_t)TP_TIE_OBJS &&
          g.object < (int32_t)TP_TIE_OBJS) {
        if (r.prim == g.prim)
          c.tie_same[r.object][g.object]++;
        else
          c.tie_diff[r.object][g.object]++;
      }
      else {
        c.tie_out_of_range++;
      }
    }
    else {
      disagree = true;
    }

    /* Q6. Miss rows: Cycles' traversal initialises t, u, v, prim and object
     * and leaves `type` alone (src/kernel/bvh/traversal.h:51-55); the caller
     * then forces prim = PRIM_NONE
     * (src/kernel/integrator/intersect_closest.h:379-391) and never reads
     * `type` unless prim != PRIM_NONE (intersect_closest.h:309). So the value
     * the gate asserts on a miss row is PRIMITIVE_NONE = 0
     * (src/kernel/types.h:686) - which is what traverse.slang:237 writes.
     * Hit rows: PRIMITIVE_TRIANGLE (types.h:687), as written by
     * src/kernel/geom/triangle_intersect.h:109. */
    if (agree_row) {
      const int32_t want = r_miss ? PRIMITIVE_NONE : r.type;
      if (g.type != want) {
        if (r_miss)
          c.type_bad_miss++;
        else
          c.type_bad_hit++;
        if (details)
          LOG_INFO << string_printf(
              "Vulkan: traverse probe:   TYPE mismatch on %s row (%u,%u): "
              "ref type %d gpu type %d",
              r_miss ? "miss" : "hit", (unsigned)(i % w), (unsigned)(i / w),
              want, g.type);
        if (c.examples.size() < 5)
          c.examples.push_back(i);
      }
    }

    if (!disagree)
      continue;
    float d[3];
    tp_pixel_dir(cam, (uint32_t)(i % w), (uint32_t)(i / w), w, h, d);
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
          (unsigned)(i % w), (unsigned)(i / w), (double)r.t,
          r.object, r.prim, (double)g.t, g.object, g.prim, e.margin);
    }
    if (c.examples.size() < 5)
      c.examples.push_back(i);
  }
  return c;
}

/* Gate tolerance (m2c_main.cpp:223-226): no hard mismatches, no shader
 * errors, and at most max(8, n/50000) edge-grazing disagreements. UNCHANGED
 * by D1 - the Pluecker reference swapped in above is the only change to the
 * verdict rule. */
static bool tp_cmp_ok(const TpCmp &c, size_t n)
{
  const size_t grazing_limit = std::max<size_t>(8, n / 50000);
  return c.bad == 0 && c.errors == 0 && c.grazing <= grazing_limit;
}

/* tp_cmp_ok plus the Q6 type assertion. Used for the main run AND for the
 * negative tests, so a negative that only shows up as a type mismatch still
 * counts as detected (and a type mismatch in the main run still fails). */
static bool tp_all_ok(const TpCmp &c, size_t n)
{
  return tp_cmp_ok(c, n) && c.type_bad_hit == 0 && c.type_bad_miss == 0;
}

static void tp_print_cmp(const char *title, const TpCmp &c, size_t n, uint32_t w)
{
  LOG_INFO << "Vulkan: traverse probe: " << title << " (" << n << " pixels)";
  LOG_INFO << "Vulkan: traverse probe:   both missed:                   " << c.miss;
  LOG_INFO << "Vulkan: traverse probe:   same (object,prim), same t:    " << c.same;
  LOG_INFO << "Vulkan: traverse probe:   edge ties (acceptable):        " << c.ties;
  LOG_INFO << "Vulkan: traverse probe:   edge-grazing disagreements:    " << c.grazing;
  LOG_INFO << "Vulkan: traverse probe:   MISMATCHES (not near an edge): " << c.bad;
  LOG_INFO << "Vulkan: traverse probe:   shader error codes:            " << c.errors;
  LOG_INFO << "Vulkan: traverse probe:   type mismatches (hit/miss):    "
           << c.type_bad_hit << "/" << c.type_bad_miss;
  for (size_t i : c.examples) {
    LOG_INFO << "Vulkan: traverse probe:   example pixel (" << (i % w)
             << "," << (i / w) << ")";
  }
}

/* Decision 2: report WHERE the ties come from. Prints one line per
 * non-empty (ref object, gpu object) pair, split by same/different prim. */
static void tp_print_tie_breakdown(const TpCmp &c, const char *label)
{
  LOG_INFO << "Vulkan: traverse probe: tie breakdown (" << label << "): "
           << c.ties << " ties";
  for (uint32_t r = 0; r < TP_TIE_OBJS; r++) {
    for (uint32_t g = 0; g < TP_TIE_OBJS; g++) {
      const size_t same = c.tie_same[r][g];
      const size_t diff = c.tie_diff[r][g];
      if (same == 0 && diff == 0)
        continue;
      LOG_INFO << string_printf(
          "Vulkan: traverse probe:   ref obj %u vs gpu obj %u: %zu ties "
          "(%zu same prim, %zu different prim)",
          r, g, same + diff, same, diff);
    }
  }
  if (c.tie_out_of_range)
    LOG_INFO << "Vulkan: traverse probe:   ties with an object index outside "
             << (TP_TIE_OBJS - 1) << ": " << c.tie_out_of_range;
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
  /* run 9: per-pixel instance-visit counter (instrumentation only). */
  VkBuffer visits_buf = VK_NULL_HANDLE;
  VkDeviceMemory hdr_mem = VK_NULL_HANDLE, out_mem = VK_NULL_HANDLE;
  VkDeviceMemory status_mem = VK_NULL_HANDLE, cksum_mem = VK_NULL_HANDLE;
  VkDeviceMemory visits_mem = VK_NULL_HANDLE;
  void *hdr_map = nullptr, *out_map = nullptr;
  void *status_map = nullptr, *cksum_map = nullptr;
  void *visits_map = nullptr;
  uint64_t hdr_addr = 0, out_addr = 0, status_addr = 0, cksum_addr = 0;
  uint64_t visits_addr = 0;

  ~TpResources()
  {
    if (device == VK_NULL_HANDLE)
      return;
    if (hdr_map) vkUnmapMemory(device, hdr_mem);
    if (out_map) vkUnmapMemory(device, out_mem);
    if (status_map) vkUnmapMemory(device, status_mem);
    if (cksum_map) vkUnmapMemory(device, cksum_mem);
    if (visits_map) vkUnmapMemory(device, visits_mem);
    if (hdr_buf) vkDestroyBuffer(device, hdr_buf, nullptr);
    if (out_buf) vkDestroyBuffer(device, out_buf, nullptr);
    if (status_buf) vkDestroyBuffer(device, status_buf, nullptr);
    if (cksum_buf) vkDestroyBuffer(device, cksum_buf, nullptr);
    if (visits_buf) vkDestroyBuffer(device, visits_buf, nullptr);
    if (hdr_mem) vkFreeMemory(device, hdr_mem, nullptr);
    if (out_mem) vkFreeMemory(device, out_mem, nullptr);
    if (status_mem) vkFreeMemory(device, status_mem, nullptr);
    if (cksum_mem) vkFreeMemory(device, cksum_mem, nullptr);
    if (visits_mem) vkFreeMemory(device, visits_mem, nullptr);
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

  /* 3. Reference world triangles + camera + the CPU references.
   *
   * DECISION 1: the GATE camera is the yawed one (128x128, 7 degrees about
   * the vertical axis). The un-yawed 64x64 camera is kept but is now only an
   * INFORMATIONAL run: with the camera aimed straight down a cube's edges,
   * every silhouette pixel puts the ray exactly on a triangle edge, so the
   * two triangle tests legitimately disagree there (run 7: 19 grazing,
   * bad=0; the yawed run: 0 grazing, bad=0). A gate should not sit on a
   * degenerate configuration, so the verdict moved to the yawed run. */
  std::vector<TpWorldTri> world_tris;
  if (!tp_build_world_tris(dump, world_tris, err))
    return fail(err.c_str());
  TpCam cam;             /* informational only (decision 1) */
  TpCam cam_gate;        /* the verdict camera */
  tp_build_camera(cam, 0.0);
  tp_build_camera(cam_gate, TP_D3_YAW_DEG);
  const float fov_deg = (float)(2.0 * std::atan(TP_PLANE_HALF) * 180.0 / 3.14159265358979323846);
  LOG_INFO << string_printf(
      "Vulkan: traverse probe: GATE camera: origin (%.9g, %.9g, %.9g), yaw %.4g deg, "
      "128x128 rays, plane half-extent %.9g at depth 1, FOV %.4f deg",
      cam_gate.o[0], cam_gate.o[1], cam_gate.o[2], TP_D3_YAW_DEG, TP_PLANE_HALF,
      fov_deg);
  LOG_INFO << string_printf(
      "Vulkan: traverse probe:   plane lower-left (%.9g, %.9g, %.9g), width vec (%.9g, %.9g, %.9g), height vec (%.9g, %.9g, %.9g)",
      cam_gate.l[0], cam_gate.l[1], cam_gate.l[2], cam_gate.u[0], cam_gate.u[1],
      cam_gate.u[2], cam_gate.v[0], cam_gate.v[1], cam_gate.v[2]);
  LOG_INFO << string_printf(
      "Vulkan: traverse probe: INFO ONLY: aligned 64x64 camera kept as a "
      "degenerate-config run (origin %.9g, %.9g, %.9g, no yaw)",
      cam.o[0], cam.o[1], cam.o[2]);

  /* Gate references over the yawed 128x128 grid:
   *  ref_gate    - watertight Pluecker, the SAME test the GPU runs: the D1
   *                verdict reference;
   *  ref_gate_mt - Moller-Trumbore: D2 says informational only, never verdict.
   * ref_aligned  - Pluecker over the un-yawed 64x64 grid, informational only. */
  std::vector<TpHit> ref_gate((size_t)TP_MAX_PIXELS);
  std::vector<TpHit> ref_gate_mt((size_t)TP_MAX_PIXELS);
  for (uint32_t y = 0; y < TP_MAX_HEIGHT; y++) {
    for (uint32_t x = 0; x < TP_MAX_WIDTH; x++) {
      float d[3];
      tp_pixel_dir(cam_gate, x, y, TP_MAX_WIDTH, TP_MAX_HEIGHT, d);
      const size_t i = (size_t)y * TP_MAX_WIDTH + x;
      ref_gate[i] = tp_trace_cpu_plucker(world_tris, cam_gate.o, d);
      ref_gate_mt[i] = tp_trace_cpu(world_tris, cam_gate.o, d);
    }
  }
  std::vector<TpHit> ref_aligned(TP_PIXELS);
  for (uint32_t y = 0; y < TP_HEIGHT; y++) {
    for (uint32_t x = 0; x < TP_WIDTH; x++) {
      float d[3];
      tp_pixel_dir(cam, x, y, TP_WIDTH, TP_HEIGHT, d);
      ref_aligned[(size_t)y * TP_WIDTH + x] =
          tp_trace_cpu_plucker(world_tris, cam.o, d);
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
   *    prepare checksum block.
   *    The output/status buffers are sized for the LARGEST run (D3, 128x128)
   *    and the canary zone always sits at that fixed offset, so the 64x64
   *    gate run and the 128x128 informational run share one allocation. The
   *    slots past the 64x64 run's rays are never dispatched and so stay at
   *    TP_UNWRITTEN_WORD. */
  const VkDeviceSize out_words = (size_t)(TP_MAX_PIXELS + TP_CANARY_RAYS) * 6;
  const VkDeviceSize out_bytes = out_words * sizeof(uint32_t);
  const VkDeviceSize status_bytes =
      (VkDeviceSize)(TP_MAX_PIXELS + TP_CANARY_RAYS) * sizeof(int32_t);
  /* run 9: one uint per pixel for the instance-visit histogram. No canary
   * zone needed - this buffer is informational and never part of a verdict,
   * so a plain TP_MAX_PIXELS allocation is enough. */
  const VkDeviceSize visits_bytes =
      (VkDeviceSize)TP_MAX_PIXELS * sizeof(uint32_t);
  if (!tp_create_mapped_buffer(dev, TP_HEADER_BYTES, res.hdr_buf, res.hdr_mem,
                               &res.hdr_map, &res.hdr_addr) ||
      !tp_create_mapped_buffer(dev, out_bytes, res.out_buf, res.out_mem,
                               &res.out_map, &res.out_addr) ||
      !tp_create_mapped_buffer(dev, status_bytes, res.status_buf, res.status_mem,
                               &res.status_map, &res.status_addr) ||
      !tp_create_mapped_buffer(dev, visits_bytes, res.visits_buf, res.visits_mem,
                               &res.visits_map, &res.visits_addr) ||
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
    /* D4: never write into the repo root. The blob file is only produced when
     * CYCLES_VULKAN_TRAVERSE_DUMP_DIR is set, and it goes INTO that directory
     * (which is normally the temp-dir sidecar dump). */
    if (dump_env) {
      const std::string blob_path = dump_dir + "/m4c_live_kerneldata.bin";
      if (FILE *f = fopen(blob_path.c_str(), "wb")) {
        fwrite(&ck[5], 4, 640, f);
        fclose(f);
        LOG_INFO << "Vulkan: traverse probe: live KernelData blob written to "
                 << blob_path << " (640 words)";
      }
      else {
        LOG_WARNING << "Vulkan: traverse probe: cannot write " << blob_path
                    << " (continuing; the blob dump is a debugging aid only)";
      }
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
   *    canaries, dispatch, verify, copy out. `w`/`h`/`run_cam`/`run_ref` let
   *    the same helper serve the 64x64 gate run, the three 64x64 negative
   *    tests and the 128x128 D3 run without duplicating the Vulkan calls. */
  auto run_probe = [&](uint32_t visibility, uint32_t test_mode, uint32_t w,
                       uint32_t h, const TpCam &run_cam,
                       const std::vector<TpHit> &run_ref, TpCmp &cmp,
                       size_t &unwritten, size_t &err_pixels,
                       bool &canaries_ok, bool details,
                       std::vector<uint32_t> *visits_out = nullptr) -> bool {
    uint32_t *outw = (uint32_t *)res.out_map;
    const size_t real_words = (size_t)w * h * 6; /* Intersection = 6 words */
    const size_t canary_words = (size_t)TP_CANARY_RAYS * 6;
    /* The canary zone sits at a FIXED offset (past the largest run), so a
     * 64x64 dispatch can never be mistaken for a canary overwrite. */
    const size_t max_words = (size_t)TP_MAX_PIXELS * 6;
    for (size_t i = 0; i < max_words; i++)
      outw[i] = TP_UNWRITTEN_WORD;
    for (size_t i = 0; i < canary_words; i++)
      outw[max_words + i] = TP_CANARY_WORD;
    int32_t *statw = (int32_t *)res.status_map;
    for (size_t i = 0; i < TP_MAX_PIXELS; i++)
      statw[i] = (int32_t)TP_UNWRITTEN_WORD;
    /* run 9: the visits buffer gets the same unwritten sentinel, so a thread
     * that never ran shows up as 0xFFFFFFFF in the histogram sanity check. */
    uint32_t *visw = (uint32_t *)res.visits_map;
    for (size_t i = 0; i < TP_MAX_PIXELS; i++)
      visw[i] = TP_UNWRITTEN_WORD;

    /* Visibility, test_mode, resolution and camera live in the header the
     * host owns; the shader reads width/height/cam from it
     * (traverse_probe.slang:162-176), so this is how a run picks its grid. */
    TpHeader *hd = (TpHeader *)res.hdr_map;
    hd->visibility = visibility;
    hd->test_mode = test_mode;
    hd->width = w;
    hd->height = h;
    for (int i = 0; i < 3; i++) {
      hd->cam[0 + i] = run_cam.o[i];
      hd->cam[3 + i] = run_cam.l[i];
      hd->cam[6 + i] = run_cam.u[i];
      hd->cam[9 + i] = run_cam.v[i];
    }

    TpProbePush pp{};
    pp.hdr = res.hdr_addr;
    pp.out = res.out_addr;
    pp.status = res.status_addr;
    pp.visits = res.visits_addr; /* run 9 */
    auto no_barrier = [](VkCommandBuffer cmd) { (void)cmd; };
    if (!tp_dispatch(dev, res.pool, res.pipe_probe, res.layout_probe,
                     TP_PROBE_PUSH_BYTES, &pp, (w + 7) / 8, (h + 7) / 8,
                     no_barrier))
      return false;

    canaries_ok = true;
    for (size_t i = 0; i < canary_words; i++)
      if (outw[max_words + i] != TP_CANARY_WORD)
        canaries_ok = false;

    std::vector<TpIntersection> gpu((size_t)w * h);
    std::vector<int32_t> status((size_t)w * h);
    memcpy(gpu.data(), res.out_map, real_words * sizeof(uint32_t));
    memcpy(status.data(), res.status_map, (size_t)w * h * sizeof(int32_t));
    /* run 9: hand the per-ray instance-visit counts back to the caller (the
     * gate run builds the histogram from these). */
    if (visits_out) {
      visits_out->assign((size_t)w * h, 0u);
      memcpy(visits_out->data(), res.visits_map,
             (size_t)w * h * sizeof(uint32_t));
    }
    unwritten = 0;
    err_pixels = 0;
    for (size_t i = 0; i < (size_t)w * h; i++) {
      const uint32_t tbits = ((const uint32_t *)&gpu[i])[0];
      if (tbits == TP_UNWRITTEN_WORD && ((const uint32_t *)&gpu[i])[4] ==
                                            TP_UNWRITTEN_WORD)
        unwritten++;
      if (status[i] != 0)
        err_pixels++;
    }
    cmp = tp_compare(run_ref, gpu, status, world_tris, run_cam, details, w, h);
    return true;
  };

  /* 11. GATE run: visibility 0xFFFFFFFF, no faults (Q5), 128x128 rays from
   *    the YAWED camera (decision 1). */
  TpCmp main_cmp;
  size_t unwritten = 0, err_pixels = 0;
  bool canaries_ok = false;
  /* run 9: the gate run's per-ray instance-visit counts (instrumentation). */
  std::vector<uint32_t> gate_visits;
  if (!run_probe(0xFFFFFFFFu, 0, TP_MAX_WIDTH, TP_MAX_HEIGHT, cam_gate, ref_gate,
                 main_cmp, unwritten, err_pixels, canaries_ok, /*details=*/true,
                 &gate_visits))
    return fail("ray dispatch failed");
  tp_print_cmp("D1 verdict (GATE, 128x128 yawed camera): watertight Pluecker "
               "reference (same test as the GPU) vs GPU (ported traversal)",
               main_cmp, TP_MAX_PIXELS, TP_MAX_WIDTH);
  tp_print_tie_breakdown(main_cmp, "gate run");
  LOG_INFO << "Vulkan: traverse probe: canaries: "
           << (canaries_ok ? "INTACT" : "OVERWRITTEN !!!")
           << ", pixels never written: " << unwritten
           << ", shader error pixels: " << err_pixels;

  /* 11b. run 9 INSTRUMENTATION (informational only, never a verdict input):
   *    histogram of rays by how many instance leaves they entered, plus a
   *    sanity count of visits words the shader never wrote. This answers
   *    whether the gate camera's rays exercise the instance POP at all. */
  {
    size_t hist[5] = {0, 0, 0, 0, 0}; /* index 4 = "4 or more" */
    size_t vis_unwritten = 0;
    for (uint32_t v : gate_visits) {
      if (v == TP_UNWRITTEN_WORD) {
        vis_unwritten++;
        continue;
      }
      hist[v < 4u ? v : 4]++;
    }
    LOG_INFO << string_printf(
        "Vulkan: traverse probe: INFO ONLY (run 9) instance-visit histogram "
        "(gate camera): 0=%zu 1=%zu 2=%zu 3=%zu 4+=%zu (visits never written: %zu)",
        hist[0], hist[1], hist[2], hist[3], hist[4], vis_unwritten);
  }

  /* 11c. run 9 INSTRUMENTATION: snapshot the CLEAN gate result so each
   *    negative below can be diffed against it (task step 1b). Copied out of
   *    the still-mapped buffers; informational only. */
  std::vector<TpIntersection> clean_gate(TP_MAX_PIXELS);
  std::vector<int32_t> clean_status(TP_MAX_PIXELS);
  memcpy(clean_gate.data(), res.out_map,
         (size_t)TP_MAX_PIXELS * sizeof(TpIntersection));
  memcpy(clean_status.data(), res.status_map,
         (size_t)TP_MAX_PIXELS * sizeof(int32_t));

  /* D2: Moller-Trumbore kept as an INFORMATIONAL line only. It is a
   * different algorithm class from the port, so it never enters the verdict;
   * its `bad` count must stay 0, because a non-zero `bad` would mean the two
   * triangle tests disagree somewhere that is not near an edge. The output
   * buffer still holds the gate run's result (the aligned run and the
   * negatives below overwrite it), so the MT comparison is read straight out
   * of the mapped buffers. */
  {
    std::vector<TpIntersection> gpu((size_t)TP_MAX_PIXELS);
    std::vector<int32_t> status((size_t)TP_MAX_PIXELS);
    memcpy(gpu.data(), res.out_map,
           (size_t)TP_MAX_PIXELS * sizeof(TpIntersection));
    memcpy(status.data(), res.status_map,
           (size_t)TP_MAX_PIXELS * sizeof(int32_t));
    const TpCmp mt = tp_compare(ref_gate_mt, gpu, status, world_tris, cam_gate,
                                /*details=*/false, TP_MAX_WIDTH, TP_MAX_HEIGHT);
    LOG_INFO << string_printf(
        "Vulkan: traverse probe: INFO ONLY (D2, not part of the verdict) "
        "Moller-Trumbore vs GPU (gate rays): same=%zu grazing=%zu bad=%zu errors=%zu",
        mt.same, mt.grazing, mt.bad, mt.errors);
  }

  /* Per-cube hit counts (gate requirement): GPU hits grouped by object,
   * labelled with each object's world-space x centre. Reads the GATE run's
   * output still sitting in the mapped buffer. */
  {
    std::vector<size_t> hits((size_t)dump.object_count, 0);
    const TpIntersection *gpu = (const TpIntersection *)res.out_map;
    const int32_t *status = (const int32_t *)res.status_map;
    for (size_t i = 0; i < TP_MAX_PIXELS; i++) {
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

  /* 12. Negative tests (gate). Q7 assumption, restated: a negative test
   *    passes only when the gate reports FAIL for the deliberately broken
   *    input, and a negative that goes undetected fails the whole gate. */
  auto run_negative = [&](const char *label, uint32_t visibility,
                          uint32_t test_mode) -> bool {
    TpCmp c;
    size_t uw = 0, ep = 0;
    bool ck = false;
    if (!run_probe(visibility, test_mode, TP_MAX_WIDTH, TP_MAX_HEIGHT, cam_gate,
                   ref_gate, c, uw, ep, ck, /*details=*/false)) {
      LOG_ERROR << "Vulkan: traverse probe: NEGATIVE TEST " << label
                << ": dispatch failed";
      return false;
    }
    /* tp_all_ok = the unchanged tolerances (tp_cmp_ok) + the Q6 type
     * assertion, so a fault that only shows up as a wrong primitive type
     * still counts as detected. */
    const bool detected = !tp_all_ok(c, TP_MAX_PIXELS);
    LOG_INFO << string_printf(
        "Vulkan: traverse probe: NEGATIVE TEST %-28s: %s (bad=%zu errors=%zu "
        "grazing=%zu type=%zu/%zu)",
        label, detected ? "DETECTED (gate failure, as expected)" : "NOT DETECTED !!!",
        c.bad, c.errors, c.grazing, c.type_bad_hit, c.type_bad_miss);

    /* run 9 INSTRUMENTATION (task step 1b, INFORMATIONAL ONLY - it does not
     * enter `detected`): count pixels whose (object, prim, t) differ from the
     * CLEAN gate result of the same run. This is a direct fault-effect
     * measure: 0 means the fault changed nothing at all, independent of the
     * reference tolerances. t is compared with a small epsilon so a
     * sub-ulp wobble does not count as an effect. */
    {
      const TpIntersection *faulty = (const TpIntersection *)res.out_map;
      const int32_t *fstat = (const int32_t *)res.status_map;
      size_t diff = 0;
      for (size_t i = 0; i < TP_MAX_PIXELS; i++) {
        const bool status_diff = (fstat[i] != 0) != (clean_status[i] != 0);
        const bool obj_diff = faulty[i].object != clean_gate[i].object;
        const bool prim_diff = faulty[i].prim != clean_gate[i].prim;
        const bool t_diff = fabsf(faulty[i].t - clean_gate[i].t) > 1e-6f;
        if (status_diff || obj_diff || prim_diff || t_diff)
          diff++;
      }
      LOG_INFO << string_printf(
          "Vulkan: traverse probe: INFO ONLY (run 9) NEGATIVE %-28s: "
          "%zu / %d pixels differ from the clean GPU result",
          label, diff, (int)TP_MAX_PIXELS);
    }
    return detected;
  };
  const bool neg1 = run_negative("1 skip instance pop", 0xFFFFFFFFu, 1u /*TEST_MODE_SKIP_POP*/);
  const bool neg2 = run_negative("2 corrupt object_node", 0xFFFFFFFFu, 2u /*TEST_MODE_CORRUPT_OBJECT_NODE*/);
  const bool neg3 = run_negative("3 visibility mask 0", 0u, 0u);

  /* 12b. INFORMATIONAL ONLY (decision 1): the un-yawed 64x64 run, kept
   *    because it is the configuration that EXPOSED the degeneracy. It is a
   *    known degenerate: the camera looks straight down the cube edges, so
   *    the silhouette rays lie exactly on triangle edges and the two triangle
   *    tests legitimately disagree there. Never part of the verdict. */
  TpCmp aligned_cmp;
  {
    size_t uw2 = 0, ep2 = 0;
    bool ck2 = false;
    const bool ok = run_probe(0xFFFFFFFFu, 0, TP_WIDTH, TP_HEIGHT, cam,
                              ref_aligned, aligned_cmp, uw2, ep2, ck2,
                              /*details=*/false);
    tp_print_cmp("INFO ONLY, KNOWN DEGENERATE (rays lie on cube edges): "
                 "un-yawed 64x64 run, Pluecker reference vs GPU",
                 aligned_cmp, TP_PIXELS, TP_WIDTH);
    tp_print_tie_breakdown(aligned_cmp, "aligned 64x64 run");
    if (!ok || uw2 != 0 || ep2 != 0 || !ck2)
      LOG_ERROR << "Vulkan: traverse probe: aligned run did not complete cleanly";
  }

  /* 13. Verdict: the GATE run (128x128 yawed, Pluecker reference) + the three
   *    negatives only. The aligned 64x64 run and the Moller-Trumbore line are
   *    informational and never decide it.
   *    The tie count is reported and, per decision 3, ties above the same
   *    max(8, n/50000) limit mean STOP and report the breakdown. */
  const bool main_ok = tp_all_ok(main_cmp, TP_MAX_PIXELS) && canaries_ok &&
                       unwritten == 0 && err_pixels == 0;
  const size_t ties_limit = std::max<size_t>(8, (size_t)TP_MAX_PIXELS / 50000);
  if (main_cmp.ties > ties_limit)
    LOG_WARNING << "Vulkan: traverse probe: TIES " << main_cmp.ties
                << " exceed the " << ties_limit
                << " limit - STOP AND REPORT THE BREAKDOWN (decision 3)";
  const bool pass = main_ok && neg1 && neg2 && neg3 && infra_ok;
  LOG_INFO << "Vulkan: traverse probe: "
           << (pass ? "PASS - gate run matches the reference, all three faults detected"
                    : "FAIL - see the numbers above");
  return pass;
}

CCL_NAMESPACE_END

#endif /* WITH_VULKAN */
