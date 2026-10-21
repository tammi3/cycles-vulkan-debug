/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 * SPDX-License-Identifier: Apache-2.0 */

#ifdef WITH_VULKAN

#include "device/vulkan/vulkan_pipeline.h"

#include "util/log.h"

#include <cstring>
#include <fstream>
#include <vector>

CCL_NAMESPACE_BEGIN

namespace {

bool read_whole_file(const std::string &path, std::vector<char> &out)
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

VkResult create_host_visible_buffer(VkDevice device,
                                    uint32_t memory_type_bits_from_reqs_unused,
                                    VkBufferUsageFlags usage,
                                    VkDeviceSize size,
                                    VkBuffer *out_buffer,
                                    VkDeviceMemory *out_memory,
                                    VulkanDevice *dev)
{
  VkBufferCreateInfo binfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  binfo.size = size;
  binfo.usage = usage;
  binfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VkResult res = vkCreateBuffer(device, &binfo, nullptr, out_buffer);
  if (res != VK_SUCCESS) return res;

  VkMemoryRequirements reqs;
  vkGetBufferMemoryRequirements(device, *out_buffer, &reqs);

  VkMemoryAllocateInfo ainfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ainfo.allocationSize = reqs.size;
  ainfo.memoryTypeIndex = dev->find_memory_type(
      reqs.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  res = vkAllocateMemory(device, &ainfo, nullptr, out_memory);
  if (res != VK_SUCCESS) {
    vkDestroyBuffer(device, *out_buffer, nullptr);
    return res;
  }
  return vkBindBufferMemory(device, *out_buffer, *out_memory, 0);
}

} /* namespace */

bool VulkanPipeline::init(VulkanDevice *d, const std::string &spirv_dir, std::string *error)
{
  device_ = d;
  const VkDevice device = d->vk_device();

  /* 1. Load SPIR-V produced by slangc at build time. */
  std::vector<char> spirv;
  const std::string spv_path = spirv_dir + "/path_trace.spv";
  if (!read_whole_file(spv_path, spirv)) {
    if (error) *error = "Vulkan: cannot open compiled kernel '" + spv_path + "'";
    return false;
  }

  VkShaderModuleCreateInfo sm_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  sm_info.codeSize = spirv.size();
  sm_info.pCode = reinterpret_cast<const uint32_t *>(spirv.data());
  if (vkCreateShaderModule(device, &sm_info, nullptr, &shader_module) != VK_SUCCESS) {
    if (error) *error = "Vulkan: vkCreateShaderModule failed for path_trace.spv";
    return false;
  }

  /* 2. Descriptor layout: one storage buffer (the params struct). */
  const VkDescriptorSetLayoutBinding binding{
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
  };
  VkDescriptorSetLayoutCreateInfo dsl_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dsl_info.bindingCount = 1;
  dsl_info.pBindings = &binding;
  if (vkCreateDescriptorSetLayout(device, &dsl_info, nullptr, &ds_layout) != VK_SUCCESS) {
    if (error) *error = "Vulkan: descriptor set layout creation failed";
    return false;
  }

  VkPipelineLayoutCreateInfo pl_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pl_info.setLayoutCount = 1;
  pl_info.pSetLayouts = &ds_layout;
  if (vkCreatePipelineLayout(device, &pl_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
    if (error) *error = "Vulkan: pipeline layout creation failed";
    return false;
  }

  /* 3. Compute pipeline. Entry point name emitted by slangc -Xspirv-entry-point. */
  const VkComputePipelineCreateInfo cp_info{
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage =
          {
              .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
              .stage = VK_SHADER_STAGE_COMPUTE_BIT,
              .module = shader_module,
              .pName = "path_trace_kernel",
          },
      .layout = pipeline_layout,
  };
  if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cp_info, nullptr, &pipeline) != VK_SUCCESS) {
    if (error) *error = "Vulkan: compute pipeline creation failed (bad entry point name?)";
    return false;
  }

  /* 4. Params buffer + pool + single descriptor set. */
  if (create_host_visible_buffer(device, 0,
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                 sizeof(VulkanKernelParams),
                                 &params_buffer, &params_memory, d) != VK_SUCCESS) {
    if (error) *error = "Vulkan: params buffer allocation failed";
    return false;
  }

  const VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
  VkDescriptorPoolCreateInfo dp_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dp_info.maxSets = 1;
  dp_info.poolSizeCount = 1;
  dp_info.pPoolSizes = &pool_size;
  if (vkCreateDescriptorPool(device, &dp_info, nullptr, &ds_pool) != VK_SUCCESS) {
    if (error) *error = "Vulkan: descriptor pool creation failed";
    return false;
  }

  VkDescriptorSetAllocateInfo ds_alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  ds_alloc.descriptorPool = ds_pool;
  ds_alloc.descriptorSetCount = 1;
  ds_alloc.pSetLayouts = &ds_layout;
  if (vkAllocateDescriptorSets(device, &ds_alloc, &descriptor_set) != VK_SUCCESS) {
    if (error) *error = "Vulkan: descriptor set allocation failed";
    return false;
  }

  const VkDescriptorBufferInfo dbi{params_buffer, 0, sizeof(VulkanKernelParams)};
  const VkWriteDescriptorSet write{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = descriptor_set,
      .dstBinding = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &dbi,
  };
  vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);

  return true;
}

void VulkanPipeline::destroy()
{
  if (!device_) return;
  const VkDevice device = device_->vk_device();
  if (params_buffer) vkDestroyBuffer(device, params_buffer, nullptr);
  if (params_memory) vkFreeMemory(device, params_memory, nullptr);
  if (ds_pool) vkDestroyDescriptorPool(device, ds_pool, nullptr);
  if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
  if (pipeline_layout) vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
  if (ds_layout) vkDestroyDescriptorSetLayout(device, ds_layout, nullptr);
  if (shader_module) vkDestroyShaderModule(device, shader_module, nullptr);
  *this = VulkanPipeline{};
}

bool VulkanPipeline::dispatch_tile(VulkanDevice *d,
                                   const VulkanKernelParams &params,
                                   VkCommandBuffer cmd,
                                   std::string *error)
{
  /* Upload params via mapped host-visible memory (coherent). */
  void *mapped = nullptr;
  if (vkMapMemory(d->vk_device(), params_memory, 0, sizeof(params), 0, &mapped) != VK_SUCCESS) {
    if (error) *error = "Vulkan: failed to map params buffer";
    return false;
  }
  memcpy(mapped, &params, sizeof(params));
  vkUnmapMemory(d->vk_device(), params_memory);

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1,
                          &descriptor_set, 0, nullptr);

  /* Workgroup size is fixed at 8x8 in the Slang kernel. */
  const uint32_t gx = (params.tile_width + 7) / 8;
  const uint32_t gy = (params.tile_height + 7) / 8;
  vkCmdDispatch(cmd, gx, gy, 1);
  return true;
}

CCL_NAMESPACE_END

#endif /* WITH_VULKAN */
