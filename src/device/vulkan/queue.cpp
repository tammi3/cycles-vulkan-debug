/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 * SPDX-License-Identifier: Apache-2.0 */

#ifdef WITH_VULKAN

#include "device/vulkan/queue.h"
#include "device/vulkan/device_impl.h"
#include "device/vulkan/vulkan_pipeline.h"
#include "util/log.h"
#include <algorithm>

CCL_NAMESPACE_BEGIN

VulkanQueue::VulkanQueue(VulkanDevice *d) : DeviceQueue(d), vulkan_device_(d) {}

int VulkanQueue::num_concurrent_states(const size_t s) const {
  return (int)std::min<size_t>(1 << 20, (512ull << 20) / s);
}

int VulkanQueue::num_concurrent_busy_states(const size_t s) const { 
  return num_concurrent_states(s) / 4; 
}

void VulkanQueue::init_execution() {
  debug_init_execution();
  /* M4: Pipeline is already initialized in load_kernels(). Nothing to do here. */
}

void VulkanQueue::load_image_info() {
  // M4: Will be implemented when texture handling is added
}

bool VulkanQueue::enqueue(DeviceKernel k, const int n, const DeviceKernelArguments &args) {
  debug_enqueue_begin(k, n);

  /* M4: For now we only support the path_trace kernel (DEVICE_KERNEL_INTEGRATOR_PATH_TRACE).
   * The args contain: path_index, render_buffer, work_size.
   * We need to build VulkanKernelParams from the current scene state and dispatch. */
  if (k != DEVICE_KERNEL_INTEGRATOR_PATH_TRACE) {
    vulkan_device_->set_error("Vulkan: unsupported kernel type");
    debug_enqueue_end();
    return false;
  }

  /* Extract film buffer from args (render_buffer is the output film tile). */
  device_memory *film_mem = static_cast<device_memory *>(args.render_buffer);
  if (!film_mem || !film_mem->device_pointer) {
    vulkan_device_->set_error("Vulkan: invalid film buffer in enqueue");
    debug_enqueue_end();
    return false;
  }

  /* Build kernel params from device state. */
  VulkanKernelParams params{};
  params.film_width = film_mem->data_width;
  params.film_height = film_mem->data_height;
  params.tile_offset_x = 0;  // TODO: support tiling
  params.tile_offset_y = 0;
  params.tile_width = params.film_width;
  params.tile_height = params.film_height;
  params.samples_per_pixel = 1;  // TODO: get from integrator settings
  params.max_bounces = 2;        // TODO: get from integrator settings
  params.film_buffer = (uint64_t)film_mem->device_pointer;

  /* Camera parameters - need to be set from scene. For now use defaults. */
  params.cam_origin[0] = 0.0f; params.cam_origin[1] = 0.0f; params.cam_origin[2] = 5.0f;
  params.cam_aspect = float(params.film_width) / float(params.film_height);
  params.cam_forward[0] = 0.0f; params.cam_forward[1] = 0.0f; params.cam_forward[2] = -1.0f;
  params.cam_focal_length = 35.0f;
  params.cam_right[0] = 1.0f; params.cam_right[1] = 0.0f; params.cam_right[2] = 0.0f;
  params.cam_sensor_width = 36.0f;
  params.cam_up[0] = 0.0f; params.cam_up[1] = 1.0f; params.cam_up[2] = 0.0f;
  params.pad_cam = 0.0f;

  /* Scene data pointers - need to be populated from constant buffer / device memory. */
  params.bvh_nodes = vulkan_device_->get_bvh_nodes_address();
  params.bvh_prim_offsets = vulkan_device_->get_bvh_prim_offsets_address();
  params.bvh_prim_count = vulkan_device_->get_bvh_prim_count_address();
  params.tri_indices = vulkan_device_->get_tri_indices_address();
  params.vert_co = vulkan_device_->get_vert_co_address();
  params.object_transforms = 0;
  params.light_prim_offsets = 0;
  params.light_prim_counts = 0;
  params.light_emission = 0;
  params.num_triangles = vulkan_device_->get_num_triangles();
  params.num_objects = 0;
  params.num_lights = 0;
  params.pad_scene = 0;

  /* World background color. */
  params.bg_color[0] = 0.0f; params.bg_color[1] = 0.0f; params.bg_color[2] = 0.0f;
  params.pad_bg = 0.0f;

  /* RNG seed. */
  params.rng_seed = 0x12345678;  // TODO: proper seed per frame
  params.reserved_0 = 0;
  params.reserved_1 = 0;
  params.reserved_2 = 0;

  /* Create and submit command buffer. */
  VkCommandBufferAllocateInfo cmd_alloc{};
  cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cmd_alloc.commandPool = vulkan_device_->get_cmd_pool();
  cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cmd_alloc.commandBufferCount = 1;
  VkCommandBuffer cmd;
  if (vkAllocateCommandBuffers(vulkan_device_->vk_device(), &cmd_alloc, &cmd) != VK_SUCCESS) {
    vulkan_device_->set_error("Vulkan: failed to allocate command buffer");
    debug_enqueue_end();
    return false;
  }

  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  if (vkBeginCommandBuffer(cmd, &begin_info) != VK_SUCCESS) {
    vulkan_device_->set_error("Vulkan: failed to begin command buffer");
    vkFreeCommandBuffers(vulkan_device_->vk_device(), vulkan_device_->get_cmd_pool(), 1, &cmd);
    debug_enqueue_end();
    return false;
  }

  std::string dispatch_error;
  if (!vulkan_device_->pipeline().dispatch_tile(vulkan_device_, params, cmd, &dispatch_error)) {
    vulkan_device_->set_error("Vulkan: dispatch failed: " + dispatch_error);
    vkEndCommandBuffer(cmd);
    vkFreeCommandBuffers(vulkan_device_->vk_device(), vulkan_device_->get_cmd_pool(), 1, &cmd);
    debug_enqueue_end();
    return false;
  }

  if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
    vulkan_device_->set_error("Vulkan: failed to end command buffer");
    vkFreeCommandBuffers(vulkan_device_->vk_device(), vulkan_device_->get_cmd_pool(), 1, &cmd);
    debug_enqueue_end();
    return false;
  }

  vulkan_device_->submit_and_wait(cmd);

  debug_enqueue_end();
  return true;
}

bool VulkanQueue::synchronize() { 
  debug_synchronize();
  return true; 
}

void VulkanQueue::zero_to_device(device_memory &m) { 
  vulkan_device_->mem_zero(m); 
}

void VulkanQueue::copy_to_device(device_memory &m) { 
  vulkan_device_->mem_copy_to(m); 
}

void VulkanQueue::copy_from_device(device_memory &m) { 
  vulkan_device_->mem_copy_from(m, 0, m.data_width, m.data_height == 0 ? 1 : m.data_height, m.data_elements); 
}

void *VulkanQueue::copy_from_device_synchronized(device_memory &m, vector<uint8_t> &s) {
  s.resize(m.memory_size());
  vulkan_device_->mem_copy_from(m, 0, m.data_width, m.data_height == 0 ? 1 : m.data_height, m.data_elements);
  memcpy(s.data(), m.host_pointer, s.size());
  return s.data();
}

CCL_NAMESPACE_END

#endif /* WITH_VULKAN */