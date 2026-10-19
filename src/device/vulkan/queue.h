/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#ifdef WITH_VULKAN

#include "device/queue.h"
#include <algorithm>

CCL_NAMESPACE_BEGIN

class VulkanDevice;

class VulkanQueue : public DeviceQueue {
 public:
  explicit VulkanQueue(VulkanDevice *device);
  ~VulkanQueue() override = default;

  int num_concurrent_states(const size_t state_size) const override;
  int num_concurrent_busy_states(const size_t state_size) const override;
  void init_execution() override;
  void load_image_info() override;
  bool enqueue(DeviceKernel kernel, const int work_size, const DeviceKernelArguments &args) override;
  bool synchronize() override;
  void zero_to_device(device_memory &mem) override;
  void copy_to_device(device_memory &mem) override;
  void copy_from_device(device_memory &mem) override;
  void *copy_from_device_synchronized(device_memory &mem, vector<uint8_t> &storage) override;

 protected:
  VulkanDevice *vulkan_device_;
};

CCL_NAMESPACE_END

#endif /* WITH_VULKAN */