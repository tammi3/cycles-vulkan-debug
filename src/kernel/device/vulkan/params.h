/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

/* Fixed "kernel params" layout for the Vulkan backend (host C++ only).
 *
 * Cycles uploads exactly one value per kernel global through
 * Device::const_copy_to(name, host, size): "data" (KernelData), one call per
 * KERNEL_DATA_ARRAY (a device pointer, 8 bytes) and "integrator_state"
 * (IntegratorStateGPU). The device buffer must have a layout the shader can
 * compute without asking the host where each value landed, so this struct
 * mirrors Cycles' KernelParamsHIP (source: src/kernel/device/hip/globals.h:26-35)
 * and names are resolved with offsetof, the same way HIPDevice::const_copy_to
 * does it (source: src/device/hip/device_impl.cpp:640-648).
 *
 * Array slots are uint64_t instead of typed pointers: identical 8-byte size
 * (source: src/kernel/device/hip/globals.h:29-30), but host code can never
 * dereference them. Each slot holds a Buffer Device Address value. */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "kernel/types.h"

#include "kernel/integrator/state.h"
#include "kernel/util/profiler.h"

#include "util/color.h"
#include "util/types_image.h"

CCL_NAMESPACE_BEGIN

/* Field order must stay identical to KernelParamsHIP: data, then the
 * KERNEL_DATA_ARRAY list in kernel/data_arrays.h order, then integrator_state. */
struct KernelParamsVulkan {
  KernelData data;
#define KERNEL_DATA_ARRAY(type, name) uint64_t name;
#define KERNEL_DATA_ARRAY_WRITABLE(type, name) uint64_t name;
#include "kernel/data_arrays.h"
  IntegratorStateGPU integrator_state;
};

/* Layout sanity. KernelData is ccl_align(16) with its own static_assert
 * (source: src/kernel/types.h:1367, 1408), so alignment is 16 and the size is
 * a multiple of 16. Everything after it is 8 bytes wide, so no slot can land
 * at a non-8-byte offset. */
static_assert(alignof(KernelParamsVulkan) == 16,
              "KernelParamsVulkan must carry KernelData's 16-byte alignment");
static_assert(sizeof(KernelParamsVulkan) % 16 == 0,
              "KernelParamsVulkan size must be a multiple of 16");
static_assert(offsetof(KernelParamsVulkan, data) == 0,
              "KernelData must be the first member of KernelParamsVulkan");

/* Resolve a kernel global name to its byte offset and expected upload size in
 * KernelParamsVulkan. Returns false for names that are not part of the layout;
 * `name` is exactly what Cycles passes to Device::const_copy_to. */
inline bool kernel_params_offset(const char *name, size_t *out_offset, size_t *out_size)
{
#define KERNEL_DATA_ARRAY(data_type, data_name) \
  if (strcmp(name, #data_name) == 0) { \
    *out_offset = offsetof(KernelParamsVulkan, data_name); \
    *out_size = sizeof(((KernelParamsVulkan *)nullptr)->data_name); \
    return true; \
  }
#define KERNEL_DATA_ARRAY_WRITABLE(data_type, data_name) KERNEL_DATA_ARRAY(data_type, data_name)
  KERNEL_DATA_ARRAY(KernelData, data)
  KERNEL_DATA_ARRAY(IntegratorStateGPU, integrator_state)
#include "kernel/data_arrays.h"

  return false;
}

CCL_NAMESPACE_END
