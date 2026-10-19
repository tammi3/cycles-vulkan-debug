/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "util/string.h"
#include "util/unique_ptr.h"
#include "util/vector.h"

CCL_NAMESPACE_BEGIN

class Device;
class DeviceInfo;
class Profiler;
class Stats;

bool device_vulkan_init();
unique_ptr<Device> device_vulkan_create(const DeviceInfo &info,
                                        Stats &stats,
                                        Profiler &profiler,
                                        bool headless);
void device_vulkan_info(vector<DeviceInfo> &devices);
string device_vulkan_capabilities();

CCL_NAMESPACE_END