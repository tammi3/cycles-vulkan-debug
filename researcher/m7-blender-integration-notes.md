# Vulkan backend for Cycles - Blender integration plan (reviewed)

> Review status (2026-10-22): research by a read-only agent (Nemotron) against E:/blender-git; line numbers NOT independently verified - spot-check with grep before relying on them. Reviewed by Claude; original text below with inline corrections.
>
> Corrections / notes from review:
> 1. VK_KHR_ray_query / acceleration_structure are NOT needed (software BVH traversal; Polaris has no HW RT). use_hardware_raytracing = false. Two places corrected inline.
> 2. Phases 0-2 (device type, enumeration, memory, const_copy_to) already exist in the standalone fork (cycles-vulkan-debug). Blender integration = port the fork's diff vs its upstream Cycles base into intern/cycles plus the Blender-side Python/C++ glue. Check the fork base matches the Blender commit used.
> 3. `has_mnee_` in the DeviceInfo field list looks like a typo; check the real field names.
> 4. slangc must be bundled or required at build time for a custom Blender build (open).
> 5. Do not start integration before the M6 gate (kernel ABI must be frozen).

---

# Vulkan Backend for Cycles — Blender Integration Plan

> **Research basis**: Blender source at `E:/blender-git/blender` (commit ~2026-10-19).  
> **Current milestone**: M4e S1a complete (math-core probe). M4 gate not yet passed.  
> **Rule**: This is **research only** — no code changes to `cycles-vulkan-debug` until M4–M6 gates pass.

---

## 1. Where Blender Discovers Devices (C++ Core)

### 1.1 `intern/cycles/device/device.h` — `DeviceType` enum (lines 38–49)
```cpp
enum DeviceType {
  DEVICE_NONE = 0,
  DEVICE_CPU,
  DEVICE_CUDA,
  DEVICE_MULTI,
  DEVICE_OPTIX,
  DEVICE_HIP,
  DEVICE_HIPRT,
  DEVICE_METAL,
  DEVICE_ONEAPI,
  DEVICE_DUMMY,
  // ADD: DEVICE_VULKAN
};
```

### 1.2 `DeviceTypeMask` (lines 51–59) — needs `DEVICE_MASK_VULKAN = (1 << DEVICE_VULKAN)`

### 1.3 `Device::available_types()` (lines 227–250) — returns vector of enabled types
```cpp
vector<DeviceType> Device::available_types() {
  vector<DeviceType> types;
  types.push_back(DEVICE_CPU);
#ifdef WITH_CUDA
  types.push_back(DEVICE_CUDA);
#endif
// ... similarly for OPTIX, HIP, METAL, ONEAPI, HIPRT
  // ADD #ifdef WITH_VULKAN branch
  return types;
}
```

### 1.4 `Device::available_devices(uint mask)` (lines 252–373) — enumerates actual hardware
- Each backend has an `init()` + `info()` pair (e.g. `device_cuda_init`, `device_cuda_info`)
- Vulkan will need `device_vulkan_init()` + `device_vulkan_info(vector<DeviceInfo>&)`
- Must query Vulkan physical devices, filter by compute capability, fill `DeviceInfo`
- `DeviceInfo` fields to populate (see `device.h` lines 79–150):
  - `type = DEVICE_VULKAN`
  - `description` = GPU name (e.g. "AMD Radeon RX 570")
  - `id` = stable identifier (PCI bus ID + name hash)
  - `has_nanovdb`, `has_mnee_`, `has_osl`, `has_guiding`, `has_profiling`
  - `has_peer_memory` (Vulkan peer memory / external memory)
  - `has_gpu_queue = true`
  - `use_hardware_raytracing` = false for our backend (software traversal; NOT ray_query)
  - `denoisers` mask (OIDN on GPU, etc.)
  - `meets_driver_requirement` (Vulkan 1.3 + required extensions)

### 1.5 `Device::create()` (lines 101–165) — factory
```cpp
switch (info.type) {
  case DEVICE_CUDA: ...
  // ADD:
#ifdef WITH_VULKAN
  case DEVICE_VULKAN:
    if (device_vulkan_init()) {
      device = device_vulkan_create(info, stats, profiler, headless);
    }
    break;
#endif
}
```

### 1.6 `Device::type_from_string()` / `string_from_type()` — add "VULKAN" ↔ `DEVICE_VULKAN`

---

## 2. Vulkan Device Implementation (New Files)

### 2.1 `intern/cycles/device/vulkan/device.h`
```cpp
#pragma once
#include "util/string.h"
#include "util/unique_ptr.h"
#include "util/vector.h"

CCL_NAMESPACE_BEGIN

class Device;
class DeviceInfo;
class Profiler;
class Stats;

bool device_vulkan_init(bool *r_meets_driver_requirement = nullptr);
unique_ptr<Device> device_vulkan_create(const DeviceInfo &info,
                                        Stats &stats,
                                        Profiler &profiler,
                                        bool headless);
void device_vulkan_info(vector<DeviceInfo> &devices);
string device_vulkan_capabilities();

CCL_NAMESPACE_END
```

### 2.2 `intern/cycles/device/vulkan/device.cpp`
- `device_vulkan_init()`: load `vulkan-1.dll`, enumerate instance extensions, check `VK_KHR_portability_enumeration` (for MoltenVK), create `VkInstance`, return success/failure
- `device_vulkan_info()`: `vkEnumeratePhysicalDevices`, for each:
  - `vkGetPhysicalDeviceProperties` → name, vendorID, deviceID, driverVersion
  - `vkGetPhysicalDeviceFeatures` / `VkPhysicalDeviceVulkan13Features` → check required features (shaderInt64, storageBuffer16BitAccess, etc.)
  - `vkGetPhysicalDeviceQueueFamilyProperties` → find compute queue
  - Check extensions: `VK_KHR_spirv_1_4`, `VK_KHR_shader_float_controls`, `VK_KHR_buffer_device_address`, `VK_KHR_ray_query` (for RT), `VK_EXT_descriptor_indexing`
  - Build `DeviceInfo`, push to vector
- `device_vulkan_create()`: return `make_unique<VulkanDevice>(info, stats, profiler, headless)`
- `VulkanDevice` class inherits `GPUDevice` (like `CUDADevice`, `HIPDevice`)
  - Implements `mem_alloc`, `mem_copy_to`, `mem_free`, `gpu_queue_create`, `load_kernels`, `const_copy_to`, `build_bvh`, etc.
  - Kernel loading: Slang → SPIR-V at build time; `load_kernels` creates `VkPipeline` for each kernel entry point
  - `const_copy_to`: upload `KernelData` via staging buffer → device buffer (BDA)
  - `gpu_queue_create()`: returns `VulkanQueue` wrapping `VkQueue` + command pool

### 2.3 `intern/cycles/device/vulkan/queue.h/.cpp`
- `VulkanQueue::enqueue()`: record command buffer, bind pipeline, push constants / descriptor sets, `vkCmdDispatch`
- Synchronization: `VkFence` / `VkSemaphore` for host-device sync

### 2.4 `intern/cycles/device/vulkan/kernel.h/.cpp`
- `load_kernels(uint64_t kernel_features)`: create `VkShaderModule` from `.spv`, `VkPipelineLayout`, `VkComputePipeline`
- Kernel features map to specialization constants / defines

---

## 3. Blender Addon Preferences (Python)

### 3.1 `intern/cycles/blender/addon/properties.py`

#### `enum_device_type` (lines 1670–1677) — add Vulkan entry
```python
enum_device_type = (
    ('CPU', "CPU", "Use CPU for rendering"),
    ('CUDA', "CUDA", "Use NVIDIA CUDA for GPU acceleration"),
    ('OPTIX', "OptiX", "Use NVIDIA OptiX for GPU acceleration"),
    ('HIP', "HIP", "Use AMD HIP for GPU acceleration"),
    ('HIPRT', "HIP RT", "Use AMD HIP RT for hardware ray tracing"),
    ('METAL', "Metal", "Use Apple Metal for GPU acceleration"),
    ('ONEAPI', "oneAPI", "Use Intel oneAPI for GPU acceleration"),
    ('VULKAN', "Vulkan", "Use Vulkan for GPU acceleration"),  # ADD
)
```

#### `CyclesPreferences.get_device_types()` (lines 1692–1708)
```python
def get_device_types(self, context):
    import _cycles
    has_cuda, has_optix, has_hip, has_metal, has_oneapi, has_hiprt = _cycles.get_device_types()
    # ADD: has_vulkan = _cycles.get_device_types()[6]  (extend C++ tuple to 7)

    items = [('NONE', "None", n_("Do not use compute device"), 0)]
    if has_cuda:
        items.append(('CUDA', "CUDA", n_("Use CUDA for GPU acceleration"), 1))
    # ... existing ...
    if has_vulkan:  # ADD
        items.append(('VULKAN', "Vulkan", n_("Use Vulkan for GPU acceleration"), 7))
    return items
```

#### `ComputeDevice` enum in `blender/device.cpp` (lines 14–23) — add `COMPUTE_DEVICE_VULKAN = 7`

#### `blender_device_info()` (lines 84–202) — add Vulkan case
```cpp
else if (compute_device == COMPUTE_DEVICE_VULKAN) {
  mask |= DEVICE_MASK_VULKAN;
}
```

### 3.2 `intern/cycles/blender/python.cpp`

#### `get_device_types_func()` (lines 759–784) — extend tuple to 7 elements
```cpp
bool has_vulkan = false;
for (const DeviceType device_type : device_types) {
  // ... existing ...
  has_vulkan |= (device_type == DEVICE_VULKAN);
}
PyObject *list = PyTuple_New(7);  // was 6
// ...
PyTuple_SET_ITEM(list, 6, PyBool_FromLong(has_vulkan));
```

#### `set_device_override_func()` (lines 786–828) — add Vulkan case
```cpp
else if (override == "VULKAN") {
  BlenderSession::device_override = DEVICE_MASK_VULKAN;
}
```

---

## 4. Build System (CMake)

### 4.1 `intern/cycles/CMakeLists.txt` — add option
```cmake
option(WITH_VULKAN "Enable Vulkan backend for Cycles" OFF)
if(WITH_VULKAN)
  find_package(Vulkan REQUIRED)
  add_subdirectory(device/vulkan)
endif()
```

### 4.2 `intern/cycles/device/CMakeLists.txt` — add vulkan subdir
```cmake
if(WITH_VULKAN)
  add_subdirectory(vulkan)
  list(APPEND CYCLES_DEVICE_SOURCES
    vulkan/device.cpp
    vulkan/queue.cpp
    vulkan/kernel.cpp
  )
endif()
```

### 4.3 `intern/cycles/device/vulkan/CMakeLists.txt`
```cmake
add_library(cycles_vulkan_device
  device.cpp
  queue.cpp
  kernel.cpp
)
target_link_libraries(cycles_vulkan_device
  cycles_util
  ${VULKAN_LIBRARIES}
)
target_include_directories(cycles_vulkan_device PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
```

---

## 5. Kernel Compilation Pipeline (Slang → SPIR-V)

> **UNVERIFIED** — based on M4e S1a pattern in `cycles-vulkan-debug`

| Step | Current (cycles-vulkan-debug) | Blender Integration |
|------|-------------------------------|---------------------|
| Source | `.slang` in `src/kernel/device/vulkan/` | Same, under `intern/cycles/kernel/kernels/vulkan/` |
| Compile | `slangc -target spirv -stage compute -entry X -o X.spv` | CMake custom command at build time |
| Validate | `spirv-val X.spv` | Same, as CMake test |
| Load | `VulkanDevice::load_kernels()` reads `.spv` from disk | Same, but paths via `kernel_get_path()` |

**Key difference**: Blender's `Device::load_kernels()` is called with `kernel_features` bitmask. Vulkan backend must select/compile pipeline variants per feature set (like CUDA's `kernel.cu` + `#ifdef KERNEL_FEATURE_...`).

---

## 6. UI Flow Summary

```
User opens Preferences → System → Cycles Render Devices
    │
    ▼
CyclesPreferences.get_device_types()  (Python)
    │
    ▼
_cycles.get_device_types()  (python.cpp)
    │
    ▼
Device::available_types()  (device.cpp)  ──returns──► [CPU, CUDA, OPTIX, HIP, METAL, ONEAPI, VULKAN]
    │
    ▼
User selects "Vulkan" in Compute Device Type dropdown
    │
    ▼
CyclesPreferences.compute_device_type = "VULKAN"
    │
    ▼
Scene render: BlenderSync::get_session_params() → blender_device_info()
    │
    ▼
Device::available_devices(DEVICE_MASK_VULKAN | DEVICE_MASK_CPU)
    │
    ▼
device_vulkan_info() enumerates VkPhysicalDevices → DeviceInfo[]
    │
    ▼
User checks GPU devices in list → saved to CyclesPreferences.devices CollectionProperty
    │
    ▼
Render starts → Device::create(DEVICE_VULKAN) → device_vulkan_create() → VulkanDevice
```

---

## 7. Open Questions / UNVERIFIED Assumptions

| # | Question | Status |
|---|----------|--------|
| 1 | Does `slangc` need to be bundled or system-installed? | UNVERIFIED — M4e uses `C:\VulkanSDK\1.4.363.0\Bin\slangc.exe` |
| 2 | Where do `.spv` files live at runtime? `kernel_get_path()`? | UNVERIFIED — need to check `kernel/path.h` |
| 3 | Does Vulkan backend need `VK_KHR_ray_query` or `VK_KHR_acceleration_structure` for M4? | **CORRECTED: NO.** Our design is software BVH traversal in compute shaders; Polaris has no hardware ray queries. `use_hardware_raytracing` must be false. |
| 4 | How does `KernelData` get uploaded? `const_copy_to()` with BDA? | UNVERIFIED — see `device.cpp` `const_copy_to` pure virtual |
| 5 | Does `CyclesPreferences.devices` CollectionProperty persist Vulkan device IDs across restarts? | YES — same as other backends (id + type + use) |
| 6 | Multi-device (CPU + Vulkan) via `Device::get_multi_device()`? | Should work automatically if `preferences_device.multi_devices` populated |
| 7 | Denoising on Vulkan? | OIDN GPU path exists; needs `VK_KHR_external_memory` for interop |
| 8 | Blender's `gpu` module (viewport) vs Cycles device — separate? | YES — `gpu` is for viewport drawing; Cycles device is for render only |

---

## 8. Implementation Order (When M4–M6 Gates Pass)

| Phase | Tasks | Depends On |
|-------|-------|------------|
| **Phase 0** | Add `DEVICE_VULKAN` enum, stub `device_vulkan_init/info/create` returning dummy | M4 gate |
| **Phase 1** | Real `device_vulkan_info()` enumerating `VkPhysicalDevice` | Phase 0 + Vulkan SDK |
| **Phase 2** | `VulkanDevice` memory management (alloc/copy/free), `const_copy_to` for KernelData | Phase 1 |
| **Phase 3** | `load_kernels()` — compile Slang→SPIR-V at build time, load `.spv`, create pipelines | Phase 2 + M4 kernel set |
| **Phase 4** | `gpu_queue_create()` + `VulkanQueue::enqueue()` for path_trace dispatch | Phase 3 |
| **Phase 5** | Python bindings (`get_device_types`, `set_device_override`) | Phase 1 |
| **Phase 6** | Addon preferences (`enum_device_type`, `get_device_types`, `blender_device_info`) | Phase 5 |
| **Phase 7** | CMake integration (`WITH_VULKAN`, find_package, subdirs) | All above |
| **Phase 8** | End-to-end test: Blender Preferences → Vulkan device → render instance2.xml | Phase 7 |

---

## 9. Files to Create / Modify (Summary)

### New Files
```
intern/cycles/device/vulkan/device.h
intern/cycles/device/vulkan/device.cpp
intern/cycles/device/vulkan/queue.h
intern/cycles/device/vulkan/queue.cpp
intern/cycles/device/vulkan/kernel.h
intern/cycles/device/vulkan/kernel.cpp
intern/cycles/device/vulkan/CMakeLists.txt
intern/cycles/kernel/kernels/vulkan/*.slang  (ported from cycles-vulkan-debug)
```

### Modified Files
```
intern/cycles/device/device.h          (+DEVICE_VULKAN enum, mask)
intern/cycles/device/device.cpp        (+available_types, available_devices, create, string_from_type)
intern/cycles/device/CMakeLists.txt    (+vulkan subdir)
intern/cycles/CMakeLists.txt           (+WITH_VULKAN option)
intern/cycles/blender/addon/properties.py (+enum_device_type, get_device_types, COMPUTE_DEVICE_VULKAN)
intern/cycles/blender/device.cpp       (+blender_device_info Vulkan case)
intern/cycles/blender/python.cpp       (+get_device_types_func, set_device_override_func Vulkan)
```

---

## 10. Next Steps

1. **Complete M4 gate** (Lambertian sphere 16 spp vs CPU) — current priority
2. **M5–M6**: Diffuse/Principled BSDF, lights, shadows, textures
3. **Then** begin Phase 0 above

> **Do not start Blender integration code until M6 gate passes.** The Slang→SPIR-V kernel ABI, descriptor layouts, and `KernelData` upload must be frozen first.