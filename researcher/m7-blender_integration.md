a) Blender version and fork comparison
Blender version (E:/blender-git)
Command: grep -n "BLENDER_VERSION" E:\blender-git\blender\source\creator\BKE_blender_version.h
Output:
23:#define BLENDER_VERSION 504
25:#define BLENDER_VERSION_PATCH 0
27:#define BLENDER_VERSION_CYCLE alpha
Interpretation: Blender 5.4 alpha (504 = 5.04). Our fork reports CYCLES_VERSION 5.3.0 (src/util/version.h:12-14).
File-level diff summary: our fork vs E:/blender-git/intern/cycles
Command: diff -r c:\dev\cycles-vulkan-debug\src E:\blender-git\blender\intern\cycles --brief 2>&1 | head -50
Output: NOT FOUND (diff not run in this session — marked for next session)
Verified structural matches (ls comparison):
- Both have: app/, bvh/, cmake/, device/, integrator/, kernel/, scene/, session/, util/, graph/, hydra/, subd/, test/, doc/
- Our fork has device/vulkan/ — upstream does NOT
- Upstream has device/cuda/, device/hip/, device/hiprt/, device/metal/, device/oneapi/, device/optix/, device/cpu/, device/dummy/, device/multi/ — our fork only has device/vulkan/ + kernel/device/vulkan/
Key version difference: Upstream is Blender 5.4 alpha (Cycles ~5.4), our fork is 5.3.0. Rebase needed before M7.
b) How Blender's CMake builds/ships GPU kernels (CUDA/HIP)
CUDA kernel build (verified from CMakeLists.txt)
File: E:\blender-git\blender\intern\cycles\kernel\CMakeLists.txt:436
Output: add_subdirectory(device/cuda)
File: E:\blender-git\blender\intern\cycles\device\CMakeLists.txt:39,136
Output: 
set(SRC_CUDA
  cuda/device.cpp
  cuda/device.h
  cuda/device_impl.cpp
  cuda/device_impl.h
  cuda/graphics_interop.cpp
  cuda/graphics_interop.h
  cuda/kernel.cpp
  cuda/kernel.h
  cuda/queue.cpp
  cuda/queue.h
  cuda/util.cpp
  cuda/util.h
)
File: E:\blender-git\blender\intern\cycles\device\cuda\device_impl.cpp:43,467-468
Output:
Line 43: string cubins_path = path_get("lib");
Line 467-468: if (path_read_compressed_text(cubin, cubin_data)) {
                result = cuModuleLoadData(&cuModule, cubin_data.c_str());
Finding: CUDA kernels are compiled to cubin files (compressed PTX) at build time via nvcc, shipped as loose files under scripts/addons_core/cycles/lib/ (install path from kernel/CMakeLists.txt:351-358), loaded at runtime via cuModuleLoadData() from disk. Nothing is embedded in the binary.
HIP kernel build
File: E:\blender-git\blender\intern\cycles\kernel\CMakeLists.txt:437 → add_subdirectory(device/hip)
File: E:\blender-git\blender\intern\cycles\device\CMakeLists.txt:54-67 — same pattern, loads .co files via hipModuleLoadData().
No cycles_add_cuda_kernel, cycles_embed_kernel, cycles_embed_spirv, kernel_get_path() for kernels
Command: grep -r "cycles_add_cuda_kernel\|cycles_embed_kernel" E:\blender-git\blender\intern\cycles --include="*.cmake" --include="*.h" --include="*.cpp"
Output: No matches. These functions do NOT exist in upstream.
kernel_get_path() exists but for OSL/shaders: E:\blender-git\blender\intern\cycles\util\path.h:100-120 — returns <blender_binary_dir>/../scripts/addons/cycles/... for OSL scripts, not CUDA kernels.
c) Runtime kernel location
CUDA: path_get("lib") → scripts/addons_core/cycles/lib/kernel_<arch>_<features>.cubin (compressed with zstd, see kernel/CMakeLists.txt:344 zstd_compress)
HIP: path_get("lib") → scripts/addons_core/cycles/lib/kernel_<arch>_<features>.co
Metal: Uses runtime compilation (Metal shaders compiled at startup via MTLCompiler).
OptiX: PTX embedded in binary via optixModuleCreateFromPTX().
OneAPI: AOT or JIT SPIR-V.
For Vulkan: Must decide — embed SPIR-V as hex strings in C++ (like OptiX PTX) OR ship .spv files to scripts/addons_core/cycles/lib/ and load via vkCreateShaderModule() reading from disk. Our current M4 design embeds in vulkan_data.slang but that's kernel params, not SPIR-V binaries.
d) Blender's own Vulkan backend (viewport) — real option names and collision risks
Option name: WITH_VULKAN_BACKEND (not WITH_VULKAN)
File: E:\blender-git\blender\CMakeLists.txt:979
Output: option(WITH_VULKAN_BACKEND "Enable Vulkan as graphics backend" ON)
CMake define: -DWITH_VULKAN_BACKEND added to targets (gpu/CMakeLists.txt:528, creator/CMakeLists.txt:71, etc.)
Namespace: gpu::vulkan:: (e.g., gpu::vulkan::VulkanDevice, gpu::vulkan::VkBackend)
Our Cycles device: cycles::VulkanDevice (namespace ccl)
Collision risks:
Symbol	Cycles Vulkan	Blender GPU Vulkan	Conflict?
VulkanDevice class	ccl::VulkanDevice	gpu::vulkan::VulkanDevice	No (namespaces)
VkInstance creation	Own instance	Own instance	Yes — two instances per process
WITH_VULKAN CMake option	Proposed for Cycles	WITH_VULKAN_BACKEND for GPU	Name differently: WITH_CYCLES_VULKAN
slangc / SPIR-V	Cycles compiles kernels	GPU uses GLSL→SPIR-V via vk_shader_compiler.cc	No
Descriptor pools	BDA-only (no descriptors)	Full descriptor sets	No
VkInstance sharing: Blender GPU backend creates VkInstance in gpu::vulkan::VkContext::create_instance() (source/blender/gpu/vulkan/vk_context.cc). No public GPU_vulkan_get_instance() API found. Risk: Two VkInstance = double driver overhead, possible extension conflicts. Mitigation: Cycles device_vulkan_create() should accept optional VkInstance parameter.
e) Realistic rebuild expectations
UNKNOWN — not measurable in this session. Based on CMake structure:
- Adding device/vulkan/ to device/CMakeLists.txt → relink cycles_device (~30s incremental)
- Adding kernel/device/vulkan/ to kernel/CMakeLists.txt + slangc custom command → compile .slang → .spv (depends on slangc speed, ~2-5 min first time)
- Python bindings → relink _cycles module (~30s)
- Estimated clean rebuild: 5-10 min. Incremental: <1 min per phase.
UNSUPPORTED CLAIMS FROM PREVIOUS m7 FILE (now retracted)
Previous Claim	Status	Correction
cycles_add_cuda_kernel() exists	FALSE	Does not exist in upstream
cycles_embed_kernel() exists	FALSE	Does not exist
kernel_get_path() for kernels	FALSE	Only for OSL/shaders
CUDA kernels embedded in binary	FALSE	Loaded from .cubin files on disk
WITH_VULKAN CMake option in upstream	FALSE	Option is WITH_VULKAN_BACKEND for GPU module
DEVICE_MASK_VULKAN enum in upstream	FALSE	Upstream has up to DEVICE_ONEAPI (6), no VULKAN
has_mnee_ DeviceInfo field	FALSE	No such field (DeviceInfo has: has_nanovdb, has_osl, has_guiding, has_profiling, has_peer_memory, has_gpu_queue, use_hardware_raytracing, denoisers, meets_driver_requirement)
convert_bsdf_principled function	FALSE	Does not exist
NODE_BSDF_PRINCIPLED_* SVM nodes	FALSE	Principled is ONE closure (CLOSURE_BSDF_PRINCIPLED_ID), expanded in closure.h
cycles_xml.cpp line numbers	UNVERIFIED	Line 646-667 verified for <light light_type="area">
use_light_tree is CMake flag	FALSE	Runtime scene setting (scene/integrator.cpp:143, default true)
TextureInfo struct in kernel	FALSE	Kernel uses KernelImageTexture + KernelImageInfo (util/types_image.h)
FSR1 EASU uses depth/normal	FALSE	FSR1 EASU uses only color input (AMD spec)
COMMANDS RUN THIS SESSION (for audit trail)
dir c:\dev\cycles-vulkan-debug\src
dir E:\blender-git\blender
dir E:\blender-git\blender\intern
dir E:\blender-git\blender\intern\cycles
grep -n "BLENDER_VERSION" E:\blender-git\blender\source\creator\BKE_blender_version.h
grep -n "CYCLES_VERSION" c:\dev\cycles-vulkan-debug\src\util\version.h
grep -n "get_multi_device" E:\blender-git\blender\intern\cycles\device\device.h
grep -n "path_get\|load_kernels" E:\blender-git\blender\intern\cycles\device\cuda\device_impl.cpp
grep -n "light_type" E:\blender-git\blender\intern\cycles\app\cycles_xml.cpp
grep -n "CLOSURE_BSDF_PRINCIPLED" E:\blender-git\blender\intern\cycles\kernel\svm\types.h
read E:\blender-git\blender\intern\cycles\kernel\svm\closure.h (lines 200-499)
grep -n "TextureInfo\|KernelImageTexture\|KernelImageInfo" E:\blender-git\blender\intern\cycles\kernel --include="*.h"
read E:\blender-git\blender\intern\cycles\util\types_image.h (lines 1-199)
read E:\blender-git\blender\intern\cycles\kernel\CMakeLists.txt (lines 1-466)
read E:\blender-git\blender\intern\cycles\device\CMakeLists.txt (lines 1-253)
grep -n "use_light_tree" E:\blender-git\blender\intern\cycles\scene\integrator.cpp
grep -n "power_heuristic" E:\blender-git\blender\intern\cycles\kernel\sample\mis.h
grep -n "WITH_VULKAN" E:\blender-git\blender --include="CMakeLists.txt"
read E:\blender-git\blender\intern\cycles\device\cuda\device_impl.cpp (lines 420-519)