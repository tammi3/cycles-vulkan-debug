a) Verified upscaling infrastructure

Item	Source	Status

DenoiseParams.upscale\_factor	E:\\blender-git\\blender\\intern\\cycles\\device\\denoise.h:90	SURVIVED — float upscale\_factor = 1.0f;

OptiX uses upscale\_factor	E:\\blender-git\\blender\\intern\\cycles\\device\\denoiser\_optix.cpp:64,83-96	SURVIVED — checks == 2.0f for UPSCALE2X model

DLSS uses upscale\_factor	E:\\blender-git\\blender\\intern\\cycles\\device\\denoiser\_dlss.cpp:230-258	SURVIVED — sets input/output width

RenderScheduler resolution\_divider	E:\\blender-git\\blender\\intern\\cycles\\render\\render\_scheduler.cpp:316,358	SURVIVED — \*= denoiser\_params\_.upscale\_factor

UI DenoiserUpscaleQuality enum	E:\\blender-git\\blender\\intern\\cycles\\blender\\addon\\properties.py:1158-1247	SURVIVED — viewport only

OIDN GPU on Vulkan	oidn/include/OpenImageDenoise/oidn.hpp	CORRECTED — OIDN 2.x has CUDA/HIP/SYCL/Metal, no Vulkan backend

b) FSR1 pipeline position (corrected)

PathTrace (low-res: width/scale × height/scale)

&#x20;   │

&#x20;   ▼

RenderBuffers (low-res) — COMBINED, ALBEDO, NORMAL, DEPTH, MOTION

&#x20;   │

&#x20;   ▼

\[NEW] VulkanUpscaler (Slang compute)

&#x20;   │   Input:  low-res COMBINED (color only for EASU)

&#x20;   │   Output: high-res COMBINED (full resolution)

&#x20;   │

&#x20;   ▼

RenderBuffers (high-res) — COMBINED now at full res

&#x20;   │

&#x20;   ▼

Denoiser (OIDN CPU) — runs on high-res COMBINED + ALBEDO + NORMAL

&#x20;   │

&#x20;   ▼

Film convert/display

RenderBuffers: Single instance with full\_width/full\_height ≠ width/height (buffers.h:200-300). Upscaler writes to high-res COMBINED pass.

c) FSR1 spec — ONLY from AMD source

AMD FidelityFX FSR1: https://github.com/GPUOpen-Effects/FidelityFX-FSR1

License: MIT

If AMD source not readable in this session: NOT FOUND — do not describe from memory.

DLSS: No implementation in our fork. Removed.

d) NCNN measurement plan (unchanged, no integration design)

e) Task split (corrected paths)

Task M8.1: FSR1 EASU Slang pass (kernel/device/vulkan/fsr1\_easu.slang)

&#x20; → Host probe: 100×100 input → 200×200 output vs CPU reference (AMD EASU)

&#x20; → Gate: max pixel diff < 1/255

&#x20; → Negative: wrong edge weights → ringing



Task M8.2: FSR1 RCAS Slang pass (kernel/device/vulkan/fsr1\_rcas.slang)

&#x20; → Host probe: same image → sharpened vs CPU reference (AMD RCAS)

&#x20; → Gate: max pixel diff < 1/255



Task M8.3: VulkanUpscaler class (device/vulkan/upscaler.h/cpp)

&#x20; → Host probe: low-res render → upscale → compare to CPU FSR1

&#x20; → Gate: end-to-end match



Task M8.4: RenderScheduler integration (render/scheduler.cpp)

&#x20; → Host probe: render at 1/2 res, upscale to full, compare buffer params

&#x20; → Gate: full-res output dimensions correct, passes preserved



Task M8.5: M8 Gate — FSR1 end-to-end

&#x20; → Scene: Cornell box (M5) at 320×240 → upscale 2× to 640×480

&#x20; → Compare: GPU FSR1 vs CPU FSR1 (same low-res input)

&#x20; → Tolerance: pixel-perfect match (deterministic)

roadmap.md — CORRECTED session estimates

Task	Est. Sessions	Basis

M4 completion	4-5	Observed: 2 sessions for stride table + stride check + partial accessors

M5.1-M5.4	3-4	\~200 lines each, simple probes

M5.5-M5.8	8-10	\~200-300 lines each, stateful kernels

M5.9 (M5 gate)	2	End-to-end

M6-P2 (GGX metallic+specular)	5-6	\~400 lines, microfacet math

M6-P3 (GGX transmission)	4-5	\~300 lines

M6-P4 (Subsurface)	6-8	\~500 lines, BSSRDF

M6-P5 (Sheen)	2-3	\~150 lines

M6-P6 (Clearcoat)	2-3	\~150 lines

M6-P7 (Anisotropic)	3-4	\~200 lines

M6 texture fetch	5-6	\~200 lines, format handling

M6 lights (point/spot/sun/bg)	4-5	\~300 lines total

M7 (Blender integration)	8-10	CMake + Python + device enum

M8.1 (FSR1 EASU)	3-4	\~200 lines Slang

M8.2 (FSR1 RCAS)	2-3	\~100 lines Slang

M8.3 (VulkanUpscaler)	3-4	C++ host + dispatch

M8.4 (Scheduler integration)	3-4	Buffer swap logic

M8.5 (M8 gate)	2	End-to-end

All marked ESTIMATES — based on observed \~3-4 sessions per small task pace.

COMPLETE LIST OF RETRACTED CLAIMS (all previous files)

&#x20;1. convert\_bsdf\_principled function exists — FALSE

&#x20;2. NODE\_BSDF\_PRINCIPLED\_\* SVM nodes exist — FALSE

&#x20;3. cycles\_add\_cuda\_kernel() CMake function exists — FALSE

&#x20;4. cycles\_embed\_kernel() CMake function exists — FALSE

&#x20;5. CUDA kernels embedded in binary via cycles\_embed\_kernel — FALSE

&#x20;6. kernel\_get\_path() used for kernel binaries — FALSE

&#x20;7. TextureInfo struct used in kernel — FALSE

&#x20;8. use\_light\_tree is a CMake compile flag — FALSE

&#x20;9. Light tree mandatory for M5 — FALSE

10\. FSR1 EASU uses depth/normal inputs — FALSE

11\. FSR1 includes temporal accumulation — FALSE

12\. DLSS integration exists in our fork — FALSE

13\. WITH\_VULKAN CMake option in upstream Blender — FALSE

14\. DEVICE\_MASK\_VULKAN enum in upstream — FALSE

15\. has\_mnee\_ DeviceInfo field — FALSE

16\. cycles\_xml.cpp:571 for light parsing — FALSE (actual: 646-667)

17\. Emissive meshes are area lights — FALSE (triangle lights)

18\. get\_multi\_device takes DeviceInfo vector by value — CORRECTED (takes const vector<DeviceInfo>\&)

19\. Principled compiles to multiple SVM nodes — FALSE (one closure, lobes at runtime)

20\. TextureInfo has cudaTextureObject\_t — FALSE (that's host-side CUDA struct)

