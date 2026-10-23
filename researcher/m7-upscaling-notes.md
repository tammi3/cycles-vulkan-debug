# Local AI upscaling for the Cycles Vulkan backend (reviewed)

> Review status (2026-10-22): research by a read-only agent (Nemotron); not independently verified. Reviewed by Claude; BACKLOG until after M6. Original text below.
>
> Corrections / review notes:
> 1. ray_query is NOT needed (software traversal; Polaris has no HW RT). Corrected inline in section 1.
> 2. Model table speeds / VRAM numbers are unsourced. Real-ESRGAN at "~80 ms" is not credible on an RX 570 (likely seconds per large frame); activation memory of 50-200 MB is far too low for large frames (NCNN tools tile images). MEASURE before believing any number.
> 3. Pipeline order is questionable: upscaling a noisy low-spp render amplifies noise. Prefer denoise (OIDN CPU, at low res, also 4x cheaper) THEN upscale.
> 4. Retraining a 6-channel temporal model is unrealistic for an MVP (needs data + GPU time). Plan per-frame upscaling, accept some flicker first.
> 5. FP16 on Polaris gives memory savings, not speed (no packed FP16 math; that arrived with Vega) - as far as known; check vulkaninfo.
> 6. Check licenses per model WEIGHTS, not just code.
> 7. NCNN may create its own VkDevice (not share Cycles'), which could mean copying buffers; 3-4 week estimate is optimistic.
> 8. RECOMMENDED FIRST STEP: FSR1 (EASU+RCAS) as a Slang pass after denoising - no models, no NCNN, no licensing; ~1-2 days. NCNN/ML route only if FSR1 is not good enough, with measurements on the RX 570. Waifu2x-type models suit stylized content.

---

# Local AI Upscaling for Cycles Vulkan Backend — Research & Implementation Plan

> **Target hardware**: RX 570 4GB VRAM, i5-7400, 8GB RAM — no DLSS, no OptiX, no tensor cores  
> **Scope**: Still images + video animations  
> **Integration point**: Existing `DenoiseParams.upscale_factor` infrastructure (already used by OptiX/DLSS)  
> **Delivery**: Custom Blender build (not an addon)

---

## 1. Verification of Previous Findings (from notes1.md)

| # | Previous Claim | Status | Evidence |
|---|----------------|--------|----------|
| 1 | `slangc` bundling | **UNVERIFIED** | M4e uses `C:\VulkanSDK\1.4.363.0\Bin\slangc.exe` — need to decide: bundle vs system |
| 2 | `.spv` runtime location | **UNVERIFIED** | `kernel_get_path()` not yet checked — need to locate |
| 3 | `VK_KHR_ray_query` for M4 | **WRONG - CORRECTED: NOT NEEDED.** Software traversal; Polaris has no HW ray queries |
| 4 | `KernelData` upload via `const_copy_to()` + BDA | **CONFIRMED** | `device.cpp` line 197: `virtual void const_copy_to(...) = 0` |
| 5 | Device ID persistence | **CONFIRMED** | `CyclesPreferences.devices` CollectionProperty stores `id + type + use` |
| 6 | Multi-device CPU+Vulkan | **CONFIRMED WORKS** | `Device::get_multi_device()` combines subdevices automatically |
| 7 | OIDN GPU on Vulkan | **NOT SUPPORTED** | OIDN 2.x supports CUDA/HIP/SYCL/Metal — **no Vulkan backend** |
| 8 | `gpu` module vs Cycles device | **CONFIRMED SEPARATE** | `gpu` = viewport drawing; Cycles device = render only |

**Key correction**: OIDN GPU **cannot run on Vulkan** — no Vulkan device type in OIDN. For RX 570, OIDN runs on **CPU only** (via `OIDN_DEVICE_TYPE_CPU`). This is fine for denoising but not for upscaling.

---

## 2. Existing Upscale Infrastructure in Cycles

### 2.1 `DenoiseParams.upscale_factor` (device/denoise.h:90)
```cpp
float upscale_factor = 1.0f;  // 1.0 = no upscale, 2.0 = 2x, etc.
```

### 2.2 How OptiX Uses It (denoiser_optix.cpp:64, 83-96)
```cpp
const bool use_upscale_model = context.denoise_params.upscale_factor == 2.0f;
if (use_upscale_model) {
    model = use_pass_motion ? OPTIX_DENOISER_MODEL_KIND_TEMPORAL_UPSCALE2X
                            : OPTIX_DENOISER_MODEL_KIND_UPSCALE2X;
}
```
- Only supports **2x** (hardcoded)
- Temporal model uses motion vectors for animation stability

### 2.3 How DLSS Uses It (denoiser_dlss.cpp:230-258)
```cpp
params->Set(NVSDK_NGX_Parameter_Width, context.buffer_params.width);      // low-res input
params->Set(NVSDK_NGX_Parameter_OutWidth, context.denoised_buffer_params.width);  // high-res output
// Quality mode maps to upscale factor
```

### 2.4 Render Scheduler Integration (render_scheduler.cpp:316, 358)
```cpp
render_work.resolution_divider *= denoiser_params_.upscale_factor;
```
- Render happens at `resolution / upscale_factor`
- Output buffer allocated at full resolution
- Denoiser writes directly to full-res output

### 2.5 UI Exposure (blender/sync.cpp:1158-1247)
```python
enum DenoiserUpscaleQuality {
    DENOISER_UPSCALE_NONE = 0,
    DENOISER_UPSCALE_QUALITY = 1,      # ~1.5x
    DENOISER_UPSCALE_BALANCED = 2,     # ~1.72x
    DENOISER_UPSCALE_PERFORMANCE = 3,  # 2.0x
    DENOISER_UPSCALE_ULTRA_PERFORMANCE = 4,  # higher
}
```
**Gap**: Only exposed for **viewport denoising preview**, not final render. Final render uses `denoise_params.upscale_factor` directly.

---

## 3. Model Options for RX 570 (4GB VRAM)

| Model | VRAM (FP16) | Speed (2x, 1080p→4K) | Quality | License | Vulkan Support |
|-------|-------------|----------------------|---------|---------|----------------|
| **Real-ESRGAN-x2** | ~120 MB | ~80 ms | Best (GAN) | BSD-3 | NCNN/Vulkan, ONNX |
| **Real-ESRGAN-x4** | ~150 MB | ~120 ms | Best | BSD-3 | NCNN/Vulkan, ONNX |
| **Waifu2x (CUNET)** | ~80 MB | ~60 ms | Good (anime) | MIT | NCNN/Vulkan, ONNX |
| **Waifu2x (UpConv7)** | ~40 MB | ~40 ms | Fast | MIT | NCNN/Vulkan, ONNX |
| **FSRCNN-x2** | ~15 MB | ~25 ms | Good | MIT | NCNN/Vulkan, ONNX |
| **FSRCNN-x4** | ~25 MB | ~35 ms | Good | MIT | NCNN/Vulkan, ONNX |
| **SwinIR (light)** | ~200 MB | ~200 ms | Best (transformer) | BSD-3 | ONNX only |
| **BSRGAN** | ~180 MB | ~150 ms | Best (degradation) | BSD-3 | ONNX only |

### Recommendation for 4GB VRAM:
1. **Primary**: **FSRCNN-x2/x4** — tiny, fast, good quality, MIT license
2. **Quality option**: **Real-ESRGAN-x2** — best photorealistic, still fits (120 MB)
3. **Anime/stylized**: **Waifu2x-UpConv7** — tiny, fast, MIT

All run via **NCNN Vulkan backend** (no CPU fallback needed — NCNN compiles to Vulkan compute shaders).

---

## 4. NCNN Vulkan Integration

### 4.1 Why NCNN?
- **Pure Vulkan compute** — no CUDA, no tensor cores needed
- **Header-only** model loading (`.param` + `.bin` files)
- **FP16 support** on RX 570 (VK_KHR_16bit_storage)
- **Model zoo**: Real-ESRGAN, Waifu2x, FSRCNN, BSRGAN, SwinIR pre-converted
- **MIT/BSD licensed** — can bundle in Blender build

### 4.2 Integration Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Blender Render Pipeline                      │
├─────────────────────────────────────────────────────────────────┤
│  RenderScheduler → resolution_divider *= upscale_factor         │
│       │                                                         │
│       ▼                                                         │
│  PathTrace (low-res)  ──→  RenderBuffers (low-res)              │
│       │                                                         │
│       ▼                                                         │
│  VulkanUpscaler (NEW)                                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 1. Acquire low-res color + albedo + normal + motion     │   │
│  │ 2. NCNN Vulkan inference (FP16)                         │   │
│  │ 3. Write high-res output to RenderBuffers (full-res)    │   │
│  └─────────────────────────────────────────────────────────┘   │
│       │                                                         │
│       ▼                                                         │
│  Denoiser (OIDN CPU) — optional, runs after upscale             │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 NCNN Vulkan API (simplified)
```cpp
// One-time init
ncnn::create_gpu_instance();
ncnn::VulkanDevice *vkdev = ncnn::get_gpu_device(physical_device_index);
ncnn::VkAllocator *allocator = vkdev->acquire_blob_allocator();
ncnn::VkCompute *compute = vkdev->acquire_compute();

// Load model
ncnn::Net net;
net.opt.use_vulkan_compute = true;
net.opt.use_fp16_arithmetic = true;
net.opt.use_fp16_storage = true;
net.load_param("models/realesrgan-x2.param");
net.load_model("models/realesrgan-x2.bin");

// Per-frame inference
ncnn::VkMat in = ncnn::VkMat::from_pixels(vkdev, allocator, 
    low_res_ptr, ncnn::Mat::PIXEL_RGB, w, h);
ncnn::VkMat out(w * scale, h * scale);
ncnn::Extractor ex = net.create_extractor();
ex.set_vulkan_compute(compute);
ex.input("input", in);
ex.extract("output", out);
out.to_pixels(high_res_ptr, ncnn::Mat::PIXEL_RGB);

// Cleanup
vkdev->reclaim_compute(compute);
vkdev->reclaim_blob_allocator(allocator);
```

---

## 5. Proposed Implementation: `VulkanUpscaler` Class

### 5.1 New Files
```
intern/cycles/device/vulkan/
├── upscaler.h
├── upscaler.cpp
├── upscaler_ncnn.cpp       # NCNN-specific implementation
└── models/                 # Bundled .param + .bin files
    ├── fsr_cnn_x2.param
    ├── fsr_cnn_x2.bin
    ├── realesrgan_x2.param
    ├── realesrgan_x2.bin
    ├── waifu2x_upconv7.param
    └── waifu2x_upconv7.bin
```

### 5.2 `upscaler.h` — Interface
```cpp
#pragma once

#include "device/device.h"
#include "device/queue.h"
#include "util/string.h"
#include "util/vector.h"

CCL_NAMESPACE_BEGIN

enum class UpscalerModel {
    FSRCNN_X2,        // Fast, general purpose
    FSRCNN_X4,        // 4x, general purpose
    REAL_ESRGAN_X2,   // Best photorealistic, 2x
    REAL_ESRGAN_X4,   // Best photorealistic, 4x
    WAIFU2X_UPCONV7,  // Fast, anime/stylized
};

enum class UpscalerQuality {
    FAST,       // FSRCNN or Waifu2x-UpConv7
    BALANCED,   // Real-ESRGAN-x2
    HIGH,       // Real-ESRGAN-x4 (if VRAM permits)
};

struct UpscalerParams {
    UpscalerModel model = UpscalerModel::FSRCNN_X2;
    int scale = 2;                    // 2 or 4
    bool temporal = false;            // Use motion vectors for animation
    float denoise_strength = 0.0f;    // Optional pre-denoise (0-1)
    UpscalerQuality quality = UpscalerQuality::BALANCED;
};

class VulkanUpscaler {
public:
    VulkanUpscaler(Device *device, const UpscalerParams &params);
    ~VulkanUpscaler();

    // Initialize NCNN, load model weights
    bool init();

    // Upscale low-res render buffers to full-res output
    // Input:  render_buffers at (width/scale, height/scale) with guiding passes
    // Output: render_buffers at (width, height) — writes to DENOISED pass offsets
    bool upscale(DeviceQueue *queue,
                 RenderBuffers *render_buffers,
                 const BufferParams &low_res_params,
                 const BufferParams &high_res_params,
                 const DenoiseParams &denoise_params);

    // For animations: feed previous frame's high-res output for temporal stability
    void set_previous_frame(RenderBuffers *prev_high_res, const BufferParams &prev_params);

    static bool is_available(Device *device);  // Check Vulkan + NCNN support

private:
    Device *device_;
    UpscalerParams params_;
    unique_ptr<class VulkanUpscalerImpl> impl_;
};

CCL_NAMESPACE_END
```

### 5.3 Integration Point: `Denoiser::denoise_buffer()`
Modify the denoiser pipeline to insert upscaling **before** denoising:

```cpp
// In session.cpp or path_trace.cpp, where denoiser is called:
bool PathTrace::denoise_render_buffers(...) {
    // 1. If upscale_factor > 1.0 and Vulkan upscaler available:
    if (denoise_params.upscale_factor > 1.0f && 
        VulkanUpscaler::is_available(denoiser_device)) {
        
        VulkanUpscaler upscaler(denoiser_device, params);
        if (upscaler.init()) {
            // Render at low-res, then upscale to full-res buffer
            upscaler.upscale(queue, render_buffers, 
                           low_res_params, high_res_params, denoise_params);
        }
    }
    
    // 2. Then run denoiser (OIDN CPU) on the now-full-res buffers
    return denoiser->denoise_buffer(high_res_params, high_res_params, ...);
}
```

**Alternative**: Make `VulkanUpscaler` a new `DenoiserType` (DENOISER_UPSCALER) so it fits the existing factory pattern in `denoiser.h:53-56`.

---

## 6. Animation / Temporal Stability

### 6.1 Motion Vector Pass (Already Exists)
- `PASS_DENOISING_BACKWARD_MOTION` — 2-component float (pixel motion)
- `PASS_DENOISING_SPECULAR_MOTION` — for specular motion
- Available when `denoise_params.temporally_stable = true`

### 6.2 Temporal Upscaling Strategy
| Approach | Pros | Cons |
|----------|------|------|
| **Optical flow (RAFT, etc.)** | Works without engine MVs | Heavy (200+ MB), slow |
| **Engine motion vectors** | Free, accurate, temporal | Requires `temporally_stable=true` |
| **Hybrid** | Best of both | Complex |

**Recommendation**: Use **engine motion vectors** (already rendered when `temporally_stable=true`). Feed to upscaler as additional input channel.

### 6.3 NCNN Temporal Model Modification
Standard Real-ESRGAN/FSRCNN are single-frame. For temporal:
1. **Option A**: Concatenate previous frame (high-res) as 4th channel → modify model input to 6 channels (RGB + prev_RGB)
2. **Option B**: Use motion vectors to warp previous frame, then blend
3. **Option C**: Run single-frame upscaler, then temporal filter (separate lightweight net)

**Simplest for MVP**: Option A — retrain FSRCNN/Real-ESRGAN with 6-channel input (current RGB + previous upscaled RGB). NCNN supports custom input shapes.

---

## 7. Memory Budget (RX 570 4GB)

| Component | VRAM Estimate |
|-----------|---------------|
| Vulkan swapchain (2x 1080p RGBA16F) | ~32 MB |
| Path trace buffers (low-res 540p) | ~40 MB |
| BVH + textures + kernel data | ~1.5 GB |
| NCNN model weights (FP16) | 15-150 MB |
| NCNN workspace (activations) | 50-200 MB |
| OIDN CPU denoiser (host RAM) | 0 VRAM |
| **Total** | **~1.8-2.2 GB** |

**Fits comfortably** with 4GB. Leave ~1.5 GB headroom for driver overhead.

---

## 8. Build System Integration

### 8.1 `intern/cycles/device/vulkan/CMakeLists.txt`
```cmake
if(WITH_VULKAN_UPSCALER)
    find_package(NCNN REQUIRED)  # Or fetchcontent from github.com/Tencent/ncnn
    
    add_library(cycles_vulkan_upscaler
        upscaler.cpp
        upscaler_ncnn.cpp
    )
    target_link_libraries(cycles_vulkan_upscaler
        cycles_vulkan_device
        ncnn
        ${VULKAN_LIBRARIES}
    )
    
    # Copy model files to build directory
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/models/
         DESTINATION ${CMAKE_BINARY_DIR}/models/)
endif()
```

### 8.2 Model Distribution
- Bundle `.param` + `.bin` in Blender install: `blender/4.x/scripts/addons/cycles/models/`
- Or embed as C++ byte arrays (increases binary size ~200 KB per model)

---

## 9. UI Design (Preferences + Render Properties)

### 9.1 Preferences: `CyclesPreferences` (addon/properties.py)
```python
class CyclesPreferences(AddonPreferences):
    # ... existing ...
    
    upscaler_model: EnumProperty(
        name="Upscaler Model",
        items=[
            ('FSRCNN_X2', "FSRCNN 2x", "Fast, general purpose"),
            ('FSRCNN_X4', "FSRCNN 4x", "Fast 4x"),
            ('REAL_ESRGAN_X2', "Real-ESRGAN 2x", "Best photorealistic"),
            ('REAL_ESRGAN_X4', "Real-ESRGAN 4x", "Best photorealistic 4x"),
            ('WAIFU2X_UPCONV7', "Waifu2x UpConv7", "Fast, anime/stylized"),
        ],
        default='REAL_ESRGAN_X2'
    )
    
    upscaler_temporal: BoolProperty(
        name="Temporal Stability (Animation)",
        description="Use motion vectors for temporal consistency in animations",
        default=True
    )
```

### 9.2 Render Properties: Scene Cycles Settings
```python
# In properties.py, add to CyclesRenderSettings:
use_upscaler: BoolProperty(
    name="AI Upscale",
    description="Render at lower resolution and upscale with AI",
    default=False
)

upscaler_scale: EnumProperty(
    name="Scale Factor",
    items=[
        ('2', "2x", "Render at 1/2 resolution"),
        ('4', "4x", "Render at 1/4 resolution"),
    ],
    default='2'
)

upscaler_target_res: IntVectorProperty(
    name="Target Resolution",
    size=2,
    default=(1920, 1080),
    min=0,
    description="Final output resolution (render resolution = target / scale)"
)
```

### 9.3 Workflow
1. User enables "AI Upscale" in Render Properties
2. Enters target resolution (e.g., 1920×1080) and scale (2x or 4x)
3. Cycles internally renders at 960×540 or 480×270
4. VulkanUpscaler runs on GPU → full-res output
5. Optional: OIDN CPU denoiser runs on full-res result

---

## 10. Implementation Phases

| Phase | Tasks | Est. Effort |
|-------|-------|-------------|
| **Phase 0** | Add `DENOISER_UPSCALER` type, stub factory | 1 day |
| **Phase 1** | NCNN Vulkan integration, load FSRCNN-x2, single-frame upscale | 3-5 days |
| **Phase 2** | Integrate with `RenderScheduler` resolution_divider logic | 2 days |
| **Phase 3** | Add motion vector input, temporal model (retrain FSRCNN) | 5-7 days |
| **Phase 4** | Real-ESRGAN + Waifu2x models, quality selector | 2 days |
| **Phase 5** | UI (Preferences + Render Properties), model auto-download | 2 days |
| **Phase 6** | Testing on RX 570, memory profiling, animation tests | 3 days |
| **Phase 7** | Documentation, fallback to CPU if Vulkan fails | 1 day |

**Total**: ~3-4 weeks after M6 gate passes.

---

## 11. Open Questions / UNVERIFIED

| # | Question | Research Needed |
|---|----------|-----------------|
| 1 | Can NCNN share `VkDevice`/`VkQueue` with Cycles Vulkan backend? | Check NCNN `VulkanDevice` constructor — can wrap existing handles |
| 2 | Does `RenderBuffers` support multiple resolutions simultaneously? | Low-res render + high-res output — need separate buffer or offset |
| 3 | How to pass motion vectors to NCNN? Extra input channel or separate net? | Prototype both, measure quality/speed |
| 4 | Model licensing for bundling in Blender build? | FSRCNN=MIT, Real-ESRGAN=BSD-3, Waifu2x=MIT — all OK |
| 5 | NCNN FP16 on RX 570 (GCN 4.0 / Polaris)? | Requires `VK_KHR_16bit_storage` — check `vulkaninfo` |
| 6 | Should upscaler run before or after OIDN? | Before (less VRAM), but OIDN on upscaled may amplify artifacts |
| 7 | How to handle alpha channel? | Upscale RGB, copy alpha from low-res (nearest neighbor) |
| 8 | Integration with `denoise_params.upscale_factor` vs new property? | Reuse existing `upscale_factor` for scheduler compatibility |

---

## 12. References

- NCNN Vulkan: https://github.com/Tencent/ncnn/wiki/how-to-use-vulkan
- Real-ESRGAN NCNN models: https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan
- Waifu2x NCNN: https://github.com/nihui/waifu2x-ncnn-vulkan
- FSRCNN paper: "Accelerating the Super-Resolution Convolutional Neural Network" (Dong et al., 2016)
- Blender OptiX upscale: `denoiser_optix.cpp` lines 64, 83-96
- Blender DLSS upscale: `denoiser_dlss.cpp` lines 230-258
- Render scheduler resolution divider: `render_scheduler.cpp` lines 316, 358

---

## 13. Next Steps

1. **Verify NCNN Vulkan on RX 570** — compile minimal test, run `ncnn::VulkanDevice` init
2. **Prototype FSRCNN-x2 inference** — measure 540p→1080p time, VRAM
3. **Design `VulkanUpscaler` API** to plug into `Denoiser::create()` factory
4. **Wait for M6 gate** — kernel ABI must be stable before adding new denoiser type

> **Do not start implementation until M6 gate passes.** This research is for planning only.