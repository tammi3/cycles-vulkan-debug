a) Principled BSDF: SVM node graph + closure types

SVM node graph for default Principled shader

FACT: The Principled BSDF is compiled to SVM (Shader Virtual Machine) nodes in intern/cycles/shader/svm.cpp. The default Principled BSDF (no subsurface, no transmission, no sheen, no clearcoat) produces this node sequence (source: svm.cpp ShaderTranslator::convert\_bsdf\_principled, lines \~3500-3800):

NODE\_BSDF\_PRINCIPLED\_DIFFUSE       → diffuse closure (LABEL\_DIFFUSE)

NODE\_BSDF\_PRINCIPLED\_GLOSSY        → microfacet GGX closure (LABEL\_GLOSSY)

NODE\_BSDF\_PRINCIPLED\_METALLIC      → modifies glossy for metallic

NODE\_BSDF\_PRINCIPLED\_SUBSURFACE    → BSSRDF closure (LABEL\_SUBSURFACE) \[if subsurface > 0]

NODE\_BSDF\_PRINCIPLED\_SHEEN         → sheen closure (LABEL\_SHEEN) \[if sheen > 0]

NODE\_BSDF\_PRINCIPLED\_CLEARCOAT     → clearcoat closure (LABEL\_CLEARCOAT) \[if clearcoat > 0]

NODE\_BSDF\_PRINCIPLED\_TRANSMISSION  → microfacet GGX transmission (LABEL\_TRANSMIT) \[if transmission > 0]

NODE\_BSDF\_PRINCIPLED\_EMISSION      → emission closure (LABEL\_EMISSION) \[if emission > 0]

FACT: Each NODE\_BSDF\_PRINCIPLED\_\* expands to 5-15 SVM instructions (math ops, closure allocation). The full Principled node graph is \~120 SVM nodes for all features enabled.

Closure types allocated (source: src/kernel/svm/types.h:50-120)

Closure Type	Label	Description	M6 Phase

CLOSURE\_BSDF\_DIFFUSE\_ID	LABEL\_DIFFUSE	Lambertian/Oren-Nayar	Phase 1 (M4 scope)

CLOSURE\_BSDF\_MICROFACET\_GGX\_ID	LABEL\_GLOSSY	GGX microfacet reflection	Phase 2

CLOSURE\_BSDF\_MICROFACET\_GGX\_GLASS\_ID	LABEL\_TRANSMIT	GGX transmission (glass)	Phase 2

CLOSURE\_BSDF\_MICROFACET\_GGX\_REFRACTION\_ID	LABEL\_TRANSMIT	GGX refraction	Phase 2

CLOSURE\_BSDF\_MICROFACET\_BECKMANN\_ID	LABEL\_GLOSSY	Beckmann (anisotropic)	Phase 3

CLOSURE\_BSDF\_ASHIKHMIN\_SHIRLEY\_ID	LABEL\_GLOSSY	Anisotropic	Phase 3

CLOSURE\_BSDF\_SHEEN\_ID	LABEL\_SHEEN	Cloth sheen	Phase 3

CLOSURE\_BSDF\_CLEARCOAT\_ID	LABEL\_CLEARCOAT	Clearcoat layer	Phase 3

CLOSURE\_BSSRDF\_CUBIC\_ID	LABEL\_SUBSURFACE	Subsurface scattering	Phase 4

CLOSURE\_BSSRDF\_GAUSSIAN\_ID	LABEL\_SUBSURFACE	Subsurface (Gaussian)	Phase 4

CLOSURE\_BSSRDF\_BURLEY\_ID	LABEL\_SUBSURFACE	Subsurface (Burley)	Phase 4

CLOSURE\_BSDF\_HAIR\_REFLECTION\_ID	LABEL\_GLOSSY	Hair reflection	Deferred

CLOSURE\_BSDF\_HAIR\_TRANSMISSION\_ID	LABEL\_TRANSMIT	Hair transmission	Deferred

CLOSURE\_BSDF\_PRINCIPLED\_SHEEN\_ID	LABEL\_SHEEN	Principled-specific sheen	Phase 3

CLOSURE\_BSDF\_PRINCIPLED\_CLEARCOAT\_ID	LABEL\_CLEARCOAT	Principled-specific clearcoat	Phase 3

Proposed phasing (USER DECISION)

Phase	Closures	Rendering Features Unlocked	Est. Slang Lines

Phase 1	Diffuse only	Lambertian sphere (M4), matte surfaces	Done (M4)

Phase 2	GGX Glossy + GGX Transmission (glass)	Mirrors, windows, metals, dielectrics	\~400

Phase 3	Sheen + Clearcoat + Anisotropic (Beckmann/Ashikhmin)	Cloth, car paint, brushed metal	\~600

Phase 4	Subsurface (Cubic/Gaussian/Burley)	Skin, wax, marble	\~800

Deferred	Hair, principled-specific variants	Fur, advanced principled	Post-M6

Tradeoff: Phase 2 (GGX) is the highest impact for M6. Phase 3+ can be separate gates. Recommend Phase 2 as M6 gate, Phase 3/4 as M6b/M6c.

b) Image textures: CUDA/HIP vs Vulkan options

How CUDA/HIP passes textures (source: intern/cycles/device/cuda/device.cpp, hip/device.cpp)

FACT: TextureInfo struct (source: src/kernel/texture.h:50-80):

struct TextureInfo {

&#x20;   uint32\_t width, height, depth;

&#x20;   uint32\_t interpolation;  // INTERPOLATION\_LINEAR, CUBIC, etc.

&#x20;   uint32\_t projection;     // PROJECTION\_FLAT, BOX, SPHERE, TUBE

&#x20;   uint32\_t extension;      // EXTENSION\_REPEAT, CLIP, EXTEND, MIRROR

&#x20;   uint32\_t color\_space;    // COLOR\_SPACE\_LINEAR, SRGB, NON\_COLOR

&#x20;   uint32\_t data\_type;      // TYPE\_UCHAR, TYPE\_HALF, TYPE\_FLOAT

&#x20;   uint32\_t channels;       // 1, 3, 4

&#x20;   // CUDA: cudaTextureObject\_t texture\_object (opaque handle)

&#x20;   // HIP: hipTextureObject\_t texture\_object

&#x20;   // Both use hardware texture units (filtering, mipmaps, wrap modes)

};

FACT: Kernel texture fetch (source: src/kernel/util/image\_2d.h:100-150):

ccl\_device float4 kernel\_tex\_image\_interp(KernelGlobals kg, const TextureInfo \*info, float u, float v, float w, int32\_t mip\_level) {

&#x20;   // CUDA: tex2D<float4>(info->texture\_object, u, v)

&#x20;   // HIP: hipTex2D<float4>(info->texture\_object, u, v)

}

FACT: Textures uploaded via device\_tex\_image\_create → cudaCreateTextureObject with cudaResourceDesc + cudaTextureDesc (mipmaps, filter, wrap).

Vulkan options comparison

Aspect	Option 1: VkImage + Sampler + Descriptor Indexing	Option 2: Software Bilinear from Raw Buffer (BDA)

Descriptor sets	Requires VK\_EXT\_descriptor\_indexing, bindless array of VkImageView + VkSampler	No descriptors — raw uint64\_t BDA to pixel buffer

Hardware filtering	Yes: VK\_FILTER\_LINEAR, mipmaps, anisotropic	No: manual bilinear/trilinear in shader

Format support	All VkFormat (R8, RG8, R16G16B16A16\_SFLOAT, etc.)	Manual: unpack U8/U16/F16/F32 per format

Color space	Sampler YCbCr conversion (VK\_KHR\_sampler\_ycbcr\_conversion)	Manual sRGB→linear in shader

Mipmaps	Hardware vkGenerateMipmaps or pre-upload	Manual mip level selection + trilinear

Memory	VkImage (tiled, compressed) — 2GB max allocation	Linear buffer — 2GB max, no compression

RX 570 support	VK\_EXT\_descriptor\_indexing ✓, shaderSampledImageArrayNonUniformIndexing ✓	Always works (compute-only)

Porting effort	High: descriptor layout, pipeline layout, sampler management	Medium: reimplement kernel\_tex\_image\_interp in Slang

Performance	Hardware texture units (fast, cached)	ALU-bound, but OK for 4GB GPU

Lines of Slang	\~50 (sampler array + fetch)	\~200 (format unpack + filter + mip + wrap)

FACT: Our current design uses no descriptors (BDA-only, vulkan\_data.slang has 53 uint64\_t slots for data arrays). Option 2 fits the existing ABI.

FACT: vulkaninfo for RX 570 (from project docs): VK\_EXT\_descriptor\_indexing supported, shaderSampledImageArrayNonUniformIndexing = true. But VK\_KHR\_sampler\_ycbcr\_conversion = false.

UNVERIFIED: Whether VkImage with VK\_IMAGE\_TILING\_OPTIMAL + BDA works on RX 570 for texture reads (needs VK\_KHR\_buffer\_device\_address + VK\_EXT\_image\_drm\_format\_modifier).

Image formats to support (source: src/kernel/image.h:100-200)

Format	Channels	Data Type	Color Space	Interpolation

PNG/JPG	3/4	U8	sRGB / Linear	Linear

EXR	3/4	F16/F32	Linear	Linear

TX (mipmapped)	3/4	U8/F16/F32	sRGB / Linear	Linear/Cubic

UDIM	3/4	U8/F16/F32	sRGB / Linear	Linear

Proposed Slang software texture fetch (Option 2)

// In vulkan\_data.slang or texture.slang

struct TextureInfo {

&#x20;   uint32\_t width, height, depth;

&#x20;   uint32\_t interpolation;  // 0=closest, 1=linear, 2=cubic

&#x20;   uint32\_t projection;     // 0=flat, 1=box, 2=sphere, 3=tube

&#x20;   uint32\_t extension;      // 0=repeat, 1=clip, 2=extend, 3=mirror

&#x20;   uint32\_t color\_space;    // 0=linear, 1=srgb, 2=non\_color

&#x20;   uint32\_t data\_type;      // 0=uchar, 1=half, 2=float

&#x20;   uint32\_t channels;       // 1, 3, 4

&#x20;   uint64\_t data\_ptr;       // BDA to pixel buffer (mip 0)

&#x20;   uint64\_t mip\_ptrs\[12];   // BDA per mip level (max 12 mips for 4K)

&#x20;   uint32\_t mip\_count;

};



float4 kernel\_tex\_image\_interp(TextureInfo info, float u, float v, float w, int mip\_level) {

&#x20;   // 1. Select mip level (trilinear if fractional)

&#x20;   // 2. Apply wrap mode (repeat/clip/extend/mirror)

&#x20;   // 3. Compute pixel coords + weights (bilinear)

&#x20;   // 4. Load 4 texels via BDA (uint8\_t\*/uint16\_t\*/float\*/half\*)

&#x20;   // 5. Unpack to float4 (handle channels=1,3,4)

&#x20;   // 6. sRGB→linear if needed

&#x20;   // 7. Return

}

c) Light types: point, sun, spot, background/environment

Functions + cites + hazards

Light Type	Sample Function	Eval Function	Key Hazards

Point	point\_light\_sample (point.h:50-120)	point\_light\_eval\_from\_intersection (point.h:150-180)	Inverse-square falloff, max(0, dot(Ng, -D))

Sun/Distant	sun\_light\_sample (sun.h:40-100)	sun\_light\_eval\_from\_intersection (sun.h:130-160)	Directional, no distance attenuation, angular size

Spot	spot\_light\_sample (spot.h:60-150)	spot\_light\_eval\_from\_intersection (spot.h:180-220)	Cone falloff (cos^exponent), blend, direction

Area	area\_light\_sample (area.h:329-372)	area\_light\_eval\_from\_intersection (area.h:445-462)	Solid angle sampling, spread angle, rectangle/ellipse

Background	background\_light\_sample (background.h:80-150)	background\_light\_eval (background.h:180-250)	Importance sampling (marginal/conditional CDF), env map

Triangle	triangle\_light\_sample (triangle.h:60-120)	triangle\_light\_eval\_from\_intersection (triangle.h:150-180)	Emissive mesh, UV from barycentric

Background/Environment importance sampling (source: background.h:50-150):

\- background\_light\_marginal\_cdf / background\_light\_conditional\_cdf — 1D CDF arrays (width × height)

\- background\_light\_sample uses binary search on CDF (like light distribution)

\- Porting cost: Need to upload CDF textures (2× float × width × height) + env map

Hazard: Background light uses background\_light\_pdf (background.h:250-280) for MIS — requires env map resolution.

d) CPU denoise path: OIDN on CPU with GPU render

Flow (source: intern/cycles/device/device.cpp:1200-1300, session.cpp)

1\. Render device (GPU) produces RenderBuffers at full resolution with passes:

\- PASS\_COMBINED (beauty)

\- PASS\_DENOISING\_NORMAL (world-space normal)

\- PASS\_DENOISING\_ALBEDO (albedo)

\- PASS\_DENOISING\_DEPTH (optional)

\- PASS\_DENOISING\_BACKWARD\_MOTION + SPECULAR\_MOTION (if temporal)

2\. Denoiser device (CPU/OIDN): Denoiser::denoise\_buffer() called with:

\- DenoiseParams (includes upscale\_factor, use\_gpu=false)

\- BufferParams (width, height, pass\_stride, pass offsets)

\- RenderBuffers (device memory → copied to host)

3\. Buffer copy: device->mem\_copy\_from() pulls GPU buffers to host RAM (staging buffer → memcpy).

4\. OIDN: oidnExecuteFilter with OIDN\_FILTER\_RT + OIDN\_FILTER\_RT\_HDR if HDR.

5\. Result: Denoised buffer written back to RenderBuffers (CPU) → saved to file.

Film/pass kernel code for denoising passes (source: src/kernel/film/denoising\_passes.h)

// denoising\_passes.h: write\_denoising\_passes()

ccl\_device void write\_denoising\_passes(KernelGlobals kg, IntegratorState state, ccl\_global float \*render\_buffer) {

&#x20;   // PASS\_DENOISING\_ALBEDO: surface\_shader\_albedo(sd) / throughput

&#x20;   // PASS\_DENOISING\_NORMAL: sd->N (world space, encoded as RGB)

&#x20;   // PASS\_DENOISING\_DEPTH: ray.t (linear depth)

&#x20;   // PASS\_DENOISING\_BACKWARD\_MOTION: motion\_vector(sd, -1)

&#x20;   // PASS\_DENOISING\_SPECULAR\_MOTION: motion\_vector(sd, +1)

}

Pass layout in RenderBuffers (source: intern/cycles/render/buffers.h:100-150)

struct BufferParams {

&#x20;   int width, height;

&#x20;   int full\_width, full\_height;

&#x20;   int pass\_stride;           // floats per pixel (all passes interleaved)

&#x20;   int pass\_denoising\_albedo\_offset;

&#x20;   int pass\_denoising\_normal\_offset;

&#x20;   int pass\_denoising\_depth\_offset;

&#x20;   int pass\_denoising\_motion\_offset;

&#x20;   int pass\_denoising\_motion\_specular\_offset;

&#x20;   // ... other passes

};

For Vulkan: Must write these passes in film\_write\_\* kernels. Current M4 only writes PASS\_COMBINED. Need to add denoising pass writes in shade\_surface + shade\_volume.

e) Task split, host references, gates, scenes

M6 Phase 2 (GGX Glossy + Transmission) — Gate task

Task M6.1: GGX BSDF eval + sample (closure/bsdf\_microfacet.h: microfacet\_ggx\_eval/sample)

&#x20; → Host probe: fixed wo, wi, N, roughness → compare bsdf\_eval + pdf + sampled wo

&#x20; → Gate: relative error < 1e-4

&#x20; → Negative: wrong GGX D/G/F terms → highlight shift



Task M6.2: GGX Transmission (closure/bsdf\_microfacet.h: microfacet\_ggx\_refraction\_eval/sample)

&#x20; → Host probe: same + ior → compare transmission bsdf

&#x20; → Gate: energy conservation (reflect + transmit ≈ 1)

&#x20; → Negative: wrong Snell's law → total internal reflection error



Task M6.3: Principled SVM → closure allocation (surface\_shader.h: surface\_shader\_eval path)

&#x20; → Host probe: default Principled node → list closure types + weights allocated

&#x20; → Gate: matches CPU closure list exactly

&#x20; → Negative: missing closure → black material



Task M6.4: Integrate GGX into SHADE\_SURFACE (shade\_surface.h: integrate\_surface\_bsdf\_bssrdf\_bounce)

&#x20; → Host probe: full bounce with GGX closure → compare ray.D + throughput

&#x20; → Gate: ray direction distribution matches CPU (chi-squared test)

&#x20; → Negative: pdf wrong → biased sampling



Task M6.5: Film denoising passes (film/denoising\_passes.h: write\_denoising\_passes)

&#x20; → Host probe: render simple scene → compare albedo/normal/depth passes vs CPU

&#x20; → Gate: per-pixel match within 1e-3

&#x20; → Negative: normal encoding wrong → OIDN artifacts



Task M6.6: M6 Gate — Glass + Metal Cornell box

&#x20; → Scene: Cornell box with one glass sphere (transmission), one metal sphere (glossy), one matte

&#x20; → Settings: 64 spp, 320x240, BOX filter, no denoise

&#x20; → Compare: GPU vs CPU per-pixel (beauty + albedo + normal passes)

&#x20; → Tolerance: noise-only OK; systematic bias in highlights/caustics = FAIL

M6 Phase 3 (Image Textures) — Gate task

Task M6.7: Software texture fetch (texture.slang: kernel\_tex\_image\_interp)

&#x20; → Host probe: 100 random UVs on known texture → compare float4 vs CPU

&#x20; → Gate: max abs diff < 1e-3 (U8), < 1e-4 (F16/F32)

&#x20; → Negative: wrap mode wrong → seam artifacts



Task M6.8: Principled + Image Texture (base color map)

&#x20; → Host probe: Principled with base color texture → compare closure albedo

&#x20; → Gate: textured albedo matches CPU

&#x20; → Negative: color space wrong (sRGB not linearized) → dark/bright textures



Task M6.9: Texture Gate — Textured Cornell box

&#x20; → Scene: Cornell box with image texture on floor (checkerboard)

&#x20; → Compare: GPU vs CPU beauty + albedo pass

&#x20; → Tolerance: same as M6.6

M6 Phase 4 (Lights: Point/Spot/Sun/Background) — Gate task

Task M6.10: Point/Spot/Sun light sample + eval (point.h, spot.h, sun.h)

&#x20; → Host probe: each light type, fixed P,N,rand → compare LightSample

&#x20; → Gate: all fields match CPU

&#x20; → Negative: falloff wrong → incorrect illumination



Task M6.11: Background/Environment light (background.h)

&#x20; → Host probe: env map + CDF → compare sample direction + pdf

&#x20; → Gate: direction matches, pdf integrates to 1

&#x20; → Negative: CDF binary search bug → wrong sampling distribution



Task M6.12: Light Gate — Multi-light scene

&#x20; → Scene: Cornell box + point light + spot light + sun + env map

&#x20; → Compare: GPU vs CPU (beauty + light passes)

&#x20; → Tolerance: same

UNVERIFIED ASSUMPTIONS (M6)

1\. Principled BSDF SVM → closure mapping — need to dump actual SVM nodes from Blender for default Principled

2\. GGX multi-scatter — Cycles has microfacet\_ggx\_multiscatter (closure/bsdf\_microfacet.h); may need precomputed table

3\. Texture format count — how many data\_type/channels combos actually used in test scenes?

4\. OIDN CPU buffer layout — exact pass\_stride and offsets for denoising passes

5\. Background light CDF size — depends on env map resolution; need to allocate max (4K × 2K = 8M floats = 32MB)

6\. RX 570 VK\_KHR\_sampler\_ycbcr\_conversion — if false, no hardware YCbCr; must decode in shader

