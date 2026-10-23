roadmap.md

Dependency graph across M4→M8

M4 (Lambertian sphere, 16 spp, env light)

│

├─→ M5 (Cornell box + area light: NEE, shadow rays, MIS)

│   │

│   ├─→ M6 Phase 2 (GGX Glossy + Transmission)

│   │   │

│   │   ├─→ M6 Phase 3 (Sheen + Clearcoat + Anisotropic + Image Textures)

│   │   │   │

│   │   │   ├─→ M6 Phase 4 (Point/Spot/Sun/Background lights + Subsurface)

│   │   │   │   │

│   │   │   │   └─→ M7 (Blender integration: device enum, Python, CMake)

│   │   │   │       │

│   │   │   │       └─→ M8 Phase 1 (FSR1 EASU+RCAS Slang passes)

│   │   │   │           │

│   │   │   │           └─→ M8 Phase 2 (RenderScheduler integration + end-to-end)

│   │   │   │               │

│   │   │   │               └─→ M8 Phase 3 (NCNN measurement spike — parallel, optional)

Critical path: M4 → M5 → M6.2 → M6.3 → M6.4 → M7 → M8.1 → M8.2  

Parallelizable: M6.3/6.4 can overlap; M8.3 (NCNN) is fully independent after M6.

PARALLEL SPIKES (standalone Vulkan/Slang, run on RX 570 + SwiftShader)

Spike	Purpose	Size Est.	Gate (host reference)	Unblocks

S1: Wave64 active-index + prefix-sum	Compact active paths, sort by shader	\~150 lines Slang	CPU prefix-sum (exclusive scan) on 10K random uints → match indices	M5 queue compaction, M6 shader sorting

S2: CAS-based float atomic add	Film buffer accumulation (no native float atomic on RX 570)	\~80 lines Slang	CPU atomicAdd emulation on float array → bitwise match	M4 film write, M5 direct light write, M6 denoising passes

S3: Software bilinear texture fetch via BDA	Option 2 from M6.b (no descriptors)	\~200 lines Slang	CPU kernel\_tex\_image\_interp on 100 random UVs → max diff < 1e-3	M6 Phase 3 image textures

S4: FSR1 EASU+RCAS on test image	Verify Slang port matches AMD reference	\~300 lines Slang	CPU FSR1 (port from HLSL) on zone plate → pixel-perfect	M8 Phase 1

S5: GGX multi-scatter table lookup	Energy-conserving GGX (Heitz 2014)	\~100 lines Slang + 4KB table	CPU GGX multiscatter eval → relative error < 1e-3	M6 Phase 2 GGX

S6: Light tree traversal (simplified)	Validate tree walk + reservoir sampling	\~250 lines Slang	CPU light\_tree\_sample on 3-light scene → same emitter index	M5/M6 light tree option

S7: OIDN CPU denoiser pipeline	Verify buffer layout + pass requirements	Host C++ only	Blender CPU denoise on test render → output saved	M6 denoising passes, M8 post-upscale denoise

All spikes: Compile with slangc -target spirv, validate with spirv-val, run on RX 570 + SwiftShader (VK\_ICD\_FILENAMES=.../vk\_swiftshader\_icd.json). Zero VUID errors required.

Risk ranking (highest first)

Rank	Risk	Reason

1	RX 570 driver miscompiles dynamic OpAccessChain on fixed arrays in PhysicalStorageBuffer (AGENTS.md DEVICE RULE)	Confirmed in M4e S1b-2; affects ALL BDA struct member arrays. Workaround: base pointers / pointer members / raw uint\* views. Must verify every dynamic index.

2	SwiftShader Int64 capability crash (S1b-3 step 2)	OpTypeInt 64 crashes SwiftShader Subzero. Avoid strided uint64\_t data arrays; pointer members OK.

3	Packed float3 (12B) vs Slang float3 (16B) stride mismatch	S1b-3 step 1 confirmed SoA buffers use 12B stride. Accessors must use float\* base + 3\*i or confirmed float3\* (stride 12 on both devices).

4	No native float atomic add on RX 570	Film buffer, direct light accumulation need CAS emulation. Unverified performance at scale.

5	Light tree porting complexity (\~500 lines, complex structs, MIS PDF requires second traversal)	Defer to post-M6; flat distribution sufficient for M5 gate.

6	Principled BSDF SVM → closure mapping divergence	Cycles SVM is dynamic; our explicit switch(closure\_type) must match exactly. Missing closure = black material.

7	Texture format combinatorics (U8/U16/F16/F32 × 1/3/4 channels × sRGB/linear × wrap modes)	Software fetch must handle all; test coverage critical.

8	Blender integration CMake + Python binding drift	Upstream Cycles moves fast; our fork base may diverge. Rebase needed before M7.

9	VkInstance sharing with Blender GPU module	Two instances = double driver overhead, possible resource conflicts. Need GPU\_vulkan\_get\_instance() API.

10	NCNN VkDevice sharing unproven	If NCNN creates own device, buffer copies needed (VRAM + latency).

Session estimates (ESTIMATES — reasoning included)

Task	Est. Sessions	Reasoning

M4 completion (accessors + gate)	3-4	S1b-3 items 3-5: accessors (\~400 lines), gate probe, negatives. Stride verified.

M5.1 Flat light distribution	1	\~60 lines, simple binary search, trivial probe.

M5.2 Area light sample/eval	2	\~300 lines, solid-angle math, rectangle/ellipse, spread clamp. Probe: LightSample struct.

M5.3 MIS weights	1	\~50 lines, deterministic power\_heuristic. Probe: feed known PDFs.

M5.4 Shadow ray setup	1	\~100 lines, ray offset + self-intersection skip. Probe: Ray struct.

M5.5 INTERSECT\_SHADOW kernel	2	\~180 lines, opaque + transparent paths, sort intersections. Probe: packed\_num\_hits + throughput.

M5.6 SHADE\_SHADOW kernel	2	\~290 lines, transparent eval loop, surface + volume. Probe: throughput after eval.

M5.7 SHADE\_LIGHT\_NEE kernel	1	\~150 lines, light eval + MIS + throughput update. Probe: eval + throughput.

M5.8 SHADE\_SURFACE NEE branch	2	Part of shade\_surface; NEE spawns shadow\_state. Probe: shadow\_state match.

M5.9 M5 Gate (Cornell box)	1	End-to-end render, compare to CPU.

M6.1 GGX BSDF eval/sample	3	\~400 lines, GGX D/G/F, sampling, multi-scatter table. Probe: bsdf\_eval + pdf + wo.

M6.2 GGX Transmission	2	\~300 lines, Snell's law, TIR, transmission eval/sample. Probe: energy conservation.

M6.3 Principled → closure allocation	2	SVM node graph → closure list. Probe: closure types + weights.

M6.4 GGX in SHADE\_SURFACE bounce	2	Integrate closure into path state. Probe: ray direction distribution.

M6.5 Film denoising passes	1	\~100 lines, albedo/normal/depth/motion. Probe: pass values vs CPU.

M6.6 M6 Gate (Glass+Metal Cornell)	1	64 spp, compare beauty + passes.

M6.7 Texture fetch (software)	3	\~200 lines, format unpack, bilinear, mip, wrap, sRGB. Probe: 100 UVs.

M6.8 Principled + texture	1	Integration test. Probe: textured albedo.

M6.9 Texture Gate	1	Textured Cornell box.

M6.10 Point/Spot/Sun lights	2	\~300 lines total. Probe: LightSample per type.

M6.11 Background/Environment	2	\~250 lines, CDF binary search, env map. Probe: sample direction + pdf.

M6.12 Light Gate	1	Multi-light scene.

M7 Phase 0-2 (device enum + info + memory)	3	C++ only, follows existing backend patterns.

M7 Phase 3 (load\_kernels + SPIR-V embed)	2	CMake cycles\_add\_spirv\_kernel(), pipeline creation.

M7 Phase 4 (queue + enqueue)	2	Vulkan command buffer, push constants, dispatch.

M7 Phase 5-6 (Python + addon)	2	Tuple extension, preferences UI.

M7 Phase 7-8 (CMake + E2E test)	1	Integration.

M8.1 FSR1 EASU	2	\~200 lines, edge detection, directional weights. Probe: CPU reference.

M8.2 FSR1 RCAS	1	\~100 lines, local contrast, 5-tap sharpen. Probe: CPU reference.

M8.3 VulkanUpscaler class	2	C++ host, buffer management, dispatch. Probe: low-res → high-res match.

M8.4 RenderScheduler integration	2	Buffer swap, resolution\_divider, pass preservation. Probe: buffer params.

M8.5 M8 Gate (FSR1 E2E)	1	Cornell box 2× upscale.

M8.6 NCNN measurement spike	3-4	Standalone, not on critical path. Build NCNN, measure table.

Total critical path estimate: \~45-55 sessions (M4→M8 gate)  

With parallel spikes: +8-10 sessions (can overlap with main path)

QUESTIONS FOR THE USER (all USER DECISIONs in one place)

\#	Decision	Options	Recommendation

1	M5 light sampling: Light tree or flat distribution for gate?	A) Flat distribution (simple, \~60 lines)<br>B) Light tree (full, \~500 lines)	A — single area light, tree unjustified

2	M6 Principled phasing: Which phases in M6 gate?	A) Phase 2 only (GGX glossy+transmission)<br>B) Phase 2+3 (add sheen/clearcoat/aniso)<br>C) Phase 2+3+4 (add subsurface)	A — highest impact, \~400 lines each phase

3	M6 Image textures: Vulkan descriptors (Option 1) or software BDA (Option 2)?	A) VkImage + descriptor indexing (hardware filtering)<br>B) Software bilinear from raw buffer (fits BDA design)	B — consistent with no-descriptor ABI, \~200 lines

4	M7 slangc: Bundle in Blender repo or require Vulkan SDK at build?	A) Bundle slangc binary + DLLs (\~15 MB)<br>B) Require VULKAN\_SDK env var + slangc in PATH	A — zero-config for artists, matches Blender's CUDA toolkit bundling

5	M7 SPIR-V loading: Embed as hex strings or load .spv files at runtime?	A) Embed via cycles\_embed\_spirv() (like PTX)<br>B) Load from scripts/addons/cycles/kernels/vulkan/	A — simpler deploy, no file I/O at render start

6	M8 Upscaler order: FSR1 first, then NCNN if needed?	A) FSR1 only (no ML, \~2 days)<br>B) FSR1 + NCNN FSRCNN (measure first)<br>C) NCNN only (skip FSR1)	A — FSR1 is MIT, no deps, \~300 lines Slang. NCNN only if FSR1 quality insufficient.

7	M8 Temporal upscaling: Engine motion vectors or optical flow?	A) Engine MVs (free, accurate, needs temporally\_stable=true)<br>B) Optical flow (heavy, generic)<br>C) Per-frame only (accept flicker)	A — MVs already rendered for OIDN temporal; feed to custom temporal pass

8	M4→M5 transition: Any M4 cleanup before M5?	A) Clean up M4 diffs, finalize ABI<br>B) Start M5 immediately on current M4 state	A — ABI must be frozen (kernel params, integrator state layout) before M5 kernels depend on it

Files written in this research session

research/m5\_area\_light.md        ✓ Complete

research/m6\_materials\_textures\_lights.md  ✓ Complete

research/m7\_blender\_integration.md        ✓ Complete

research/m8\_upscaler.md          ✓ Complete

research/roadmap.md              ✓ Complete

Top 5 risks (summary)

1\. RX 570 driver bug — dynamic OpAccessChain on fixed arrays in PhysicalStorageBuffer miscompiled. Workaround mandatory everywhere.

2\. SwiftShader Int64 crash — avoid strided uint64\_t arrays; pointer members OK.

3\. Packed float3 (12B) stride — confirmed on both devices; accessors must use float\* base + 3\*i or float3\*.

4\. No float atomic add — CAS emulation required for all film/light accumulation.

5\. Light tree complexity — defer; flat distribution for M5/M6 gates.

