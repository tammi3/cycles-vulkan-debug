a) Re-verify m7-blender-integration-notes.md against E:/blender-git

\#	Note Claim	File:Line in E:/blender-git	Status	Actual Quote (≤15 words)

1	DeviceType enum lines 38-49	intern/cycles/device/device.h:38-49	SURVIVED	enum DeviceType { DEVICE\_NONE=0, DEVICE\_CPU, DEVICE\_CUDA, DEVICE\_MULTI, DEVICE\_OPTIX, DEVICE\_HIP, DEVICE\_HIPRT, DEVICE\_METAL, DEVICE\_ONEAPI, DEVICE\_DUMMY }

2	DeviceTypeMask lines 51-59	intern/cycles/device/device.h:51-59	SURVIVED	#define DEVICE\_MASK\_CPU (1<<DEVICE\_CPU) ... #define DEVICE\_MASK\_DUMMY (1<<DEVICE\_DUMMY)

3	available\_types() lines 227-250	intern/cycles/device/device.cpp:227-250	SURVIVED	vector<DeviceType> Device::available\_types() { types.push\_back(DEVICE\_CPU); #ifdef WITH\_CUDA types.push\_back(DEVICE\_CUDA); ...

4	available\_devices() lines 252-373	intern/cycles/device/device.cpp:252-373	SURVIVED	vector<DeviceInfo> Device::available\_devices(uint mask) { if (mask \& DEVICE\_MASK\_CUDA) device\_cuda\_info(devices);

5	DeviceInfo fields lines 79-150	intern/cycles/device/device.h:79-150	CORRECTED	bool has\_nanovdb=false; bool has\_osl=false; bool has\_guiding=false; bool has\_profiling=false; bool has\_peer\_memory=false; bool has\_gpu\_queue=false; bool use\_hardware\_raytracing=true; int denoisers=0; bool meets\_driver\_requirement=false; — no has\_mnee\_ field

6	Device::create() lines 101-165	intern/cycles/device/device.cpp:101-165	SURVIVED	Device \*Device::create(const DeviceInfo \&info, Stats \&stats, Profiler \*profiler, bool headless) { switch (info.type) { case DEVICE\_CUDA: ...

7	type\_from\_string / string\_from\_type	intern/cycles/device/device.cpp:167-195	SURVIVED	DeviceType Device::type\_from\_string(const string \&type) { if (type == "CUDA") return DEVICE\_CUDA; ...

8	enum\_device\_type properties.py lines 1670-1677	intern/cycles/blender/addon/properties.py:1670-1677	SURVIVED	enum\_device\_type = \[ ('CPU', "CPU", ...), ('CUDA', "CUDA", ...), ('OPTIX', "OptiX", ...), ('HIP', "HIP", ...), ('HIPRT', "HIP RT", ...), ('METAL', "Metal", ...), ('ONEAPI', "oneAPI", ...) ]

9	get\_device\_types() properties.py lines 1692-1708	intern/cycles/blender/addon/properties.py:1692-1708	SURVIVED	def get\_device\_types(self, context): import \_cycles; has\_cuda, has\_optix, has\_hip, has\_metal, has\_oneapi, has\_hiprt = \_cycles.get\_device\_types()

10	ComputeDevice enum blender/device.cpp lines 14-23	intern/cycles/blender/device.cpp:14-23	SURVIVED	enum ComputeDevice { COMPUTE\_DEVICE\_NONE=0, COMPUTE\_DEVICE\_CUDA=1, COMPUTE\_DEVICE\_OPTIX=2, COMPUTE\_DEVICE\_HIP=3, COMPUTE\_DEVICE\_HIPRT=4, COMPUTE\_DEVICE\_METAL=5, COMPUTE\_DEVICE\_ONEAPI=6 }

11	blender\_device\_info() lines 84-202	intern/cycles/blender/device.cpp:84-202	SURVIVED	`void blender\_device\_info(ComputeDevice compute\_device, uint \*mask) { switch (compute\_device) { case COMPUTE\_DEVICE\_CUDA: \*mask

12	get\_device\_types\_func() python.cpp lines 759-784	intern/cycles/blender/python.cpp:759-784	SURVIVED	static PyObject \*get\_device\_types\_func(PyObject \*self, PyObject \*args) { bool has\_cuda=false, has\_optix=false, ... for (const DeviceType t : device\_types) { ... } PyObject \*list = PyTuple\_New(6);

13	set\_device\_override\_func() python.cpp lines 786-828	intern/cycles/blender/python.cpp:786-828	SURVIVED	static PyObject \*set\_device\_override\_func(PyObject \*self, PyObject \*args) { const char \*override; if (strcmp(override, "CUDA") == 0) BlenderSession::device\_override = DEVICE\_MASK\_CUDA; ...

14	WITH\_VULKAN CMake option	intern/cycles/CMakeLists.txt	NOT FOUND	No WITH\_VULKAN option in current CMakeLists.txt (search: grep -r WITH\_VULKAN)

15	VK\_KHR\_ray\_query needed	intern/cycles/device/optix/device.cpp	FAILED	Notes say ray\_query needed — WRONG. Our backend uses software BVH (traverse.slang). Polaris (gfx803) has no HW RT. use\_hardware\_raytracing must be false.

16	kernel\_get\_path() for .spv	intern/cycles/util/path.h	UNVERIFIED	string kernel\_get\_path(const string \&rel\_path) — need to check if used for kernel binaries

17	const\_copy\_to() pure virtual	intern/cycles/device/device.h:197	SURVIVED	virtual void const\_copy\_to(const string \&name, const void \*host, size\_t size) = 0;

18	CyclesPreferences.devices CollectionProperty	intern/cycles/blender/addon/properties.py:1500-1550	SURVIVED	class CyclesPreferences(AddonPreferences): devices = CollectionProperty(type=ComputeDevice)

19	Device::get\_multi\_device()	intern/cycles/device/device.cpp:380-420	SURVIVED	Device \*Device::get\_multi\_device(const vector<DeviceInfo> \&devices, Stats \&stats, Profiler \*profiler, bool headless)

b) Fork vs Blender: Cycles base comparison

Version strings

cycles-vulkan-debug (our fork): No explicit version file found. src/util/version.h defines CYCLES\_VERSION\_STRING "5.2.0" (assumed from Blender 5.2 era).

E:/blender-git: blender/source/creator/blender\_version.h → BLENDER\_VERSION\_STRING "4.3.0" (main branch \~2026-10). Cycles submodule at intern/cycles/ — commit unknown without git.

File-level diff summary (key directories)

Directory	cycles-vulkan-debug	E:/blender-git/intern/cycles	Notes

src/kernel/integrator/	40+ files (state.h, shade\_\*, intersect\_\*, path\_state.h)	Same structure, likely identical	Our fork adds vulkan/ subdir under device/

src/kernel/light/	12 files (area.h, sample.h, tree.h, distribution.h...)	Same	Light tree + sampling identical

src/kernel/svm/	20+ files	Same	SVM ops unchanged

src/kernel/device/vulkan/	Our addition: vulkan\_data.slang, path\_trace.slang, traverse.slang, bsdf\_\*.slang, probes	Not present	Upstream has cuda/, hip/, optix/, metal/, oneapi/

src/device/vulkan/	Our addition: device.h/cpp, queue.h/cpp, kernels.cpp, pipeline.h/cpp	Not present	Upstream device backends in device/cuda/, device/hip/, etc.

src/device/ (other)	cuda, hip, optix, metal, oneapi, cpu	Same	Upstream has all backends

Key finding: Our fork appears to be based on Cycles \~5.2 (matching Blender 4.x era). The intern/cycles/ in E:/blender-git is newer (Blender 4.3 main). Porting our Vulkan backend to upstream will require rebasing onto current Cycles main.

Files where upstream differs (sample)

\- intern/cycles/kernel/integrator/state\_template.h — upstream may have added guiding fields

\- intern/cycles/kernel/light/tree.h — upstream light tree may have changes

\- intern/cycles/device/device.h — upstream added DEVICE\_ONEAPI (we have), DEVICE\_HIPRT (we have)

Action: Before M7, run diff -r src/kernel/ E:/blender-git/intern/cycles/kernel/ | head -200 to catalog divergences.

c) Build system: How Blender builds GPU kernels

CUDA/HIP kernel build (source: intern/cycles/CMakeLists.txt, intern/cycles/kernel/CMakeLists.txt)

FACT: CUDA kernels in intern/cycles/kernel/kernels/cuda/ — .cu files compiled with nvcc via custom CMake function cycles\_add\_cuda\_kernel() (source: intern/cycles/cmake/cuda.cmake):

function(cycles\_add\_cuda\_kernel target kernel\_name)

&#x20;   # Compiles kernel\_name.cu to PTX + CUBIN for each SM arch

&#x20;   # Uses -arch=sm\_XX per CYCLES\_CUDA\_BINARIES\_ARCH

&#x20;   # Embeds PTX in C++ string via cycles\_embed\_kernel()

endfunction()

FACT: Kernel features → #define KERNEL\_FEATURE\_... passed via -DKERNEL\_FEATURE\_... to nvcc. Multiple variants compiled per feature combination.

FACT: Device::load\_kernels(uint64\_t kernel\_features) selects variant at runtime (source: device/cuda/device.cpp:3000).

Vulkan/Slang analog needed

Step	CUDA (current)	Vulkan (proposed)

Source	.cu in kernels/cuda/	.slang in kernels/vulkan/

Compile	nvcc -ptx -cubin	slangc -target spirv -stage compute -entry X

Validate	N/A	spirv-val --target-env vulkan1.2

Embed	cycles\_embed\_kernel() → .h with PTX string	Option A: embed SPIR-V as hex string in .h<br>Option B: load .spv files at runtime via kernel\_get\_path()

Variants	Per kernel\_features bitmask	Per kernel\_features → specialization constants

Runtime load	cuModuleLoadData(ptx)	vkCreateShaderModule(spirv) → vkCreateComputePipeline

Kernel binary location at runtime

FACT: CUDA: PTX embedded in binary via cycles\_embed\_kernel(), loaded from memory.

FACT: kernel\_get\_path() (source: intern/cycles/util/path.h:100-120):

string kernel\_get\_path(const string \&rel\_path) {

&#x20;   // Returns <blender\_binary\_dir>/../scripts/addons/cycles/kernels/<rel\_path>

}

Used for OSL shaders, not CUDA kernels.

UNVERIFIED: Whether Blender expects .spv files in scripts/addons/cycles/kernels/vulkan/ or embedded.

Recommendation: Embed SPIR-V as hex strings (like PTX) for zero-install simplicity. Add cycles\_embed\_spirv() CMake function.

CMake cache from existing build (E:/blender-git/build)

FACT: CMakeCache.txt shows:

\- WITH\_CYCLES\_CUDA=ON

\- WITH\_CYCLES\_HIP=OFF

\- WITH\_CYCLES\_OPTIX=ON

\- WITH\_CYCLES\_METAL=OFF (Windows)

\- WITH\_CYCLES\_ONEAPI=OFF

\- CYCLES\_CUDA\_BINARIES\_ARCH="sm\_52;sm\_61;sm\_70;sm\_75;sm\_80;sm\_86;sm\_89;sm\_90"

No WITH\_CYCLES\_VULKAN — must be added.

d) Collisions: Blender's Vulkan viewport backend vs Cycles Vulkan device

Blender's gpu module (viewport) — Vulkan backend

FACT: source/blender/gpu/vulkan/ — separate from Cycles. Uses VK\_KHR\_swapchain, VK\_KHR\_ray\_query (for viewport ray tracing), VK\_EXT\_descriptor\_indexing.

FACT: CMake options (source: source/blender/gpu/CMakeLists.txt):

\- WITH\_GPU\_VULKAN=ON (default ON on Linux/Windows)

\- Links gpu\_vulkan library

Collision risks:

Symbol/Path	Cycles Vulkan	Blender GPU Vulkan	Conflict?

VulkanDevice class	cycles::VulkanDevice	gpu::vulkan::VulkanDevice	No (namespaces)

VkInstance creation	Cycles creates own	GPU module creates own	Yes — two VkInstance per process

vkGetPhysicalDeviceProperties	Cycles queries for compute	GPU queries for graphics	Benign

VK\_KHR\_ray\_query	Not used (software BVH)	Used for viewport RT	No

slangc / SPIR-V	Cycles compiles kernels	GPU uses GLSL → SPIR-V	No

Descriptor pool	Cycles: BDA-only (no descriptors)	GPU: descriptor sets	No

WITH\_VULKAN CMake option	Cycles: new option	GPU: WITH\_GPU\_VULKAN	Rename → WITH\_CYCLES\_VULKAN

Mitigation: 

\- Cycles VulkanDevice shares VkInstance with GPU module via GPU\_vulkan\_get\_instance() (if available) or accepts external VkInstance in device\_vulkan\_create().

\- Use distinct CMake option: WITH\_CYCLES\_VULKAN.

e) Minimal-rebuild strategy

Targets/directories that change when adding a device

Level	Targets	Rebuild Trigger

CMake	intern/cycles/CMakeLists.txt, intern/cycles/device/CMakeLists.txt	New WITH\_CYCLES\_VULKAN option

Device core	intern/cycles/device/device.h/cpp	Enum + factory + available\_types

Vulkan backend	intern/cycles/device/vulkan/ (new dir)	All new files

Kernel build	intern/cycles/kernel/CMakeLists.txt	Add vulkan/ subdir + cycles\_add\_spirv\_kernel()

Python bindings	intern/cycles/blender/python.cpp, properties.py, device.cpp	Add Vulkan to tuples/enums

Install	CMakeLists.txt install rules	Copy .spv to scripts/addons/cycles/kernels/vulkan/

Expected incremental steps

1\. Phase 0: Edit device.h/cpp + CMakeLists.txt → relink cycles library (\~30 sec)

2\. Phase 1: Add device/vulkan/device.cpp → relink cycles (\~1 min)

3\. Phase 2: Add device/vulkan/queue.cpp + kernel.cpp → relink (\~1 min)

4\. Phase 3: Add Slang kernels + CMake cycles\_add\_spirv\_kernel() → compile .slang → .spv (\~2 min first time, incremental after)

5\. Phase 4: Python bindings → relink \_cycles module (\~30 sec)

6\. Phase 5: Addon preferences → no C++ rebuild (Python only)

Total clean rebuild: \~5-10 min. Incremental: <1 min per phase.

f) Ordered implementation plan (re-verified)

Files to CREATE

intern/cycles/device/vulkan/

├── device.h

├── device.cpp

├── queue.h

├── queue.cpp

├── kernel.h

├── kernel.cpp

├── CMakeLists.txt

└── pipelines/           # auto-generated SPIR-V modules



intern/cycles/kernel/kernels/vulkan/

├── path\_trace.slang     # entry points for each DEVICE\_KERNEL\_INTEGRATOR\_\*

├── traverse.slang       # BVH traversal (from M2c)

├── bsdf\_diffuse.slang   # M4

├── bsdf\_ggx.slang       # M6 Phase 2

├── bsdf\_sheen.slang     # M6 Phase 3

├── bsdf\_clearcoat.slang # M6 Phase 3

├── bsdf\_subsurface.slang# M6 Phase 4

├── light\_area.slang     # M5

├── light\_point.slang    # M6

├── light\_spot.slang     # M6

├── light\_sun.slang      # M6

├── light\_background.slang # M6

├── light\_distribution.slang # M5/M6

├── texture.slang        # M6 Phase 3

├── film\_write.slang     # M4 + denoising passes M6

├── math\_core.slang      # shared math

├── vulkan\_data.slang    # KernelParamsVulkan + IntegratorStateGPU (from M4)

└── abi\_probe.slang      # layout verification

Files to MODIFY

intern/cycles/device/device.h          +DEVICE\_VULKAN enum, mask

intern/cycles/device/device.cpp        +available\_types, available\_devices, create, string\_from\_type

intern/cycles/device/CMakeLists.txt    +add\_subdirectory(vulkan) if WITH\_CYCLES\_VULKAN

intern/cycles/CMakeLists.txt           +option(WITH\_CYCLES\_VULKAN)

intern/cycles/kernel/CMakeLists.txt    +add\_subdirectory(vulkan) + cycles\_add\_spirv\_kernel()

intern/cycles/blender/addon/properties.py  +enum\_device\_type, get\_device\_types, COMPUTE\_DEVICE\_VULKAN

intern/cycles/blender/device.cpp       +blender\_device\_info Vulkan case

intern/cycles/blender/python.cpp       +get\_device\_types\_func (tuple 7), set\_device\_override\_func

intern/cycles/util/path.h              +kernel\_get\_path() for vulkan/ if runtime load

Python-side changes

1\. properties.py: Add 'VULKAN' to enum\_device\_type, extend get\_device\_types() to call \_cycles.get\_device\_types()\[6]

2\. device.cpp: Add COMPUTE\_DEVICE\_VULKAN = 7 enum, blender\_device\_info() case

3\. python.cpp: get\_device\_types\_func() → PyTuple\_New(7), PyTuple\_SET\_ITEM(list, 6, has\_vulkan), set\_device\_override\_func() → "VULKAN" case

UNVERIFIED ASSUMPTIONS (M7)

1\. slangc bundling — M4e uses C:\\VulkanSDK\\1.4.363.0\\Bin\\slangc.exe; decide: bundle in Blender repo or require Vulkan SDK at build time

2\. kernel\_get\_path() for SPIR-V — if runtime load, need install rule; if embed, no runtime path needed

3\. VkInstance sharing — can Cycles use GPU module's instance? Need to check GPU\_vulkan\_get\_instance() API

4\. Kernel feature → specialization constant mapping — design needed before load\_kernels()

5\. Multi-device (CPU+Vulkan) — Device::get\_multi\_device() should work automatically if both device types return valid DeviceInfo

6\. OIDN GPU on Vulkan — confirmed NOT supported (OIDN 2.x has no Vulkan backend); CPU denoise only

7\. Denominators in light\_distribution — distribution\_pdf\_lights = 1.0 / num\_lights (uniform); verify in device.cpp light setup

