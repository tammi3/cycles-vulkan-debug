// SPDX-FileCopyrightText: 2011-2022 Blender Foundation
// SPDX-License-Identifier: Apache-2.0

/* M4b kernel ABI shared between the C++ host and the Slang kernels.
 *
 * This mirrors the host-side KernelParamsVulkan
 * (source: src/kernel/device/vulkan/params.h:38-44), which Cycles fills
 * exclusively through Device::const_copy_to(name, host, size):
 *   - "data": the KernelData blob (2560 bytes, opaque here), offset 0
 *   - one 64-bit Buffer Device Address slot per KERNEL_DATA_ARRAY entry
 *     (53 of them, source: src/kernel/data_arrays.h:15-97), offset 2560
 *   - "integrator_state": IntegratorStateGPU
 *     (source: src/kernel/integrator/state.h:140), offset 2984
 *   - total: 5120 bytes
 *
 * Every offset quoted below is locked on the C++ side by static_asserts in
 * src/device/vulkan/kernels.cpp (M4b ABI probe). If Cycles' layout changes,
 * the host build breaks there; at runtime the CYCLES_VULKAN_ABI_PROBE=1
 * dispatch (see abi_probe.slang) re-checks the Slang layout against the C++
 * layout value by value.
 *
 * Device pointers are uint64_t Buffer Device Addresses, exactly like the
 * host's uint64_t slots (source: src/kernel/device/vulkan/params.h:40-41)
 * — true for the KernelParamsVulkan slots below; the integrator_state
 * mirror uses typed pointers instead (see the M4e S1b-7 note). Raw pointer
 * members and pointer arithmetic are only formed at the use
 * site, in the style of the M2c spike (agentreference/bvh_traverse.slang),
 * the only Slang code known to compile and run on the RX 570.
 *
 * Slang's struct layout rules place each member at its natural alignment,
 * the same rules the host's MSVC x64 ABI applies to this all-8-byte-tail
 * struct; no float3/vector types appear anywhere below, so the two sides
 * cannot disagree on padding. */

/* Opaque KernelData blob: sizeof(KernelData) == 2560 bytes
 * (source: src/kernel/device/vulkan/kernels.cpp static_assert). We never
 * interpret its contents from Slang in M4; only the raw size matters here. */
static const uint VULKAN_KERNELDATA_BYTES = 2560u;
static const uint VULKAN_KERNELDATA_WORDS = VULKAN_KERNELDATA_BYTES / 4u;

struct KernelDataBlob
{
    uint words[VULKAN_KERNELDATA_WORDS]; // offset 0..2559
};

/* integrator_state: mirror of IntegratorStateGPU
 * (source: src/kernel/integrator/state.h:140-213).
 *
 * On the host every member of this struct is an 8-byte device pointer:
 * state.h:170 expands each KERNEL_STRUCT_MEMBER to `ccl_global type *name;`
 * and `ccl_global` is empty in host C++ (Apple-only packed layout,
 * __INTEGRATOR_GPU_PACKED_STATE__, is not active here: src/kernel/types.h:626).
 * We mirror each pointer as its TYPED device pointer (see the M4e S1b-7 note
 * below); the KernelParamsVulkan slots further down keep uint64_t for now —
 * they are retyped when a consumer lands (retype-on-landing policy).
 *
 * The C++ struct uses C-style anonymous nested structs
 * (`struct { ... } path;`); Slang gets explicit type names instead, which
 * does not change the layout. Member order and counts come from
 * src/kernel/integrator/state_template.h and shadow_state_template.h; the
 * offsets in the comments are relative to the start of IntegratorStateGPU. */

/* M4e S1b-6/S1b-7: SoA address members are declared as TYPED device
 * pointers instead of uint64_t. A typed pointer is 8 bytes with 8-byte
 * alignment — identical to uint64_t — so every offset and sizeof in this
 * file is UNCHANGED (the M4b/M4e host static_asserts and the runtime ABI
 * probe lock that). state.h:170 expands each KERNEL_STRUCT_MEMBER to
 * `ccl_global type *name`, so the host member is always a POINTER to an
 * array of `type`, and the mirror's element type follows state_template.h
 * / shadow_state_template.h exactly:
 *     uint32_t -> uint*,     int    -> int*,      float -> float*,
 *     packed_float3 / PackedSpectrum (== packed_float3, util/types_spectrum.h:14)
 *                        -> float3*   (12-byte elements),
 *     uint16_t -> Subword16T*,  uint8_t / bool -> Subword8T*  (macro-dependent
 *     word vs element view, see below).
 * WHY the full retype (M4e S1b-7, user decision): even an UNREAD uint64_t
 * member forces SPIR-V OpCapability Int64, and SwiftShader neither exposes
 * shaderInt64 nor supports OpConvertPtrToU (observed 2026-10-24:
 * gate_swiftshader.log -> 0xC00000FD). With every reachable uint64 gone, a
 * kernel that only LOADS through this mirror compiles Int64-free and runs on
 * both reference devices. Members whose pointed-to element really is 8 bytes
 * of opaque data use uint2* (same 8-byte stride, no Int64): see
 * path_segment below. */

/* Sub-word (uint16/uint8/bool) SoA view type, switched by the same macro the
 * accessors use (VULKAN_SUBWORD_NARROW, default 0). Default: a uint* WORD view
 * (2 uint16 or 4 uint8 elements per 32-bit word) for the portable atomic
 * AND/OR fallback — no Int16/Int8 capability needed. Narrow mode: the true
 * element pointer (uint16_t or uint8_t POINTERS, requires shaderInt16/shaderInt8; the
 * accessor layer's store/load_u16/u8 switch in lockstep). Both views address
 * the same buffer the host allocated as uint16_t/uint8_t (path_trace_work_gpu.cpp),
 * only the Slang-side VIEW differs. */
#ifndef VULKAN_SUBWORD_NARROW
#define VULKAN_SUBWORD_NARROW 0
#endif
#if VULKAN_SUBWORD_NARROW
typedef uint16_t Subword16T;
typedef uint8_t  Subword8T;
#else
typedef uint     Subword16T;
typedef uint     Subword8T;
#endif

/* state_template.h:7-74, 29 pointer members -> 232 bytes */
struct IntegratorStatePath
{
    uint*    render_pixel_index;         // offset   0
    uint*    sample;                     // offset   8
    Subword16T* bounce;                  // offset  16
    Subword16T* transparent_bounce;      // offset  24
    Subword16T* diffuse_bounce;          // offset  32 (uint16_t, state_template.h:21)
    Subword16T* glossy_bounce;           // offset  40 (uint16_t, :23)
    Subword16T* transmission_bounce;     // offset  48 (uint16_t, :25)
    Subword16T* volume_bounce;           // offset  56 (uint16_t, :27)
    Subword16T* volume_bounds_bounce;    // offset  64 (uint16_t, :29)
    Subword16T* portal_bounce;           // offset  72 (uint16_t, :31)
    Subword16T* queued_kernel;           // offset  80
    uint*    rng_pixel;                  // offset  88 (uint32_t, state_template.h:35)
    Subword16T* rng_offset;              // offset  96
    Subword8T*  visibility;              // offset 104
    uint*    flag;                       // offset 112
    Subword8T*  mnee;                    // offset 120
    int*   mnee_shadow_state;            // offset 128 (int, state_template.h:45)
    float*   optical_depth;              // offset 136
    float*   mis_ray_pdf;                // offset 144
    int*   mis_ray_object;               // offset 152 (int, state_template.h:54)
    float3*  mis_origin_n;               // offset 160
    float*   min_ray_pdf;                // offset 168
    float*   continuation_probability;   // offset 176
    float3*  throughput;                 // offset 184
    float* unguided_throughput;          // offset 192 (float, state_template.h:65)
    float3*  pass_diffuse_weight;        // offset 200
    float3*  pass_glossy_weight;         // offset 208 (PackedSpectrum, :68)
    float3*  denoising_feature_throughput;// offset 216 (PackedSpectrum, :70)
    uint*    shader_sort_key;            // offset 224
};                                       // size 232

/* state_template.h:78-87, 8 pointer members -> 64 bytes */
struct IntegratorStateRay
{
    float3* P;           // offset 0
    float*  dP;          // offset 8
    float3* D;           // offset 16
    float*  dD;          // offset 24
    float*  tmin;        // offset 32
    float*  tmax;        // offset 40
    float*  time;        // offset 48
    float* previous_dt; // offset 56 (state_template.h:86)
};                        // size 64

/* state_template.h:92-99, 6 pointer members -> 48 bytes */
struct IntegratorStateIsect
{
    float* t;      // offset 0
    float* u;      // offset 8  (state_template.h:94)
    float* v;      // offset 16 (state_template.h:95)
    int*  prim;    // offset 24
    int*  object;  // offset 32
    int*  type;    // offset 40
};                   // size 48

/* state_template.h:103-108, 4 pointer members -> 32 bytes */
struct IntegratorStateSubsurface
{
    float3* albedo;   // offset 0  (PackedSpectrum, state_template.h:104)
    float3* radius;   // offset 8  (PackedSpectrum, state_template.h:105)
    float*  anisotropy;// offset 16 (float, state_template.h:106)
    float3* N;        // offset 24 (packed_float3, state_template.h:107)
};                     // size 32

/* state_template.h:112-117, 2 array members x MAX_VOLUME_STACK_SIZE (32,
 * source: src/kernel/types.h:847). M4e S1b-5: this is ARRAY-OF-STRUCTS, NOT
 * struct-of-arrays. state.h:174 maps KERNEL_STRUCT_ARRAY_MEMBER to
 * KERNEL_STRUCT_MEMBER (ccl_global type *name) and state.h:178-180 maps
 * KERNEL_STRUCT_END_ARRAY to "} name[gpu_size];", so the member is
 *   struct { int *object; int *shader; } volume_stack[32];
 * i.e. 32 elements of 16 bytes (a PAIR of pointers per slot), element i at byte
 * i*16: object at i*16+0, shader at i*16+8. The host allocator
 * (path_trace_work_gpu.cpp:174-186) writes volume_stack[i].object and
 * volume_stack[i].shader as separate SoA allocations, confirming the pair-per-slot
 * shape. Total size 32*16 = 512, unchanged from the old flat layout. */
struct IntegratorStateVolumeStackElement
{
    int* object; // offset 0 (device_ptr -> int[] for slot i)
    int* shader; // offset 8 (device_ptr -> int[] for slot i)
};                   // size 16

struct IntegratorStateVolumeStack
{
    IntegratorStateVolumeStackElement slot[32]; // offset 0, 32*16 = 512 bytes
};                                              // size 512

/* state_template.h:120-147, 8 pointer members -> 64 bytes
 * (path_segment is a pointer on the host whether or not __PATH_GUIDING__ is
 * defined: state_template.h:123-129) */
struct IntegratorStateGuiding
{
    /* path_segment: state.h:170 makes the host member `uint64_t *path_segment`
     * (state_template.h:129, non-openpgl branch) — a device pointer to 8-byte
     * elements. uint2* keeps that exact 8-byte element stride without
     * OpCapability Int64 (SwiftShader has no shaderInt64). The element is
     * never dereferenced: KERNEL_FEATURE_PATH_GUIDING gates all users off. */
    uint2*  path_segment;                    // offset 0
    Subword8T*  use_surface_guiding;         // offset 8  (bool, state_template.h:132)
    float*  sample_surface_guiding_rand;     // offset 16 (float, :135)
    float*  surface_guiding_sampling_prob;   // offset 24 (float, :137)
    float*  bssrdf_sampling_prob;            // offset 32 (float, :139)
    Subword8T*  use_volume_guiding;          // offset 40 (bool, :141)
    float*  sample_volume_guiding_rand;      // offset 48 (float, :144)
    float*  volume_guiding_sampling_prob;    // offset 56 (float, :146)
};                                             // size 64

/* state_template.h:151-156, 3 pointer members -> 24 bytes */
struct IntegratorStateShadowLink
{
    float* dedicated_light_weight; // offset 0  (state_template.h:152)
    int*   last_isect_prim;        // offset 8  (state_template.h:154)
    int*   last_isect_object;      // offset 16 (state_template.h:155)
};                                   // size 24

/* shadow_state_template.h:7-74, 24 pointer members -> 192 bytes */
struct IntegratorStateShadowPath
{
    uint*    render_pixel_index;               // offset   0
    uint*    sample;                          // offset   8 (uint32_t, shadow_state_template.h:11)
    uint*    rng_pixel;                       // offset  16 (uint32_t, :13)
    Subword16T* rng_offset;                   // offset  24 (uint16_t, :15)
    Subword16T* bounce;                       // offset  32 (uint16_t, :17)
    Subword16T* transparent_bounce;           // offset  40 (uint16_t, :19)
    Subword16T* diffuse_bounce;               // offset  48 (uint16_t, :21)
    Subword16T* glossy_bounce;                // offset  56 (uint16_t, :23)
    Subword16T* transmission_bounce;          // offset  64 (uint16_t, :25)
    Subword16T* volume_bounds_bounce;         // offset  72 (uint16_t, :27)
    Subword16T* portal_bounce;                // offset  80 (uint16_t, :29)
    Subword16T* queued_kernel;                 // offset  88
    Subword8T*  visibility;                    // offset  96 (uint8_t, :33)
    uint*    flag;                             // offset 104
    float3*  throughput;                       // offset 112
    float3*  unshadowed_throughput;            // offset 120 (PackedSpectrum, :40)
    float3*  pass_diffuse_weight;              // offset 128 (PackedSpectrum, :44)
    float3*  pass_glossy_weight;               // offset 136 (PackedSpectrum, :45)
    Subword16T* packed_num_hits;               // offset 144 (uint16_t, :51)
    Subword8T*  lightgroup;                    // offset 152
    float3*  unlit_throughput;                 // offset 160 (PackedSpectrum, :55)
    /* Same uint64_t-element pointer as guiding.path_segment (state.h:170 with
     * shadow_state_template.h:62) -> uint2* keeps the 8-byte stride, no Int64. */
    uint2*  path_segment;                      // offset 168
    float*  guiding_light_linking_mis_weight;  // offset 176 (float, :65)
    float*  bsdf_eval_average;                 // offset 184 (float, :72)
};                                             // size 192

/* shadow_state_template.h:78-88, 9 pointer members -> 72 bytes */
struct IntegratorStateShadowRay
{
    float3* P;                // offset 0
    float3* D;                // offset 8
    float*  tmin;             // offset 16
    float* tmax;             // offset 24 (shadow_state_template.h:82)
    float* time;             // offset 32 (shadow_state_template.h:83)
    float3* dP;              // offset 40 (packed_float3, :84)
    float3* dD;              // offset 48 (packed_float3, :85)
    int*  self_light_object; // offset 56 (int, :86)
    int*  self_light_prim;   // offset 64 (int, :87)
};                             // size 72

/* shadow_state_template.h:94-103, 6 array members x INTEGRATOR_SHADOW_ISECT_SIZE_GPU
 * (4, source: src/kernel/types.h:59). M4e S1b-5: ARRAY-OF-STRUCTS, NOT struct-of-arrays.
 * state.h:174 + state.h:178-180 give
 *   struct { float *t; float *u; float *v;
 *            int *prim; int *object; int *type; } shadow_isect[4];
 * i.e. 4 elements of 48 bytes (a 6-pointer tuple per slot), element i at byte i*48:
 * t@i*48+0, u@+8, v@+16, prim@+24, object@+32, type@+40. The host allocator
 * (path_trace_work_gpu.cpp:174-186) writes shadow_isect[i].t etc. as separate SoA
 * allocations, confirming the tuple-per-slot shape. Total size 4*48 = 192, unchanged. */
struct IntegratorStateShadowIsectElement
{
    float*  t;      // offset 0  (device_ptr -> float[])
    float*  u;      // offset 8  (device_ptr -> float[], shadow_state_template.h:96)
    float*  v;      // offset 16 (device_ptr -> float[], :97)
    int*    prim;   // offset 24 (device_ptr -> int[],    :98)
    int*  object;   // offset 32 (device_ptr -> int[])
    int*    type;   // offset 40 (device_ptr -> int[],    :100)
};                   // size 48

struct IntegratorStateShadowIsect
{
    IntegratorStateShadowIsectElement slot[4]; // offset 0, 4*48 = 192 bytes
};                                             // size 192

/* shadow_state_template.h:107-112, same AoS shape as volume_stack (M4e S1b-5):
 * struct { int *object; int *shader; } shadow_volume_stack[32]; 32*16 = 512 bytes. */
struct IntegratorStateShadowVolumeStackElement
{
    int* object; // offset 0 (device_ptr -> int[] for slot i)
    int* shader; // offset 8 (device_ptr -> int[] for slot i)
};                   // size 16

struct IntegratorStateShadowVolumeStack
{
    IntegratorStateShadowVolumeStackElement slot[32]; // offset 0, 32*16 = 512 bytes
};                                                    // size 512

/* state.h:100-103: struct IntegratorQueueCounter { int num_queued[18];
 * int cache_miss; }. Mirrored so queue_counter below can be a TYPED pointer
 * (a uint64_t* pointee would re-introduce OpCapability Int64). The pointee is
 * never touched by the gate. DEVICE_GPU_KERNEL_INTEGRATOR_NUM = 18
 * (types.h:1745-1763,1827). */
struct IntegratorQueueCounter
{
    int num_queued[18]; // offset 0, 72 bytes
    int cache_miss;     // offset 72, size 76
};

/* IntegratorStateGPU itself (source: src/kernel/integrator/state.h:140-213).
 * 265 pointer members = 2120 bytes + uint sort_partition_divisor (state.h:212)
 * = 2124, padded to 2128 by the struct's 8-byte alignment. Locked by
 * static_assert(sizeof(IntegratorStateGPU) == 2128) in kernels.cpp. */
struct IntegratorStateGPU
{
    IntegratorStatePath path;                         // offset    0
    IntegratorStateRay ray;                           // offset  232
    IntegratorStateIsect isect;                       // offset  296
    IntegratorStateSubsurface subsurface;             // offset  344
    IntegratorStateVolumeStack volume_stack;          // offset  376
    IntegratorStateGuiding guiding;                   // offset  888
    IntegratorStateShadowLink shadow_link;            // offset  952
    IntegratorStateShadowPath shadow_path;            // offset  976
    IntegratorStateShadowRay shadow_ray;              // offset 1168
    IntegratorStateShadowIsect shadow_isect;          // offset 1240
    IntegratorStateShadowVolumeStack shadow_volume_stack;// offset 1432
    IntegratorQueueCounter* queue_counter;    // offset 1944 // state.h:197
    int*    sort_key_counter[18];             // offset 1952 // state.h:200: `int *sort_key_counter[18]` — an ARRAY OF 18 int pointers (18*8 = 144 bytes)
    int*    next_shadow_path_index;           // offset 2096 // state.h:203
    int*    next_main_path_index;             // offset 2104 // state.h:206
    int*    sort_partition_key_offsets;       // offset 2112 // state.h:209
    uint sort_partition_divisor;                      // offset 2120 // state.h:212
    /* MSVC pads the struct to its 8-byte alignment (2124 -> 2128); slangc
     * 2026.17 reports sizeof WITHOUT that tail padding (observed: 2124), so
     * we add it explicitly to keep both sides at 2128. */
    uint pad_tail;                                    // offset 2124 (implicit host padding)
};                                                     // size 2128

/* The M4b kernel params struct, mirroring KernelParamsVulkan
 * (source: src/kernel/device/vulkan/params.h:38-44). */
struct KernelParamsVulkan
{
    KernelDataBlob data;      // offset    0, 2560 bytes

    /* The 53 KERNEL_DATA_ARRAY slots, in kernel/data_arrays.h:15-97 order.
     * Each is a Buffer Device Address (host: uint64_t, params.h:40-41). */
    uint64_t bvh_nodes;                              // offset 2560
    uint64_t bvh_leaf_nodes;                         // offset 2568
    uint64_t prim_type;                              // offset 2576
    uint64_t prim_visibility;                        // offset 2584
    uint64_t prim_index;                             // offset 2592
    uint64_t prim_object;                            // offset 2600
    uint64_t object_node;                            // offset 2608
    uint64_t prim_time;                              // offset 2616
    uint64_t objects;                                // offset 2624
    uint64_t object_motion_pass;                     // offset 2632
    uint64_t object_motion;                          // offset 2640
    uint64_t object_flag;                            // offset 2648
    uint64_t object_prim_offset;                     // offset 2656
    uint64_t camera_motion;                          // offset 2664
    uint64_t tri_shader;                             // offset 2672
    uint64_t tri_vindex;                             // offset 2680
    uint64_t tri_verts;                              // offset 2688
    uint64_t curves;                                 // offset 2696
    uint64_t curve_keys;                             // offset 2704
    uint64_t curve_segments;                         // offset 2712
    uint64_t points;                                 // offset 2720
    uint64_t points_shader;                          // offset 2728
    uint64_t attributes_map;                         // offset 2736
    uint64_t attributes_float;                       // offset 2744
    uint64_t attributes_float2;                      // offset 2752
    uint64_t attributes_float3;                      // offset 2760
    uint64_t attributes_float4;                      // offset 2768
    uint64_t attributes_uchar4;                      // offset 2776
    uint64_t attributes_normal;                      // offset 2784
    uint64_t light_distribution;                     // offset 2792
    uint64_t lights;                                 // offset 2800
    uint64_t light_background_marginal_cdf;          // offset 2808
    uint64_t light_background_conditional_cdf;       // offset 2816
    uint64_t light_tree_nodes;                       // offset 2824
    uint64_t light_tree_emitters;                    // offset 2832
    uint64_t light_to_tree;                          // offset 2840
    uint64_t object_lookup_offset;                   // offset 2848
    uint64_t triangle_to_tree;                       // offset 2856
    uint64_t particles;                              // offset 2864
    uint64_t svm_nodes;                              // offset 2872
    uint64_t shaders;                                // offset 2880
    uint64_t lookup_table;                           // offset 2888
    uint64_t sample_pattern_lut;                     // offset 2896
    uint64_t ies;                                    // offset 2904
    uint64_t volume_tree_nodes;                      // offset 2912
    uint64_t volume_tree_roots;                      // offset 2920
    uint64_t volume_tree_root_ids;                   // offset 2928
    uint64_t volume_step_size;                       // offset 2936
    uint64_t image_textures;                         // offset 2944
    uint64_t image_texture_tile_descriptors;         // offset 2952
    uint64_t image_texture_tile_access_state;        // offset 2960
    uint64_t image_texture_udims;                    // offset 2968
    uint64_t image_info;                             // offset 2976

    IntegratorStateGPU integrator_state;             // offset 2984, 2128 bytes

    /* MSVC rounds the struct size up to its 16-byte alignment (5112 -> 5120);
     * slangc 2026.17 does not add that tail padding by itself (observed:
     * sizeof = 5108 before the explicit pads above and below), so we spell
     * it out. Host: sizeof(KernelParamsVulkan) == 5120, see the static_assert
     * in src/device/vulkan/kernels.cpp. */
    uint pad_params[2];                              // offset 5112 (implicit host padding)
};                                                   // size 5120
