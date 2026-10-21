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
 * host's uint64_t slots (source: src/kernel/device/vulkan/params.h:40-41).
 * Raw pointer members and pointer arithmetic are only formed at the use
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
 * We mirror each pointer as a uint64_t Buffer Device Address.
 *
 * The C++ struct uses C-style anonymous nested structs
 * (`struct { ... } path;`); Slang gets explicit type names instead, which
 * does not change the layout. Member order and counts come from
 * src/kernel/integrator/state_template.h and shadow_state_template.h; the
 * offsets in the comments are relative to the start of IntegratorStateGPU. */

/* state_template.h:7-74, 29 pointer members -> 232 bytes */
struct IntegratorStatePath
{
    uint64_t render_pixel_index;         // offset   0
    uint64_t sample;                     // offset   8
    uint64_t bounce;                     // offset  16
    uint64_t transparent_bounce;         // offset  24
    uint64_t diffuse_bounce;             // offset  32
    uint64_t glossy_bounce;              // offset  40
    uint64_t transmission_bounce;        // offset  48
    uint64_t volume_bounce;              // offset  56
    uint64_t volume_bounds_bounce;       // offset  64
    uint64_t portal_bounce;              // offset  72
    uint64_t queued_kernel;              // offset  80
    uint64_t rng_pixel;                  // offset  88
    uint64_t rng_offset;                 // offset  96
    uint64_t visibility;                 // offset 104
    uint64_t flag;                       // offset 112
    uint64_t mnee;                       // offset 120
    uint64_t mnee_shadow_state;          // offset 128
    uint64_t optical_depth;              // offset 136
    uint64_t mis_ray_pdf;                // offset 144
    uint64_t mis_ray_object;             // offset 152
    uint64_t mis_origin_n;               // offset 160
    uint64_t min_ray_pdf;                // offset 168
    uint64_t continuation_probability;   // offset 176
    uint64_t throughput;                 // offset 184
    uint64_t unguided_throughput;        // offset 192
    uint64_t pass_diffuse_weight;        // offset 200
    uint64_t pass_glossy_weight;         // offset 208
    uint64_t denoising_feature_throughput;// offset 216
    uint64_t shader_sort_key;            // offset 224
};                                       // size 232

/* state_template.h:78-87, 8 pointer members -> 64 bytes */
struct IntegratorStateRay
{
    uint64_t P;           // offset 0
    uint64_t dP;          // offset 8
    uint64_t D;           // offset 16
    uint64_t dD;          // offset 24
    uint64_t tmin;        // offset 32
    uint64_t tmax;        // offset 40
    uint64_t time;        // offset 48
    uint64_t previous_dt; // offset 56
};                        // size 64

/* state_template.h:92-99, 6 pointer members -> 48 bytes */
struct IntegratorStateIsect
{
    uint64_t t;      // offset 0
    uint64_t u;      // offset 8
    uint64_t v;      // offset 16
    uint64_t prim;   // offset 24
    uint64_t object; // offset 32
    uint64_t type;   // offset 40
};                   // size 48

/* state_template.h:103-108, 4 pointer members -> 32 bytes */
struct IntegratorStateSubsurface
{
    uint64_t albedo;   // offset 0
    uint64_t radius;   // offset 8
    uint64_t anisotropy;// offset 16
    uint64_t N;        // offset 24
};                     // size 32

/* state_template.h:112-117, 2 array members x MAX_VOLUME_STACK_SIZE (32,
 * source: src/kernel/types.h:846-847) -> 64 pointers, 512 bytes */
struct IntegratorStateVolumeStack
{
    uint64_t object[32]; // offset 0
    uint64_t shader[32]; // offset 256
};                       // size 512

/* state_template.h:120-147, 8 pointer members -> 64 bytes
 * (path_segment is a pointer on the host whether or not __PATH_GUIDING__ is
 * defined: state_template.h:123-129) */
struct IntegratorStateGuiding
{
    uint64_t path_segment;                    // offset 0
    uint64_t use_surface_guiding;             // offset 8
    uint64_t sample_surface_guiding_rand;     // offset 16
    uint64_t surface_guiding_sampling_prob;   // offset 24
    uint64_t bssrdf_sampling_prob;            // offset 32
    uint64_t use_volume_guiding;              // offset 40
    uint64_t sample_volume_guiding_rand;      // offset 48
    uint64_t volume_guiding_sampling_prob;    // offset 56
};                                             // size 64

/* state_template.h:151-156, 3 pointer members -> 24 bytes */
struct IntegratorStateShadowLink
{
    uint64_t dedicated_light_weight; // offset 0
    uint64_t last_isect_prim;        // offset 8
    uint64_t last_isect_object;      // offset 16
};                                   // size 24

/* shadow_state_template.h:7-74, 24 pointer members -> 192 bytes */
struct IntegratorStateShadowPath
{
    uint64_t render_pixel_index;               // offset   0
    uint64_t sample;                           // offset   8
    uint64_t rng_pixel;                        // offset  16
    uint64_t rng_offset;                       // offset  24
    uint64_t bounce;                           // offset  32
    uint64_t transparent_bounce;               // offset  40
    uint64_t diffuse_bounce;                   // offset  48
    uint64_t glossy_bounce;                    // offset  56
    uint64_t transmission_bounce;              // offset  64
    uint64_t volume_bounds_bounce;             // offset  72
    uint64_t portal_bounce;                    // offset  80
    uint64_t queued_kernel;                    // offset  88
    uint64_t visibility;                       // offset  96
    uint64_t flag;                             // offset 104
    uint64_t throughput;                       // offset 112
    uint64_t unshadowed_throughput;            // offset 120
    uint64_t pass_diffuse_weight;              // offset 128
    uint64_t pass_glossy_weight;               // offset 136
    uint64_t packed_num_hits;                  // offset 144
    uint64_t lightgroup;                       // offset 152
    uint64_t unlit_throughput;                 // offset 160
    uint64_t path_segment;                     // offset 168
    uint64_t guiding_light_linking_mis_weight; // offset 176
    uint64_t bsdf_eval_average;                // offset 184
};                                             // size 192

/* shadow_state_template.h:78-88, 9 pointer members -> 72 bytes */
struct IntegratorStateShadowRay
{
    uint64_t P;                // offset 0
    uint64_t D;                // offset 8
    uint64_t tmin;             // offset 16
    uint64_t tmax;             // offset 24
    uint64_t time;             // offset 32
    uint64_t dP;               // offset 40
    uint64_t dD;               // offset 48
    uint64_t self_light_object;// offset 56
    uint64_t self_light_prim;  // offset 64
};                             // size 72

/* shadow_state_template.h:94-103, 6 array members x INTEGRATOR_SHADOW_ISECT_SIZE_GPU
 * (4, source: src/kernel/types.h:59) -> 24 pointers, 192 bytes */
struct IntegratorStateShadowIsect
{
    uint64_t t[4];      // offset 0
    uint64_t u[4];      // offset 32
    uint64_t v[4];      // offset 64
    uint64_t prim[4];   // offset 96
    uint64_t object[4]; // offset 128
    uint64_t type[4];   // offset 160
};                      // size 192

/* shadow_state_template.h:107-112, same shape as volume_stack -> 512 bytes */
struct IntegratorStateShadowVolumeStack
{
    uint64_t object[32]; // offset 0
    uint64_t shader[32]; // offset 256
};                       // size 512

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
    uint64_t queue_counter;                           // offset 1944 // state.h:197
    uint64_t sort_key_counter[18];                    // offset 1952 // state.h:200, DEVICE_GPU_KERNEL_INTEGRATOR_NUM = 18 (types.h:1745-1763,1827)
    uint64_t next_shadow_path_index;                  // offset 2096 // state.h:203
    uint64_t next_main_path_index;                    // offset 2104 // state.h:206
    uint64_t sort_partition_key_offsets;              // offset 2112 // state.h:209
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
