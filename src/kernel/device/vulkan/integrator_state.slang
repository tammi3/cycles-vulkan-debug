// SPDX-FileCopyrightText: 2011-2022 Blender Foundation
// SPDX-License-Identifier: Apache-2.0

/* M4e S1b-6 Task 1: integrator state ACCESSORS (Slang equivalent of Cycles'
 * INTEGRATOR_STATE / INTEGRATOR_STATE_WRITE macros).
 *
 * Cycles' macro (source: src/kernel/integrator/state.h:273-276, the NON-packed
 * __KERNEL_GPU__ branch, which is the active one on this Windows target because
 * __INTEGRATOR_GPU_PACKED_STATE__ is Apple-ARM64-only: src/kernel/types.h:626):
 *
 *   INTEGRATOR_STATE(state, nested, member)
 *       == kernel_integrator_state.nested.member[state]
 *   INTEGRATOR_STATE_ARRAY(state, nested, i, member)
 *       == kernel_integrator_state.nested[i].member[state]
 *
 * `kernel_integrator_state` is an IntegratorStateGPU whose members are POINTERS
 * to Structure-of-Arrays (SoA) buffers in device memory: every member is its own
 * array of `max_num_paths` elements (source: src/integrator/path_trace_work_gpu.cpp:
 * 145-156, one device_only_memory<type> per KERNEL_STRUCT_MEMBER). Reading a
 * member = load its SoA data pointer, then index that pointer by `state`.
 *
 * ADDRESSING (all three patterns compile+validate with slangc 2026.17 +
 * spirv-val, proven in cast_test.slang, M4e S1b-6):
 *
 *  NOTE on cast<>: `cast<T[]>` / `cast<T*>` do NOT exist in this slangc build
 *  (error E30015), and there is NO uint64->typed-pointer conversion (functional
 *  `T*(addr)` parses as multiplication). The mirror (vulkan_data.slang) therefore
 *  stores the SoA addresses of every member reached below as TYPED device
 *  pointers (uint, float, float3, int, Subword16T and Subword8T POINTERS), and
 *  each
 *  accessor loads the typed pointer member and indexes it directly. A typed
 *  pointer is 8 bytes with 8-byte alignment = uint64_t, so the byte layout and
 *  every ABI-probe offset are unchanged (verified: RX 570 ABI probe still
 *  ALL VALUES MATCH after the retype).
 *
 *  (A) SCALAR / sub-word / packed3 members:
 *        T* data_ptr = ist-><group>.<member>;   // scalar pointer member: safe
 *        data_ptr[state] = v;                   // local ptr + dynamic index:
 *                                               // OpPtrAccessChain, safe.
 *      The SLANG RULE forbids only the INLINE form `ist-><group>.arr[i]` on a
 *      fixed ARRAY member with a dynamic index (miscompiles to element 0 on the
 *      RX 570 driver). Loading a scalar member first is the proven-safe shape.
 *
 *  (B) AoS ARRAY members (volume_stack/shadow_isect/shadow_volume_stack,
 *      layout proven M4e S1b-5: array-of-structs-of-pointers, base offsets
 *      376/1240/1432, element strides 16/48/16, locked by host static_asserts
 *      in kernels.cpp): take the address of slot 0 with a CONSTANT index, then
 *      index that local base pointer with the DYNAMIC slot index:
 *        IntegratorStateVolumeStackElement* base = &ist->volume_stack.slot[0];
 *        int* soa = base[i].object;   // i dynamic -> OpPtrAccessChain (safe)
 *        soa[state] = v;
 *      `&slot[0]` on a device-pointer struct is allowed (constant index);
 *      arrays do not decay automatically, and push-constant-embedded structs
 *      are immutable, which is why the entry point passes IntegratorStateGPU*
 *      (a device pointer), not an embedded struct.
 *
 *  (C) packed3 stride: float3* indexes at ArrayStride 12 (confirmed from
 *      disassembly of a typed float3* member), matching Cycles'
 *      sizeof(packed_float3)==12 (types_float3.h:198 static_assert) and
 *      PackedSpectrum==packed_float3 (types_spectrum.h:14). packed3/PackedSpectrum
 *      members are therefore plain float3* arrays; no float*+3i workaround.
 *
 *  Sub-word members (uint16_t/uint8_t/bool elements, sizes from the S1b-3
 *  stride table, kernels.cpp:2105-2508) go through the accessor layer below:
 *  atomic AND/OR-on-containing-word by DEFAULT (portable: no Int16/Int8
 *  capability, race-free for adjacent states sharing a word — proven by the
 *  Stage A probe subword_word.slang, modes 0/1 PASS on RX 570 AND SwiftShader;
 *  the non-atomic RMW negative, modes 2/3, was DETECTED). The compile-time
 *  macro VULKAN_SUBWORD_NARROW (defined in vulkan_data.slang, default 0)
 *  switches the mirror member type AND these helpers in lockstep to true
 *  narrow stores (Int16/Int8, RX 570-only) — the divergence entry in
 *  DIVERGENCES.md documents that the narrow variant is to be measured on the
 *  RX 570 after M5.
 *
 * Every `// source:` below cites the Cycles header that defines the member's
 * type and name. Members NOT covered here (and why) are listed at the bottom.
 *
 * The SoA DATA pointers are always LOADED from the IntegratorStateGPU struct
 * (never pushed as raw addresses), so the gate can confirm packed3 stride
 * stays 12 when the pointer comes from the struct. The struct's own base
 * address arrives as the IntegratorStateGPU* parameter, exactly the way
 * Cycles kernels receive `kernel_integrator_state`. */

#include "vulkan_data.slang"

/* ==================================================================== *
 * Sub-word accessor layer (M4e S1b-4 item 4 decision, user):           *
 *   - writes: atomic AND/OR-on-containing-word by DEFAULT, portable    *
 *     (feature-independent, runs on RX 570 and SwiftShader),           *
 *     race-free: disjoint-bit atomics commute with a neighbour's.      *
 *   - reads: plain 32-bit word load + shift/mask (no Int16/Int8).      *
 *     Aligned 32-bit word loads do not tear; Stage A verified reads.   *
 *   - VULKAN_SUBWORD_NARROW: true narrow element store/load            *
 *     (RX 570-only; measurement of that variant deferred to post-M5    *
 *     per the task; see DIVERGENCES.md).                               *
 *                                                                     *
 * Element t of a uint16 array lives in word k = t>>1 at shift          *
 * (t&1)*16; a uint8 element lives in word k = t>>2 at shift (t&3)*8.   *
 * Two adjacent states share one 32-bit word, so a store must not       *
 * clobber its neighbour. Atomic form copied verbatim from the proven   *
 * Stage A probe (subword_word.slang modes 0/1, PASS both devices):     *
 *   InterlockedAnd(word, int(mask)); InterlockedOr(word, int(slot));   *
 * The helpers take Subword16T and Subword8T POINTERS (vulkan_data.slang): *
 * default = uint word view, narrow = uint16_t or uint8_t element view. *
 * ==================================================================== */

/* Read a uint16 element t from the sub-word SoA array. */
uint load_u16(Subword16T* base, uint t)
{
#if VULKAN_SUBWORD_NARROW
    return uint(base[t]);                    // direct 16-bit element load
#else
    uint k = t >> 1;                         // containing 32-bit word
    uint shift = (t & 1u) * 16u;             // low or high half
    return (base[k] >> shift) & 0xFFFFu;
#endif
}

/* Write a uint16 element t. Atomic word fallback by default. */
void store_u16(Subword16T* base, uint t, uint value)
{
    value &= 0xFFFFu;
#if VULKAN_SUBWORD_NARROW
    base[t] = Subword16T(value);             // true 16-bit store (Int16)
#else
    uint k = t >> 1;
    uint shift = (t & 1u) * 16u;
    uint mask = ~(0xFFFFu << shift);         // all bits EXCEPT ours
    uint slot = value << shift;              // our value in place
    InterlockedAnd(base[k], int(mask));      // clear our 16 bits
    InterlockedOr(base[k], int(slot));       // set them
#endif
}

/* Read a uint8 element t (4 per word). */
uint load_u8(Subword8T* base, uint t)
{
#if VULKAN_SUBWORD_NARROW
    return uint(base[t]);                    // direct 8-bit element load
#else
    uint k = t >> 2;
    uint shift = (t & 3u) * 8u;
    return (base[k] >> shift) & 0xFFu;
#endif
}

/* Write a uint8 element t. Atomic word fallback by default. */
void store_u8(Subword8T* base, uint t, uint value)
{
    value &= 0xFFu;
#if VULKAN_SUBWORD_NARROW
    base[t] = Subword8T(value);              // true 8-bit store (Int8)
#else
    uint k = t >> 2;
    uint shift = (t & 3u) * 8u;
    uint mask = ~(0xFFu << shift);
    uint slot = value << shift;
    InterlockedAnd(base[k], int(mask));
    InterlockedOr(base[k], int(slot));
#endif
}

/* `bool` members are 1 byte on the host (state_template.h:132 uses `bool`
 * for the guiding flags; same 1-byte sub-word layout as uint8_t). Stored
 * as a uint8 element, normalized to 0/1. */
uint load_bool(Subword8T* base, uint t) { return load_u8(base, t) & 1u; }
void store_bool(Subword8T* base, uint t, bool value)
{
    store_u8(base, t, value ? 1u : 0u);
}

/* ==================================================================== *
 * Sub-word SLICE-POINTER loader - narrow-mode driver-bug workaround   *
 * (M4e narrow pre-flight, 2026-10-27; probe evidence v1-v7 in         *
 * lastsession.txt).                                                   *
 *                                                                     *
 * A sub-word mirror member such as path.bounce is a uint16_t* SLOT     *
 * inside the IntegratorStateGPU struct. Reading it as written          *
 * (`ist->path.bounce`) lowers to OpAccessChain + OpLoad with a         *
 * POINTER result type, and the amdvlk driver (31.0.21925.1001) sizes   *
 * that load by the INNERMOST pointee element instead of 8 bytes:       *
 *   uint32_t* member -> 4 B (full only because addresses < 4 GB here)  *
 *   uint16_t* member -> 2 B  (v3: slice ptr loads as low 2 bytes)      *
 *   uint8_t*  member -> 1 B  (v4: 0x...0128 loads as 0x28)             *
 * The truncated pointer then addresses near physical page 0: reads     *
 * return 0, writes vanish (v2/v3), and narrow gate mode 0 fails with   *
 * 64 payload mismatches. The SPIR-V is spec-valid (spirv-val passes);  *
 * it is purely an amdvlk miscompile. NOTE the double-pointer form      *
 * `*(&member)` does NOT help: the direct load's operand already has an *
 * 8-byte immediate element (ptr<ptr<ushort>>) and is still truncated to *
 * 2 B - the driver recurses to the base element type, so the slot must *
 * be read as a SCALAR of the full width.                               *
 *                                                                     *
 * SECOND DRIVER BUG (probe v7): taking the slot address as             *
 * `&(m)` and converting it with OpConvertPtrToU DROPS the accesschain  *
 * element offset - the address echo came back as the OUTER base        *
 * (0x18600000 instead of 0x18600010), so every member read mirror+0    *
 * and all 8 pointers resolved to the render slice. Therefore the slot  *
 * address is built as plain INTEGER ARITHMETIC from the struct base:   *
 *   slot = (uint64_t)ist + MEMBER_OFFSET   (no accesschain involved)   *
 * v7 verified this form for offsets 16 / 104 / 1064 (bounce,           *
 * visibility, shadow_path.queued_kernel) - each returned the exact    *
 * slice address. Param->uint64_t, OpConvertUToPtr and the 8-byte       *
 * SCALAR load were all proven correct in v6/v7 (0 VUID, canaries       *
 * intact). Neither conversion is an OpBitcast: pointer OpBitcast with  *
 * a 16/8-bit pointee crashes amdvlk at vkCreateComputePipelines        *
 * (v1/noAB/noF bisect).                                               *
 *                                                                     *
 * Slot offsets (vulkan_data.slang member comments; byte positions of  *
 * the pointer slots inside IntegratorStateGPU):                       *
 *   path @ 0        bounce 16 (:109)  transparent_bounce 24 (:110)     *
 *                   queued_kernel 80 (:117)  rng_offset 96 (:119)      *
 *                   visibility 104 (:120)  mnee 120 (:122)             *
 *   shadow_path @976 (:327): queued_kernel +88 (:235) = 1064,          *
 *                   lightgroup +152 (:243) = 1128                      *
 * The gate re-verifies every offset byte-for-byte after any change.    *
 *                                                                     *
 * Result-preserving: the helpers yield the exact 8-byte address the    *
 * member holds - same value the untruncated load would produce, no     *
 * logic change, CPU-comparison tolerances untouched. Default mode      *
 * keeps the direct load (the macro's `member` argument expands         *
 * verbatim there; the narrow expansion discards it - a macro argument  *
 * that appears in neither replacement list is never evaluated): its    *
 * 4-byte-pointee load is full for this card and is the configuration   *
 * proven on RX 570 and SwiftShader. NEW sub-word accessors must go     *
 * through SUBWORD16_PTR / SUBWORD8_PTR, never a raw member load.       *
 * ==================================================================== */
#if VULKAN_SUBWORD_NARROW
static const uint BOUNCE_SLOT_OFF = 16u;                  /* path.bounce */
static const uint TRANSPARENT_BOUNCE_SLOT_OFF = 24u;      /* path.transparent_bounce */
static const uint QUEUED_KERNEL_SLOT_OFF = 80u;          /* path.queued_kernel */
static const uint RNG_OFFSET_SLOT_OFF = 96u;             /* path.rng_offset */
static const uint VISIBILITY_SLOT_OFF = 104u;            /* path.visibility */
static const uint MNEE_SLOT_OFF = 120u;                  /* path.mnee */
/* M4f slice 2 (path_state_init_integrator's u16 bounce counters; slot bytes
 * from vulkan_data.slang IntegratorStatePath comments, path @0): */
static const uint DIFFUSE_BOUNCE_SLOT_OFF = 32u;
static const uint GLOSSY_BOUNCE_SLOT_OFF = 40u;
static const uint TRANSMISSION_BOUNCE_SLOT_OFF = 48u;
static const uint VOLUME_BOUNCE_SLOT_OFF = 56u;
static const uint VOLUME_BOUNDS_BOUNCE_SLOT_OFF = 64u;
static const uint PORTAL_BOUNCE_SLOT_OFF = 72u;
static const uint SHADOW_PATH_BASE_OFF = 976u;           /* shadow_path struct */
static const uint SHADOW_QUEUED_KERNEL_SLOT_OFF = SHADOW_PATH_BASE_OFF + 88u;
static const uint SHADOW_LIGHTGROUP_SLOT_OFF = SHADOW_PATH_BASE_OFF + 152u;

Subword16T* subword16_ptr(uint64_t slot_addr)
{
    /* integer -> uint64_t* (spec opcode, NOT OpBitcast) */
    uint64_t* slot = spirv_asm { result:$$uint64_t* = OpConvertUToPtr $slot_addr; };
    uint64_t slice_addr = *slot; /* 8-byte SCALAR load of the stored slice address */
    return spirv_asm { result:$$Subword16T* = OpConvertUToPtr $slice_addr; };
}
Subword8T* subword8_ptr(uint64_t slot_addr)
{
    uint64_t* slot = spirv_asm { result:$$uint64_t* = OpConvertUToPtr $slot_addr; };
    uint64_t slice_addr = *slot;
    return spirv_asm { result:$$Subword8T* = OpConvertUToPtr $slice_addr; };
}
/* (ist, member, off): narrow uses ist+off (integer add - see v7 note);
 * `member` is discarded here and only supplies the default-mode load. */
#define SUBWORD16_PTR(ist, member, off) subword16_ptr((uint64_t)(ist) + (off))
#define SUBWORD8_PTR(ist, member, off) subword8_ptr((uint64_t)(ist) + (off))
#else
/* Default word view: expand to the original expression verbatim, so the
 * default-mode SPIR-V is unchanged by this workaround (`off` discarded). */
#define SUBWORD16_PTR(ist, member, off) (member)
#define SUBWORD8_PTR(ist, member, off) (member)
#endif

/* ===================== path struct accessors ===================== */
/* state_template.h:7-74. */

uint read_path_render_pixel_index(IntegratorStateGPU* ist, uint state)
{
    // source: state_template.h:13 (uint32_t)
    uint* data_ptr = ist->path.render_pixel_index;
    return data_ptr[state];
}
void write_path_render_pixel_index(IntegratorStateGPU* ist, uint state, uint v)
{
    uint* data_ptr = ist->path.render_pixel_index;
    data_ptr[state] = v;
}

uint read_path_sample(IntegratorStateGPU* ist, uint state)
{
    // source: state_template.h:15 (uint32_t)
    uint* data_ptr = ist->path.sample;
    return data_ptr[state];
}
void write_path_sample(IntegratorStateGPU* ist, uint state, uint v)
{
    uint* data_ptr = ist->path.sample;
    data_ptr[state] = v;
}

uint read_path_bounce(IntegratorStateGPU* ist, uint state)
{
    // source: state_template.h:17 (uint16_t) -> sub-word
    Subword16T* data_ptr = SUBWORD16_PTR(ist, ist->path.bounce, BOUNCE_SLOT_OFF);
    return load_u16(data_ptr, state);
}
void write_path_bounce(IntegratorStateGPU* ist, uint state, uint v)
{
    Subword16T* data_ptr = SUBWORD16_PTR(ist, ist->path.bounce, BOUNCE_SLOT_OFF);
    store_u16(data_ptr, state, v);
}

uint read_path_transparent_bounce(IntegratorStateGPU* ist, uint state)
{
    // source: state_template.h:19 (uint16_t) -> sub-word
    Subword16T* data_ptr = SUBWORD16_PTR(ist, ist->path.transparent_bounce, TRANSPARENT_BOUNCE_SLOT_OFF);
    return load_u16(data_ptr, state);
}
void write_path_transparent_bounce(IntegratorStateGPU* ist, uint state, uint v)
{
    Subword16T* data_ptr = SUBWORD16_PTR(ist, ist->path.transparent_bounce, TRANSPARENT_BOUNCE_SLOT_OFF);
    store_u16(data_ptr, state, v);
}

uint read_path_queued_kernel(IntegratorStateGPU* ist, uint state)
{
    // source: state_template.h:33 (uint16_t) -> queue path (kernel.h:66,422)
    Subword16T* data_ptr = SUBWORD16_PTR(ist, ist->path.queued_kernel, QUEUED_KERNEL_SLOT_OFF);
    return load_u16(data_ptr, state);
}
void write_path_queued_kernel(IntegratorStateGPU* ist, uint state, uint v)
{
    Subword16T* data_ptr = SUBWORD16_PTR(ist, ist->path.queued_kernel, QUEUED_KERNEL_SLOT_OFF);
    store_u16(data_ptr, state, v);
}

uint read_path_rng_offset(IntegratorStateGPU* ist, uint state)
{
    // source: state_template.h:37 (uint16_t) -> sub-word
    Subword16T* data_ptr = SUBWORD16_PTR(ist, ist->path.rng_offset, RNG_OFFSET_SLOT_OFF);
    return load_u16(data_ptr, state);
}
void write_path_rng_offset(IntegratorStateGPU* ist, uint state, uint v)
{
    Subword16T* data_ptr = SUBWORD16_PTR(ist, ist->path.rng_offset, RNG_OFFSET_SLOT_OFF);
    store_u16(data_ptr, state, v);
}

uint read_path_visibility(IntegratorStateGPU* ist, uint state)
{
    // source: state_template.h:39 (uint8_t) -> sub-word
    Subword8T* data_ptr = SUBWORD8_PTR(ist, ist->path.visibility, VISIBILITY_SLOT_OFF);
    return load_u8(data_ptr, state);
}
void write_path_visibility(IntegratorStateGPU* ist, uint state, uint v)
{
    Subword8T* data_ptr = SUBWORD8_PTR(ist, ist->path.visibility, VISIBILITY_SLOT_OFF);
    store_u8(data_ptr, state, v);
}

uint read_path_flag(IntegratorStateGPU* ist, uint state)
{
    // source: state_template.h:41 (uint32_t); OR-updates: intersect_closest.h:40,44,72,77
    uint* data_ptr = ist->path.flag;
    return data_ptr[state];
}
void write_path_flag(IntegratorStateGPU* ist, uint state, uint v)
{
    uint* data_ptr = ist->path.flag;
    data_ptr[state] = v;
}

uint read_path_mnee(IntegratorStateGPU* ist, uint state)
{
    // source: state_template.h:43 (uint8_t) -> sub-word
    Subword8T* data_ptr = SUBWORD8_PTR(ist, ist->path.mnee, MNEE_SLOT_OFF);
    return load_u8(data_ptr, state);
}
void write_path_mnee(IntegratorStateGPU* ist, uint state, uint v)
{
    Subword8T* data_ptr = SUBWORD8_PTR(ist, ist->path.mnee, MNEE_SLOT_OFF);
    store_u8(data_ptr, state, v);
}

float read_path_mis_ray_pdf(IntegratorStateGPU* ist, uint state)
{
    // source: state_template.h:52 (float)
    float* data_ptr = ist->path.mis_ray_pdf;
    return data_ptr[state];
}
void write_path_mis_ray_pdf(IntegratorStateGPU* ist, uint state, float v)
{
    float* data_ptr = ist->path.mis_ray_pdf;
    data_ptr[state] = v;
}

float read_path_min_ray_pdf(IntegratorStateGPU* ist, uint state)
{
    // source: state_template.h:58 (float)
    float* data_ptr = ist->path.min_ray_pdf;
    return data_ptr[state];
}
void write_path_min_ray_pdf(IntegratorStateGPU* ist, uint state, float v)
{
    float* data_ptr = ist->path.min_ray_pdf;
    data_ptr[state] = v;
}

float read_path_continuation_probability(IntegratorStateGPU* ist, uint state)
{
    // source: state_template.h:60 (float); written by intersect_closest.h:62
    float* data_ptr = ist->path.continuation_probability;
    return data_ptr[state];
}
void write_path_continuation_probability(IntegratorStateGPU* ist, uint state, float v)
{
    float* data_ptr = ist->path.continuation_probability;
    data_ptr[state] = v;
}

float read_path_optical_depth(IntegratorStateGPU* ist, uint state)
{
    // source: state_template.h:47 (float); read by state_flow.h:62
    float* data_ptr = ist->path.optical_depth;
    return data_ptr[state];
}
void write_path_optical_depth(IntegratorStateGPU* ist, uint state, float v)
{
    float* data_ptr = ist->path.optical_depth;
    data_ptr[state] = v;
}

uint read_path_shader_sort_key(IntegratorStateGPU* ist, uint state)
{
    // source: state_template.h:73 (uint32_t) -> queue path (state_flow.h:114,171)
    uint* data_ptr = ist->path.shader_sort_key;
    return data_ptr[state];
}
void write_path_shader_sort_key(IntegratorStateGPU* ist, uint state, uint v)
{
    uint* data_ptr = ist->path.shader_sort_key;
    data_ptr[state] = v;
}

/* PackedSpectrum == packed_float3, 12 bytes (types_spectrum.h:14,
 * types_float3.h:198 static_assert sizeof==12). float3* indexes at stride 12
 * (disassembly: ArrayStride 12) — the gate re-confirms it when the pointer is
 * LOADED from IntegratorStateGPU. */
float3 read_path_throughput(IntegratorStateGPU* ist, uint state)
{
    // source: state_template.h:62 (PackedSpectrum); read: shade_surface.h:107,243
    float3* data_ptr = ist->path.throughput;
    return data_ptr[state];
}
void write_path_throughput(IntegratorStateGPU* ist, uint state, float3 v)
{
    float3* data_ptr = ist->path.throughput;
    data_ptr[state] = v;
}

float3 read_path_mis_origin_n(IntegratorStateGPU* ist, uint state)
{
    // source: state_template.h:56 (packed_float3)
    float3* data_ptr = ist->path.mis_origin_n;
    return data_ptr[state];
}
void write_path_mis_origin_n(IntegratorStateGPU* ist, uint state, float3 v)
{
    float3* data_ptr = ist->path.mis_origin_n;
    data_ptr[state] = v;
}

float3 read_path_pass_diffuse_weight(IntegratorStateGPU* ist, uint state)
{
    // source: state_template.h:67 (PackedSpectrum)
    float3* data_ptr = ist->path.pass_diffuse_weight;
    return data_ptr[state];
}
void write_path_pass_diffuse_weight(IntegratorStateGPU* ist, uint state, float3 v)
{
    float3* data_ptr = ist->path.pass_diffuse_weight;
    data_ptr[state] = v;
}

/* ===================== M4f slice-2 accessors ===========================
 * Members written by path_state_init_integrator (path_state.h:40-108) that
 * had no accessor when slice 1 landed. Every source cite is the
 * state_template.h line defining the member type. The u16 members MUST go
 * through SUBWORD16_PTR (Form-A loader, see the driver-bug note above) -
 * never a raw pointer-member load. */
uint read_path_rng_pixel(IntegratorStateGPU* ist, uint state)
{
    // source: state_template.h:35 (uint32_t)
    uint* data_ptr = ist->path.rng_pixel;
    return data_ptr[state];
}
void write_path_rng_pixel(IntegratorStateGPU* ist, uint state, uint v)
{
    uint* data_ptr = ist->path.rng_pixel;
    data_ptr[state] = v;
}

uint read_path_diffuse_bounce(IntegratorStateGPU* ist, uint state)
{
    // source: state_template.h:21 (uint16_t) -> sub-word
    Subword16T* data_ptr = SUBWORD16_PTR(ist, ist->path.diffuse_bounce, DIFFUSE_BOUNCE_SLOT_OFF);
    return load_u16(data_ptr, state);
}
void write_path_diffuse_bounce(IntegratorStateGPU* ist, uint state, uint v)
{
    Subword16T* data_ptr = SUBWORD16_PTR(ist, ist->path.diffuse_bounce, DIFFUSE_BOUNCE_SLOT_OFF);
    store_u16(data_ptr, state, v);
}

uint read_path_glossy_bounce(IntegratorStateGPU* ist, uint state)
{
    // source: state_template.h:23 (uint16_t) -> sub-word
    Subword16T* data_ptr = SUBWORD16_PTR(ist, ist->path.glossy_bounce, GLOSSY_BOUNCE_SLOT_OFF);
    return load_u16(data_ptr, state);
}
void write_path_glossy_bounce(IntegratorStateGPU* ist, uint state, uint v)
{
    Subword16T* data_ptr = SUBWORD16_PTR(ist, ist->path.glossy_bounce, GLOSSY_BOUNCE_SLOT_OFF);
    store_u16(data_ptr, state, v);
}

uint read_path_transmission_bounce(IntegratorStateGPU* ist, uint state)
{
    // source: state_template.h:25 (uint16_t) -> sub-word
    Subword16T* data_ptr = SUBWORD16_PTR(ist, ist->path.transmission_bounce, TRANSMISSION_BOUNCE_SLOT_OFF);
    return load_u16(data_ptr, state);
}
void write_path_transmission_bounce(IntegratorStateGPU* ist, uint state, uint v)
{
    Subword16T* data_ptr = SUBWORD16_PTR(ist, ist->path.transmission_bounce, TRANSMISSION_BOUNCE_SLOT_OFF);
    store_u16(data_ptr, state, v);
}

uint read_path_volume_bounce(IntegratorStateGPU* ist, uint state)
{
    // source: state_template.h:27 (uint16_t) -> sub-word
    Subword16T* data_ptr = SUBWORD16_PTR(ist, ist->path.volume_bounce, VOLUME_BOUNCE_SLOT_OFF);
    return load_u16(data_ptr, state);
}
void write_path_volume_bounce(IntegratorStateGPU* ist, uint state, uint v)
{
    Subword16T* data_ptr = SUBWORD16_PTR(ist, ist->path.volume_bounce, VOLUME_BOUNCE_SLOT_OFF);
    store_u16(data_ptr, state, v);
}

uint read_path_volume_bounds_bounce(IntegratorStateGPU* ist, uint state)
{
    // source: state_template.h:29 (uint16_t) -> sub-word
    Subword16T* data_ptr = SUBWORD16_PTR(ist, ist->path.volume_bounds_bounce, VOLUME_BOUNDS_BOUNCE_SLOT_OFF);
    return load_u16(data_ptr, state);
}
void write_path_volume_bounds_bounce(IntegratorStateGPU* ist, uint state, uint v)
{
    Subword16T* data_ptr = SUBWORD16_PTR(ist, ist->path.volume_bounds_bounce, VOLUME_BOUNDS_BOUNCE_SLOT_OFF);
    store_u16(data_ptr, state, v);
}

uint read_path_portal_bounce(IntegratorStateGPU* ist, uint state)
{
    // source: state_template.h:31 (uint16_t) -> sub-word
    Subword16T* data_ptr = SUBWORD16_PTR(ist, ist->path.portal_bounce, PORTAL_BOUNCE_SLOT_OFF);
    return load_u16(data_ptr, state);
}
void write_path_portal_bounce(IntegratorStateGPU* ist, uint state, uint v)
{
    Subword16T* data_ptr = SUBWORD16_PTR(ist, ist->path.portal_bounce, PORTAL_BOUNCE_SLOT_OFF);
    store_u16(data_ptr, state, v);
}

int read_path_mis_ray_object(IntegratorStateGPU* ist, uint state)
{
    // source: state_template.h:54 (int, KERNEL_FEATURE_LIGHT_LINKING)
    int* data_ptr = ist->path.mis_ray_object;
    return data_ptr[state];
}
void write_path_mis_ray_object(IntegratorStateGPU* ist, uint state, int v)
{
    int* data_ptr = ist->path.mis_ray_object;
    data_ptr[state] = v;
}

float3 read_path_denoising_feature_throughput(IntegratorStateGPU* ist, uint state)
{
    // source: state_template.h:70 (PackedSpectrum, KERNEL_FEATURE_DENOISING)
    float3* data_ptr = ist->path.denoising_feature_throughput;
    return data_ptr[state];
}
void write_path_denoising_feature_throughput(IntegratorStateGPU* ist, uint state, float3 v)
{
    float3* data_ptr = ist->path.denoising_feature_throughput;
    data_ptr[state] = v;
}

/* ===================== ray struct accessors ===================== */
/* state_template.h:78-87. init_from_camera writes via
 * integrator_state_write_ray (state_util.h:53-59); shade_surface rewrites
 * the next bounce ray (shade_surface.h:181-190). */

float3 read_ray_P(IntegratorStateGPU* ist, uint state)
{
    // source: state_template.h:79 (packed_float3)
    float3* data_ptr = ist->ray.P;
    return data_ptr[state];
}
void write_ray_P(IntegratorStateGPU* ist, uint state, float3 v)
{
    float3* data_ptr = ist->ray.P;
    data_ptr[state] = v;
}

float3 read_ray_D(IntegratorStateGPU* ist, uint state)
{
    // source: state_template.h:81 (packed_float3)
    float3* data_ptr = ist->ray.D;
    return data_ptr[state];
}
void write_ray_D(IntegratorStateGPU* ist, uint state, float3 v)
{
    float3* data_ptr = ist->ray.D;
    data_ptr[state] = v;
}

float read_ray_dP(IntegratorStateGPU* ist, uint state)
{
    // source: state_template.h:80 (float; differential compact scale)
    float* data_ptr = ist->ray.dP;
    return data_ptr[state];
}
void write_ray_dP(IntegratorStateGPU* ist, uint state, float v)
{
    float* data_ptr = ist->ray.dP;
    data_ptr[state] = v;
}

float read_ray_dD(IntegratorStateGPU* ist, uint state)
{
    // source: state_template.h:82 (float)
    float* data_ptr = ist->ray.dD;
    return data_ptr[state];
}
void write_ray_dD(IntegratorStateGPU* ist, uint state, float v)
{
    float* data_ptr = ist->ray.dD;
    data_ptr[state] = v;
}

float read_ray_tmin(IntegratorStateGPU* ist, uint state)
{
    // source: state_template.h:83 (float)
    float* data_ptr = ist->ray.tmin;
    return data_ptr[state];
}
void write_ray_tmin(IntegratorStateGPU* ist, uint state, float v)
{
    float* data_ptr = ist->ray.tmin;
    data_ptr[state] = v;
}

float read_ray_tmax(IntegratorStateGPU* ist, uint state)
{
    // source: state_template.h:84 (float)
    float* data_ptr = ist->ray.tmax;
    return data_ptr[state];
}
void write_ray_tmax(IntegratorStateGPU* ist, uint state, float v)
{
    float* data_ptr = ist->ray.tmax;
    data_ptr[state] = v;
}

float read_ray_time(IntegratorStateGPU* ist, uint state)
{
    // source: state_template.h:85 (float)
    float* data_ptr = ist->ray.time;
    return data_ptr[state];
}
void write_ray_time(IntegratorStateGPU* ist, uint state, float v)
{
    float* data_ptr = ist->ray.time;
    data_ptr[state] = v;
}

/* ===================== isect struct accessors ===================== */
/* state_template.h:92-99. INTERSECT_CLOSEST fills all six through
 * bvh_intersection / triangle_intersect writes. */

float read_isect_t(IntegratorStateGPU* ist, uint state)
{
    // source: state_template.h:93 (float)
    float* data_ptr = ist->isect.t;
    return data_ptr[state];
}
void write_isect_t(IntegratorStateGPU* ist, uint state, float v)
{
    float* data_ptr = ist->isect.t;
    data_ptr[state] = v;
}

int read_isect_prim(IntegratorStateGPU* ist, uint state)
{
    // source: state_template.h:96 (int); read: intersect_closest.h:363
    int* data_ptr = ist->isect.prim;
    return data_ptr[state];
}
void write_isect_prim(IntegratorStateGPU* ist, uint state, int v)
{
    int* data_ptr = ist->isect.prim;
    data_ptr[state] = v;
}

int read_isect_object(IntegratorStateGPU* ist, uint state)
{
    // source: state_template.h:97 (int); read: intersect_closest.h:364;
    // written OBJECT_NONE by shade_surface.h:180
    int* data_ptr = ist->isect.object;
    return data_ptr[state];
}
void write_isect_object(IntegratorStateGPU* ist, uint state, int v)
{
    int* data_ptr = ist->isect.object;
    data_ptr[state] = v;
}

int read_isect_type(IntegratorStateGPU* ist, uint state)
{
    // source: state_template.h:98 (int)
    int* data_ptr = ist->isect.type;
    return data_ptr[state];
}
void write_isect_type(IntegratorStateGPU* ist, uint state, int v)
{
    int* data_ptr = ist->isect.type;
    data_ptr[state] = v;
}

/* ===================== volume_stack (AoS) accessors ===================== */
/* state_template.h:112-117. Layout: struct { int *object; int *shader; }[32],
 * base offset 376, element stride 16, object@0 / shader@8 inside the slot
 * (M4e S1b-5 host static_asserts in kernels.cpp). Pattern (B) from the header:
 * constant-index address-of, then dynamic index on the LOCAL base pointer. */

int read_volume_stack_object(IntegratorStateGPU* ist, uint i, uint state)
{
    // source: state_template.h:113 (int, AoS array member)
    IntegratorStateVolumeStackElement* base = &ist->volume_stack.slot[0];
    int* soa = base[i].object;      // i NON-constant -> OpPtrAccessChain
    return soa[state];
}
void write_volume_stack_object(IntegratorStateGPU* ist, uint i, uint state, int v)
{
    IntegratorStateVolumeStackElement* base = &ist->volume_stack.slot[0];
    int* soa = base[i].object;
    soa[state] = v;
}

int read_volume_stack_shader(IntegratorStateGPU* ist, uint i, uint state)
{
    // source: state_template.h:114 (int, AoS array member)
    IntegratorStateVolumeStackElement* base = &ist->volume_stack.slot[0];
    int* soa = base[i].shader;
    return soa[state];
}
void write_volume_stack_shader(IntegratorStateGPU* ist, uint i, uint state, int v)
{
    IntegratorStateVolumeStackElement* base = &ist->volume_stack.slot[0];
    int* soa = base[i].shader;
    soa[state] = v;
}

/* ===================== shadow_path accessors ===================== */
/* shadow_state_template.h:7-74. SHADE_SURFACE (diffuse) writes these when it
 * spawns the shadow ray (shade_surface.h:247-261 and the bounce copies). */

uint read_shadow_path_render_pixel_index(IntegratorStateGPU* ist, uint state)
{
    // source: shadow_state_template.h:9 (uint32_t); copy: shade_surface.h:250
    uint* data_ptr = ist->shadow_path.render_pixel_index;
    return data_ptr[state];
}
void write_shadow_path_render_pixel_index(IntegratorStateGPU* ist, uint state, uint v)
{
    uint* data_ptr = ist->shadow_path.render_pixel_index;
    data_ptr[state] = v;
}

uint read_shadow_path_queued_kernel(IntegratorStateGPU* ist, uint state)
{
    // source: shadow_state_template.h:31 (uint16_t) -> queue path (state_flow.h:125)
    Subword16T* data_ptr = SUBWORD16_PTR(ist, ist->shadow_path.queued_kernel, SHADOW_QUEUED_KERNEL_SLOT_OFF);
    return load_u16(data_ptr, state);
}
void write_shadow_path_queued_kernel(IntegratorStateGPU* ist, uint state, uint v)
{
    Subword16T* data_ptr = SUBWORD16_PTR(ist, ist->shadow_path.queued_kernel, SHADOW_QUEUED_KERNEL_SLOT_OFF);
    store_u16(data_ptr, state, v);
}

uint read_shadow_path_flag(IntegratorStateGPU* ist, uint state)
{
    // source: shadow_state_template.h:35 (uint32_t)
    uint* data_ptr = ist->shadow_path.flag;
    return data_ptr[state];
}
void write_shadow_path_flag(IntegratorStateGPU* ist, uint state, uint v)
{
    uint* data_ptr = ist->shadow_path.flag;
    data_ptr[state] = v;
}

uint read_shadow_path_lightgroup(IntegratorStateGPU* ist, uint state)
{
    // source: shadow_state_template.h:53 (uint8_t) -> sub-word
    Subword8T* data_ptr = SUBWORD8_PTR(ist, ist->shadow_path.lightgroup, SHADOW_LIGHTGROUP_SLOT_OFF);
    return load_u8(data_ptr, state);
}
void write_shadow_path_lightgroup(IntegratorStateGPU* ist, uint state, uint v)
{
    Subword8T* data_ptr = SUBWORD8_PTR(ist, ist->shadow_path.lightgroup, SHADOW_LIGHTGROUP_SLOT_OFF);
    store_u8(data_ptr, state, v);
}

float3 read_shadow_path_throughput(IntegratorStateGPU* ist, uint state)
{
    // source: shadow_state_template.h:37 (PackedSpectrum)
    float3* data_ptr = ist->shadow_path.throughput;
    return data_ptr[state];
}
void write_shadow_path_throughput(IntegratorStateGPU* ist, uint state, float3 v)
{
    float3* data_ptr = ist->shadow_path.throughput;
    data_ptr[state] = v;
}

/* ===================== shadow_ray accessors ===================== */
/* shadow_state_template.h:78-88; filled by integrator_state_write_shadow_ray
 * (state_util.h:83-87). */

float3 read_shadow_ray_P(IntegratorStateGPU* ist, uint state)
{
    // source: shadow_state_template.h:79 (packed_float3)
    float3* data_ptr = ist->shadow_ray.P;
    return data_ptr[state];
}
void write_shadow_ray_P(IntegratorStateGPU* ist, uint state, float3 v)
{
    float3* data_ptr = ist->shadow_ray.P;
    data_ptr[state] = v;
}

float3 read_shadow_ray_D(IntegratorStateGPU* ist, uint state)
{
    // source: shadow_state_template.h:80 (packed_float3)
    float3* data_ptr = ist->shadow_ray.D;
    return data_ptr[state];
}
void write_shadow_ray_D(IntegratorStateGPU* ist, uint state, float3 v)
{
    float3* data_ptr = ist->shadow_ray.D;
    data_ptr[state] = v;
}

float read_shadow_ray_tmin(IntegratorStateGPU* ist, uint state)
{
    // source: shadow_state_template.h:81 (float)
    float* data_ptr = ist->shadow_ray.tmin;
    return data_ptr[state];
}
void write_shadow_ray_tmin(IntegratorStateGPU* ist, uint state, float v)
{
    float* data_ptr = ist->shadow_ray.tmin;
    data_ptr[state] = v;
}

/* ===================== shadow_isect (AoS) accessors ===================== */
/* shadow_state_template.h:94-103. Layout: 6-pointer tuple [4], base 1240,
 * element stride 48, t@0/u@8/v@16/prim@24/object@32/type@40 (M4e S1b-5). */

float read_shadow_isect_t(IntegratorStateGPU* ist, uint i, uint state)
{
    // source: shadow_state_template.h:95 (float, AoS array member)
    IntegratorStateShadowIsectElement* base = &ist->shadow_isect.slot[0];
    float* soa = base[i].t;
    return soa[state];
}
void write_shadow_isect_t(IntegratorStateGPU* ist, uint i, uint state, float v)
{
    IntegratorStateShadowIsectElement* base = &ist->shadow_isect.slot[0];
    float* soa = base[i].t;
    soa[state] = v;
}

int read_shadow_isect_object(IntegratorStateGPU* ist, uint i, uint state)
{
    // source: shadow_state_template.h:99 (int, AoS array member)
    IntegratorStateShadowIsectElement* base = &ist->shadow_isect.slot[0];
    int* soa = base[i].object;
    return soa[state];
}
void write_shadow_isect_object(IntegratorStateGPU* ist, uint i, uint state, int v)
{
    IntegratorStateShadowIsectElement* base = &ist->shadow_isect.slot[0];
    int* soa = base[i].object;
    soa[state] = v;
}

/* ===================== shadow_volume_stack (AoS) accessors ===================== */
/* shadow_state_template.h:107-112. Same shape as volume_stack: base 1432,
 * stride 16, object@0 / shader@8 (M4e S1b-5). */

int read_shadow_volume_stack_object(IntegratorStateGPU* ist, uint i, uint state)
{
    // source: shadow_state_template.h:108 (int, AoS array member)
    IntegratorStateShadowVolumeStackElement* base = &ist->shadow_volume_stack.slot[0];
    int* soa = base[i].object;
    return soa[state];
}
void write_shadow_volume_stack_object(IntegratorStateGPU* ist, uint i, uint state, int v)
{
    IntegratorStateShadowVolumeStackElement* base = &ist->shadow_volume_stack.slot[0];
    int* soa = base[i].object;
    soa[state] = v;
}

int read_shadow_volume_stack_shader(IntegratorStateGPU* ist, uint i, uint state)
{
    // source: shadow_state_template.h:109 (int, AoS array member)
    IntegratorStateShadowVolumeStackElement* base = &ist->shadow_volume_stack.slot[0];
    int* soa = base[i].shader;
    return soa[state];
}
void write_shadow_volume_stack_shader(IntegratorStateGPU* ist, uint i, uint state, int v)
{
    IntegratorStateShadowVolumeStackElement* base = &ist->shadow_volume_stack.slot[0];
    int* soa = base[i].shader;
    soa[state] = v;
}

/* ====================================================================== *
 * MEMBERS NOT DONE HERE (and why). These follow the identical patterns   *
 * above (scalar / sub-word / packed3 / AoS) and are either not touched by *
 * INIT_FROM_CAMERA, INTERSECT_CLOSEST, SHADE_SURFACE(diffuse) or the      *
 * queue path, or are compile-time feature-gated OFF in this build:       *
 *   - subsurface.* (albedo/radius/anisotropy/N; state_template.h:103-108) *
 *     -> only SUBSURFACE kernels.                                         *
 *   - shadow_link.* (dedicated_light_weight/last_isect_*;                 *
 *     state_template.h:151-156) -> only SHADOW_LINKING kernels.           *
 *   - guiding.* (path_segment/use_*_guiding/sampling probs;               *
 *     state_template.h:120-147) -> __PATH_GUIDING__ is OFF in this build.*
 *   - path.unguided_throughput (state_template.h:65) -> PATH_GUIDING only.*
 *   - path.mnee_shadow_state (state_template.h:45) -> MNEE only.          *
 *   - path.rng_pixel (state_template.h:35) and path.denoising_feature_    *
 *     throughput (state_template.h:70) -> same scalar/packed3 pattern;    *
 *     added when the INIT_FROM_CAMERA port needs them.                    *
 *   - ray.previous_dt (state_template.h:86) -> LIGHT_TREE only.           *
 *   - isect.u / isect.v (state_template.h:94-95) -> same float pattern as *
 *     isect.t; not needed by the diffuse path until M5.                   *
 *   - shadow_path.sample/rng_pixel/rng_offset/bounce/transparent_bounce/  *
 *     volume_bounds_bounce/visibility/unshadowed_throughput/              *
 *     pass_diffuse_weight/pass_glossy_weight/packed_num_hits/             *
 *     unlit_throughput/path_segment/guiding_light_linking_mis_weight/     *
 *     bsdf_eval_average (shadow_state_template.h:11-73) -> the            *
 *     shade_surface shadow-copy set, added per-kernel during M5 shadow    *
 *     work (mirror members stay uint64_t until then — each retyped when   *
 *     its accessor lands, keeping the file's rule: retype = reached).     *
 *   - shadow_ray.time/dP/dD/tmax/self_light_object/self_light_prim        *
 *     (shadow_state_template.h:82-87) -> same float/int pattern.          *
 *   - shadow_isect.u/v/prim/type (shadow_state_template.h:96-100) -> same *
 *     float/int AoS pattern as shadow_isect.t.                            *
 *   - the IntegratorStateGPU scalars (queue_counter/sort_key_counter/     *
 *     next_*_path_index/sort_partition_*; state.h:197-212) are NOT SoA    *
 *     members and are host-managed; not needed by these kernels.          *
 * ====================================================================== */
