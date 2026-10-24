// SPDX-FileCopyrightText: 2011-2022 Blender Foundation
// SPDX-License-Identifier: Apache-2.0

/* !!! COMPILE STATUS: DOES NOT COMPILE YET (M4e S1b-6, context ran short) !!!
 *
 * Every accessor below currently uses `cast<T[]>`, which is UNAVAILABLE in
 * slangc 2026.17 (error E30015). To make this file compile, the next session
 * must, for each accessor:
 *   1. Retype the corresponding member in vulkan_data.slang from `uint64_t` to
 *      the TYPED device pointer (uint* / float3* / float* / int* / uint16_t* /
 *      uint8_t*). A typed pointer is 8 bytes = uint64_t, so byte layout and the
 *      ABI probe offsets are UNCHANGED.
 *   2. Replace `let data_ptr = params->...member; return cast<T[]>(data_ptr)[state];`
 *      with `T* data_ptr = params->...member; return data_ptr[state];`
 *      (load the typed pointer member, index it directly).
 *   3. For AoS members, use the proven base-pointer pattern:
 *        Elem* base = &params->...slot[0];  int* p = base[i].object;  p[state]=v;
 *   4. Fix the 3 ABI-probe reads in abi_probe.slang that load these members to
 *      wrap in `uint64_t(...)` (proven to compile).
 * The full proven design is documented in the header below and in
 * lastsession.txt (M4e S1b-6). float3* stride is 12 (confirmed by disassembly).
 */

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
 * addressing strategies, both chosen to satisfy the DEVICE RULE / SLANG RULE
 * (never inline-DYNAMIC-index a fixed array member reached through a device
 * pointer — that miscompiles to element 0 on the RX 570 driver):
 *
 *  *** IMPORTANT, PROVEN THIS SESSION (M4e S1b-6) ***
 *  `cast<T[]>` / `cast<T*>` are UNAVAILABLE in slangc 2026.17 (error E30015,
 *  "undefined identifier 'cast'"). There is NO uint64->typed-pointer conversion
 *  in this Slang build (functional T*(addr) parses as multiplication). Therefore
 *  the SoA members in vulkan_data.slang MUST be declared as TYPED device
 *  pointers (`uint*`, `float3*`, `float*`, `int*`, `uint16_t*`, `uint8_t*`),
 *  NOT `uint64_t`. A typed pointer is 8 bytes, same as uint64_t, so the byte
 *  layout is IDENTICAL (ABI probe offsets unchanged). The M4b ABI probe reads
 *  the raw address back with `uint64_t(typed_ptr)` (proven to compile).
 *
 *  The PROVEN indexing pattern (cast_test.slang, slangc+spirv-val exit 0):
 *    IstSub ist = pc.params->ist;   // load struct via typed device pointer
 *    ist.throughput[s] = float3(...); // dynamic index on a TYPED pointer member
 *  emits `%_ptr_PhysicalStorageBuffer_v3float` + OpPtrAccessChain. SAFE.
 *
 *  float3* stride is 12 (ArrayStride 12 confirmed by disassembly) — correct for
 *  Cycles packed_float3/PackedSpectrum (types_float3.h:198). packed3 members use
 *  `float3*` directly; NO float*+3i workaround needed.
 *
 *  (A) SCALAR / sub-word / packed3 members: the SoA pointer is a TYPED pointer
 *      field of the struct. Load it via the typed view
 *      `params->integrator_state.<group>.<member>` (a scalar pointer field, not a
 *      fixed array) and index it directly by `state` (dynamic). Proven byte-correct
 *      by the M4b ABI probe (render_pixel_index @2984 MATCH).
 *
 *  (B) AoS ARRAY members (volume_stack/shadow_isect/shadow_volume_stack): the
 *      runtime array_index is dynamic, and `slot` is a FIXED array inside a
 *      device-pointer struct. The PROVEN pattern (cast_test.slang) is:
 *        SlotElem* base = &params->integrator_state.volume_stack.slot[0];
 *        int* obj = base[i].object;   // i NON-constant -> OpPtrAccessChain
 *        obj[state] = value;
 *      `&slot[0]` (constant index) yields a typed base pointer; dynamic `base[i]`
 *      on that LOCAL base pointer is SAFE (OpPtrAccessChain), unlike the inline
 *      `params->...slot[i]` form.
 *
 * Every `// source:` below cites the Cycles header that defines the member's type
 * and name. Members NOT covered here (and why) are listed at the bottom of this
 * file and in the M4e S1b-6 report.
 *
 * The SoA DATA pointer is always LOADED from the IntegratorStateGPU struct (never
 * pushed as a raw address), so the gate can confirm packed3 stride stays 12 when
 * the pointer comes from the struct. Only the struct's own base address `ist_addr`
 * (case B) is passed in, because a struct field's address cannot be formed without
 * a base. */

#include "vulkan_data.slang"

/* ------------------------------------------------------------------ *
 * Sub-word accessor layer (M4e S1b-4 item 4 DECISION, user):          *
 *   - writes: atomic AND/OR-on-containing-word fallback by DEFAULT,   *
 *     because it is feature-independent (runs on RX 570 AND            *
 *     SwiftShader) and was proven race-free by the Stage A probe.     *
 *   - reads: plain 32-bit load + shift/mask (no Int16/Int8 needed).   *
 *   - a compile-time macro switches writes to true narrow stores      *
 *     (shaderInt16/shaderInt8, RX 570-only; the narrow variant stays  *
 *     verified by the Stage A subword_narrow.slang probe).            *
 *                                                                     *
 * Element t of a uint16 array lives in word k = t>>1 at shift         *
 * (t&1)*16; a uint8 element lives in word k = t>>2 at shift (t&3)*8.  *
 * Two adjacent states share one 32-bit word, so a store must not      *
 * clobber its neighbour. The atomic fallback clears then sets only    *
 * THIS element's bits: InterlockedAnd(mask) then InterlockedOr(value).*
 * Disjoint-bit atomics commute, so neighbours race-free (Stage A:     *
 * atomic16/8 PASS on both devices; non-atomic RMW NEGATIVE lost       *
 * ~half the updates and was DETECTED).                                *
 * ------------------------------------------------------------------ */

/* Set to 1 to use true narrow (Int16/Int8) stores instead of the atomic
 * fallback. Default 0 = atomic (portable). The narrow path is RX 570-only. */
#ifndef VULKAN_SUBWORD_NARROW
#define VULKAN_SUBWORD_NARROW 0
#endif

/* Read a uint16 element t from a uint16 SoA array at device address base_u64.
 * Plain 32-bit load of the containing word + shift/mask (no Int16 capability). */
uint load_u16(uint64_t base_u64, uint t)
{
    let word_ptr = cast<uint[]>(base_u64);        // flat uint32 view of the array
    uint k = t >> 1;                               // containing 32-bit word
    uint shift = (t & 1u) * 16u;                   // low or high half
    return (word_ptr[k] >> shift) & 0xFFFFu;
}

/* Write a uint16 element t. Atomic fallback by default. */
void store_u16(uint64_t base_u64, uint t, uint value)
{
    value &= 0xFFFFu;
#if VULKAN_SUBWORD_NARROW
    /* True narrow store (RX 570-only): single 16-bit write, no race because each
     * element is its own 16-bit lane and the hardware handles sub-word writes to
     * distinct lanes. Requires shaderInt16 (Stage A: true16 PASS on RX 570). */
    let elem_ptr = cast<ushort[]>(base_u64);
    elem_ptr[t] = (ushort)value;
#else
    /* Atomic AND/OR fallback (portable, default). */
    let word_ptr = cast<uint[]>(base_u64);
    uint k = t >> 1;
    uint shift = (t & 1u) * 16u;
    uint mask = ~(0xFFFFu << shift);               // all bits EXCEPT ours
    uint slot = value << shift;                    // our value in place
    InterlockedAnd(word_ptr[k], int(mask));        // clear our 16 bits
    InterlockedOr(word_ptr[k], int(slot));         // set them
#endif
}

/* Read a uint8 element t (4 per word). */
uint load_u8(uint64_t base_u64, uint t)
{
    let word_ptr = cast<uint[]>(base_u64);
    uint k = t >> 2;
    uint shift = (t & 3u) * 8u;
    return (word_ptr[k] >> shift) & 0xFFu;
}

/* Write a uint8 element t. Atomic fallback by default. */
void store_u8(uint64_t base_u64, uint t, uint value)
{
    value &= 0xFFu;
#if VULKAN_SUBWORD_NARROW
    let elem_ptr = cast<uchar[]>(base_u64);
    elem_ptr[t] = (uchar)value;
#else
    let word_ptr = cast<uint[]>(base_u64);
    uint k = t >> 2;
    uint shift = (t & 3u) * 8u;
    uint mask = ~(0xFFu << shift);
    uint slot = value << shift;
    InterlockedAnd(word_ptr[k], int(mask));
    InterlockedOr(word_ptr[k], int(slot));
#endif
}

/* `bool` in the C++ state is 1 byte (source: state_template.h:132 uses `bool`).
 * We store it as a uint8 element and normalize to 0/1. */
uint load_bool(uint64_t base_u64, uint t) { return load_u8(base_u64, t) & 1u; }
void store_bool(uint64_t base_u64, uint t, bool value)
{
    store_u8(base_u64, t, value ? 1u : 0u);
}

/* ------------------------------------------------------------------ *
 * AoS slot pointer loader (case B).                                   *
 * Reads the SoA data pointer for member `inner_off` of array slot `i` *
 * from a FLAT uint64_t* at ist_addr + base + i*stride + inner_off.    *
 * Flat pointer + dynamic offset => OpPtrAccessChain (safe).           *
 * ------------------------------------------------------------------ */
uint64_t aos_soa_ptr(uint64_t ist_addr, uint base, uint stride,
                     uint inner_off, uint i)
{
    let slots = cast<uint64_t[]>(ist_addr + base);  // flat view of the slot area
    uint idx = (i * stride + inner_off) / 8u;       // 64-bit words
    return slots[idx];
}

/* ===================== path struct accessors ===================== */
/* state_template.h:7-74. */

uint read_path_render_pixel_index(KernelParamsVulkan* params, uint state)
{
    // source: state_template.h:13 (uint32_t)
    let data_ptr = params->integrator_state.path.render_pixel_index;
    return cast<uint[]>(data_ptr)[state];
}
void write_path_render_pixel_index(KernelParamsVulkan* params, uint state, uint v)
{
    let data_ptr = params->integrator_state.path.render_pixel_index;
    cast<uint[]>(data_ptr)[state] = v;
}

uint read_path_sample(KernelParamsVulkan* params, uint state)
{
    // source: state_template.h:15 (uint32_t)
    let data_ptr = params->integrator_state.path.sample;
    return cast<uint[]>(data_ptr)[state];
}
void write_path_sample(KernelParamsVulkan* params, uint state, uint v)
{
    let data_ptr = params->integrator_state.path.sample;
    cast<uint[]>(data_ptr)[state] = v;
}

uint read_path_bounce(KernelParamsVulkan* params, uint state)
{
    // source: state_template.h:17 (uint16_t) -> sub-word
    let data_ptr = params->integrator_state.path.bounce;
    return load_u16(data_ptr, state);
}
void write_path_bounce(KernelParamsVulkan* params, uint state, uint v)
{
    let data_ptr = params->integrator_state.path.bounce;
    store_u16(data_ptr, state, v);
}

uint read_path_transparent_bounce(KernelParamsVulkan* params, uint state)
{
    // source: state_template.h:19 (uint16_t)
    let data_ptr = params->integrator_state.path.transparent_bounce;
    return load_u16(data_ptr, state);
}
void write_path_transparent_bounce(KernelParamsVulkan* params, uint state, uint v)
{
    let data_ptr = params->integrator_state.path.transparent_bounce;
    store_u16(data_ptr, state, v);
}

uint read_path_queued_kernel(KernelParamsVulkan* params, uint state)
{
    // source: state_template.h:33 (uint16_t) -> queue path
    let data_ptr = params->integrator_state.path.queued_kernel;
    return load_u16(data_ptr, state);
}
void write_path_queued_kernel(KernelParamsVulkan* params, uint state, uint v)
{
    let data_ptr = params->integrator_state.path.queued_kernel;
    store_u16(data_ptr, state, v);
}

uint read_path_rng_offset(KernelParamsVulkan* params, uint state)
{
    // source: state_template.h:37 (uint16_t)
    let data_ptr = params->integrator_state.path.rng_offset;
    return load_u16(data_ptr, state);
}
void write_path_rng_offset(KernelParamsVulkan* params, uint state, uint v)
{
    let data_ptr = params->integrator_state.path.rng_offset;
    store_u16(data_ptr, state, v);
}

uint read_path_visibility(KernelParamsVulkan* params, uint state)
{
    // source: state_template.h:39 (uint8_t) -> sub-word
    let data_ptr = params->integrator_state.path.visibility;
    return load_u8(data_ptr, state);
}
void write_path_visibility(KernelParamsVulkan* params, uint state, uint v)
{
    let data_ptr = params->integrator_state.path.visibility;
    store_u8(data_ptr, state, v);
}

uint read_path_flag(KernelParamsVulkan* params, uint state)
{
    // source: state_template.h:41 (uint32_t)
    let data_ptr = params->integrator_state.path.flag;
    return cast<uint[]>(data_ptr)[state];
}
void write_path_flag(KernelParamsVulkan* params, uint state, uint v)
{
    let data_ptr = params->integrator_state.path.flag;
    cast<uint[]>(data_ptr)[state] = v;
}

uint read_path_mnee(KernelParamsVulkan* params, uint state)
{
    // source: state_template.h:43 (uint8_t) -> sub-word
    let data_ptr = params->integrator_state.path.mnee;
    return load_u8(data_ptr, state);
}
void write_path_mnee(KernelParamsVulkan* params, uint state, uint v)
{
    let data_ptr = params->integrator_state.path.mnee;
    store_u8(data_ptr, state, v);
}

float read_path_mis_ray_pdf(KernelParamsVulkan* params, uint state)
{
    // source: state_template.h:52 (float)
    let data_ptr = params->integrator_state.path.mis_ray_pdf;
    return cast<float[]>(data_ptr)[state];
}
void write_path_mis_ray_pdf(KernelParamsVulkan* params, uint state, float v)
{
    let data_ptr = params->integrator_state.path.mis_ray_pdf;
    cast<float[]>(data_ptr)[state] = v;
}

float read_path_min_ray_pdf(KernelParamsVulkan* params, uint state)
{
    // source: state_template.h:58 (float)
    let data_ptr = params->integrator_state.path.min_ray_pdf;
    return cast<float[]>(data_ptr)[state];
}
void write_path_min_ray_pdf(KernelParamsVulkan* params, uint state, float v)
{
    let data_ptr = params->integrator_state.path.min_ray_pdf;
    cast<float[]>(data_ptr)[state] = v;
}

float read_path_continuation_probability(KernelParamsVulkan* params, uint state)
{
    // source: state_template.h:60 (float)
    let data_ptr = params->integrator_state.path.continuation_probability;
    return cast<float[]>(data_ptr)[state];
}
void write_path_continuation_probability(KernelParamsVulkan* params, uint state, float v)
{
    let data_ptr = params->integrator_state.path.continuation_probability;
    cast<float[]>(data_ptr)[state] = v;
}

float read_path_optical_depth(KernelParamsVulkan* params, uint state)
{
    // source: state_template.h:47 (float)
    let data_ptr = params->integrator_state.path.optical_depth;
    return cast<float[]>(data_ptr)[state];
}
void write_path_optical_depth(KernelParamsVulkan* params, uint state, float v)
{
    let data_ptr = params->integrator_state.path.optical_depth;
    cast<float[]>(data_ptr)[state] = v;
}

uint read_path_shader_sort_key(KernelParamsVulkan* params, uint state)
{
    // source: state_template.h:73 (uint32_t) -> queue path
    let data_ptr = params->integrator_state.path.shader_sort_key;
    return cast<uint[]>(data_ptr)[state];
}
void write_path_shader_sort_key(KernelParamsVulkan* params, uint state, uint v)
{
    let data_ptr = params->integrator_state.path.shader_sort_key;
    cast<uint[]>(data_ptr)[state] = v;
}

/* PackedSpectrum == packed_float3, 12 bytes (source: types_spectrum.h:14,
 * types_float3.h:198 static_assert sizeof==12). SoA stride 12; float3* via
 * cast<float3[]>. The gate measures that the stride is 12, not 16. */
float3 read_path_throughput(KernelParamsVulkan* params, uint state)
{
    // source: state_template.h:62 (PackedSpectrum)
    let data_ptr = params->integrator_state.path.throughput;
    return cast<float3[]>(data_ptr)[state];
}
void write_path_throughput(KernelParamsVulkan* params, uint state, float3 v)
{
    let data_ptr = params->integrator_state.path.throughput;
    cast<float3[]>(data_ptr)[state] = v;
}

float3 read_path_mis_origin_n(KernelParamsVulkan* params, uint state)
{
    // source: state_template.h:56 (packed_float3)
    let data_ptr = params->integrator_state.path.mis_origin_n;
    return cast<float3[]>(data_ptr)[state];
}
void write_path_mis_origin_n(KernelParamsVulkan* params, uint state, float3 v)
{
    let data_ptr = params->integrator_state.path.mis_origin_n;
    cast<float3[]>(data_ptr)[state] = v;
}

float3 read_path_pass_diffuse_weight(KernelParamsVulkan* params, uint state)
{
    // source: state_template.h:67 (PackedSpectrum)
    let data_ptr = params->integrator_state.path.pass_diffuse_weight;
    return cast<float3[]>(data_ptr)[state];
}
void write_path_pass_diffuse_weight(KernelParamsVulkan* params, uint state, float3 v)
{
    let data_ptr = params->integrator_state.path.pass_diffuse_weight;
    cast<float3[]>(data_ptr)[state] = v;
}

/* ===================== ray struct accessors ===================== */
/* state_template.h:78-87. Packed: P/D are packed_float3 (stride 12). */

float3 read_ray_P(KernelParamsVulkan* params, uint state)
{
    // source: state_template.h:79 (packed_float3)
    let data_ptr = params->integrator_state.ray.P;
    return cast<float3[]>(data_ptr)[state];
}
void write_ray_P(KernelParamsVulkan* params, uint state, float3 v)
{
    let data_ptr = params->integrator_state.ray.P;
    cast<float3[]>(data_ptr)[state] = v;
}

float3 read_ray_D(KernelParamsVulkan* params, uint state)
{
    // source: state_template.h:81 (packed_float3)
    let data_ptr = params->integrator_state.ray.D;
    return cast<float3[]>(data_ptr)[state];
}
void write_ray_D(KernelParamsVulkan* params, uint state, float3 v)
{
    let data_ptr = params->integrator_state.ray.D;
    cast<float3[]>(data_ptr)[state] = v;
}

float read_ray_dP(KernelParamsVulkan* params, uint state)
{
    // source: state_template.h:80 (float)
    let data_ptr = params->integrator_state.ray.dP;
    return cast<float[]>(data_ptr)[state];
}
void write_ray_dP(KernelParamsVulkan* params, uint state, float v)
{
    let data_ptr = params->integrator_state.ray.dP;
    cast<float[]>(data_ptr)[state] = v;
}

float read_ray_dD(KernelParamsVulkan* params, uint state)
{
    // source: state_template.h:82 (float)
    let data_ptr = params->integrator_state.ray.dD;
    return cast<float[]>(data_ptr)[state];
}
void write_ray_dD(KernelParamsVulkan* params, uint state, float v)
{
    let data_ptr = params->integrator_state.ray.dD;
    cast<float[]>(data_ptr)[state] = v;
}

float read_ray_tmin(KernelParamsVulkan* params, uint state)
{
    // source: state_template.h:83 (float)
    let data_ptr = params->integrator_state.ray.tmin;
    return cast<float[]>(data_ptr)[state];
}
void write_ray_tmin(KernelParamsVulkan* params, uint state, float v)
{
    let data_ptr = params->integrator_state.ray.tmin;
    cast<float[]>(data_ptr)[state] = v;
}

float read_ray_tmax(KernelParamsVulkan* params, uint state)
{
    // source: state_template.h:84 (float)
    let data_ptr = params->integrator_state.ray.tmax;
    return cast<float[]>(data_ptr)[state];
}
void write_ray_tmax(KernelParamsVulkan* params, uint state, float v)
{
    let data_ptr = params->integrator_state.ray.tmax;
    cast<float[]>(data_ptr)[state] = v;
}

float read_ray_time(KernelParamsVulkan* params, uint state)
{
    // source: state_template.h:85 (float)
    let data_ptr = params->integrator_state.ray.time;
    return cast<float[]>(data_ptr)[state];
}
void write_ray_time(KernelParamsVulkan* params, uint state, float v)
{
    let data_ptr = params->integrator_state.ray.time;
    cast<float[]>(data_ptr)[state] = v;
}

/* ===================== isect struct accessors ===================== */
/* state_template.h:92-99. INTERSECT_CLOSEST writes all six. */

float read_isect_t(KernelParamsVulkan* params, uint state)
{
    // source: state_template.h:93 (float)
    let data_ptr = params->integrator_state.isect.t;
    return cast<float[]>(data_ptr)[state];
}
void write_isect_t(KernelParamsVulkan* params, uint state, float v)
{
    let data_ptr = params->integrator_state.isect.t;
    cast<float[]>(data_ptr)[state] = v;
}

int read_isect_prim(KernelParamsVulkan* params, uint state)
{
    // source: state_template.h:96 (int)
    let data_ptr = params->integrator_state.isect.prim;
    return cast<int[]>(data_ptr)[state];
}
void write_isect_prim(KernelParamsVulkan* params, uint state, int v)
{
    let data_ptr = params->integrator_state.isect.prim;
    cast<int[]>(data_ptr)[state] = v;
}

int read_isect_object(KernelParamsVulkan* params, uint state)
{
    // source: state_template.h:97 (int)
    let data_ptr = params->integrator_state.isect.object;
    return cast<int[]>(data_ptr)[state];
}
void write_isect_object(KernelParamsVulkan* params, uint state, int v)
{
    let data_ptr = params->integrator_state.isect.object;
    cast<int[]>(data_ptr)[state] = v;
}

int read_isect_type(KernelParamsVulkan* params, uint state)
{
    // source: state_template.h:98 (int)
    let data_ptr = params->integrator_state.isect.type;
    return cast<int[]>(data_ptr)[state];
}
void write_isect_type(KernelParamsVulkan* params, uint state, int v)
{
    let data_ptr = params->integrator_state.isect.type;
    cast<int[]>(data_ptr)[state] = v;
}

/* ===================== volume_stack (AoS) accessors ===================== */
/* state_template.h:112-117. layout: struct { int*object; int*shader; }[32].
 * base offset 376, stride 16, inner object@0 / shader@8 (M4e S1b-5 static_asserts). */
static const uint VS_BASE = 376u;
static const uint VS_STRIDE = 16u;

int read_volume_stack_object(KernelParamsVulkan* params, uint64_t ist_addr,
                             uint i, uint state)
{
    // source: state_template.h:113 (int, AoS array member)
    let data_ptr = aos_soa_ptr(ist_addr, VS_BASE, VS_STRIDE, 0u, i);
    return cast<int[]>(data_ptr)[state];
}
void write_volume_stack_object(KernelParamsVulkan* params, uint64_t ist_addr,
                               uint i, uint state, int v)
{
    let data_ptr = aos_soa_ptr(ist_addr, VS_BASE, VS_STRIDE, 0u, i);
    cast<int[]>(data_ptr)[state] = v;
}

int read_volume_stack_shader(KernelParamsVulkan* params, uint64_t ist_addr,
                             uint i, uint state)
{
    // source: state_template.h:114 (int, AoS array member)
    let data_ptr = aos_soa_ptr(ist_addr, VS_BASE, VS_STRIDE, 8u, i);
    return cast<int[]>(data_ptr)[state];
}
void write_volume_stack_shader(KernelParamsVulkan* params, uint64_t ist_addr,
                               uint i, uint state, int v)
{
    let data_ptr = aos_soa_ptr(ist_addr, VS_BASE, VS_STRIDE, 8u, i);
    cast<int[]>(data_ptr)[state] = v;
}

/* ===================== shadow_path accessors ===================== */
/* shadow_state_template.h:7-74. SHADE_SURFACE (diffuse) writes these. */

uint read_shadow_path_render_pixel_index(KernelParamsVulkan* params, uint state)
{
    // source: shadow_state_template.h:9 (uint32_t)
    let data_ptr = params->integrator_state.shadow_path.render_pixel_index;
    return cast<uint[]>(data_ptr)[state];
}
void write_shadow_path_render_pixel_index(KernelParamsVulkan* params, uint state, uint v)
{
    let data_ptr = params->integrator_state.shadow_path.render_pixel_index;
    cast<uint[]>(data_ptr)[state] = v;
}

uint read_shadow_path_queued_kernel(KernelParamsVulkan* params, uint state)
{
    // source: shadow_state_template.h:31 (uint16_t) -> queue path
    let data_ptr = params->integrator_state.shadow_path.queued_kernel;
    return load_u16(data_ptr, state);
}
void write_shadow_path_queued_kernel(KernelParamsVulkan* params, uint state, uint v)
{
    let data_ptr = params->integrator_state.shadow_path.queued_kernel;
    store_u16(data_ptr, state, v);
}

uint read_shadow_path_flag(KernelParamsVulkan* params, uint state)
{
    // source: shadow_state_template.h:35 (uint32_t)
    let data_ptr = params->integrator_state.shadow_path.flag;
    return cast<uint[]>(data_ptr)[state];
}
void write_shadow_path_flag(KernelParamsVulkan* params, uint state, uint v)
{
    let data_ptr = params->integrator_state.shadow_path.flag;
    cast<uint[]>(data_ptr)[state] = v;
}

uint read_shadow_path_lightgroup(KernelParamsVulkan* params, uint state)
{
    // source: shadow_state_template.h:53 (uint8_t) -> sub-word
    let data_ptr = params->integrator_state.shadow_path.lightgroup;
    return load_u8(data_ptr, state);
}
void write_shadow_path_lightgroup(KernelParamsVulkan* params, uint state, uint v)
{
    let data_ptr = params->integrator_state.shadow_path.lightgroup;
    store_u8(data_ptr, state, v);
}

float3 read_shadow_path_throughput(KernelParamsVulkan* params, uint state)
{
    // source: shadow_state_template.h:37 (PackedSpectrum)
    let data_ptr = params->integrator_state.shadow_path.throughput;
    return cast<float3[]>(data_ptr)[state];
}
void write_shadow_path_throughput(KernelParamsVulkan* params, uint state, float3 v)
{
    let data_ptr = params->integrator_state.shadow_path.throughput;
    cast<float3[]>(data_ptr)[state] = v;
}

/* ===================== shadow_ray accessors ===================== */
/* shadow_state_template.h:78-88. Packed: P/D are packed_float3 (stride 12). */

float3 read_shadow_ray_P(KernelParamsVulkan* params, uint state)
{
    // source: shadow_state_template.h:79 (packed_float3)
    let data_ptr = params->integrator_state.shadow_ray.P;
    return cast<float3[]>(data_ptr)[state];
}
void write_shadow_ray_P(KernelParamsVulkan* params, uint state, float3 v)
{
    let data_ptr = params->integrator_state.shadow_ray.P;
    cast<float3[]>(data_ptr)[state] = v;
}

float3 read_shadow_ray_D(KernelParamsVulkan* params, uint state)
{
    // source: shadow_state_template.h:80 (packed_float3)
    let data_ptr = params->integrator_state.shadow_ray.D;
    return cast<float3[]>(data_ptr)[state];
}
void write_shadow_ray_D(KernelParamsVulkan* params, uint state, float3 v)
{
    let data_ptr = params->integrator_state.shadow_ray.D;
    cast<float3[]>(data_ptr)[state] = v;
}

float read_shadow_ray_tmin(KernelParamsVulkan* params, uint state)
{
    // source: shadow_state_template.h:81 (float)
    let data_ptr = params->integrator_state.shadow_ray.tmin;
    return cast<float[]>(data_ptr)[state];
}
void write_shadow_ray_tmin(KernelParamsVulkan* params, uint state, float v)
{
    let data_ptr = params->integrator_state.shadow_ray.tmin;
    cast<float[]>(data_ptr)[state] = v;
}

/* ===================== shadow_isect (AoS) accessors ===================== */
/* shadow_state_template.h:94-103. layout: 6-pointer tuple [4], stride 48.
 * base 1240, t@0/u@8/v@16/prim@24/object@32/type@40 (M4e S1b-5 static_asserts). */
static const uint SI_BASE = 1240u;
static const uint SI_STRIDE = 48u;

float read_shadow_isect_t(KernelParamsVulkan* params, uint64_t ist_addr,
                          uint i, uint state)
{
    // source: shadow_state_template.h:95 (float, AoS array member)
    let data_ptr = aos_soa_ptr(ist_addr, SI_BASE, SI_STRIDE, 0u, i);
    return cast<float[]>(data_ptr)[state];
}
void write_shadow_isect_t(KernelParamsVulkan* params, uint64_t ist_addr,
                          uint i, uint state, float v)
{
    let data_ptr = aos_soa_ptr(ist_addr, SI_BASE, SI_STRIDE, 0u, i);
    cast<float[]>(data_ptr)[state] = v;
}

int read_shadow_isect_object(KernelParamsVulkan* params, uint64_t ist_addr,
                             uint i, uint state)
{
    // source: shadow_state_template.h:99 (int, AoS array member)
    let data_ptr = aos_soa_ptr(ist_addr, SI_BASE, SI_STRIDE, 32u, i);
    return cast<int[]>(data_ptr)[state];
}
void write_shadow_isect_object(KernelParamsVulkan* params, uint64_t ist_addr,
                               uint i, uint state, int v)
{
    let data_ptr = aos_soa_ptr(ist_addr, SI_BASE, SI_STRIDE, 32u, i);
    cast<int[]>(data_ptr)[state] = v;
}

/* ===================== shadow_volume_stack (AoS) accessors ===================== */
/* shadow_state_template.h:107-112. Same shape as volume_stack: stride 16.
 * base 1432, object@0 / shader@8 (M4e S1b-5 static_asserts). */
static const uint SVS_BASE = 1432u;
static const uint SVS_STRIDE = 16u;

int read_shadow_volume_stack_object(KernelParamsVulkan* params, uint64_t ist_addr,
                                    uint i, uint state)
{
    // source: shadow_state_template.h:108 (int, AoS array member)
    let data_ptr = aos_soa_ptr(ist_addr, SVS_BASE, SVS_STRIDE, 0u, i);
    return cast<int[]>(data_ptr)[state];
}
void write_shadow_volume_stack_object(KernelParamsVulkan* params, uint64_t ist_addr,
                                      uint i, uint state, int v)
{
    let data_ptr = aos_soa_ptr(ist_addr, SVS_BASE, SVS_STRIDE, 0u, i);
    cast<int[]>(data_ptr)[state] = v;
}

int read_shadow_volume_stack_shader(KernelParamsVulkan* params, uint64_t ist_addr,
                                    uint i, uint state)
{
    // source: shadow_state_template.h:109 (int, AoS array member)
    let data_ptr = aos_soa_ptr(ist_addr, SVS_BASE, SVS_STRIDE, 8u, i);
    return cast<int[]>(data_ptr)[state];
}
void write_shadow_volume_stack_shader(KernelParamsVulkan* params, uint64_t ist_addr,
                                      uint i, uint state, int v)
{
    let data_ptr = aos_soa_ptr(ist_addr, SVS_BASE, SVS_STRIDE, 8u, i);
    cast<int[]>(data_ptr)[state] = v;
}

/* ====================================================================== *
 * MEMBERS NOT DONE HERE (and why).                                       *
 * These follow the identical patterns above (scalar/sub-word/packed3/AoS)*
 * and are NOT touched by INIT_FROM_CAMERA, INTERSECT_CLOSEST,             *
 * SHADE_SURFACE(diffuse) or the queue path, or are compile-time feature-  *
 * gated OFF in this build:                                               *
 *   - subsurface.* (albedo/radius/anisotropy/N; state_template.h:103-108)*
 *     -> only SUBSURFACE kernels.                                         *
 *   - shadow_link.* (dedicated_light_weight/last_isect_*;                 *
 *     state_template.h:151-156) -> only SHADOW_LINKING kernels.           *
 *   - guiding.* (path_segment/use_*_guiding/sampling probs;               *
 *     state_template.h:120-147) -> __PATH_GUIDING__ is OFF in this build,*
 *     so init_from_camera/shade_surface do not touch them.                *
 *   - path.unguided_throughput (state_template.h:65) -> PATH_GUIDING only.*
 *   - path.mnee_shadow_state (state_template.h:45) -> MNEE only.          *
 *   - path.rng_pixel (state_template.h:35) and path.denoising_feature_    *
 *     throughput (state_template.h:70) -> same scalar/packed3 pattern as  *
 *     members above; left for the INIT_FROM_CAMERA port to add if needed. *
 *   - ray.previous_dt (state_template.h:86) -> LIGHT_TREE only.           *
 *   - isect.u / isect.v (state_template.h:94-95) -> same float pattern as  *
 *     isect.t above; not read/written by the diffuse path until M5.       *
 *   - shadow_path.rng_pixel/rng_offset/sample/bounce/transparent_bounce/  *
 *     volume_bounds_bounce/glossy_bounce/transmission_bounce/diffuse_     *
 *     bounce/visibility/unshadowed_throughput/pass_diffuse_weight/        *
 *     pass_glossy_weight/packed_num_hits/unlit_throughput/path_segment/   *
 *     guiding_light_linking_mis_weight/bsdf_eval_average                  *
 *     (shadow_state_template.h:11-73) -> same sub-word/packed3/scalar     *
 *     patterns as shadow_path above; added per-kernel during M5 shadow    *
 *     work.                                                               *
 *   - shadow_ray.time/dP/dD/self_light_object/self_light_prim             *
 *     (shadow_state_template.h:82-87) -> same float/int pattern.          *
 *   - shadow_isect.u/v/prim/type (shadow_state_template.h:96-100) -> same *
 *     float/int AoS pattern as shadow_isect.t above.                      *
 *   - the IntegratorStateGPU scalars (queue_counter/sort_key_counter/     *
 *     next_*_path_index/sort_partition_*; state.h:197-212) are NOT SoA    *
 *     members and are host-managed; not needed by these kernels.          *
 * ====================================================================== */
