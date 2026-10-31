// SPDX-FileCopyrightText: 2011-2022 Blender Foundation
// SPDX-License-Identifier: Apache-2.0

/* film_queue.slang - M4f SLICE 3: the film + queue layer of the
 * INIT_FROM_CAMERA device kernel, ported faithfully (GPU semantics).
 *
 * WHAT IS PORTED (one small cluster, AGENTS.md section 3A):
 *   film_pass_pixel_render_buffer()   src/kernel/film/write.h:24-31 (state overload)
 *   film_need_sample_pixel()          src/kernel/film/adaptive_sampling.h:13-25
 *   film_write_sample()               src/kernel/film/light_passes.h:158-173
 *   write_optical_depth()             src/kernel/integrator/state_flow.h:51-67
 *   integrator_path_init()            src/kernel/integrator/state_flow.h:71-76 (GPU branch)
 *   integrator_path_next()            src/kernel/integrator/state_flow.h:78-86 (GPU branch)
 *   integrator_path_terminate()       src/kernel/integrator/state_flow.h:88-98 (GPU branch)
 *   integrator_path_cache_miss()      src/kernel/integrator/state_flow.h:100-105 (GPU branch)
 *
 * GATE: film_queue_probe.slang + vulkan_film_queue_probe (kernels.cpp).
 *
 * SIGNATURE MAPPING (AGENTS.md section 4):
 *   kg (kernel_data.film.*) -> explicit cited int args: pass_stride,
 *       pass_sample_count, pass_adaptive_aux_buffer, pass_volume_majorant,
 *       pass_volume_majorant_sample_count. KernelData stays an opaque host
 *       blob (M4e S1b-0 decision), so the five values the film functions read
 *       are arguments - same pattern as slice 2's kernel_features.
 *   kg (kernel_integrator_state.queue_counter) -> the mirror's TYPED pointer
 *       slot `ist->queue_counter` (vulkan_data.slang:331, host offset 1944,
 *       state.h:197). The host probe points it at a real counter block.
 *   state -> IntegratorStateGPU* ist + uint state (slice-1/2 accessors)
 *   render_buffer -> float* device pointer (push constant)
 *   next_kernel / current_kernel -> explicit uint args carrying the cited
 *       DeviceKernel enum values (types.h:1745+: INIT_FROM_CAMERA=0,
 *       INTERSECT_CLOSEST=2, INTERSECT_VOLUME_STACK=5)
 *
 * NOT PORTED (documented, not oversight):
 *   - integrator_path_init_sorted / path_next_sorted / path_cache_miss_sorted
 *     (state_flow.h:107-117,163-202): not called by init_from_camera; the
 *     shader-sort layer is a later milestone. DEFERRED.
 *   - the shadow-path functions (state_flow.h:119-157): init_from_camera
 *     creates no shadow paths. DEFERRED to the NEE kernels.
 *   - PROFILING_INIT (init_from_camera.h:65): host-side profiling macro only.
 *
 * PLATFORM CONSTRAINTS (documented divergences, both result-preserving):
 *   1. uint64_t render-buffer offset (write.h:28) -> uint32 multiply. Cycles
 *      casts render_pixel_index to uint64_t before multiplying by pass_stride.
 *      This backend compiles Int64-free by design (M4e S1b-7, user decision:
 *      SwiftShader exposes no shaderInt64, vulkan_data.slang:76-80), and on a
 *      <=4 GB device every in-buffer float offset is < 2^30, so the 32-bit
 *      multiply cannot overflow for any buffer that exists in device memory.
 *   2. atomic_add_and_fetch_float (write.h:61-62 under __ATOMIC_PASS_WRITE__,
 *      CUDA spelling atomicAdd, util/atomic.h:30) -> CAS loop on the bit-cast
 *      uint word. Slang's InterlockedAdd on float lowers to OpAtomicFAddEXT
 *      (SPV_EXT_shader_atomic_float_add), which needs the
 *      VK_EXT_shader_atomic_float_features feature this backend does NOT
 *      enable at device create (device.cpp:176-190 requests only BDA, Int8,
 *      Int16, Int64) and SwiftShader does not support. The CAS loop is core
 *      SPIR-V (OpAtomicCompareExchange) and computes the SAME IEEE float add;
 *      with a single writer per slot (the gate's layout, and one state per
 *      pixel slot in the real scheduler's counter-free film slots) the result
 *      bits are identical to Cycles' atomicAdd.
 *   3. `if (!render_buffer) return;` (state_flow.h:54) -> an explicit
 *      `render_buffer_valid` uint PARAMETER (0 = null buffer). A pointer ==
 *      null comparison lowers to OpConvertPtrToU on this Slang, which forces
 *      OpCapability Int64; SwiftShader exposes no shaderInt64 (M4e S1b-7,
 *      vulkan_data.slang:76-80) and the recorded S1b-7 run crashed on
 *      OpConvertPtrToU (0xC00000FD, 2026-10-24). Three spellings were tried
 *      in this session and all lower to OpConvertPtrToU or fail to compile:
 *      `== nullptr`, `!ptr`, and pointer==pointer variables; Slang's
 *      [op] / [__target_op] / __spirv("OpPtrEqual") intrinsic attributes are
 *      all unknown in slangc 2026.20, so OpPtrEqual cannot be emitted.
 *      The flag is wired 1:1 by the host from the very pointer it passes, so
 *      the check cannot disagree with the pointer; the null branch IS
 *      exercised by the gate (host reference calls the REAL
 *      integrator_path_terminate with literal nullptr, exactly as
 *      intersect_subsurface.h:21 does, while the probe gets valid=0 with a
 *      canary-filled buffer that must stay untouched).
 */

#include "integrator_state.slang"  // accessors + the IntegratorStateGPU mirror

/* --------------------------------------------------------------------------- *
 * Constants, values cited from the real enums.
 * --------------------------------------------------------------------------- */
// source: src/kernel/types.h:52 (#define PASS_UNUSED (~0)) -> int -1
static const int FILM_PASS_UNUSED = -1;
// source: src/kernel/types.h:276 (PATH_RAY_VOLUME_PRIMARY_TRANSMIT = 1U << 24)
static const uint PATH_RAY_VOLUME_PRIMARY_TRANSMIT = 1u << 24u;

/* DeviceKernel enum values the queue functions receive as arguments
 * (source: src/kernel/types.h:1745-1763; the probe passes these explicitly):
 *   DEVICE_KERNEL_INTEGRATOR_INIT_FROM_CAMERA = 0
 *   DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST = 2
 *   DEVICE_KERNEL_INTEGRATOR_INTERSECT_VOLUME_STACK = 5
 * DEVICE_GPU_KERNEL_INTEGRATOR_NUM = 18 (types.h:1827 = MEGAKERNEL's value);
 * num_queued has exactly 18 slots [0..17] (state.h:101, mirrored at
 * vulkan_data.slang:310) - every kernel index used below is bounds-checked
 * against it (AGENTS section 4: raw device pointers have no bounds check). */

/* =========================================================================== *
 *  film_pass_pixel_render_buffer - FAITHFUL PORT (with constraint 1 above)
 *  source: src/kernel/film/write.h:24-31
 * =========================================================================== *
 * Cycles:
 *   const uint32_t render_pixel_index = INTEGRATOR_STATE(state, path,
 *                                                            render_pixel_index);
 *   const uint64_t render_buffer_offset = (uint64_t)render_pixel_index *
 *                                         kernel_data.film.pass_stride;
 *   return render_buffer + render_buffer_offset;
 * The uint64_t cast becomes a uint32 multiply - see constraint 1 in the file
 * header (Int64-free backend; overflow impossible for an in-memory buffer). */
float *film_pass_pixel_render_buffer(IntegratorStateGPU *ist,
                                     uint state,
                                     float *render_buffer,
                                     const int pass_stride)
{
    uint render_pixel_index = read_path_render_pixel_index(ist, state);
    uint render_buffer_offset = render_pixel_index * (uint)pass_stride;
    return render_buffer + render_buffer_offset;
}

/* =========================================================================== *
 *  film_need_sample_pixel - FAITHFUL PORT
 *  source: src/kernel/film/adaptive_sampling.h:13-25
 * =========================================================================== *
 * Cycles:
 *   if (kernel_data.film.pass_adaptive_aux_buffer == PASS_UNUSED) return true;
 *   buffer = film_pass_pixel_render_buffer(...);
 *   aux_w_offset = pass_adaptive_aux_buffer + 3;
 *   return buffer[aux_w_offset] == 0.0f;   // the .w of the aux float4
 * No branch is feature-compiled out; the PASS_UNUSED test is runtime. */
bool film_need_sample_pixel(IntegratorStateGPU *ist,
                            uint state,
                            float *render_buffer,
                            const int pass_adaptive_aux_buffer,
                            const int pass_stride)
{
    if (pass_adaptive_aux_buffer == FILM_PASS_UNUSED) {
        return true;
    }

    float *buffer = film_pass_pixel_render_buffer(ist, state, render_buffer, pass_stride);

    uint aux_w_offset = (uint)pass_adaptive_aux_buffer + 3u;
    return buffer[aux_w_offset] == 0.0f;
}

/* =========================================================================== *
 *  film_write_sample - FAITHFUL PORT
 *  source: src/kernel/film/light_passes.h:158-173
 * =========================================================================== *
 * Cycles:
 *   if (kernel_data.film.pass_sample_count == PASS_UNUSED) return sample;
 *   buffer = film_pass_pixel_render_buffer(...);
 *   return atomic_fetch_and_add_uint32(
 *              (ccl_global uint *)(buffer) + kernel_data.film.pass_sample_count, 1)
 *          + sample_offset;
 * The count slot is addressed as UINT (the counter bits live in a float-typed
 * slot; read back via __float_as_uint at adaptive_sampling.h:58). InterlockedAdd
 * matches HLSL's uint overload and lowers to core OpAtomicIAdd. Bounds check:
 * pass_sample_count is an offset in the pixel's pass row, which must be
 * < pass_stride for the slot to exist; the gate only feeds in-range values and
 * the real kernel data comes from the film layout (checked host-side). */
int film_write_sample(IntegratorStateGPU *ist,
                      uint state,
                      float *render_buffer,
                      const int sample,
                      const int sample_offset,
                      const int pass_sample_count,
                      const int pass_stride)
{
    if (pass_sample_count == FILM_PASS_UNUSED) {
        return sample;
    }

    float *buffer = film_pass_pixel_render_buffer(ist, state, render_buffer, pass_stride);

    uint *count_slot = (uint *)buffer + (uint)pass_sample_count;
    uint old;
    InterlockedAdd(count_slot[0], 1u, old);
    return int(old) + sample_offset;
}

/* =========================================================================== *
 *  atomic_add_float - PORTABLE spelling of atomic_add_and_fetch_float
 *  (implementation of constraint 2 in the file header; the CALLER below is the
 *  faithful port, this helper replaces the CUDA/HIP atomicAdd)
 *  source of the semantics: src/kernel/film/write.h:61-62 -> util/atomic.h:30
 *  (atomicAdd returns the OLD value, so fetch-and-add = old + x)
 * =========================================================================== *
 * CAS loop on the bit-cast uint word: load bits, compute the IEEE sum, and
 * CAS it in; retry with the winner's value if another writer interfered. */
float atomic_add_float_cas(float *slot, float x)
{
    uint *p = (uint *)slot;  // bit-cast: same 4 bytes, integer view for the CAS
    uint old_bits = p[0];
    float sum;
    uint new_bits;
    uint prev;
    while (true)
    {
        sum = asfloat(old_bits) + x;
        new_bits = asuint(sum);
        InterlockedCompareExchange(p[0], old_bits, new_bits, prev);
        if (prev == old_bits) {
            break;           // our write landed; prev held the value we added to
        }
        old_bits = prev;     // someone else won - retry against their value
    }
    return sum;
}

/* =========================================================================== *
 *  write_optical_depth - FAITHFUL PORT
 *  source: src/kernel/integrator/state_flow.h:51-67
 * =========================================================================== *
 * Cycles:
 *   if (!render_buffer) return;                       // null check kept
 *   if (state->path.flag & PATH_RAY_VOLUME_PRIMARY_TRANSMIT) {
 *     kernel_assert(kernel_data.film.pass_volume_majorant != PASS_UNUSED);
 *     optical_depth = state->path.optical_depth;
 *     buffer = film_pass_pixel_render_buffer(...);
 *     film_write_pass_float(buffer + pass_volume_majorant, optical_depth);
 *     film_write_pass_float(buffer + pass_volume_majorant_sample_count, 1.0f);
 *   }
 * film_write_pass_float is the GPU-atomic variant (write.h:15-18 defines
 * __ATOMIC_PASS_WRITE__ for __KERNEL_GPU__) -> atomic_add_float_cas above.
 * kernel_assert is a debug-only macro with no Slang equivalent in this
 * backend; it is intentionally NOT ported (a wrong offset would be caught by
 * the gate's raw-bit compare anyway).
 * The null check of state_flow.h:54 becomes the render_buffer_valid argument
 * - platform constraint 3 in the file header (pointer-null comparison forces
 * OpCapability Int64, which SwiftShader cannot run). The caller wires the
 * flag from the same pointer it passes. */
void write_optical_depth(IntegratorStateGPU *ist,
                         uint state,
                         float *render_buffer,
                         const uint render_buffer_valid,
                         const int pass_volume_majorant,
                         const int pass_volume_majorant_sample_count,
                         const int pass_stride)
{
    if (render_buffer_valid == 0u) {
        return;  // = Cycles' `if (!render_buffer) return;` (state_flow.h:54)
    }

    uint flag = read_path_flag(ist, state);
    if ((flag & PATH_RAY_VOLUME_PRIMARY_TRANSMIT) != 0u) {
        float optical_depth = read_path_optical_depth(ist, state);
        float *buffer = film_pass_pixel_render_buffer(ist, state, render_buffer, pass_stride);

        /* source: state_flow.h:64-65 via film_write_pass_float (write.h:58-66) */
        atomic_add_float_cas(buffer + (uint)pass_volume_majorant, optical_depth);
        atomic_add_float_cas(buffer + (uint)pass_volume_majorant_sample_count, 1.0f);
    }
}

/* ---- Queue layer: the __KERNEL_GPU__ branch of state_flow.h:69-117.
 * The CPU branch (:206-244) omits every counter update - that is why the
 * gate's expected counter values are computed as net sums of THESE documented
 * atomic ops (see kernels.cpp's probe doc), while queued_kernel writes are
 * compared against the real CPU calls (both branches write it identically).
 *
 * Base-pointer rule (M4e device rule, AGENTS.md section 11): num_queued is a
 * fixed array INSIDE a PhysicalStorageBuffer struct; a dynamic inline index
 * through it (queue_counter->num_queued[kernel]) is miscompiled by the RX 570
 * driver (always element 0). Take the base once with a CONSTANT index, then
 * index through the plain int* - this lowers to OpPtrAccessChain, the proven
 * safe shape. ---- */

/* Bounds check helper: kernel index must be < 18 (num_queued slots,
 * state.h:101). Out-of-range returns false and the caller skips the op
 * instead of writing out of bounds (AGENTS section 4). */
bool queue_kernel_index_valid(uint kernel)
{
    return kernel < 18u;
}

/* =========================================================================== *
 *  integrator_path_init - FAITHFUL PORT (GPU branch)
 *  source: src/kernel/integrator/state_flow.h:71-76
 * =========================================================================== *
 * Cycles:
 *   atomic_fetch_and_add_uint32(
 *       &kernel_integrator_state.queue_counter->num_queued[next_kernel], 1);
 *   INTEGRATOR_STATE_WRITE(state, path, queued_kernel) = next_kernel; */
void integrator_path_init(IntegratorStateGPU *ist,
                          uint state,
                          const uint next_kernel)
{
    if (!queue_kernel_index_valid(next_kernel)) {
        return;  // defensive: refuse the out-of-range counter write
    }
    int *base = &ist->queue_counter->num_queued[0];  // CONSTANT index (device rule)
    InterlockedAdd(base[next_kernel], 1);
    write_path_queued_kernel(ist, state, next_kernel);
}

/* =========================================================================== *
 *  integrator_path_next - FAITHFUL PORT (GPU branch)
 *  source: src/kernel/integrator/state_flow.h:78-86
 * =========================================================================== *
 * Cycles:
 *   atomic_fetch_and_sub_uint32(&...num_queued[current_kernel], 1);
 *   atomic_fetch_and_add_uint32(&...num_queued[next_kernel], 1);
 *   INTEGRATOR_STATE_WRITE(state, path, queued_kernel) = next_kernel;
 * atomic_fetch_and_sub is add-of-negative on every Cycles backend
 * (util/atomic.h:17,33: dec = add -1); InterlockedAdd with -1 matches. */
void integrator_path_next(IntegratorStateGPU *ist,
                          uint state,
                          const uint current_kernel,
                          const uint next_kernel)
{
    if (!queue_kernel_index_valid(current_kernel) ||
        !queue_kernel_index_valid(next_kernel)) {
        return;  // defensive: refuse out-of-range counter writes
    }
    int *base = &ist->queue_counter->num_queued[0];  // CONSTANT index (device rule)
    InterlockedAdd(base[current_kernel], -1);
    InterlockedAdd(base[next_kernel], 1);
    write_path_queued_kernel(ist, state, next_kernel);
}

/* =========================================================================== *
 *  integrator_path_terminate - FAITHFUL PORT (GPU branch)
 *  source: src/kernel/integrator/state_flow.h:88-98
 * =========================================================================== *
 * Cycles:
 *   write_optical_depth(kg, state, render_buffer);
 *   atomic_fetch_and_sub_uint32(&...num_queued[current_kernel], 1);
 *   INTEGRATOR_STATE_WRITE(state, path, queued_kernel) = 0;
 * render_buffer_valid carries the null-ness of render_buffer into
 * write_optical_depth (platform constraint 3 in the file header); Cycles
 * passes the pointer itself and lets write_optical_depth test it. */
void integrator_path_terminate(IntegratorStateGPU *ist,
                               uint state,
                               float *render_buffer,
                               const uint render_buffer_valid,
                               const uint current_kernel,
                               const int pass_volume_majorant,
                               const int pass_volume_majorant_sample_count,
                               const int pass_stride)
{
    write_optical_depth(ist,
                        state,
                        render_buffer,
                        render_buffer_valid,
                        pass_volume_majorant,
                        pass_volume_majorant_sample_count,
                        pass_stride);

    if (!queue_kernel_index_valid(current_kernel)) {
        return;  // defensive: refuse the out-of-range counter write
    }
    int *base = &ist->queue_counter->num_queued[0];  // CONSTANT index (device rule)
    InterlockedAdd(base[current_kernel], -1);
    write_path_queued_kernel(ist, state, 0u);
}

/* =========================================================================== *
 *  integrator_path_cache_miss - FAITHFUL PORT (GPU branch)
 *  source: src/kernel/integrator/state_flow.h:100-105
 * =========================================================================== *
 * Cycles:
 *   // Queued kernel and counter is unmodified, so it will be re-executed.
 *   kernel_integrator_state.queue_counter->cache_miss = true;
 * `true` into the int member (state.h:102) is the value 1. The CPU branch
 * (:240-244) deliberately asserts instead - the reference for this write is
 * the documented constant, not a host call. */
void integrator_path_cache_miss(IntegratorStateGPU *ist, uint state)
{
    (void)state;  // the GPU branch does not touch the state itself
    ist->queue_counter->cache_miss = 1;
}
