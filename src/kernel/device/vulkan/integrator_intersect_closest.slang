// SPDX-FileCopyrightText: 2011-2022 Blender Foundation
// SPDX-License-Identifier: Apache-2.0

/* integrator_intersect_closest.slang - the PRODUCTION entry point of the
 * DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST device kernel (M4f slice 6).
 *
 * Cycles source (verbatim contract):
 *   src/kernel/device/gpu/kernel.h:118-132:
 *
 *     ccl_gpu_kernel(GPU_KERNEL_BLOCK_NUM_THREADS, GPU_KERNEL_MAX_REGISTERS)
 *         ccl_gpu_kernel_signature(integrator_intersect_closest,
 *                                  const ccl_global int *path_index_array,
 *                                  ccl_global float *render_buffer,
 *                                  const int work_size)
 *     {
 *       const int global_index = ccl_gpu_global_id_x();
 *       if (ccl_gpu_kernel_within_bounds(global_index, work_size)) {
 *         const int state = (path_index_array) ? path_index_array[global_index] : global_index;
 *         ccl_gpu_kernel_call(integrator_intersect_closest(nullptr, state, render_buffer));
 *       }
 *     }
 *
 *   src/kernel/integrator/intersect_closest.h:351-446:
 *     - reads ray (P, D, tmin, tmax, time) and visibility from integrator state
 *     - calls scene_intersect(kg, &ray, visibility, &isect)
 *     - writes intersection result via integrator_state_write_isect(state, &isect)
 *     - calls integrator_intersect_next_kernel to schedule:
 *         hit  -> DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE (with shader_sort_key)
 *         miss -> DEVICE_KERNEL_INTEGRATOR_SHADE_BACKGROUND
 *
 * Mapping (AGENTS.md sections 4-5):
 *   - Push constants carry raw device addresses (BDA) for params, render_buffer,
 *     path_index_array, and TraverseHeader.
 *   - BVH2 traversal is performed by the verified traverse() implementation in
 *     traverse.slang (aligned nodes + triangles + instance push/pop).
 *   - Path state transitions go through integrator_path_next (film_queue.slang),
 *     which atomically updates the GPU queue counters (num_queued[INTERSECT_CLOSEST]--,
 *     num_queued[NEXT]++).
 *   - Pointer null checks become host-wired flags (has_path_index_array, has_th,
 *     has_tri_shader) per platform constraint 3 (avoid OpConvertPtrToU).
 */

#include "integrator_state.slang"
#include "film_queue.slang"
#include "traverse.slang"

/* DeviceKernel constants (src/kernel/types.h:1744-1752) */
static const uint DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST = 2u;
static const uint DEVICE_KERNEL_INTEGRATOR_SHADE_BACKGROUND = 3u;
static const uint DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE = 4u;

/* Push block: 56 bytes (< 128-byte spec minimum) */
struct QueueIntersectPush
{
    KernelParamsVulkan* params;               // offset  0
    float*              render_buffer;        // offset  8
    int*                path_index_array;     // offset 16
    TraverseHeader*     th;                   // offset 24
    uint*               tri_shader;           // offset 32
    int                 work_size;            // offset 40
    uint                has_path_index_array; // offset 44
    uint                has_th;               // offset 48
    uint                has_tri_shader;       // offset 52
};

[[vk::push_constant]]
ConstantBuffer<QueueIntersectPush> pc;

[shader("compute")]
[numthreads(64, 1, 1)]
void integrator_intersect_closest_main(uint3 tid : SV_DispatchThreadID)
{
    const int global_index = (int)tid.x;
    if (global_index >= pc.work_size) {
        return;
    }

    const int state = (pc.has_path_index_array != 0) ? pc.path_index_array[global_index] : global_index;
    IntegratorStateGPU* ist = &pc.params->integrator_state;

    /* 1. Read ray from integrator state */
    float3 P = read_ray_P(ist, (uint)state);
    float3 D = read_ray_D(ist, (uint)state);
    float tmin = read_ray_tmin(ist, (uint)state);
    float tmax = read_ray_tmax(ist, (uint)state);
    uint visibility = read_path_visibility(ist, (uint)state);

    /* 2. Traverse scene BVH2 */
    float hit_t = tmax;
    float hit_u = 0.0f;
    float hit_v = 0.0f;
    uint hit_prim = NO_PRIM;
    uint hit_object = NO_OBJECT;
    uint hit_type = 0;
    uint instance_visits = 0;

    if (tmax > 0.0f && pc.has_th != 0) {
        traverse(pc.th, P, D, tmin, tmax, visibility,
                 hit_t, hit_u, hit_v, hit_prim, hit_object, hit_type,
                 instance_visits);
    }

    const bool hit = (hit_prim != NO_PRIM);
    if (!hit) {
        hit_prim = NO_PRIM;
        hit_object = NO_OBJECT;
        hit_type = 0;
        hit_u = 0.0f;
        hit_v = 0.0f;
    }

    /* 3. Write intersection results into integrator state (state_util.h:179-184) */
    write_isect_t(ist, (uint)state, hit_t);
    write_isect_u(ist, (uint)state, hit_u);
    write_isect_v(ist, (uint)state, hit_v);
    write_isect_prim(ist, (uint)state, (int)hit_prim);
    write_isect_object(ist, (uint)state, (int)hit_object);
    write_isect_type(ist, (uint)state, (int)hit_type);

    /* 4. Queue next kernel (intersect_closest.h:250-295) */
    if (hit) {
        uint shader = 0;
        if ((hit_type & PRIMITIVE_TRIANGLE) != 0 && pc.has_tri_shader != 0) {
            shader = pc.tri_shader[hit_prim] & 0x00FFFFFFu; // SHADER_MASK
        }
        write_path_shader_sort_key(ist, (uint)state, shader);

        integrator_path_next(ist, (uint)state,
                             DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST,
                             DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE);
    }
    else {
        write_path_shader_sort_key(ist, (uint)state, 0);

        integrator_path_next(ist, (uint)state,
                             DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST,
                             DEVICE_KERNEL_INTEGRATOR_SHADE_BACKGROUND);
    }
}
