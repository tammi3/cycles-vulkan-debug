// SPDX-FileCopyrightText: 2011-2022 Blender Foundation
// SPDX-License-Identifier: Apache-2.0

/* state_init_probe.slang : M4f SLICE 2 gate probe (host env
 * CYCLES_VULKAN_STATE_INIT_PROBE=1, SPV via
 * CYCLES_VULKAN_STATE_INIT_PROBE_SPV or ./state_init_probe.spv).
 *
 * WHAT IT PROVES: the ported state-initialization cluster (path_state.slang:
 * path_state_init_queues / path_state_init / path_state_init_integrator /
 * integrator_state_write_ray) writes EXACTLY what the REAL Cycles functions
 * write. Unlike the other probes this one has no per-case output buffer: the
 * writes into the SoA slices ARE the output, and the host compares the raw
 * slice bytes against a reference produced by calling the real Cycles
 * functions on a zeroed IntegratorStateCPU (state.h:227,234-240 - direct
 * member assignment on the host).
 *
 * 64 threads per workgroup, one case (= one SoA state index) per thread.
 *
 * neg_mode (gate-only input fault, NOT in Cycles; the ported functions
 * contain no fault switch) - one probe-side INPUT fault per pass so each
 * comparison group is proven live:
 *   1 -> sample fed to path_state_init_integrator is sample+1
 *        -> ONLY path.sample may differ;
 *   2 -> tile stride fed to path_state_init is stride+1
 *        -> ONLY path.render_pixel_index may differ;
 *   3 -> ray.time fed to integrator_state_write_ray is time+0.5
 *        -> ONLY ray.time may differ.
 * The clean pass must MATCH byte-for-byte (integers) / within 1e-5 relative
 * (floats) for every slice. */

#include "path_state.slang"  // ported functions + accessors

/* One test case. Layout locked against StateInitCase in kernels.cpp. */
struct StateInitCase
{
    int  sample;     // scheduled sample (the integrator_init_from_camera arg)
    uint x, y;       // pixel coordinates (path_state_init)
    uint rng_pixel;  // rng_pixel input (init_from_camera.h:93 produces it)
    float f[14];     // ray + throughput, see layout below
};                   // 16 + 56 = 72 bytes
/* f[] layout (host MUST match):
 *   [0..2]  ray.P      [3]  ray.dP     [4..6] ray.D    [7]  ray.dD
 *   [8]     ray.tmin   [9]  ray.tmax   [10]   ray.time  [11..13] throughput */

/* Gate params. Layout locked against StateInitParams in kernels.cpp. */
struct StateInitParams
{
    uint kernel_features;          // forced-on feature bits (see kernels.cpp)
    int  background_object_index;  // kernel_data.background.object_index
    int  background_volume_shader; // kernel_data.background.volume_shader
    int  tile_offset;              // KernelWorkTile.offset (types.h:1695)
    uint tile_stride;              // KernelWorkTile.stride
    int  neg_mode;                 // probe fault switch (see header)
    uint pad0, pad1;
};                                 // 32 bytes

/* Push block: 3 device pointers + count = 32 bytes. The `ist` pointer
 * targets the host-filled IntegratorStateGPU mirror (slice addresses). */
struct StateInitPush
{
    StateInitCase*      cases;
    IntegratorStateGPU* ist;
    StateInitParams*    params;
    uint                num_cases;
};                        // 32 bytes

[[vk::push_constant]]
ConstantBuffer<StateInitPush> pc;

[shader("compute")]
[numthreads(64, 1, 1)]
void state_init_main(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= pc.num_cases)
        return;

    StateInitCase c = pc.cases[i];
    StateInitParams p = *(pc.params);

    /* ---- probe-side input faults (header): the port has NO switch ---- */
    int sample_arg = c.sample;
    if (p.neg_mode == 1)
        sample_arg += 1;

    uint stride_arg = p.tile_stride;
    if (p.neg_mode == 2)
        stride_arg += 1u;

    float ray_time = c.f[10];
    if (p.neg_mode == 3)
        ray_time += 0.5f;

    /* ---- the three ported calls, in Cycles' call order ----------------
     * (integrator_init_from_camera calls path_state_init at :81, then
     * path_state_init_integrator at :121/:130 and integrator_state_write_ray
     * at :116; the members written are disjoint, but the order is kept.) */
    path_state_init(pc.ist, i, p.tile_offset, stride_arg, (int)c.x, (int)c.y);

    path_state_init_integrator(pc.ist, i,
                               p.kernel_features,
                               p.background_object_index,
                               p.background_volume_shader,
                               sample_arg,
                               c.rng_pixel,
                               float3(c.f[11], c.f[12], c.f[13]));

    integrator_state_write_ray(pc.ist, i,
                               float3(c.f[0], c.f[1], c.f[2]), c.f[3],
                               float3(c.f[4], c.f[5], c.f[6]), c.f[7],
                               c.f[8], c.f[9], ray_time);
}
