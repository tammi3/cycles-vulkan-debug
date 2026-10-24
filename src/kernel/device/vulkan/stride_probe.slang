// SPDX-FileCopyrightText: 2011-2022 Blender Foundation
// SPDX-License-Identifier: Apache-2.0

/* Probe: does cast<float3[]> work on a flat uint64 device address, and what
 * stride does it use? Mirrors bvh2.slang's cast<T[]> pattern exactly. */

// A struct of 3 floats, exactly 12 bytes, to test struct-array stride too.
struct Packed3
{
    float x;
    float y;
    float z;
};

struct PushStride
{
    uint64_t buf_addr;   // device address of the host-visible buffer
    uint    which;       // 0 = float3[], 1 = Packed3[]
};

[[vk::push_constant]]
ConstantBuffer<PushStride> pc;

[shader("compute")]
[numthreads(1, 1, 1)]
void stride_main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x != 0u)
        return;

    uint64_t base = pc.buf_addr;

    if (pc.which == 0u)
    {
        let v = cast<float3[]>(base);
        for (uint i = 0u; i < 4u; i++)
            v[i] = float3(float(i), float(i) + 10.0, float(i) + 20.0);
    }
    else
    {
        let v = cast<Packed3[]>(base);
        for (uint i = 0u; i < 4u; i++)
            v[i] = Packed3(float(i), float(i) + 10.0, float(i) + 20.0);
    }
}
