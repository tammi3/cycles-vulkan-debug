// Measure float3* stride with dynamic index. Cycles packed_float3 = 12 bytes.
struct Push { float3* buf; uint* out; };

[[vk::push_constant]]
ConstantBuffer<Push> pc;

[shader("compute")]
[numthreads(4, 1, 1)]
void cast_test_main(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= 4u) return;
    pc.buf[i] = float3(float(i), 1, 2);       // dynamic index on float3*
    pc.out[i] = 1u;
}
