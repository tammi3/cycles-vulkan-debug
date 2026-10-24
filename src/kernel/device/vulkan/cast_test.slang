// Does a struct that CONTAINS uint64_t members (never read) still force
// OpCapability Int64 when we only load a typed-pointer member through it?
// This decides whether the retyped-mirror gate probe can run on SwiftShader
// (which crashes on OpCapability Int64).
struct Path
{
    uint64_t untouched;   // never accessed by this kernel
    float3*  throughput;  // the only member we touch
};

struct Params { Path path; };
struct Push { Params* params; uint* out; };

[[vk::push_constant]]
ConstantBuffer<Push> pc;

[numthreads(4, 1, 1)]
[shader("compute")]
void cast_test_main(uint3 tid : SV_DispatchThreadID)
{
    uint s = tid.x;
    if (s >= 4u) return;
    pc.params->path.throughput[s] = float3(float(s), 1, 2);
    pc.out[s] = 1u;
}
