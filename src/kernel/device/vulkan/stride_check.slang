// stride_check.slang : M4e S1b-3 step 2 -- GPU SLANG STRIDE CHECK.
//
// Answers, empirically, the byte STRIDE a Slang accessor uses for each element
// type the integrator-state accessors will need, on THIS compiler + driver. The
// host allocates one 512-byte device buffer, pre-fills it with a canary, pushes
// 7 TYPED pointers (one per 64-byte region), dispatches 3 threads (so the
// element index k = tid.x is NON-constant), and reads the raw bytes back to see
// exactly where each value landed.
//
// Regions / types probed (each a typed pointer pushed by the host):
//   r0 base    0 : float3*     -- what stride does a Slang float3 use? 12 or 16?
//   r1 base   64 : float* (3*i)-- the PACKED (12-byte) packed_float3 read/write
//                                 pattern: element k at float index k*3.
//   r2 base  128 : float*      -- expect stride 4.
//   r3 base  192 : uint*       -- expect stride 4.
//   r4 base  256 : int*        -- expect stride 4.
//   r5 base  320 : float4*     -- expect stride 16.
//
// (No uint64_t* DATA-array region: a strided OpTypeInt-64 array needs the Int64
// capability, which SwiftShader's Subzero backend rejects (crash). It is not
// needed either -- the accessors read pointer MEMBERS from IntegratorStateGPU at
// CONSTANT offsets (BDA pointer loads, which DO run on SwiftShader), then index
// typed data; they never have a strided uint64 data array.)
//
// Every index below is a dynamic index on a FLAT typed pointer (a pointer
// member), which is the SAFE pattern (OpPtrAccessChain) -- no fixed-array-
// member-of-a-struct is involved, so the RX 570 dynindex miscompile does not
// apply and the measured stride is purely Slang's layout for T.
//
// Values written (x/y/z/w components) are distinct per region and per k so the
// host can locate each one and compute the actual stride.

struct StridePush
{
    float3* r0;       // region 0 base (float3 stride test)
    float* r1;        // region 1 base (packed3 via float, 3*i)
    float* r2;        // region 2 base (float stride test)
    uint* r3;         // region 3 base (uint stride test)
    int* r4;          // region 4 base (int stride test)
    float4* r5;       // region 5 base (float4 stride test)
};                    // 48 bytes

[[vk::push_constant]]
ConstantBuffer<StridePush> pc;

[shader("compute")]
[numthreads(3, 1, 1)]
void stride_check_main(uint3 tid : SV_DispatchThreadID)
{
    uint k = tid.x;          // 0,1,2 -- NON-constant element index
    if (k > 2u)
        return;

    // R0: float3* stride test. x=1000+k marks the element start.
    pc.r0[k] = float3(1000.0f + k, 2000.0f + k, 3000.0f + k);

    // R1: PACKED (12-byte) float3 via a float* base + 3*i (the correct
    //     packed_float3 accessor). Element k occupies float indices k*3..k*3+2.
    pc.r1[k * 3u + 0u] = 100.0f + k;
    pc.r1[k * 3u + 1u] = 200.0f + k;
    pc.r1[k * 3u + 2u] = 300.0f + k;

    // R2: float* stride test.
    pc.r2[k] = 10.0f + k;

    // R3: uint* stride test.
    pc.r3[k] = 7000u + k;

    // R4: int* stride test.
    pc.r4[k] = -100 - int(k);

    // R5: float4* stride test.
    pc.r5[k] = float4(50.0f + k, 60.0f + k, 70.0f + k, 80.0f + k);
}
