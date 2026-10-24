// state_access_probe.slang : M4e S1b-6 gate shader (Tasks 1/3/4).
//
// One kernel that exercises EVERY accessor in integrator_state.slang, so that
// slangc emits and spirv-val validates all of them, and so the host gate can
// verify byte-for-byte that each write landed at its exact SoA offset:
//   - state index = tid.x (NON-constant, 8 states; adjacent states share one
//     32-bit word for uint16/uint8 members and are written by DIFFERENT
//     threads of the same workgroup);
//   - AoS members written at NON-zero slots (1, 2, 31 for the gate; 1, 2 for
//     the wrong-stride negative);
//   - values are a pure function of (id, slot, state): the host recomputes
//     them exactly and compares raw bytes;
//   - test_mode (push constant) injects the four Task-4 negatives:
//       0 = clean gate, 1 = 16-byte stride on packed3 (path.throughput),
//       2 = plain 4-byte store into a uint16 field (path.bounce),
//       3 = wrong AoS stride on volume_stack (element index i*2 = 32 bytes
//           instead of i = 16 bytes),
//       4 = non-atomic read-modify-write on adjacent uint16 states.
//     Each fault must be DETECTED by the host verifier; mode 0 must pass.
//     NOTE: modes 2 and 4 spell their fault through Subword16T* and are only
//     valid in the DEFAULT word-view mode (VULKAN_SUBWORD_NARROW=0, the gate
//     configuration). In narrow mode p[s] degenerates to a correct 16-bit
//     store, so these two negatives would NOT fire there.
//
// Bitcast used by mode 1 is the spirv_asm form proven in M4e S1b-6:
//   float4* f4 = spirv_asm { result:$$float4* = OpBitcast $addr; };

#include "integrator_state.slang"

static const uint STATE_COUNT = 8u;

/* Known-value formula: injective in (id, slot) at bits 8..19, state at bits
 * 0..7, and a 0x40000000 high bit so nothing looks like a canary. Low byte =
 * state (what a uint8 member keeps), low 16 bits = (id<<8)|state (uint16). */
uint gbits(uint id, uint i, uint s)
{
    return 0x40000000u + (i << 20) + (id << 8) + s;
}

float3 gbits3(uint id, uint i, uint s)
{
    return float3(asfloat(gbits(id, i, s)),
                  asfloat(gbits(id + 64u, i, s)),
                  asfloat(gbits(id + 128u, i, s)));
}

struct PushGate
{
    IntegratorStateGPU* ist;
    uint* out;
    uint test_mode;
};

[[vk::push_constant]]
ConstantBuffer<PushGate> pc;

[shader("compute")]
[numthreads(8, 1, 1)]
void state_access_main(uint3 tid : SV_DispatchThreadID)
{
    uint s = tid.x;
    if (s >= STATE_COUNT)
        return;

    IntegratorStateGPU* ist = pc.ist;
    uint mode = pc.test_mode;
    uint acc = 0u;

    /* ================= path (ids 1..17) ================= */
    write_path_render_pixel_index(ist, s, gbits(1u, 0u, s));
    acc ^= read_path_render_pixel_index(ist, s);

    write_path_sample(ist, s, gbits(2u, 0u, s));
    acc ^= read_path_sample(ist, s);

    /* bounce (uint16): negative mode 2 writes a plain 4-byte word at index s
     * (byte 4s) instead of the atomic sub-word store (byte 2s). */
    if (mode == 2u) {
        Subword16T* p = ist->path.bounce;
        p[s] = gbits(3u, 0u, s);          // FAULT: uint store into uint16 field
    }
    else {
        write_path_bounce(ist, s, gbits(3u, 0u, s));
    }
    acc ^= read_path_bounce(ist, s);

    write_path_transparent_bounce(ist, s, gbits(4u, 0u, s));
    acc ^= read_path_transparent_bounce(ist, s);

    write_path_queued_kernel(ist, s, gbits(5u, 0u, s));
    acc ^= read_path_queued_kernel(ist, s);

    /* rng_offset (uint16): negative mode 4 = non-atomic read-modify-write of
     * the shared word, repeated so a lost update is deterministic (same form
     * as the proven Stage A negative, subword_word.slang modes 2/3). */
    if (mode == 4u) {
        Subword16T* p = ist->path.rng_offset;
        for (uint iter = 0u; iter < 16u; iter++) {
            uint v = (gbits(6u, 0u, s) + iter * 0x9E37u) & 0xFFFFu;
            uint k = s >> 1;
            uint shift = (s & 1u) * 16u;
            uint w = p[k];                                  // non-atomic load
            p[k] = (w & ~(0xFFFFu << shift)) | (v << shift); // RACE: lost update
        }
    }
    else {
        write_path_rng_offset(ist, s, gbits(6u, 0u, s));
    }
    acc ^= read_path_rng_offset(ist, s);

    write_path_visibility(ist, s, gbits(7u, 0u, s));
    acc ^= read_path_visibility(ist, s);

    write_path_flag(ist, s, gbits(8u, 0u, s));
    acc ^= read_path_flag(ist, s);

    write_path_mnee(ist, s, gbits(9u, 0u, s));
    acc ^= read_path_mnee(ist, s);

    write_path_mis_ray_pdf(ist, s, asfloat(gbits(10u, 0u, s)));
    acc ^= asuint(read_path_mis_ray_pdf(ist, s));

    write_path_min_ray_pdf(ist, s, asfloat(gbits(11u, 0u, s)));
    acc ^= asuint(read_path_min_ray_pdf(ist, s));

    write_path_continuation_probability(ist, s, asfloat(gbits(12u, 0u, s)));
    acc ^= asuint(read_path_continuation_probability(ist, s));

    write_path_optical_depth(ist, s, asfloat(gbits(13u, 0u, s)));
    acc ^= asuint(read_path_optical_depth(ist, s));

    write_path_shader_sort_key(ist, s, gbits(14u, 0u, s));
    acc ^= read_path_shader_sort_key(ist, s);

    /* throughput (PackedSpectrum, stride 12): negative mode 1 writes through
     * a float4* view of the SAME address -> elements land at byte 16*s
     * instead of 12*s, and bytes 12s+12..15 clobber the next element. */
    if (mode == 1u) {
        uint64_t a = uint64_t(ist->path.throughput);
        float4* f4 = spirv_asm {
            result: $$float4* = OpBitcast $a;
        };
        f4[s] = float4(gbits3(15u, 0u, s), 0.0);            // FAULT: stride 16
    }
    else {
        write_path_throughput(ist, s, gbits3(15u, 0u, s));
    }
    acc ^= asuint(read_path_throughput(ist, s).x);

    write_path_mis_origin_n(ist, s, gbits3(16u, 0u, s));
    acc ^= asuint(read_path_mis_origin_n(ist, s).x);

    write_path_pass_diffuse_weight(ist, s, gbits3(17u, 0u, s));
    acc ^= asuint(read_path_pass_diffuse_weight(ist, s).x);

    /* ================= ray (ids 18..24) ================= */
    write_ray_P(ist, s, gbits3(18u, 0u, s));
    acc ^= asuint(read_ray_P(ist, s).x);
    write_ray_D(ist, s, gbits3(19u, 0u, s));
    acc ^= asuint(read_ray_D(ist, s).x);
    write_ray_dP(ist, s, asfloat(gbits(20u, 0u, s)));
    acc ^= asuint(read_ray_dP(ist, s));
    write_ray_dD(ist, s, asfloat(gbits(21u, 0u, s)));
    acc ^= asuint(read_ray_dD(ist, s));
    write_ray_tmin(ist, s, asfloat(gbits(22u, 0u, s)));
    acc ^= asuint(read_ray_tmin(ist, s));
    write_ray_tmax(ist, s, asfloat(gbits(23u, 0u, s)));
    acc ^= asuint(read_ray_tmax(ist, s));
    write_ray_time(ist, s, asfloat(gbits(24u, 0u, s)));
    acc ^= asuint(read_ray_time(ist, s));

    /* ================= isect (ids 25..28) ================= */
    write_isect_t(ist, s, asfloat(gbits(25u, 0u, s)));
    acc ^= asuint(read_isect_t(ist, s));
    write_isect_prim(ist, s, -int(gbits(26u, 0u, s)));
    acc ^= uint(read_isect_prim(ist, s));
    write_isect_object(ist, s, -int(gbits(27u, 0u, s)));
    acc ^= uint(read_isect_object(ist, s));
    write_isect_type(ist, s, -int(gbits(28u, 0u, s)));
    acc ^= uint(read_isect_type(ist, s));

    /* ============ volume_stack AoS (ids 29..30, NON-zero slots) ============
     * Gate writes slots 1, 2, 31 (slot 31 = last element; a stride/size bug
     * that overruns it is caught by the canary after the array).
     * Negative mode 3 writes slot i at element i*2 (byte 32i, not 16i). */
    if (mode == 3u) {
        IntegratorStateVolumeStackElement* base = &ist->volume_stack.slot[0];
        int* bad = base[s * 2u + 2u].object;                // FAULT: stride 32
        bad[s] = -int(gbits(29u, 1u, s));
        int* bad2 = base[s * 2u + 2u].shader;
        bad2[s] = -int(gbits(30u, 2u, s));
    }
    else {
        write_volume_stack_object(ist, 1u, s, -int(gbits(29u, 1u, s)));
        write_volume_stack_shader(ist, 2u, s, -int(gbits(30u, 2u, s)));
        write_volume_stack_object(ist, 31u, s, -int(gbits(29u, 31u, s)));
        write_volume_stack_shader(ist, 31u, s, -int(gbits(30u, 31u, s)));
    }
    acc ^= uint(read_volume_stack_object(ist, 1u, s));
    acc ^= uint(read_volume_stack_shader(ist, 2u, s));
    acc ^= uint(read_volume_stack_object(ist, 31u, s));

    /* ================= shadow_path (ids 31..35) ================= */
    write_shadow_path_render_pixel_index(ist, s, gbits(31u, 0u, s));
    acc ^= read_shadow_path_render_pixel_index(ist, s);
    write_shadow_path_queued_kernel(ist, s, gbits(32u, 0u, s));
    acc ^= read_shadow_path_queued_kernel(ist, s);
    write_shadow_path_flag(ist, s, gbits(33u, 0u, s));
    acc ^= read_shadow_path_flag(ist, s);
    write_shadow_path_lightgroup(ist, s, gbits(34u, 0u, s));
    acc ^= read_shadow_path_lightgroup(ist, s);
    write_shadow_path_throughput(ist, s, gbits3(35u, 0u, s));
    acc ^= asuint(read_shadow_path_throughput(ist, s).x);

    /* ================= shadow_ray (ids 36..38) ================= */
    write_shadow_ray_P(ist, s, gbits3(36u, 0u, s));
    acc ^= asuint(read_shadow_ray_P(ist, s).x);
    write_shadow_ray_D(ist, s, gbits3(37u, 0u, s));
    acc ^= asuint(read_shadow_ray_D(ist, s).x);
    write_shadow_ray_tmin(ist, s, asfloat(gbits(38u, 0u, s)));
    acc ^= asuint(read_shadow_ray_tmin(ist, s));

    /* ============ shadow_isect AoS (ids 39..40, NON-zero slots) ============ */
    write_shadow_isect_t(ist, 3u, s, asfloat(gbits(39u, 3u, s)));
    write_shadow_isect_object(ist, 2u, s, -int(gbits(40u, 2u, s)));
    acc ^= asuint(read_shadow_isect_t(ist, 3u, s));
    acc ^= uint(read_shadow_isect_object(ist, 2u, s));

    /* ====== shadow_volume_stack AoS (ids 41..42, NON-zero slots) ====== */
    write_shadow_volume_stack_object(ist, 1u, s, -int(gbits(41u, 1u, s)));
    write_shadow_volume_stack_shader(ist, 3u, s, -int(gbits(42u, 3u, s)));
    acc ^= uint(read_shadow_volume_stack_object(ist, 1u, s));
    acc ^= uint(read_shadow_volume_stack_shader(ist, 3u, s));

    /* Readback accumulator: proves the dispatch ran (host checks out[s] != 0
     * pattern) and exercises every accessor's read path in the SPIR-V. */
    pc.out[s] = acc;
}
