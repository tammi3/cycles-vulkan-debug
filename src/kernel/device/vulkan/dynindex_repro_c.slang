// dynindex_repro_c.slang : Stage 0b Form C -- the array declared as a POINTER
// MEMBER (`uint* words`), not a fixed array. `p->words[i]` loads the pointer
// (a constant-offset member load) then indexes the resulting flat `uint*`.
//
// Expected codegen: the index is applied via OpPtrAccessChain on a flat
// scalar `uint*` (the SAFE pattern), NOT OpAccessChain on an OpTypeArray.
// Runtime correctness of this form is already proven by the traverse probe
// (dynamic index on pointer-member arrays, sc->object_node[obj], 3555 correct
// instance hits) and by kdata_probe's raw view. This file exists to confirm
// the SPIR-V shape directly. NOTE: with the current repro HOST, `p` points at
// a flat {10..17} buffer, so `p->words` would be a garbage pointer -- this
// entry is compiled + disassembled only, not run.

struct Blob
{
    uint* words;   // POINTER member (device address), not a fixed array
};

struct ReproPush
{
    Blob* p;       // typed device pointer to the Blob
    uint* raw;     // data buffer (unused for the codegen check)
    uint* out;
};

[[vk::push_constant]]
ConstantBuffer<ReproPush> pc;

[shader("compute")]
[numthreads(1, 1, 1)]
void dynindex_repro_main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x != 0u)
        return;

    /* inline DYNAMIC index through a POINTER member. */
    for (uint i = 0u; i < 8u; i++)
        pc.out[i] = pc.p->words[i];

    /* raw-view control. */
    for (uint i = 0u; i < 8u; i++)
        pc.out[16u + i] = pc.raw[i];
}
