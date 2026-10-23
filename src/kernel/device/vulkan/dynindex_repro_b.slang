// dynindex_repro_b.slang : Stage 0b Form B -- ONE struct level, NO extra
// `_natural` nesting. `struct Blob { uint words[8]; }` reached DIRECTLY
// through the device pointer: `p->words[i]`. Same push/out layout as
// dynindex_repro.slang so the existing host (vulkan_dynindex_repro) runs it
// unchanged (Blob == 32 bytes of uints == the flat {10..17} the host writes).
//
// Purpose: determine whether the miscompile needs the DOUBLE nesting
// (`Params{Blob{uint[8]}}`) or also hits a single struct level.

struct Blob
{
    uint words[8];   // fixed array, one struct level deep
};

struct ReproPush
{
    Blob* p;         // typed device pointer to the Blob (flat {10..17})
    uint* raw;       // SAME address viewed as a flat uint32 array
    uint* out;       // [0..7]=inline, [8..15]=base-ptr, [16..23]=raw, [24..25]=const
};

[[vk::push_constant]]
ConstantBuffer<ReproPush> pc;

[shader("compute")]
[numthreads(1, 1, 1)]
void dynindex_repro_main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x != 0u)
        return;

    /* inline DYNAMIC index, ONE struct level. */
    for (uint i = 0u; i < 8u; i++)
        pc.out[i] = pc.p->words[i];

    /* base-pointer DYNAMIC index (control). */
    uint* base = &pc.p->words[0];
    for (uint i = 0u; i < 8u; i++)
        pc.out[8u + i] = base[i];

    /* raw-view DYNAMIC index (control). */
    for (uint i = 0u; i < 8u; i++)
        pc.out[16u + i] = pc.raw[i];

    /* CONSTANT index through the typed pointer (control). */
    pc.out[24] = pc.p->words[3];
    pc.out[25] = pc.raw[3];
}
