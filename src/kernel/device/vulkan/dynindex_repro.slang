// dynindex_repro.slang : MINIMAL, self-contained repro of the Slang
// dynamic-index miscompile confirmed by kdata_probe (M4e S1b-0/S1b-1).
//
// THE BUG
//   An inline DYNAMIC index on a fixed-size array member reached through a
//   DEVICE pointer -- `p->data.words[i]` with a NON-constant `i` -- is
//   miscompiled by slangc 2026.17-25-gc7954ebd1 (VulkanSDK 1.4.363.0). The
//   index is NOT applied correctly, so it reads back the WRONG elements.
//   The struct/array LAYOUT is byte-correct (a raw uint32 view of the same
//   bytes reads back exactly what the host wrote); only the codegen for that
//   ONE access form is wrong. Root cause is visible in the SPIR-V: the typed
//   device-pointer path compiles the index with OpAccessChain on a nested
//   `_natural`-struct-wrapped array and mangles the index; the base-pointer
//   and raw paths compile to OpPtrAccessChain on a flat `uint*` and are
//   correct. The exact distortion is context-dependent: over kdata_probe's
//   640-word blob every index returned words[0]; over this 8-word array the
//   index comes back roughly i>>2 (observed 10 10 10 10 11 11 11 11 for the
//   pattern 10..17). Both are the SAME miscompiled instruction.
//
// THIS FILE isolates the bug from all the KernelData baggage. One dispatch
// reads the same 8-word pattern {10,11,...,17} three ways, plus a constant
// index control:
//
//   (a) inline DYNAMIC    p->data.words[i]        -> WRONG  (always 10)
//   (c) base-ptr DYNAMIC  (&p->data.words[0])[i]  -> correct
//   (b) raw-view DYNAMIC  raw[i]                  -> correct (control)
//   const index           p->data.words[3]        -> correct (constant index works)
//
// (a) and (c) reach the SAME bytes through the SAME typed device pointer;
// only the access syntax differs, so any output difference is pure codegen.
//
// Compile:  slangc dynindex_repro.slang -target spirv -stage compute \
//             -entry dynindex_repro_main -o dynindex_repro.spv
// Run:      CYCLES_VULKAN_DYNINDEX_REPRO=1 (+ optional ..._SPV=<path>)
//           through the host probe kernels.cpp: vulkan_dynindex_repro().

struct Blob
{
    uint words[8];   // the fixed-size array member the bug lives on
};

struct Params
{
    Blob data;       // ...nested one struct deep, behind a device pointer
};

struct ReproPush
{
    Params* p;       // typed device pointer to the Params buffer (BDA)
    uint* raw;       // the SAME address viewed as a flat uint32 array
    uint* out;       // readback: [0..7]=a, [8..15]=c, [16..23]=b, [24]=const-a, [25]=const-b
};

[[vk::push_constant]]
ConstantBuffer<ReproPush> pc;

[shader("compute")]
[numthreads(1, 1, 1)]
void dynindex_repro_main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x != 0u)
        return;

    /* (a) inline DYNAMIC index through the typed device pointer.
     * EXPECT (bug): WRONG values -- the index is miscompiled (observed
     * 10 10 10 10 11 11 11 11 for the pattern 10..17), not 10 11 12 ... 17. */
    for (uint i = 0u; i < 8u; i++)
        pc.out[i] = pc.p->data.words[i];

    /* (c) base-pointer DYNAMIC index: take the flat base ONCE, index through
     * it. This is the sanctioned fix pattern. EXPECT: correct {10..17}. */
    uint* blob_base = &pc.p->data.words[0];
    for (uint i = 0u; i < 8u; i++)
        pc.out[8u + i] = blob_base[i];

    /* (b) raw-view DYNAMIC index (control). EXPECT: correct {10..17}. */
    for (uint i = 0u; i < 8u; i++)
        pc.out[16u + i] = pc.raw[i];

    /* CONSTANT index through the typed pointer. EXPECT: correct (== words[3] == 13),
     * proving only the DYNAMIC path is broken, not constant indexing. */
    pc.out[24] = pc.p->data.words[3];
    pc.out[25] = pc.raw[3];
}
