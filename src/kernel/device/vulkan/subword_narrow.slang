// subword_narrow.slang : M4e S1b-4 STAGE A probe, NARROW-store variants (module N).
//
// Purpose: test the OTHER sub-word candidate -- a TRUE 16-bit / 8-bit store through a
// PhysicalStorageBuffer pointer (uint16_t* / uint8_t*, element stride 2 / 1). This is
// what Cycles' SoA allocator actually lays out for uint16_t/uint8_t fields, so if a
// true narrow store is valid it is the simplest correct accessor.
//
//   mode 0 = true uint16 store, p16[t] = value   (needs SPIR-V Int16 = 22)
//   mode 1 = true uint8 store,  p8[t]  = value   (needs SPIR-V Int8  = 39)
//
// Unlike module W, THIS module emits OpCapability Int16 + Int8 (confirmed by
// disassembly). Those capabilities are only legal if the corresponding Vulkan feature
// (shaderInt16 / storageBuffer16BitAccess for Int16; shaderInt8 / storageBuffer8BitAccess
// for Int8) is ENABLED at device creation. device.cpp currently enables NEITHER, so on
// the unmodified build the validation layer is expected to flag a VUID -- which is the
// empirical citation for "which feature gates 16/8-bit access through BDA." Re-running
// after enabling the features should clear the VUID and leave the stores correct.
//
// Layout: p16 points at region A (uint16 elements, byte 0, stride 2), p8 at region B
// (uint8 elements, byte 1024, stride 1). Index t is dynamic on a FLAT typed pointer
// (a push-constant pointer member), the SAFE OpPtrAccessChain pattern.

struct PushN {
    uint16_t* p16;   // region A: uint16 elements (byte 0, stride 2)
    uint8_t* p8;     // region B: uint8 elements (byte 1024, stride 1)
    uint mode;       // 0 = uint16 true store, 1 = uint8 true store
};

[[vk::push_constant]]
ConstantBuffer<PushN> pc;

static const uint NN = 256;   // N threads, each writes element t = tid.x
static const uint MN = 64;    // repeated MN times (value depends on iter)

// Distinct value per (element, iteration), mirrored exactly by the host verifier.
uint v16(uint t, uint iter) { return (iter * 37u + t * 3u + 11u) & 0xFFFFu; }
uint v8(uint t, uint iter) { return (iter * 13u + t * 5u + 7u) & 0xFFu; }

[shader("compute")]
[numthreads(256, 1, 1)]
void narrow_main(uint3 tid : SV_DispatchThreadID)
{
    uint t = tid.x;          // 0..255, NON-constant element index
    if (t >= NN)
        return;

    if (pc.mode == 0u) {
        // True 16-bit store to element t (adjacent elements share a word, but a
        // hardware 2-byte store touches only this element -> never clobbers a neighbour).
        for (uint iter = 0u; iter < MN; iter++) {
            pc.p16[t] = uint16_t(v16(t, iter));
        }
    }
    else {
        // True 8-bit store to element t.
        for (uint iter = 0u; iter < MN; iter++) {
            pc.p8[t] = uint8_t(v8(t, iter));
        }
    }
}
