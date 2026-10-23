// subword_word.slang : M4e S1b-4 STAGE A probe, WORD-based variants (module W).
//
// Purpose: decide how Cycles' sub-word SoA fields (uint16_t bounce/queued_kernel/
// rng_offset, uint8_t visibility/mnee, bool guiding flags -- element size 2 or 1,
// see the S1b-3 stride table) should be WRITTEN from a GPU kernel. Adjacent state
// elements share one 32-bit word (2 x uint16, or 4 x uint8 per word), so a store to
// element t must not clobber its neighbour in the same word. This module tests the
// two candidates that do NOT need the Int16/Int8 SPIR-V capability:
//
//   mode 0 = atomic AND/OR-on-containing-word fallback, uint16 elements.
//   mode 1 = atomic AND/OR-on-containing-word fallback, uint8 elements.
//   mode 2 = NEGATIVE: plain non-atomic read-modify-write on the word, uint16.
//   mode 3 = NEGATIVE: plain non-atomic read-modify-write on the word, uint8.
//
// This file uses ONLY uint32 + OpAtomicAnd/OpAtomicOr + a non-atomic load/store. It
// therefore emits NO Int16/Int8 capability (confirmed by disassembly), so it runs on
// ANY BDA device -- including SwiftShader, which lacks Int64 and would crash on a
// narrow-type module. That is exactly why the atomic fallback is worth measuring: if
// it is correct here with no 8/16-bit feature enabled, it is feature-independent.
//
// The negative (mode 2/3) writes a DIFFERENT value every iteration (value depends on
// `iter`). Two adjacent threads in the same word both load the SAME stale word, each
// masks in its own slot, and both store -- one clobbers the other's update. After M
// iterations a slot holds a STALE value instead of the last one -> lost update, which
// the host detects by comparing each element to its expected last value.
//
// Layout (region A = uint16 elements at byte 0, region B = uint8 elements at byte
// 1024, both inside one host-allocated BDA buffer). wordA[k] holds uint16 elements
// 2k,2k+1; wordB[k] holds uint8 elements 4k..4k+3. Indices are dynamic on FLAT uint*
// views (OpPtrAccessChain), the SAFE pattern -- no fixed-array-in-struct, so the RX
// 570 dynindex miscompile does not apply.

struct PushW {
    uint* wordA;   // region A: uint32 words covering the uint16 elements (byte 0)
    uint* wordB;   // region B: uint32 words covering the uint8 elements (byte 1024)
    uint mode;     // 0/2 = uint16 (atomic/RMW), 1/3 = uint8 (atomic/RMW)
};

[[vk::push_constant]]
ConstantBuffer<PushW> pc;

static const uint NW = 256;   // N threads, each writes element t = tid.x
static const uint MW = 64;    // repeated MW times so an RMW race manifests

// Distinct value per (element, iteration), mirrored exactly by the host verifier.
// The LAST iteration's value (iter = MW-1) is what a correct store leaves behind.
uint v16(uint t, uint iter) { return (iter * 37u + t * 3u + 11u) & 0xFFFFu; }
uint v8(uint t, uint iter) { return (iter * 13u + t * 5u + 7u) & 0xFFu; }

[shader("compute")]
[numthreads(256, 1, 1)]
void word_main(uint3 tid : SV_DispatchThreadID)
{
    uint t = tid.x;          // 0..255, NON-constant element index
    if (t >= NW)
        return;

    uint mode = pc.mode;

    if (mode == 0u || mode == 2u) {
        // ---- uint16 elements, packed 2-per-word in region A ----
        uint k = t >> 1;              // containing word
        uint shift = (t & 1u) * 16u;  // slot within the word (low/high 16 bits)
        uint mask = ~(0xFFFFu << shift);
        for (uint iter = 0u; iter < MW; iter++) {
            uint slot = (v16(t, iter) & 0xFFFFu) << shift;
            if (mode == 0u) {
                // Atomic fallback: clear our 16 bits, then set them. Two atomic ops
                // on disjoint bit ranges -> race-free with the neighbour thread.
                InterlockedAnd(pc.wordA[k], int(mask));
                InterlockedOr(pc.wordA[k], int(slot));
            }
            else {
                // NEGATIVE: non-atomic read-modify-write on the whole word.
                uint w = pc.wordA[k];                 // load the shared word
                pc.wordA[k] = (w & mask) | slot;      // store it back (RACE w/ neighbour)
            }
        }
    }
    else {
        // ---- uint8 elements, packed 4-per-word in region B ----
        uint k = t >> 2;              // containing word
        uint shift = (t & 3u) * 8u;   // slot within the word (one of 4 bytes)
        uint mask = ~(0xFFu << shift);
        for (uint iter = 0u; iter < MW; iter++) {
            uint slot = (v8(t, iter) & 0xFFu) << shift;
            if (mode == 1u) {
                InterlockedAnd(pc.wordB[k], int(mask));
                InterlockedOr(pc.wordB[k], int(slot));
            }
            else {
                uint w = pc.wordB[k];
                pc.wordB[k] = (w & mask) | slot;
            }
        }
    }
}
