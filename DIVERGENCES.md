# DIVERGENCES.md — Vulkan/Cycles divergences

Every divergence from faithful Cycles logic gets an entry here (AGENTS.md §4):
what changed, why, the source citation of the original, the measured speedup on
the RX 570, and whether the divergence is result-preserving or result-changing.
Divergences sit behind a macro/specialization constant. **This file was created
during M4e S1b-6 (Task 2); divergences from earlier milestones have not been
inventoried yet — see "Open items" at the bottom.**

---

## Entry 1 — Sub-word state stores (uint16_t / uint8_t / bool members):
## atomic 32-bit word fallback instead of direct element stores

**Status:** decided by the user (M4e S1b-4, item 4). Recorded during M4e S1b-6.

**Source citation of the original:**
- Member types: `src/kernel/integrator/state_template.h:17,19,33,37` (uint16_t),
  `:39,43` (uint8_t); `shadow_state_template.h:15..33` (uint16_t), `:33,53`
  (uint8_t). The `bool` guiding flags use the same 1-byte layout.
- The store itself is Cycles' macro:
  `src/kernel/integrator/state.h:273-276` —
  `INTEGRATOR_STATE_WRITE(path, bounce, v)` ==
  `kernel_integrator_state.path.bounce[state] = v`, i.e. a **direct element
  store at the member's natural width** (16-bit or 8-bit).

**What changed:** `src/kernel/device/vulkan/integrator_state.slang` implements
sub-word writes as `load/store_u16`, `store_u8`, `store_bool` helpers
(`integrator_state.slang`, "Sub-word accessor layer" section):

- **Default mode (`VULKAN_SUBWORD_NARROW == 0`, the gate/production mode):**
  the SoA address is viewed as a `uint*` **word array** (2×uint16 or 4×uint8
  elements per 32-bit word). A write is two non-overlapping atomics on the
  containing word:
  `InterlockedAnd(word, ~mask); InterlockedOr(word, value << shift);`
  A read is one 32-bit word load + shift/mask. No Int16/Int8 capability.
- **Narrow mode (`VULKAN_SUBWORD_NARROW == 1`):** faithful shape — a true
  16/8-bit element store through `uint16_t*` / `uint8_t*` (emits
  `OpCapability Int16/Int8`).

The macro lives in `src/kernel/device/vulkan/vulkan_data.slang`
(`Subword16T` / `Subword8T` typedefs); the mirror member type and the
accessor helpers switch **in lockstep**, so both modes address the same host
allocation byte-for-byte.

**Why:** three reasons, in order:
1. **Driver safety on the target GPU.** The RX 570 driver
   (31.0.21925.1001) miscompiles `OpAccessChain` with a dynamic index on a
   fixed array inside a PhysicalStorageBuffer struct (AGENTS.md DEVICE RULE,
   found in M4e). Sub-word members stored as narrow typed arrays behind a
   device pointer are the same shape, so the default avoids them entirely.
2. **Portability.** SwiftShader crashes (0xC00000FD) on Int16/Int8/Int64
   kernels (AGENTS.md DEVICE RULE). The default mode needs none of them.
3. **Race-free shared words.** Adjacent states share one 32-bit word and are
   written by different threads; a read-modify-write of that word without
   atomics loses updates. Disjoint-bit atomics commute with a neighbour's, so
   the fallback is race-free by construction.

**Result-preserving or result-changing:** **result-preserving.** The final
bytes of the containing word are identical to a direct element store for
disjoint element writes (the atomics touch only our mask bits). Because the
bytes are identical, the CPU-comparison tolerances of the gates are untouched.

**Evidence (from the S1b-5 session, recorded in `lastsession.txt`):**
- Stage A probe `src/kernel/device/vulkan/subword_word.slang`: modes 0/1
  (atomic u16/u8) **PASS on RX 570 and SwiftShader**; modes 2/3 (non-atomic
  RMW) **negative DETECTED**.
- M4e S1b-6: the accessor surface compiles and validates
  (`slangc -target spirv` exit 0, `spirv-val --target-env vulkan1.2` exit 0,
  `OpAtomicAnd`/`OpAtomicOr` present in default mode, absent in narrow mode).

**Measured speedup:** **NOT YET MEASURED.** Per the S1b-6 task this is
deferred: the faithful (narrow) variant vs. the atomic fallback will be timed
on the RX 570 **after the M5 gate passes** (AGENTS.md §4: no optimizing before
M5 unless the user asks). This entry will be updated with both timings and the
same-scene/same-run-conditions numbers.

**Policy deviation to note:** AGENTS.md §4 item 3 says the *faithful* variant
is the default. Here the default is the byte-equivalent atomic fallback and
the faithful narrow variant is behind the macro — the inverse. Rationale:
reasons 1–2 above make the faithful form the riskier one on the target
hardware; the user decided this order (M4e S1b-4, item 4). Flagged here so the
user can re-decide after the post-M5 measurement.

---

## Open items

- Divergence entries for milestones before M4e S1b-6 (BVH traversal, probes,
  any M4a–M4d choices) have not been inventoried — needs a pass over the
  milestone records when context allows.
