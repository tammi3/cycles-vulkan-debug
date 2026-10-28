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

**Narrow-mode status (pre-flight, 2026-10-27):** the faithful variant cannot
currently run on the RX 570 at all. The driver (amdvlk 31.0.21925.1001)
sizes the `OpLoad` of a pointer-typed struct member by the **pointee** element
width instead of 8 bytes: `uint32_t*` members load 4 bytes (correct here,
addresses < 4 GB), `uint16_t*` members load 2, `uint8_t*` members load 1 —
so every sub-word member loads a truncated pointer and narrow mode 0 fails
with exactly 64 payload mismatches (8 sub-word slices × 8 states). Root cause
proven by probes v1–v5; a **workaround is proven end-to-end by probe v6**
(rebuild the pointer from a correctly-loading `uint32_t*` member + byte
offset, convert with `OpConvertUToPtr` in a `spirv_asm` block — genuine 16-bit
load and store verified, canaries intact, 0 VUID, layer active; details and
raw output in `lastsession.txt`). Consequences, recorded as observations:
- The planned narrow-vs-atomic timing comparison for this entry is
  **impossible on this hardware** until the workaround is applied to
  `integrator_state.slang`.
- SwiftShader cannot run narrow mode at all (no `shaderInt16`/`shaderInt8`
  → VUID + `UNSUPPORTED capability 22/39`), so word-view mode is its
  configuration; narrow validation is RX 570-only.
- The default (atomic word-view) mode is unaffected: it passes on both
  devices and needs none of this.

**Policy deviation to note:** AGENTS.md §4 item 3 says the *faithful* variant
is the default. Here the default is the byte-equivalent atomic fallback and
the faithful narrow variant is behind the macro — the inverse. Rationale:
reasons 1–2 above make the faithful form the riskier one on the target
hardware; the user decided this order (M4e S1b-4, item 4). Flagged here so the
user can re-decide after the post-M5 measurement.

---

## PROPOSED entries — inventory presented 2026-10-24, NOT YET APPROVED

**Nothing in this section is an active divergence decision.** During M4e
S1b-7 a divergence inventory (16 candidates + a stub list + 6 contradictions)
was presented to the user for review; no approval was recorded, and only the
10 candidate titles below survived on disk (in `lastsession.txt`). They are
written here so the inventory is not lost, clearly marked *proposed*. Each one
still needs the full entry treatment required by AGENTS.md §4 (source citation
of the original, what changed and why, measured RX 570 speedup,
result-preserving vs result-changing) **and the user's approval** before it
becomes an Entry.

Proposed candidates (titles as recorded on 2026-10-24):

1. Traverse safety instrumentation
2. 0xFFFFFFFF visibility mask (Q5)
3. Spike camera in gate
4. Missing `TABULATED_SOBOL` branch
5. Missing `dP/dD`
6. Dynamic-index driver workaround
7. `num_concurrent_states` formula
8. `mem_copy_from` stride heuristic
9. Stale `path_trace` prototype
10. `type==0` miss assertion

The remaining 6 candidates, the stub list and the 6 contradictions from the
same presentation were never written to disk — recover them from the user or
re-derive from the milestone records before this section can be completed.

---

## Open items

- Divergence entries for milestones before M4e S1b-6 (BVH traversal, probes,
  any M4a–M4d choices) have not been inventoried — needs a pass over the
  milestone records when context allows.
