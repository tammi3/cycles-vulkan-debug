# AGENTS.md — Vulkan backend for Cycles (cycles-vulkan-debug)

These rules do not change between sessions. The current milestone, the current task
and the state of the last session do NOT live here: they live in the task prompt and in
`lastsession.txt`.

Project: a Vulkan compute backend for Cycles (Blender), written in Slang -> SPIR-V,
target GPU is an AMD RX 570 4GB. Windows, MSVC, CMake, Vulkan SDK, Slang compiler.

- Working repo (all edits happen here): `C:\dev\cycles-vulkan-debug`
- Reference repo (READ ONLY, where M0-M2 spikes started): `C:\dev\cycles-vulkan`
- High-level context: https://github.com/tammi3/Vulkan-backend-Blender-CyclesX
- `agentreference/` holds files the user copied for you to use as references
  (for example `bvh_traverse.slang`). Read only. Never edit them.

---

## 0. Start of every session

1. Re-read this file.
2. Read `lastsession.txt` (the summary of the previous session) in the repo root.
3. Read the task prompt's gate definition. If the prompt has no gate, ask for one.

You have no memory of earlier sessions. Do not rely on any.

## 1. Who you are working with (and the style that follows)

The user knows GPU theory, shaders and path tracers, but is NEW to writing C++ and to
building it. She wants to LEARN from this repo, and others may read it later.

- Comment code heavily: say WHAT each block does and WHY. Explain unusual C++ or
  Slang constructs the first time they appear.
- Explain your decisions in plain language in your replies.

## 2. Which files you may edit

- Anything under `C:\dev\cycles-vulkan` (the reference repo)
- The verification harness: tests, tolerances and reference images (see section 6)
- Any file outside `src/` unless the task says so

**Edit only what the task lists.** A task prompt may list files as editable for that
task only. Only those files may be touched, and all other rules still apply.

If a task seems to require editing a frozen or unlisted file, STOP and report what is
blocking you. Do not edit it "just a little".


## 3. Task types

Every task is one of these. The task prompt says which.

**A. Porting task** (Cycles function -> Slang), for example M4c onward.
- One Cycles function, or one small cluster of functions it calls, per task.
- Output only the changed function(s), each as its own diff block labelled with the
  enclosing function name. Do not rewrite whole files.
- Do not port, refactor, rename or "clean up" neighbouring code.
- Scope is the milestone's scope. For M4 only the `LABEL_DIFFUSE` closure path is in
  scope. Every other closure type returns a stub (`LABEL_NONE` / no-op) with a comment
  `// DEFERRED: <closure name>`. Do not implement features beyond the milestone.
- Not more than one Cycles function plus one Slang skeleton per prompt. If the task
  needs more context, say so and propose how to split it.

**B. ABI / host-binding task** (layouts, pipeline, descriptors, push constants).
- May touch the files the task lists, across several files.
- The gate is a probe: the shader reads values through the new layout and the host
  compares them with what it sent (MATCH / MISMATCH per value).
- Do not change rendering logic.

## 4. Translation fidelity rules (Cycles -> Slang)

- **Faithful translation, not improvement.** Port the Cycles logic as written, even if it
  looks redundant or awkward. Cycles is the reference; divergence is a correctness bug.
  Cite the source: `// source: <cycles path>:<line>` on every non-obvious fact. NOTE on this specific rule,
  if the file can be optimized in Vulkan at the expense of MINOR divergence in faithfulness, you have permission to do so,
  HOWEVER, this must be kept to a minimum, this exception is only made due to the philosophy of making old hardware run properly,
  therefore, if it won't be faster on older hardware, do NOT EDIT ANYTHING unnecessary from Cycles logic while doing the translation.

DIVERGENCE POLICY (applies to the exception above)
1. Port faithfully first. A kernel must pass its gate in faithful form before any divergence is added.
2. A divergence is allowed only if measured faster on the RX 570. Report the timings (faithful vs divergent, same scene, same run conditions). "Expected to be faster" is not enough.
3. Every divergence sits behind a specialization constant or macro, with faithful as the default, and gets an entry in DIVERGENCES.md: what changed, why, source citation of the original, measured speedup, and whether it is result-preserving or result-changing.
4. Gates compare FAITHFUL mode against Cycles CPU at the strict tolerance. Result-changing divergences are compared against faithful GPU mode at a tolerance the user chooses. Never loosen the CPU-comparison tolerance to accommodate a divergence.
5. Do not start optimizing before the M5 gate passes, unless the user asks.


- **No virtual dispatch.** Cycles closure arrays and BSDF calls become an explicit
  `switch` on closure type. No function pointers, no interface types.
- **No recursion.** The call graph must be straight-line.
- **No dynamic allocation.** No growable containers, no variable-length arrays. Buffers
  are allocated by the host; sizes are compile-time constants or passed as counts.
- **Pointers.** Raw pointers that come from buffer device addresses (64-bit BDA) are
  allowed and expected, as in `agentreference/bvh_traverse.slang`. Do NOT create pointers
  to function-local memory: use `out` / `inout` parameters there.
- **Address spaces.** Map Cycles `ccl_private` -> function-local or `inout`,
  `ccl_global` -> data reached through a device pointer, `ccl_constant` -> push constant
  or constant data. Do not leave the storage class to be guessed.
- **Layouts come from the real C++.** Never assume a layout (no "float3 is 16 bytes in
  std430" assumptions). For every struct shared with the host: take the layout from the
  real C++ header, state each field's offset and size in a comment above the Slang struct,
  and lock it with `static_assert(offsetof(...))` on the host. Prefer scalar members over
  `float3` in shared structs so C++ and Slang cannot disagree on padding.
- **Opaque blobs.** For big nested host structs such as `KernelData`, do not hand-mirror
  every field. Declare an opaque block of the right size in words and lock only the
  offsets of the slots you actually read.
- **Deterministic RNG only.** Use the seeded sampler state passed into the kernel. No
  unseeded random function. If the Cycles sampler is not ported yet, use a placeholder that
  is a pure function of (pixel, sample_index, dimension) and mark it
  `// TODO: swap for ported sampler`.
- **Synchronization.** Ordering between dispatches is done on the host with
  `vkCmdPipelineBarrier` (or equivalent), never assumed. For barriers inside a shader, do
  not guess a function name: flag it `UNVERIFIED:` and ask.
- **Raw device pointers have no bounds checking.** Check every array index against its
  count before use, cap every loop, and write an error code instead of reading out of
  range (see the M2c spike for the pattern). A hung shader can reset the driver.

## 5. No inventing

- Every layout, constant, enum value or struct member must come from a file you have
  opened. Cite it: `// source: path:line`.
- If you are not sure a Slang or Vulkan function exists (a specific `normalize`, `dot`,
  `copysign`, a sampling helper, a barrier), do NOT invent a name. Write `UNVERIFIED:` and
  let the human confirm or provide the signature. Try compiling a tiny test with `slangc`
  when that is cheap.
- If the Cycles code uses a feature with no clear Slang equivalent (subsurface
  scattering, complex IES light sampling, hair BSDF, volume integration), STOP and ask. A
  plausible approximation is worse than a missing feature.
- If something you need is missing or ambiguous, STOP and ask. Do not guess.
- Do not write comments saying "validated", "verified" or "tested" unless a test actually
  ran in this session and you have shown its output.

## 6. Verification gates

- **Compile before declaring done.** Every Slang output must pass `slangc -target spirv`
  and then `spirv-val`. Show the commands and their output. If it fails, iterate; never
  report a "working" result on a failing build.
- **A passing build is not correctness.** Each task states its behavioural gate at the
  start. Current milestone gates:
  - M4: Lambertian sphere, 16 spp, compared pixel by pixel against the Cycles CPU render.
    Noise-only differences are acceptable; systematic bias is a failure.
  - M5: Cornell box with one area light, low resolution; shadow regions and light falloff
    compared against the Cycles CPU render.
- **If the gate is not met, report the failure with the diff and stop.** Do not declare
  the milestone complete.
- **Never modify the verification harness.** If the harness fails, the harness is right
  and the code is wrong. Editing tests, tolerances or reference images to make a gate pass
  is a hard violation.
- **Check every `VkResult`** in new Vulkan code. The validation layer
  (`VK_LAYER_KHRONOS_validation`) must report zero errors on gate runs, and the output must
  show it was actually active.

## 7. Build and run reference

- Build: `cmake --build build --config Release --target cycles -- /m`
- Run (PowerShell example):
  `$env:CYCLES_VULKAN_ALLOW_NO_KERNELS='1'; build\bin\Release\cycles.exe --device VULKAN --log-level info --samples 1 --background instance2.xml`
- Logging macros are `LOG_ERROR`, `LOG_WARNING`, `LOG_INFO` (see `util/log.h`). The
  `LOG(ERROR)` / `LOG(WARNING)` forms do NOT compile in this codebase.
- If you must compile a file standalone, the real `cl.exe` flags can be captured with
  `cmake --build build --config Release -- /clp:ShowCommandLine` (it needs `/std:c++20`
  and the `CCL_NAMESPACE_BEGIN` defines).
- Test scene: `instance2.xml` in the repo root.

## 8. Staying out of trouble (context and loops)

The context window is limited (about 200K tokens on the free model) and a session can
stall for a long time on a bad loop.

- **Read narrowly.** Use grep and line ranges instead of opening whole large files. Do
  not paste whole files into your replies.
- **Do one step at a time** when the task prompt has steps. Finish and report a step before
  starting the next one if the prompt says so.
- **Three strikes.** If the same command or the same fix fails 3 times, STOP and report
  what you tried and the exact error. Do not keep retrying variations.
- **Time limits on runs.** Put a timeout on anything that runs a GPU workload. Renders use
  a small `--samples` and a small resolution unless the task says otherwise.
- **Token pressure.** If you think the remaining context is not enough to finish the task
  plus compile iterations, say so BEFORE starting, so the human can split the task. (Your
  estimate may be wrong; the narrow-reading rules above matter more.)

## 9. Output discipline

- **State assumptions up front.** Before writing code, list your assumptions about buffer
  layout, RNG state and closure representation in 3 to 5 bullets, and wait for the human
  if any of them is a real choice.
- **Small diffs, existing code style, no reformatting.** Do not touch lines that do not
  need to change.
- **Deliverable format** for every task: the diff, then a list titled
  `UNVERIFIED ASSUMPTIONS`, then the exact commands you ran with their captured output.

## 10. End of every session

Write (or overwrite) `lastsession.txt` in the repo root. Keep it short. Include:

- What you did, file by file, and why
- Gate results, with the real output you saw (or "gate not run")
- Anything that diverged from the task prompt, and why
- What is still open, and the suggested next task
- Facts the next session must know (commands that worked, traps you hit)

## 11. Added after M4c

- Checkpoint the summary. Update lastsession.txt after EACH completed step, not only at the end. Sessions get interrupted and compacted; a stale summary misleads the next session.
- Resuming. After a compaction or in a new session, run git status and git diff --stat and read lastsession.txt before acting. If the repo is ahead of the summary, trust the repo and say so.
- Gate methodology belongs to the user. If a gate fails because two valid algorithms disagree only at float-rounding edges, report the evidence and stop. Never change a tolerance or the reference method on your own. (The reference must be the same algorithm class as the port for a traversal gate; other references may be printed as informational lines.)
- No elevated actions (UAC prompts, registry, gflags/page heap, admin shells) unless the current task prompt explicitly allows them.
- No generated artifacts in the repo root. Write dumps, captures and .spv files to the OpenCode temp dir or a git-ignored folder, and list any untracked files left at the end so the user can handle git.
- Open items go into lastsession.txt with the exact observation (values, file, line), not a paraphrase.

- Negative tests are only meaningful against a CLEAN baseline. A negative counts as detected only if the fault changes the output relative to the clean run of the same configuration. If the baseline itself fails the gate, no negative result is valid; report that instead of a verdict.
