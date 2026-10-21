1\. Frozen infrastructure (non-negotiable)



&#x20;   M0–M3 are frozen. Do not modify VulkanDevice, BVH traversal, the wavefront queue implementation, descriptor set layouts, or any .slang file that already passes its validation gate. If a task appears to require changing them, stop and report what's blocking you instead of editing.



This is the single most important rule. Free models will happily "improve" working code and break it. Name the files explicitly so there's no ambiguity.

2\. One function per session, scope locked



&#x20;   One translation unit per task. The input is one Cycles function (or one small cluster of functions it calls). The output is the Slang equivalent of that function only. Everything else in the file stays untouched. Do not port, refactor, rename, or "clean up" neighboring code.



&#x20;   Feature scope is explicit. For M4, only the LABEL\_DIFFUSE closure path is in scope. Every other closure type must return a stub (LABEL\_NONE / no-op) with a comment marking it as deferred. Do not implement features beyond the milestone.



3\. Translation fidelity rules (Cycles → Slang)



These are the specific things that will go wrong if you don't state them:



&#x20;   No virtual dispatch. Cycles ShaderClosure arrays and virtual BSDF calls must be flattened into an explicit switch on closure type. No function pointers, no interface types, no dynamic dispatch.



&#x20;   No general pointers. Do not use pointer parameters. Use inout / out parameters or explicit indices into SSBOs. The only pointers allowed are the buffer handles passed at entry.



&#x20;   No dynamic allocation. No new, no variable-length arrays, no growable containers. All buffers are host-allocated; size them at compile time or pass counts in a uniform.



&#x20;   No recursion. Every function must be callable in a straight-line call graph.



&#x20;   Explicit address spaces. Map Cycles' ccl\_private → function-local or inout, ccl\_global → SSBO, ccl\_constant → uniform/push constant. Do not let the compiler infer storage class silently.



&#x20;   Float vector alignment. float3 in an SSBO is 16-byte aligned in std430. Do not assume 12-byte packing. When translating a Cycles struct that goes into an SSBO, restate every field with its offset and alignment in a comment above the Slang struct.



&#x20;   Deterministic RNG only. Use the seeded sampler state passed into the kernel. Do not call any unseeded random function. If Cycles' sampler isn't ported yet, use a placeholder that is a pure function of (pixel, sample\_index, dimension) and mark it TODO: swap for ported sampler.



&#x20;   Barrier discipline. Any SSBO write that another invocation may read must be followed by an explicit controlBarrier (or a memoryBarrier with the correct scope) before the next dispatch or the next read. Do not assume the driver inserts synchronization.



&#x20;   Faithful translation, not improvement. Port the Cycles logic as written, even if it looks redundant or awkward. Cycles is the reference implementation; divergence is a correctness bug.



4\. Verification gates (the AI must run these)



&#x20;   Compile before declaring done. Every Slang output must pass slangc -target spirv and spirv-val before you report success. If compilation fails, iterate until it passes, and do not report a "working" result on a failing build.



&#x20;   Behavioral gate per milestone. A passing build is not correctness. For each unit, state the gate explicitly at the start of the task:



&#x20;       M4: Lambertian sphere, 16 spp, compared pixel-wise against Cycles CPU render. Noise-only differences acceptable; systematic bias is a failure.



&#x20;       M5: Cornell box with one area light, low resolution, shadow regions and light falloff compared against Cycles CPU.



&#x20;   If the gate isn't met, report the failure with the diff and stop. Do not declare the milestone complete.



&#x20;   Do not modify the verification harness. If the harness fails, the harness is right and the code is wrong. Editing tests, tolerances, or reference images to make a gate pass is a hard violation.



5\. Output discipline

&#x20;   Output only the changed function(s). Do not rewrite whole files. If a file must change in more than one place, list each change as a separate diff block with the enclosing function name.



&#x20;   No invented APIs. If you're unsure whether a Slang function exists (e.g., a specific dot, normalize, sample\_cosine\_hemisphere), do not invent a name and hope. Flag it as UNVERIFIED: and let the human confirm or provide the signature.



&#x20;   Stop on unmapped features. If the Cycles code uses a feature with no clear Slang equivalent (subsurface scattering closures, complex IES light sampling, hair BSDF, volume integration), stop and ask. Do not approximate silently—an approximation that looks plausible is worse than a missing feature.



&#x20;   State assumptions up front. Before writing code, list the assumptions you're making about buffer layout, RNG state, and closure representation in 3–5 bullets. The human reviews these before you proceed.



6\. Context management rules



&#x20;   Prompt size limit. Do not accept or request more than one Cycles source function plus one Slang skeleton per prompt. If a task seems to need more context, break it into sub-tasks.



&#x20;   Restate the invariant each session. Every session begins by re-reading AGENTS.md and the milestone's gate definition. Do not rely on memory from previous sessions; free models don't have it.



&#x20;   Report token pressure. If you estimate the remaining context is insufficient to complete the task plus compilation iteration, say so before starting. The human can then split the task.



&#x20;   If session is coming to an end, write a summary in a .md or .txt file explaining what you did this current session

CONTEXT:



Vulkan backend for Cycles in Blender targeting my RX 570 4GB check https://github.com/tammi3/Vulkan-backend-Blender-CyclesX for further context



Drop unnecessary tokens from context memory if needed and dynamically re-open needed context, to not exhaust token / context window too fast

