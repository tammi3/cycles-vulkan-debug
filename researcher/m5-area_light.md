m5\_area\_light.md

a) Exact kernel set needed beyond M4 for an area-lit scene

DEVICE\_KERNEL\_INTEGRATOR\_\* kernels for M5 (with roles and call chains)

Kernel	Role	Key functions/headers called	Est. line count (host)

INTERSECT\_SHADOW	Trace shadow ray, find opaque/transparent hits	intersect\_shadow.h:150 integrator\_intersect\_shadow → integrate\_intersect\_shadow\_opaque (31-54), integrate\_intersect\_shadow\_transparent (112-147), scene\_intersect\_shadow/scene\_intersect\_shadow\_all (bvh.h), sort\_shadow\_intersections (82-110), integrator\_shadow\_path\_next/terminate (state\_flow.h)	\~180 lines

SHADE\_SHADOW	Evaluate transparent shadow surfaces/volumes, write direct light	shade\_shadow.h:265 integrator\_shade\_shadow → integrate\_transparent\_shadow (183-262), integrate\_transparent\_surface\_shadow (88-140), integrate\_transparent\_volume\_shadow (143-180), surface\_shader\_eval (surface\_shader.h), volume\_shadow\_ray\_marching/volume\_shadow\_null\_scattering (volume\_shader.h), guiding\_record\_direct\_light, film\_write\_direct\_light	\~290 lines

SHADE\_LIGHT\_NEE	Evaluate light shader at NEE sample (area/point/spot/sun/bg)	shade\_light.h:228 integrator\_shade\_light\_nee → integrate\_light\_nee (115-224), area\_light\_sample/eval (area.h:329-372, 249-326), point\_light\_sample, spot\_light\_sample, sun\_light\_sample, background\_light\_sample, light\_sample\_mis\_weight\_nee (sample.h:344-362), surface\_shader\_eval<KERNEL\_FEATURE\_NODE\_MASK\_SURFACE\_LIGHT>	\~300 lines

SHADE\_LIGHT\_FORWARD	Evaluate light hit by forward path (emission)	shade\_light.h:79 integrator\_shade\_light\_forward → integrate\_light\_forward (19-76), light\_eval\_from\_intersection (light.h), light\_sample\_shader\_eval\_forward (sample.h:44-111), light\_sample\_mis\_weight\_forward\_lamp (522-560), film\_write\_surface\_emission	\~150 lines

INTERSECT\_DEDICATED\_LIGHT	Shadow linking: pick emissive mesh/light behind blocker	intersect\_dedicated\_light.h:219 integrator\_intersect\_dedicated\_light → shadow\_linking\_intersect (181-215), shadow\_linking\_pick\_light\_intersection (123-173), shadow\_linking\_pick\_mesh\_intersection (30-115), lights\_intersect\_shadow\_linked (light.h)	\~230 lines

SHADE\_DEDICATED\_LIGHT	Shade the picked light for shadow linking	shade\_dedicated\_light.h:224 integrator\_shade\_dedicated\_light → shadow\_linking\_shade (144-220), shadow\_linking\_shade\_light (76-111), shadow\_linking\_shade\_surface\_emission (113-142), light\_sample\_shader\_eval\_nee\_constant (sample.h:24-40), integrate\_direct\_light\_shadow\_init\_common (shade\_surface.h:205-307)	\~240 lines

SHADE\_SURFACE (extended)	NEE branch: integrate\_surface\_direct\_light → light\_sample\_from\_position → light\_sample (sample.h:403-432)	shade\_surface.h:311-487 template integrate\_surface\_direct\_light → light\_sample\_from\_position (sample.h:403-432) → light\_sample<false> (area.h:329, point.h, spot.h, sun.h, background.h, triangle.h)	Part of M4 scope but NEE path is M5

Total new host reference code for probes: \~1,400 lines across 7 kernels.

b) Light sampling: light tree vs distribution

Light tree mandatory or optional?

FACT: Light tree is controlled by kernel\_data.integrator.use\_light\_tree (source: src/kernel/light/tree.h:754, sample.h:416-417). The integrator setting comes from SceneIntegrator at blender/sync.cpp and device/device.h. In instance2.xml and test scenes, there's no explicit light tree toggle — it's enabled by default when WITH\_CYCLES\_LIGHT\_TREE=ON in CMake (source: intern/cycles/CMakeLists.txt line \~200).

FACT: The code path splits at sample.h:415-426:

\#ifdef \_\_LIGHT\_TREE\_\_

if (kernel\_data.integrator.use\_light\_tree) {

&#x20;   if (!light\_tree\_sample<false>(kg, rand.z, P, N, 0.0f, object\_receiver, shader\_flags, ls)) {

&#x20;       return false;

&#x20;   }

} else

\#endif

{

&#x20;   if (!light\_distribution\_sample(kg, rand.z, ls)) {

&#x20;       return false;

&#x20;   }

}

FACT: light\_distribution\_sample (distribution.h:45-53) is a simple CDF over light\_distribution array (1D, \~num\_lights entries). It's O(log N) binary search. No spatial awareness.

FACT: light\_tree\_sample (tree.h:742-810) is \~200 lines of tree traversal + reservoir sampling at leaves. Requires light\_tree\_nodes, light\_tree\_emitters buffers (built in device/device.cpp build\_light\_tree).

Porting cost comparison

Aspect	Light Tree	Flat Distribution

Host build	build\_light\_tree() (device.cpp \~150 lines) + BVH-style buffers	Trivial: copy light\_distribution CDF array

Device buffers	light\_tree\_nodes (struct KernelLightTreeNode, \~48B × nodes), light\_tree\_emitters (KernelLightTreeEmitter, \~64B × emitters), light\_to\_tree (int × lights), triangle\_to\_tree (int × triangles), object\_lookup\_offset	light\_distribution (float × num\_lights + 1)

Kernel code	\~500 lines (tree.h) + area/point/spot/tree\_params	\~60 lines (distribution.h)

Slang hazards	Recursion in traversal (while loop OK), dynamic indexing on light\_tree\_nodes array, complex struct KernelLightTreeNode with unions, pointer chains	None — flat array, simple binary search

MIS weight	light\_tree\_pdf (tree.h:813-950) — traverses tree again	light\_distribution\_pdf\_lamp = constant (distribution.h:55-58)

Area light sampling	Same area\_light\_sample/eval	Same

Recommendation for M5 gate scene

USER DECISION: For the M5 gate (Cornell box, one area light), use flat distribution. Rationale:

\- Single light → tree overhead unjustified

\- Flat distribution is \~50 lines vs \~500, zero recursion, trivial Slang port

\- Light tree can be added as M6+ feature with its own gate

\- The M5 gate only requires correct NEE + MIS for one area light

If user insists on light tree: Need to port KernelLightTreeNode/Emitter layouts, light\_tree\_sample, light\_tree\_pdf, and all \_tree\_parameters functions (area.h:513-549, spot.h, point.h, sun.h, background.h, triangle.h). Estimated 3-4 additional tasks.

c) Area light sampling/eval, MIS, shadow ray setup, shadow path state

Area light sampling/eval (source: src/kernel/light/area.h)

Function	Lines	Purpose

area\_light\_rect\_sample	19-102	Solid-angle sampling of rectangle (Ureña et al.)

area\_light\_spread\_clamp\_light	125-240	Clamp sample region for spread angle (softbox)

area\_light\_eval	250-326	Template<in\_volume\_segment>: computes eval\_fac, pdf, optionally samples light\_P

area\_light\_sample	329-372	Template<in\_volume\_segment>: calls eval + validates sampled point inside light bounds

area\_light\_mnee\_sample\_update	374-388	MNEE: update light position for fixed direction

area\_light\_intersect	390-427	Ray-quad/ellipse intersection for forward path hits

area\_light\_uv	429-443	Compute UV at intersection (for shader)

area\_light\_eval\_from\_intersection	445-462	Forward path: evaluate light at hit point

area\_light\_tree\_parameters	513-549	Light tree: centroid, distance, cos\_theta\_u for importance

MIS weight functions (source: src/kernel/light/sample.h)

Function	Lines	Formula

light\_sample\_mis\_weight\_nee	344-362	power\_heuristic(nee\_pdf, forward\_pdf) (or debug overrides)

light\_sample\_mis\_weight\_forward	327-342	power\_heuristic(forward\_pdf, nee\_pdf)

light\_sample\_mis\_weight\_forward\_lamp	522-560	Includes light tree/distribution selection PDF

light\_sample\_mis\_weight\_forward\_surface	469-520	Includes triangle light PDF + light tree

light\_sample\_mis\_weight\_forward\_distant	562-573	Sun/background (distant)

light\_sample\_mis\_weight\_forward\_background	575-615	Background light

power\_heuristic (source: src/kernel/sample/mis.h:15-20):

ccl\_device\_inline float power\_heuristic(float a, float b) {

&#x20;   a = a \* a; b = b \* b;

&#x20;   return a / (a + b);

}

Shadow ray setup (source: src/kernel/light/sample.h:265-313)

// shadow\_ray\_offset: avoids shadow terminator artifact (169-221)

// shadow\_ray\_setup: fills Ray from LightSample (265-301)

// light\_sample\_to\_surface\_shadow\_ray: combines both (304-313)

Key: ray.tmin = offset from shadow\_ray\_smooth\_surface\_offset (parabolic approx), ray.tmax = light distance, ray.self = skip self-intersection.

IntegratorShadowState members involved (source: src/kernel/integrator/shadow\_state\_template.h:7-74)

// Shadow path state (SoA on GPU, each is a pointer member)

render\_pixel\_index, sample, rng\_pixel, rng\_offset,

bounce, transparent\_bounce, diffuse\_bounce, glossy\_bounce, transmission\_bounce,

volume\_bounds\_bounce, portal\_bounce, queued\_kernel,

visibility (uint8), flag (uint32),

throughput (PackedSpectrum), unshadowed\_throughput,

pass\_diffuse\_weight, pass\_glossy\_weight,

packed\_num\_hits (uint16: hits + resume\_hit + skip\_volume),

lightgroup (uint8),

unlit\_throughput, path\_segment (guiding),

guiding\_light\_linking\_mis\_weight,

bsdf\_eval\_average (only without light tree)

Shadow ray state (packed, 12B each P/D): shadow\_ray (shadow\_state\_template.h:78-88)

Shadow intersections (AoS, 4 slots GPU): shadow\_isect (94-103)

d) Slang portability hazards list

Hazard	File:Line	Description	Proposed Slang Approach

Pointer casts	area.h:268-271 rectangle\_sample takes float3\*	light\_p modified in place via pointer	Use inout float3 light\_p parameter instead of pointer

Unions	light.h:50-60 KernelLight has union { AreaLight area; PointLight point; ... }	Discriminated by type field	Flatten: separate structs per light type, or use type switch with opaque blob + offset access

Templates	area.h:249 template<bool in\_volume\_segment> on area\_light\_eval/sample	Two instantiations: surface + volume	Generate two Slang functions: area\_light\_eval\_surface, area\_light\_eval\_volume

Function pointers	surface\_shader.h closure eval uses function tables	BSDF/emission closures called via function pointers	Replace with explicit switch(closure\_type) — already M4 pattern

Recursion	tree.h:766-784 light\_tree\_sample while-loop with continue	Tree traversal loop, not recursion — OK	Keep as while(true) loop; no recursion

Packed types	shadow\_state\_template.h:79-80 packed\_float3 P, D	12-byte elements in SoA	Use float\* base + 3\*index indexing (per S1b-3 stride=12 confirmation)

Float atomics	film\_write\_\* uses atomic\_add on float	film\_write\_direct\_light → film\_atomic\_add (film/write.h)	Emulate via CAS loop (AGENTS.md: RX 570 has no native float atomic add)

Shared memory	None in these kernels	—	N/A

Warp/subgroup ops	sort\_shadow\_intersections (shade\_shadow.h:88-105) bubble sort	GPU path uses bubble sort, no subgroup ops	Keep bubble sort; no warp ops needed

KERNEL\_FEATURE\_ gating\*	#ifdef \_\_LIGHT\_TREE\_\_, \_\_TRANSPARENT\_SHADOWS\_\_, \_\_SHADOW\_LINKING\_\_, \_\_MNEE\_\_, \_\_VOLUME\_\_	Many features gated	For M5: only enable KERNEL\_FEATURE\_PATH\_TRACING, KERNEL\_FEATURE\_LIGHT\_PASSES (optional). Disable others via specialization constants.

e) Proposed split into agent-sized tasks (≤400 lines Slang each)

Dependency order

Task M5.1: Flat light distribution sampling (distribution.h + sample.h:light\_distribution\_sample)

&#x20; → Host probe: compare light index + pdf\_selection vs CPU for 10 random seeds

&#x20; → Gate: 100% match on light index, pdf within 1e-6

&#x20; → Negative: wrong CDF array → wrong light picked



Task M5.2: Area light sampling + eval (area.h: area\_light\_sample, area\_light\_eval<false>)

&#x20; → Host probe: for fixed P,N,rand, compare LightSample (P,Ng,D,pdf,eval\_fac) vs CPU

&#x20; → Gate: all components match within 1e-5

&#x20; → Negative: wrong solid angle formula → pdf bias visible in render



Task M5.3: MIS weights (sample.h: power\_heuristic, light\_sample\_mis\_weight\_nee/forward)

&#x20; → Host probe: feed known bsdf\_pdf, nee\_pdf → compare weight

&#x20; → Gate: exact match (deterministic math)

&#x20; → Negative: swapped args → weight = 1-weight



Task M5.4: Shadow ray setup (sample.h: shadow\_ray\_offset, shadow\_ray\_setup, light\_sample\_to\_surface\_shadow\_ray)

&#x20; → Host probe: given sd, ls → compare Ray (P,D,tmin,tmax,self) vs CPU

&#x20; → Gate: all fields match

&#x20; → Negative: skip self-intersection wrong → self-shadow artifacts



Task M5.5: INTERSECT\_SHADOW kernel (intersect\_shadow.h: integrate\_intersect\_shadow\_opaque/transparent)

&#x20; → Host probe: launch with pre-populated shadow state + ray, compare packed\_num\_hits + throughput

&#x20; → Gate: opaque hit terminates, transparent records hits sorted by t

&#x20; → Negative: unsorted hits → wrong transparency accumulation



Task M5.6: SHADE\_SHADOW kernel (shade\_shadow.h: integrate\_transparent\_shadow + surface/volume eval)

&#x20; → Host probe: pre-populate shadow\_isect array, compare throughput after eval

&#x20; → Gate: throughput matches CPU for same intersections

&#x20; → Negative: cache miss handling wrong → wrong resume state



Task M5.7: SHADE\_LIGHT\_NEE kernel (shade\_light.h: integrate\_light\_nee)

&#x20; → Host probe: pre-populate shadow state with light sample, compare eval + throughput update

&#x20; → Gate: light eval matches, throughput updated correctly

&#x20; - Negative: zero eval not terminated → fireflies



Task M5.8: SHADE\_SURFACE NEE branch (shade\_surface.h: integrate\_surface\_direct\_light template)

&#x20; → Host probe: full shade\_surface with NEE enabled, compare shadow\_state spawned

&#x20; → Gate: shadow\_state matches CPU for same RNG

&#x20; → Negative: MIS weight wrong → bright/dark splotches



Task M5.9: M5 Gate — Cornell box render (end-to-end)

&#x20; → Scene: Cornell box XML (provided in f)

&#x20; → Compare: 16 spp GPU vs CPU per-pixel, shadow region + falloff

&#x20; → Tolerance: noise-only diff OK; systematic bias = FAIL

f) M5 gate scene: concrete XML + CPU settings

Cornell box XML (cornell\_area\_light.xml)

<cycles>

&#x20;   <transform translate="0 0 0">

&#x20;       <camera type="perspective" fov="0.7" full\_width="320" full\_height="240" />

&#x20;   </transform>



&#x20;   <!-- Materials -->

&#x20;   <shader name="white\_wall">

&#x20;       <diffuse\_bsdf name="bsdf" color="0.8 0.8 0.8" />

&#x20;       <connect from="bsdf bsdf" to="output surface" />

&#x20;   </shader>

&#x20;   <shader name="red\_wall">

&#x20;       <diffuse\_bsdf name="bsdf" color="0.8 0.1 0.1" />

&#x20;       <connect from="bsdf bsdf" to="output surface" />

&#x20;   </shader>

&#x20;   <shader name="green\_wall">

&#x20;       <diffuse\_bsdf name="bsdf" color="0.1 0.8 0.1" />

&#x20;       <connect from="bsdf bsdf" to="output surface" />

&#x20;   </shader>

&#x20;   <shader name="light\_shader">

&#x20;       <emission\_closure name="emit" color="1 1 1" strength="15" />

&#x20;       <connect from="emit emission" to="output surface" />

&#x20;   </shader>



&#x20;   <!-- Floor -->

&#x20;   <state shader="white\_wall">

&#x20;       <object name="floor" mesh="box\_floor" />

&#x20;   </state>



&#x20;   <!-- Ceiling -->

&#x20;   <state shader="white\_wall">

&#x20;       <object name="ceiling" mesh="box\_ceiling" />

&#x20;   </state>



&#x20;   <!-- Back wall -->

&#x20;   <state shader="white\_wall">

&#x20;       <object name="back\_wall" mesh="box\_back" />

&#x20;   </state>



&#x20;   <!-- Red wall (left) -->

&#x20;   <state shader="red\_wall">

&#x20;       <object name="left\_wall" mesh="box\_left" />

&#x20;   </state>



&#x20;   <!-- Green wall (right) -->

&#x20;   <state shader="green\_wall">

&#x20;       <object name="right\_wall" mesh="box\_right" />

&#x20;   </state>



&#x20;   <!-- Area light on ceiling (1x1 square, centered) -->

&#x20;   <transform translate="0 2.9 0">

&#x20;       <state shader="light\_shader">

&#x20;           <object name="area\_light" mesh="light\_quad" />

&#x20;       </state>

&#x20;   </transform>



&#x20;   <!-- Tall box -->

&#x20;   <state shader="white\_wall">

&#x20;       <object name="tall\_box" mesh="box\_tall" />

&#x20;   </state>



&#x20;   <!-- Short box -->

&#x20;   <state shader="white\_wall">

&#x20;       <transform translate="1.5 0 0">

&#x20;           <object name="short\_box" mesh="box\_short" />

&#x20;       </transform>

&#x20;   </state>



&#x20;   <!-- Meshes (unit cube variants) -->

&#x20;   <mesh name="box\_floor" P="2 0 2 -2 0 2 -2 0 -2 2 0 -2" nverts="4" verts="0 1 2 3" />

&#x20;   <mesh name="box\_ceiling" P="2 3 2 2 3 -2 -2 3 -2 -2 3 2" nverts="4" verts="0 1 2 3" />

&#x20;   <mesh name="box\_back" P="2 0 -2 2 3 -2 -2 3 -2 -2 0 -2" nverts="4" verts="0 1 2 3" />

&#x20;   <mesh name="box\_left" P="-2 0 2 -2 3 2 -2 3 -2 -2 0 -2" nverts="4" verts="0 1 2 3" />

&#x20;   <mesh name="box\_right" P="2 0 2 2 0 -2 2 3 -2 2 3 2" nverts="4" verts="0 1 2 3" />

&#x20;   <mesh name="light\_quad" P="1 0 1 -1 0 1 -1 0 -1 1 0 -1" nverts="4" verts="0 1 2 3" />

&#x20;   <mesh name="box\_tall" P="0.5 0 0.5 -0.5 0 0.5 -0.5 1.5 0.5 0.5 1.5 0.5 -0.5 0 -0.5 0.5 0 -0.5 -0.5 1.5 -0.5 0.5 1.5 -0.5" nverts="4 4 4 4 4 4" verts="0 1 2 3 4 5 6 7 0 1 5 4 2 3 7 6 0 4 7 3 1 5 6 2" />

&#x20;   <mesh name="box\_short" P="0.3 0 0.3 -0.3 0 0.3 -0.3 0.6 0.3 0.3 0.6 0.3 -0.3 0 -0.3 0.3 0 -0.3 -0.3 0.6 -0.3 0.3 0.6 -0.3" nverts="4 4 4 4 4 4" verts="0 1 2 3 4 5 6 7 0 1 5 4 2 3 7 6 0 4 7 3 1 5 6 2" />

</cycles>

Cycles CPU settings for valid comparison

Setting	Value	Source

sampling\_pattern	SOBOL (default)	kernel/sample/pattern.h

film\_filter	BOX (no filter, pixel-perfect compare)	kernel/film/filter.h

light\_tree	false (flat distribution)	kernel\_data.integrator.use\_light\_tree = false

integrator	PATH (not BRANCHED)	kernel/integrator/megakernel.h

samples	16	M5 gate requirement

resolution	320x240	Low-res for speed

max\_bounce	4 (diffuse=4, glossy=4, transmission=4)	Default

transparent\_max\_bounce	8	Default

use\_direct\_light	true	Required for NEE

light\_threshold	0 (no probabilistic termination)	kernel\_data.integrator.light\_inv\_rr\_threshold = 0

denoising	false	Raw render

CPU reference image production

\# From Blender source build (E:/blender-git/build)

./blender -b --factory-startup -noaudio \\

&#x20; --python-expr "

import bpy

bpy.context.scene.cycles.device = 'CPU'

bpy.context.scene.cycles.samples = 16

bpy.context.scene.cycles.use\_preview = False

bpy.context.scene.render.resolution\_x = 320

bpy.context.scene.render.resolution\_y = 240

bpy.context.scene.render.film\_transparent = False

bpy.context.scene.cycles.filter\_type = 'BOX'

bpy.context.scene.cycles.light\_sampling\_threshold = 0.0

bpy.ops.wm.open\_mainfile(filepath='cornell\_area\_light.xml')

bpy.ops.render.render(write\_still=True)

" \\

&#x20; -o //cornell\_cpu\_16spp.png

Output: cornell\_cpu\_16spp.png (RGBA float32 or 8-bit sRGB). GPU render uses same XML via cycles.exe --device VULKAN --samples 16 --background cornell\_area\_light.xml -o cornell\_gpu\_16spp.png.

Comparison: Per-pixel L2 in linear space, ignore alpha. Threshold: mean abs diff < 0.01 (1/255), max abs diff < 0.05 in shadow penumbra.

UNVERIFIED ASSUMPTIONS (M5)

1\. Light tree disabled for M5 — user must confirm flat distribution is acceptable for gate scene

2\. No transparent shadows in M5 gate — \_\_TRANSPARENT\_SHADOWS\_\_ disabled; if enabled, adds \~100 lines to SHADE\_SHADOW

3\. No shadow linking — \_\_SHADOW\_LINKING\_\_ disabled; INTERSECT\_DEDICATED\_LIGHT/SHADE\_DEDICATED\_LIGHT not needed for M5

4\. No MNEE — \_\_MNEE\_\_ disabled; simplifies shade\_surface NEE branch

5\. No volume — \_\_VOLUME\_\_ disabled; removes volume shadow + light sampling paths

6\. PackedSpectrum = packed\_float3 (12B) — confirmed in S1b-3; accessor uses float\* base + stride 12

7\. Slang float3\* stride = 12 on RX 570 + SwiftShader — confirmed in S1b-3 step 2; safe for ray.P/throughput

8\. No native float atomic add on RX 570 — must use CAS emulation for film writes

9\. SwiftShader available at C:\\Windows\\System32\\Microsoft-Edge-WebView\\vk\_swiftshader\_icd.json — confirmed in S1b-2

