// traverse.slang : M4c port of the Cycles closest-hit BVH2 traversal.
//
// Ported source (reference backend: HIP):
//   - BVH_FUNCTION_FULL_NAME(BVH)  src/kernel/bvh/traversal.h:24-261
//     (aligned nodes + triangles + instance push/pop)
//   - bvh_instance_push            src/kernel/geom/object.h:464-477
//   - bvh_instance_pop             src/kernel/geom/object.h:502-510
//
// The walking skeleton is the M2c spike (agentreference/bvh_traverse.slang),
// which is the only Slang traversal proven to compile AND run correctly on
// the RX 570. Deltas vs the spike, all required by the M4c gate:
//   - output is Intersection {float t,u,v; int prim,object,type;}
//     (source: src/kernel/types.h:614-619, 24 bytes) instead of the spike's
//     12-byte Hit; u/v/type are filled the way Cycles' triangle_intersect
//     fills them (src/kernel/geom/triangle_intersect.h:52-57);
//   - the counts the spike kept in SceneData come from the host (dump meta,
//     M4c task decision Q2) and are used only by the PROBE for bounds
//     checks; production kernels will not carry counts (as HIP does not).
//   - test_mode bits allow the gate's negative tests to inject faults
//     WITHOUT editing the shader between runs (M4c task decision Q7).
//
// DEFERRED out of M4c scope (they keep the spike's error code, ERR_INSTANCE,
// with a // DEFERRED comment, per the task prompt):
//   motion blur, curves, points, unaligned (hair) nodes, shadow rays
//   (shadow_all/local/volume), self-shadow skipping, shadow linking and the
//   shadow-ray early exit.

/* Intersection, 24 bytes. Mirrors src/kernel/types.h:614-619 exactly.
 * The host mirror is locked with static_assert(offsetof(...)) +
 * static_assert(sizeof == 24) in src/device/vulkan/kernels.cpp. */
struct Intersection
{
    float t;
    float u;
    float v;
    int prim;
    int object;
    int type;
};

/* The probe header: scene pointers + probe-only counts + camera + flags.
 * The host fills everything except the 9 pointer slots and `root` (those
 * are written by the prepare kernel from the live KernelParamsVulkan
 * buffer, so the gate always traverses the LIVE scene arrays).
 *
 * Layout is scalar-only so host C++ and Slang cannot disagree on padding
 * (AGENTS.md section 4). Every offset is locked host-side. */
struct TraverseHeader
{
    float4* bvh_nodes;        // offset   0
    float4* bvh_leaf_nodes;   // offset   8
    uint* prim_index;         // offset  16
    uint* prim_visibility;    // offset  24
    uint* prim_object;        // offset  32
    uint* tri_vindex;         // offset  40
    float* tri_verts;         // offset  48
    uint* objects;            // offset  56
    uint* object_node;        // offset  64
    int root;                 // offset  72 (read from the KernelData blob)
    uint node_count;          // offset  76 (float4s; probe bounds checks)
    uint leaf_count;          // offset  80
    uint prim_count;          // offset  84
    uint tri_count;           // offset  88
    uint vert_count;          // offset  92
    uint object_count;        // offset  96
    uint object_node_count;   // offset 100
    uint width;               // offset 104
    uint height;              // offset 108
    uint visibility;          // offset 112
    uint test_mode;           // offset 116
    float cam_ox;             // offset 120  } pinhole camera:
    float cam_oy;             // offset 124  }   origin o, image-plane
    float cam_oz;             // offset 128  }   lower-left corner l, full-
    float cam_lx;             // offset 132  }   width vector u and full-
    float cam_ly;             // offset 136  }   height vector v (the M2c
    float cam_lz;             // offset 140  }   camera convention, spike
    float cam_ux;             // offset 144  }   main() :301-307 and
    float cam_uy;             // offset 148  }   agentreference/raytri_cpu.h
    float cam_uz;             // offset 152  }   :40-91)
    float cam_vx;             // offset 156  }
    float cam_vy;             // offset 160  }
    float cam_vz;             // offset 164  }
}; // 168 bytes

/* Constants. ENTRYPOINT_SENTINEL / stack size / iteration cap come from the
 * spike (bvh_traverse.slang:49-51), which matched Cycles behaviour in M2c.
 * PRIMITIVE_TRIANGLE = (1 << 0) (src/kernel/types.h:687).
 * OBJECT_WORDS / POSITION_OFFSET_WORD: KernelObject is 256 bytes
 * (64 words) and position_offset is word 42 - counted from
 * src/kernel/types.h:1412-1435 (two Transform = 2 x 3 float4 = 24 words,
 * then 18 scalar members) and locked with host static_asserts. */
static const int ENTRYPOINT_SENTINEL = 0x76543210;
static const int BVH_STACK_SIZE = 192;
static const uint MAX_ITERATIONS = 1000000u;
static const uint OBJECT_WORDS = 64u;
static const uint POSITION_OFFSET_WORD = 42u;
static const uint PRIMITIVE_TRIANGLE = 1u;

/* Result of traverse(): 0 = ok (hit OR miss), anything else is an error id.
 * Error ids live in a different bit pattern than PRIM_NONE (-1) and than
 * any real primitive index, so an error is always distinguishable from a
 * normal miss (M4c gate decision Q7). */
static const uint NO_PRIM = 0xFFFFFFFFu; // = PRIM_NONE (~0), types.h:48
static const uint NO_OBJECT = 0xFFFFFFFFu; // = OBJECT_NONE (~0), types.h:47
static const uint ERR_LOOP = 0xFFFFFFFBu;
static const uint ERR_STACK = 0xFFFFFFFCu;
static const uint ERR_BAD_INDEX = 0xFFFFFFFDu;
static const uint ERR_INSTANCE = 0xFFFFFFFEu;

/* test_mode bits (push-constant controlled fault injection, Q7). */
static const uint TEST_MODE_SKIP_POP = 1u;
static const uint TEST_MODE_CORRUPT_OBJECT_NODE = 2u;

/* --- small helpers (identical to the spike, bvh_traverse.slang:62-91) --- */

float copysign_bits(float m, float s)
{
    return asfloat((asuint(m) & 0x7FFFFFFFu) | (asuint(s) & 0x80000000u));
}

/* bvh_clamp_direction: keep direction components out of the denormal range
 * so 1/dir never explodes (source: Cycles util/vec3.h, same constant). */
float clamp_component(float c)
{
    const float ooeps = 8.271806E-25f;
    return (abs(c) > ooeps) ? c : copysign_bits(ooeps, c);
}

/* transform_point / transform_direction on a Transform stored as three
 * float4 rows (src/util/transform.h:52-53 + util/transform.h transform_point;
 * .w of each row is the translation). */
float3 transform_point(float4 tx, float4 ty, float4 tz, float3 a)
{
    return float3(
        tx.x * a.x + tx.y * a.y + tx.z * a.z + tx.w,
        ty.x * a.x + ty.y * a.y + ty.z * a.z + ty.w,
        tz.x * a.x + tz.y * a.y + tz.z * a.z + tz.w);
}

float3 transform_direction(float4 tx, float4 ty, float4 tz, float3 a)
{
    return float3(
        tx.x * a.x + tx.y * a.y + tx.z * a.z,
        ty.x * a.x + ty.y * a.y + ty.z * a.z,
        tz.x * a.x + tz.y * a.y + tz.z * a.z);
}

float3 rt_cross(float3 a, float3 b)
{
    return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

float rt_dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

/* ray_triangle_intersect: the spike's edge-function test
 * (bvh_traverse.slang:93-117), same signature and same semantics as Cycles'
 * ray_triangle_intersect called from triangle_intersect
 * (src/kernel/geom/triangle_intersect.h:44). The visibility mask test that
 * Cycles does inside triangle_intersect (:49) is done by the caller here,
 * as in the spike (:232-233). */
bool ray_triangle_intersect(float3 P, float3 D, float tmin, float tmax,
                            float3 a, float3 b, float3 c,
                            out float ou, out float ov, out float ot)
{
    ou = 0.0f;
    ov = 0.0f;
    ot = 0.0f;
    float3 v0 = a - P, v1 = b - P, v2 = c - P;
    float3 e0 = v2 - v0, e1 = v0 - v1, e2 = v1 - v2;
    float U = rt_dot(rt_cross(e0, v2 + v0), D);
    float V = rt_dot(rt_cross(e1, v0 + v1), D);
    float W = rt_dot(rt_cross(e2, v1 + v2), D);
    float UVW = U + V + W;
    float eps = 1.1920929e-07f * abs(UVW);
    float minUVW = min(U, min(V, W));
    float maxUVW = max(U, max(V, W));
    if (!(minUVW >= -eps || maxUVW <= eps)) return false;
    float3 Ng1 = rt_cross(e1, e0);
    float3 Ng = Ng1 + Ng1;
    float den = rt_dot(Ng, D);
    if (den == 0.0f) return false;
    float T = rt_dot(v0, Ng);
    float t = T / den;
    if (!(t >= tmin && t <= tmax)) return false;
    float rcp_uvw = (abs(UVW) < 1e-18f) ? 0.0f : 1.0f / UVW;
    ou = min(U * rcp_uvw, 1.0f);
    ov = min(V * rcp_uvw, 1.0f);
    ot = t;
    return true;
}

/* -------------------------------------------------------------------------
 * traverse(): port of BVH_FUNCTION_FULL_NAME(BVH)
 * (src/kernel/bvh/traversal.h:24-261) with bvh_instance_push/pop
 * (src/kernel/geom/object.h:464-477 / :502-510) inlined, exactly as the
 * spike inlined them. The Cycles per-feature template (BVH_MOTION /
 * BVH_HAIR / BVH_POINTCLOUD, traversal.h:9-13) becomes plain comments: this
 * is the plain aligned-triangles instantiation.
 *
 * Returns 0 on completion (hit OR miss; inspect the out params), or one of
 * the ERR_* codes for an out-of-range index, stack overflow or iteration
 * cap - all of which the probe reports as FAIL. */
uint traverse(TraverseHeader* sc,
              float3 P, float3 Din, float tmin, float tmax, uint visibility,
              out float hit_t, out float hit_u, out float hit_v,
              out uint hit_prim, out uint hit_object, out uint hit_type)
{
    int stack[BVH_STACK_SIZE];
    stack[0] = ENTRYPOINT_SENTINEL;      // traversal.h:37-38
    int sp = 0;                          // traversal.h:41
    int node_addr = sc->root;            // traversal.h:42 (kernel_data.bvh.root)

    /* Saved ray state for instance pop. Cycles restores from ray->P/ray->D
     * (object.h:502-510); the spike saves the originals instead
     * (bvh_traverse.slang:126-131) - same result, one register copy. */
    uint current_object = NO_OBJECT;
    float3 orig_P = P;
    float3 orig_Din = Din;
    float orig_tmin = tmin;
    float orig_tmax = tmax;

    /* bvh_clamp_direction + bvh_inverse_direction (traversal.h:46-47). */
    float3 dir = float3(clamp_component(Din.x), clamp_component(Din.y), clamp_component(Din.z));
    float3 idir = float3(1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z);

    /* isect init, traversal.h:51-55: t = tmax, u = v = 0, prim/object =
     * NONE. `type` is NOT set on a miss (Cycles does not either); the probe
     * writes 0 and the host comparison ignores type on misses (Q6). */
    hit_t = tmax;
    hit_u = 0.0f;
    hit_v = 0.0f;
    hit_prim = NO_PRIM;
    hit_object = NO_OBJECT;
    hit_type = 0;
    uint iterations = 0;

    do {
        /* ---- inner nodes, traversal.h:59-107 ---------------------------- */
        while (node_addr >= 0 && node_addr != ENTRYPOINT_SENTINEL) {
            iterations++;
            if (iterations > MAX_ITERATIONS) return ERR_LOOP;
            /* Probe-only bounds check (counts from dump meta, Q2). */
            if ((uint)node_addr + 3u >= sc->node_count) return ERR_BAD_INDEX;

            /* One aligned inner node = cnodes + 3 float4 child AABBs
             * (traversal.h:64 + NODE_INTERSECT = bvh_aligned_node_intersect;
             * written host-side in src/bvh/bvh2.cpp:213-218). */
            float4* np = sc->bvh_nodes + (uint)node_addr;
            float4 cnodes = np[0], node0 = np[1], node1 = np[2], node2 = np[3];

            /* bvh_aligned_node_intersect, inlined like the spike
             * (bvh_traverse.slang:151-174). */
            float c0lox = (node0.x - P.x) * idir.x;
            float c0hix = (node0.z - P.x) * idir.x;
            float c0loy = (node1.x - P.y) * idir.y;
            float c0hiy = (node1.z - P.y) * idir.y;
            float c0loz = (node2.x - P.z) * idir.z;
            float c0hiz = (node2.z - P.z) * idir.z;
            float c0min = max(max(tmin, min(c0lox, c0hix)), max(min(c0loy, c0hiy), min(c0loz, c0hiz)));
            float c0max = min(min(hit_t, max(c0lox, c0hix)), min(max(c0loy, c0hiy), max(c0loz, c0hiz)));

            float c1lox = (node0.y - P.x) * idir.x;
            float c1hix = (node0.w - P.x) * idir.x;
            float c1loy = (node1.y - P.y) * idir.y;
            float c1hiy = (node1.w - P.y) * idir.y;
            float c1loz = (node2.y - P.z) * idir.z;
            float c1hiz = (node2.w - P.z) * idir.z;
            float c1min = max(max(tmin, min(c1lox, c1hix)), max(min(c1loy, c1hiy), min(c1loz, c1hiz)));
            float c1max = min(min(hit_t, max(c1lox, c1hix)), min(max(c1loy, c1hiy), max(c1loz, c1hiz)));

            float dist0 = c0min;
            float dist1 = c1min;

            /* Per-child visibility mask on cnodes.x/.y (the same test
             * bvh_aligned_node_intersect does; spike:173-174). */
            int traverse_mask = 0;
            if ((c0max >= c0min) && ((asuint(cnodes.x) & visibility) != 0u)) traverse_mask |= 1;
            if ((c1max >= c1min) && ((asuint(cnodes.y) & visibility) != 0u)) traverse_mask |= 2;

            /* Child addresses in cnodes.z / cnodes.w (traversal.h:80-81). */
            node_addr = asint(cnodes.z);
            int node_addr_child1 = asint(cnodes.w);

            if (traverse_mask == 3) {
                /* Both children hit: push the farther one (traversal.h:83-95). */
                if (dist1 < dist0) { int tmp = node_addr; node_addr = node_addr_child1; node_addr_child1 = tmp; }
                sp++;
                if (sp >= BVH_STACK_SIZE) return ERR_STACK;
                stack[sp] = node_addr_child1;
            }
            else {
                if (traverse_mask == 2) { node_addr = node_addr_child1; }
                else if (traverse_mask == 0) {
                    if (sp < 0) return ERR_STACK;
                    node_addr = stack[sp];   // traversal.h:101-105
                    sp--;
                }
            }
        }

        /* ---- leaves (geometry OR instance), traversal.h:110-244 --------- */
        if (node_addr < 0) {
            /* Negative address = leaf: index -node_addr-1 into the leaf
             * array (traversal.h:110-111). One leaf = one float4:
             * x = prim_addr (int), y = prim_addr2, w = type
             * (traversal.h:112-116). */
            int leaf_index = -node_addr - 1;
            if ((uint)leaf_index >= sc->leaf_count) return ERR_BAD_INDEX;
            float4* lp = sc->bvh_leaf_nodes + (uint)leaf_index;
            float4 leaf = lp[0];
            int prim_addr = asint(leaf.x);

            if (prim_addr >= 0) {
                /* ---- geometry leaf, traversal.h:114-228 ------------------ */
                int prim_addr2 = asint(leaf.y);
                uint type = asuint(leaf.w);
                if (sp < 0) return ERR_STACK;
                node_addr = stack[sp];       // traversal.h:120-121
                sp--;
                if (prim_addr2 < prim_addr || (uint)prim_addr2 > sc->prim_count) return ERR_BAD_INDEX;

                for (; prim_addr < prim_addr2; prim_addr++) {
                    /* Inside an instance the object is the instance's
                     * (traversal.h:125-127). */
                    uint pobj = (current_object != NO_OBJECT)
                                    ? current_object
                                    : (sc->prim_object + (uint)prim_addr)[0];
                    uint prim = (sc->prim_index + (uint)prim_addr)[0]; // traversal.h:123-124

                    if ((type & PRIMITIVE_TRIANGLE) != 0u) {
                        /* triangle_intersect
                         * (src/kernel/geom/triangle_intersect.h:24-62). */
                        if (pobj >= sc->object_count || prim >= sc->tri_count) return ERR_BAD_INDEX;
                        uint pos_off = (sc->objects + (pobj * OBJECT_WORDS + POSITION_OFFSET_WORD))[0];
                        uint* vi = sc->tri_vindex + prim * 3u; // packed_uint3 = 3 tight uints
                        uint ia = vi[0] + pos_off;
                        uint ib = vi[1] + pos_off;
                        uint ic = vi[2] + pos_off;
                        if (ia >= sc->vert_count || ib >= sc->vert_count || ic >= sc->vert_count) return ERR_BAD_INDEX;

                        /* packed_float3 = 3 tight floats (12-byte stride). */
                        float* pa = sc->tri_verts + ia * 3u;
                        float* pb = sc->tri_verts + ib * 3u;
                        float* pc = sc->tri_verts + ic * 3u;
                        float3 a = float3(pa[0], pa[1], pa[2]);
                        float3 b = float3(pb[0], pb[1], pb[2]);
                        float3 c = float3(pc[0], pc[1], pc[2]);

                        float u, v, t;
                        if (ray_triangle_intersect(P, dir, tmin, hit_t, a, b, c, u, v, t)) {
                            /* Cycles tests prim_visibility inside
                             * triangle_intersect (triangle_intersect.h:49). */
                            uint pvis = (sc->prim_visibility + (uint)prim_addr)[0];
                            if ((pvis & visibility) != 0u) {
                                hit_t = t;
                                hit_u = u;
                                hit_v = v;
                                hit_prim = prim;
                                hit_object = pobj;
                                hit_type = int(PRIMITIVE_TRIANGLE); // types.h:687
                            }
                        }
                    }
                    else {
                        /* DEFERRED: curves (traversal.h:180-204) and points
                         * (:205-225) - M4c scope is triangles only. */
                        return ERR_INSTANCE;
                    }
                }
            }
            else {
                /* ---- instance leaf push, traversal.h:229-244 -------------
                 * prim_addr < 0 means "enter object -prim_addr-1" (the
                 * task's "instance leaves have x<0"). */
                int instance_idx = -prim_addr - 1;
                if ((uint)instance_idx >= sc->prim_count) return ERR_BAD_INDEX;
                uint obj = (sc->prim_object + (uint)instance_idx)[0]; // traversal.h:231
                if (obj >= sc->object_count) return ERR_BAD_INDEX;

                uint obj_node = sc->object_node[obj]; // traversal.h:243
                /* NEGATIVE TEST 2 (gate): pretend the object_node entry for
                 * this object is corrupt. The value is far out of range, so
                 * the bounds check trips and traverse() returns
                 * ERR_BAD_INDEX - the probe reports FAIL, which is the
                 * expected outcome of this fault injection. */
                if ((sc->test_mode & TEST_MODE_CORRUPT_OBJECT_NODE) != 0u)
                    obj_node = 0x3FFFFFFFu;
                if (obj_node >= sc->node_count) return ERR_BAD_INDEX;

                /* bvh_instance_push (object.h:464-477): transform the ray
                 * into object space with the INVERSE transform. KernelObject
                 * words 12-23 are itfm (types.h:1414; counted word offsets,
                 * locked host-side). */
                uint obj_base = obj * OBJECT_WORDS;
                float4 itfm_x = float4(asfloat(sc->objects[obj_base + 12]), asfloat(sc->objects[obj_base + 13]), asfloat(sc->objects[obj_base + 14]), asfloat(sc->objects[obj_base + 15]));
                float4 itfm_y = float4(asfloat(sc->objects[obj_base + 16]), asfloat(sc->objects[obj_base + 17]), asfloat(sc->objects[obj_base + 18]), asfloat(sc->objects[obj_base + 19]));
                float4 itfm_z = float4(asfloat(sc->objects[obj_base + 20]), asfloat(sc->objects[obj_base + 21]), asfloat(sc->objects[obj_base + 22]), asfloat(sc->objects[obj_base + 23]));

                P = transform_point(itfm_x, itfm_y, itfm_z, P);              // object.h:473
                float3 new_dir_raw = transform_direction(itfm_x, itfm_y, itfm_z, dir);
                dir = float3(clamp_component(new_dir_raw.x),   // object.h:475
                             clamp_component(new_dir_raw.y),
                             clamp_component(new_dir_raw.z));
                idir = float3(1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z);     // object.h:476

                sp++;                                                        // traversal.h:239-241
                if (sp >= BVH_STACK_SIZE) return ERR_STACK;
                stack[sp] = ENTRYPOINT_SENTINEL;

                current_object = obj;
                node_addr = int(obj_node);
            }
        }

        /* ---- instance pop, traversal.h:246-257 --------------------------
         * Reaching the sentinel means we finished an object's BVH: restore
         * the world-space ray and continue with the pushed node.
         * bvh_instance_pop (object.h:502-510) re-derives P/dir from
         * ray->P/ray->D; the spike's save/restore is equivalent. */
        if (node_addr == ENTRYPOINT_SENTINEL) {
            /* NEGATIVE TEST 1 (gate): skip the instance pop. The next
             * top-level nodes get tested with the object-space ray, so the
             * traversal returns wrong hits (or an error) and the probe
             * reports FAIL - the expected outcome of this fault. */
            if ((sc->test_mode & TEST_MODE_SKIP_POP) == 0u) {
                P = orig_P;
                float3 restored = float3(clamp_component(orig_Din.x),
                                         clamp_component(orig_Din.y),
                                         clamp_component(orig_Din.z));
                dir = restored;
                idir = float3(1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z);
                tmin = orig_tmin;
                tmax = orig_tmax;
                current_object = NO_OBJECT;
            }

            if (sp >= 0) {
                node_addr = stack[sp];   // traversal.h:255-256
                sp--;
            }
        }

    } while (node_addr != ENTRYPOINT_SENTINEL);  // traversal.h:258

    /* DEFERRED (M4c scope, closest-hit only):
     * - intersection_skip_self_shadow (traversal.h:129)
     * - shadow linking (traversal.h:133-137)
     * - shadow-ray early exit (traversal.h:152-155)
     * - motion-blur instance push (object.h:482-496)
     * - unaligned/hair nodes (traversal.h:9-13, BVH_FEATURE(BVH_HAIR))
     * - shadow_all / shadow_local / volume / volume_all variants. */
    return 0u;
}
