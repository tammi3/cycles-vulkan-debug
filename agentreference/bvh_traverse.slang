// bvh_traverse.slang : walk the BVH that Cycles built, on the GPU.
// Updated for M2: Instance leaf support.
//
// Original code:
//   SPDX-FileCopyrightText: 2009-2010 NVIDIA Corporation
//   SPDX-FileCopyrightText: 2009-2012 Intel Corporation
//   SPDX-FileCopyrightText: 2011-2022 Blender Foundation
//   SPDX-License-Identifier: Apache-2.0

struct Hit { float t; uint object; uint prim; };

// M2 UPDATED: Size is now 112 bytes.
struct SceneData {
    float4* bvh_nodes;        // offset  0
    float4* bvh_leaf_nodes;   // offset  8
    uint* prim_index;         // offset 16
    uint* prim_visibility;    // offset 24
    uint* prim_object;        // offset 32
    uint* tri_vindex;         // offset 40
    float* tri_verts;         // offset 48
    uint* objects;            // offset 56
    uint* object_node;        // offset 64  <-- M2 ADDITION
    int root;                 // offset 72
    uint node_count;          // offset 76
    uint leaf_count;          // offset 80
    uint prim_count;          // offset 84
    uint tri_count;           // offset 88
    uint vert_count;          // offset 92
    uint object_count;        // offset 96
    uint object_node_count;   // offset 100 <-- M2 ADDITION
    uint pad0;                // offset 104 <-- M2 ADDITION
    uint pad1;                // offset 108 <-- M2 ADDITION
};

struct Params {
    SceneData* scene;  // offset  0
    Hit* out;          // offset  8
    uint width;        // offset 16
    uint height;       // offset 20
    uint visibility;   // offset 24
    float ox, oy, oz;  // offset 28
    float lx, ly, lz;  // offset 40
    float ux, uy, uz;  // offset 52
    float vx, vy, vz;  // offset 64
};

[[vk::push_constant]] ConstantBuffer<Params> params;

static const int ENTRYPOINT_SENTINEL = 0x76543210;
static const int BVH_STACK_SIZE = 192;
static const uint MAX_ITERATIONS = 1000000u;
static const uint OBJECT_WORDS = 64u;
static const uint POSITION_OFFSET_WORD = 42u;
static const uint PRIMITIVE_TRIANGLE = 1u;

static const uint NO_PRIM = 0xFFFFFFFFu;
static const uint ERR_LOOP = 0xFFFFFFFBu;
static const uint ERR_STACK = 0xFFFFFFFCu;
static const uint ERR_BAD_INDEX = 0xFFFFFFFDu;
static const uint ERR_INSTANCE = 0xFFFFFFFEu;

float copysign_bits(float m, float s) {
    return asfloat((asuint(m) & 0x7FFFFFFFu) | (asuint(s) & 0x80000000u));
}

float clamp_component(float c) {
    const float ooeps = 8.271806E-25f;
    return (abs(c) > ooeps) ? c : copysign_bits(ooeps, c);
}

// --- M2 ADDITION: Transform helpers ---
float3 transform_point(float4 tx, float4 ty, float4 tz, float3 a) {
    return float3(
        tx.x * a.x + tx.y * a.y + tx.z * a.z + tx.w,
        ty.x * a.x + ty.y * a.y + ty.z * a.z + ty.w,
        tz.x * a.x + tz.y * a.y + tz.z * a.z + tz.w
    );
}
float3 transform_direction(float4 tx, float4 ty, float4 tz, float3 a) {
    return float3(
        tx.x * a.x + tx.y * a.y + tx.z * a.z,
        ty.x * a.x + ty.y * a.y + ty.z * a.z,
        tz.x * a.x + tz.y * a.y + tz.z * a.z
    );
}
// --------------------------------------

float3 rt_cross(float3 a, float3 b) {
    return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
float rt_dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

bool ray_triangle_intersect(float3 P, float3 D, float tmin, float tmax, float3 a, float3 b, float3 c, out float ou, out float ov, out float ot) {
    ou = 0.0f; ov = 0.0f; ot = 0.0f;
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

uint traverse(SceneData* sc, float3 P, float3 Din, float tmin, float tmax, uint visibility,
              out float hit_t, out uint hit_prim, out uint hit_object) {
    int stack[BVH_STACK_SIZE];
    stack[0] = ENTRYPOINT_SENTINEL;
    int sp = 0;
    int node_addr = sc->root;

    // --- M2 ADDITION: Save original ray state ---
    uint current_object = 0xFFFFFFFFu;
    float3 orig_P = P;
    float3 orig_Din = Din;
    float orig_tmin = tmin;
    float orig_tmax = tmax;
    // --------------------------------------------

    float3 dir = float3(clamp_component(Din.x), clamp_component(Din.y), clamp_component(Din.z));
    float3 idir = float3(1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z);

    hit_t = tmax;
    hit_prim = NO_PRIM;
    hit_object = 0xFFFFFFFFu; // <-- ADDED: Initialize to OBJECT_NONE on miss
    uint iterations = 0;

    do {
        while (node_addr >= 0 && node_addr != ENTRYPOINT_SENTINEL) {
            iterations++;
            if (iterations > MAX_ITERATIONS) return ERR_LOOP;
            if ((uint)node_addr + 3u >= sc->node_count) return ERR_BAD_INDEX;

            float4* np = sc->bvh_nodes + (uint)node_addr;
            float4 cnodes = np[0], node0 = np[1], node1 = np[2], node2 = np[3];

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

            int traverse_mask = 0;
            if ((c0max >= c0min) && ((asuint(cnodes.x) & visibility) != 0u)) traverse_mask |= 1;
            if ((c1max >= c1min) && ((asuint(cnodes.y) & visibility) != 0u)) traverse_mask |= 2;

            node_addr = asint(cnodes.z);
            int node_addr_child1 = asint(cnodes.w);

            if (traverse_mask == 3) {
                if (dist1 < dist0) { int tmp = node_addr; node_addr = node_addr_child1; node_addr_child1 = tmp; }
                sp++;
                if (sp >= BVH_STACK_SIZE) return ERR_STACK;
                stack[sp] = node_addr_child1;
            } else {
                if (traverse_mask == 2) { node_addr = node_addr_child1; }
                else if (traverse_mask == 0) {
                    if (sp < 0) return ERR_STACK;
                    node_addr = stack[sp];
                    sp--;
                }
            }
        }

        if (node_addr < 0) {
            int leaf_index = -node_addr - 1;
            if ((uint)leaf_index >= sc->leaf_count) return ERR_BAD_INDEX;
            float4* lp = sc->bvh_leaf_nodes + (uint)leaf_index;
            float4 leaf = lp[0];
            int prim_addr = asint(leaf.x);

            if (prim_addr >= 0) {
                int prim_addr2 = asint(leaf.y);
                uint type = asuint(leaf.w);
                if (sp < 0) return ERR_STACK;
                node_addr = stack[sp];
                sp--;
                if (prim_addr2 < prim_addr || (uint)prim_addr2 > sc->prim_count) return ERR_BAD_INDEX;

                for (; prim_addr < prim_addr2; prim_addr++) {
                    // --- M2 ADDITION: Use instance object ID if inside an instance ---
                    uint pobj = (current_object != 0xFFFFFFFFu) ? current_object : (sc->prim_object + (uint)prim_addr)[0];
                    uint prim = (sc->prim_index + (uint)prim_addr)[0];

                    if ((type & PRIMITIVE_TRIANGLE) != 0u) {
                        if (pobj >= sc->object_count || prim >= sc->tri_count) return ERR_BAD_INDEX;
                        uint pos_off = (sc->objects + (pobj * OBJECT_WORDS + POSITION_OFFSET_WORD))[0];
                        uint* vi = sc->tri_vindex + prim * 3u;
                        uint ia = vi[0] + pos_off;
                        uint ib = vi[1] + pos_off;
                        uint ic = vi[2] + pos_off;
                        if (ia >= sc->vert_count || ib >= sc->vert_count || ic >= sc->vert_count) return ERR_BAD_INDEX;

                        float* pa = sc->tri_verts + ia * 3u;
                        float* pb = sc->tri_verts + ib * 3u;
                        float* pc = sc->tri_verts + ic * 3u;
                        float3 a = float3(pa[0], pa[1], pa[2]);
                        float3 b = float3(pb[0], pb[1], pb[2]);
                        float3 c = float3(pc[0], pc[1], pc[2]);

                        float u, v, t;
                        if (ray_triangle_intersect(P, dir, tmin, hit_t, a, b, c, u, v, t)) {
                            uint pvis = (sc->prim_visibility + (uint)prim_addr)[0];
                            if ((pvis & visibility) != 0u) {
                                hit_t = t;
                                hit_prim = prim;
                                hit_object = pobj; // <-- ADDED
                            }
                        }
                    } else {
                        return ERR_INSTANCE;
                    }
                }
            } else {
                // --- M2 ADDITION: INSTANCE LEAF PUSH ---
                int instance_idx = -prim_addr - 1;
                if ((uint)instance_idx >= sc->prim_count) return ERR_BAD_INDEX;
                uint obj = (sc->prim_object + (uint)instance_idx)[0];
                if (obj >= sc->object_count) return ERR_BAD_INDEX;
                if (sc->object_node[obj] >= sc->node_count) return ERR_BAD_INDEX;

                uint obj_base = obj * OBJECT_WORDS;
                float4 itfm_x = float4(asfloat(sc->objects[obj_base + 12]), asfloat(sc->objects[obj_base + 13]), asfloat(sc->objects[obj_base + 14]), asfloat(sc->objects[obj_base + 15]));
                float4 itfm_y = float4(asfloat(sc->objects[obj_base + 16]), asfloat(sc->objects[obj_base + 17]), asfloat(sc->objects[obj_base + 18]), asfloat(sc->objects[obj_base + 19]));
                float4 itfm_z = float4(asfloat(sc->objects[obj_base + 20]), asfloat(sc->objects[obj_base + 21]), asfloat(sc->objects[obj_base + 22]), asfloat(sc->objects[obj_base + 23]));

                float3 new_P = transform_point(itfm_x, itfm_y, itfm_z, P);
                float3 new_dir_raw = transform_direction(itfm_x, itfm_y, itfm_z, dir);
                float3 new_dir = float3(clamp_component(new_dir_raw.x), clamp_component(new_dir_raw.y), clamp_component(new_dir_raw.z));

                P = new_P;
                dir = new_dir;
                idir = float3(1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z);

                sp++;
                if (sp >= BVH_STACK_SIZE) return ERR_STACK;
                stack[sp] = ENTRYPOINT_SENTINEL;

                current_object = obj;
                node_addr = sc->object_node[obj];
            }
        }

        // --- M2 ADDITION: INSTANCE POP ---
        if (node_addr == ENTRYPOINT_SENTINEL) {
            P = orig_P;
            float3 restored_dir = float3(clamp_component(orig_Din.x), clamp_component(orig_Din.y), clamp_component(orig_Din.z));
            dir = restored_dir;
            idir = float3(1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z);
            tmin = orig_tmin;
            tmax = orig_tmax;
            current_object = 0xFFFFFFFFu;

            if (sp >= 0) {
                node_addr = stack[sp];
                sp--;
            }
        }

    } while (node_addr != ENTRYPOINT_SENTINEL);

    return 0u;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint x = tid.x;
    uint y = tid.y;
    if (x >= params.width || y >= params.height) return;

    float s = ((float)x + 0.5f) / (float)params.width;
    float t_img = 1.0f - ((float)y + 0.5f) / (float)params.height;
    float3 o = float3(params.ox, params.oy, params.oz);
    float3 L = float3(params.lx, params.ly, params.lz);
    float3 U = float3(params.ux, params.uy, params.uz);
    float3 V = float3(params.vx, params.vy, params.vz);
    float3 d = normalize(L + s * U + t_img * V - o);

    float hit_t;
    uint hit_prim;
    uint hit_object;
    uint status = traverse(params.scene, o, d, 1e-4f, 1e30f, params.visibility, hit_t, hit_prim, hit_object);

    Hit* slot = params.out + (y * params.width + x);
    if (status != 0u) {
        slot->t = -1.0f;
        slot->object = 0xFFFFFFFFu;
        slot->prim = status;
    } else {
        slot->t = hit_t;
        slot->object = hit_object;
        slot->prim = hit_prim;
    }
}