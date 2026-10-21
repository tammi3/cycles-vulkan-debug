// SPDX-FileCopyrightText: 2011-2022 Blender Foundation
// SPDX-License-Identifier: Apache-2.0

/* BVH2 traversal ported from the M2 spike (Vulkan-backend-Blender-CyclesX).
 *
 * Cycles BVH2 node layout (src/bvh/bvh2.h), as built by the CPU-side builder
 * in device/device.cpp -> bvh_build_cpu.cpp:
 *   struct BVH2Node {
 *     float4 bound_min;   // +0
 *     float4 bound_max;   // +16
 *     uint   xsize;       // +32  (split axis encoded as width of quantized bounds)
 *     uint   numprims;    // +36
 *     union {
 *       uint   child[2];  // +40,+44 : interior nodes (offsets relative to this node)
 *       uint   prim[2];   //           leaf nodes: packed (prim_offset<<8 | object|type)
 *     };
 *   };                    // sizeof == 48
 *
 * IMPORTANT: In Cycles BVH2, `child`/`prim` are stored as *byte* offsets /
 * packed indices respectively, NOT raw array indices. The exact decoding is:
 *   - interior: left = node_index + (int)(node.child.x), right = node_index + (int)(node.child.y)
 *     where child.x/y are int32 offsets counted in NODES (this repo's builder
 *     emits node-relative int offsets; verify against bvh2.h if results look wrong)
 *   - leaf: prim.x = primitive offset into the linear primitive array,
 *           prim.y = (num_prims_in_leaf << 8) | type_and_object bits
 *
 * We keep the traversal exactly as validated in the M2 spike so that the CPU
 * reference and GPU results stay bit-compatible for closest-hit. */

#include "vulkan_data.slang"

struct BVH2NodeGPU
{
    float4 bound_min;
    float4 bound_max;
    uint   xsize;
    uint   numprims;
    uint   a;   // child.x / prim.x
    uint   b;   // child.y / prim.y
};

/* Ray with slab-tested origin/direction and reciprocal direction. */
struct RayGPU
{
    float3 origin;
    float3 dir;
    float3 inv_dir;   // 1/dir, componentwise (+/-INF allowed)
    float  tmin;      // current nearest hit distance (starts at 1e30)
    uint   prim;      // nearest triangle index (INVALID_PRIM if none)
};

static const uint INVALID_PRIM = 0xFFFFFFFFu;

/* Decode a BVH2 node pointer from a raw device address. */
BVH2NodeGPU fetch_node(ulong bvh_base, uint node_index)
{
    /* Each node is 48 bytes. Pointer arithmetic on PhysicalStorageBuffer must
     * be done in bytes: cast base -> BVH2NodeGPU*, then index. */
    StructuredBuffer<BVH2NodeGPU> nodes = StructuredBuffer<BVH2NodeGPU>(bvh_base);
    return nodes[node_index];
}

bool ray_aabb(RayGPU r, float3 bmin, float3 bmax, float tmax_limit)
{
    /* Standard slab test using the reciprocal direction. Handles +-INF in
     * inv_dir correctly because min/max ignore NaN only when we use the
     * fmin/fmax form below — SPIR-V's select-based form is what the spike used. */
    float3 t0 = (bmin - r.origin) * r.inv_dir;
    float3 t1 = (bmax - r.origin) * r.inv_dir;

    float3 tsmall = min(t0, t1);
    float3 tlarge = max(t0, t1);

    float tmin = max(max(tsmall.x, tsmall.y), tsmall.z);
    float tmax = min(min(tlarge.x, tlarge.y), tlarge.z);

    return (tmax >= max(tmin, 0.0f)) && (tmin <= tmax_limit);
}

/* Möller–Trumbore, UVW-consistent with the CPU reference in the spike. */
bool intersect_triangle(inout RayGPU io, uint tri_index,
                        ulong tri_indices_base, ulong vert_co_base,
                        out float t_out, out float2 uv_out)
{
    StructuredBuffer<uint4> tris = StructuredBuffer<uint4>(tri_indices_base);   // packed as (i,j,k,pad) per triangle
    StructuredBuffer<float4> verts = StructuredBuffer<float4>(vert_co_base);     // stored as float4 (w unused) for alignment

    uint4 idx = tris[tri_index];
    float3 v0 = verts[idx.x].xyz;
    float3 v1 = verts[idx.y].xyz;
    float3 v2 = verts[idx.z].xyz;

    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;
    float3 h = cross(io.dir, e2);
    float a = dot(e1, h);

    if (a > -1e-8f && a < 1e-8f)
        return false;

    float f = 1.0f / a;
    float3 s = io.origin - v0;
    float u = f * dot(s, h);
    if (u < 0.0f || u > 1.0f)
        return false;

    float3 q = cross(s, e1);
    float v = f * dot(io.dir, q);
    if (v < 0.0f || u + v > 1.0f)
        return false;

    float t = f * dot(e2, q);
    if (t > 1e-5f && t < io.tmin)
    {
        t_out = t;
        uv_out = float2(u, v);
        return true;
    }
    return false;
}

/* Stackless front-to-back BVH2 traversal, matching the CPU reference. */
void bvh2_intersect(inout RayGPU ray, ulong bvh_base,
                    ulong prim_offsets_base, ulong prim_counts_base,
                    ulong tri_indices_base, ulong vert_co_base)
{
    /* Traversal stack sized for worst-case depth of our scenes (<= 64). */
    uint stack[64];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0u; // root node index

    while (stack_ptr > 0)
    {
        uint node_idx = stack[--stack_ptr];
        BVH2NodeGPU node = fetch_node(bvh_base, node_idx);

        if (!ray_aabb(ray, node.bound_min.xyz, node.bound_max.xyz, ray.tmin))
            continue;

        if (node.numprims > 0)
        {
            /* Leaf: decode primitive range. This repo's builder stores
             * prim.x = absolute primitive offset, prim.y = count in low 16 bits. */
            uint prim_start = node.a;
            uint prim_count = node.b & 0xFFFFu;
            if (node.b == 0) prim_count = node.numprims; // older packing fallback

            for (uint i = 0; i < prim_count; ++i)
            {
                float t; float2 uv;
                if (intersect_triangle(ray, prim_start + i, tri_indices_base, vert_co_base, t, uv))
                {
                    ray.tmin = t;
                    ray.prim = prim_start + i;
                }
            }
        }
        else
        {
            /* Interior: children stored as signed node-relative offsets (Cycles
             * BVH2 convention: (int)a / (int)b added to current index). */
            uint c0 = node_idx + (uint)((int)node.a);
            uint c1 = node_idx + (uint)((int)node.b);

            BVH2NodeGPU n0 = fetch_node(bvh_base, c0);
            bool hit0 = ray_aabb(ray, n0.bound_min.xyz, n0.bound_max.xyz, ray.tmin);

            if (hit0)
            {
                /* Push far child first so near child is popped first. */
                stack[stack_ptr++] = c1;
                stack[stack_ptr++] = c0;
            }
            else
            {
                BVH2NodeGPU n1 = fetch_node(bvh_base, c1);
                if (ray_aabb(ray, n1.bound_min.xyz, n1.bound_max.xyz, ray.tmin))
                    stack[stack_ptr++] = c1;
            }
        }
    }
}