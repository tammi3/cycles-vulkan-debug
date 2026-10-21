// traverse_probe.slang : M4c gate probe. Two compute entry points,
// compiled into TWO separate .spv files with slangc (see the host side,
// src/device/vulkan/kernels.cpp: vulkan_traverse_probe()):
//
//   traverse_prepare_main -> traverse_prepare.spv
//     One thread. Reads the LIVE KernelParamsVulkan buffer through the
//     Slang layout, writes the nine scene-array device pointers and the
//     BVH root into the probe header, and checksums bvh_nodes + tri_verts
//     so the host can prove the live scene matches the cycles_bvh_dump
//     sidecar (M4c decision Q1: mismatch => INCONCLUSIVE, not FAIL).
//
//   traverse_probe_main -> traverse_probe.spv
//     64x64 threads (8x8 workgroups). One pinhole ray per thread through
//     the ported traverse(); writes the Intersection (types.h:614) and a
//     per-ray status word.
//
// Syntax follows the M4b ABI probe (abi_probe.slang), the only in-tree
// shader proven to build and run on the RX 570: push constants carry raw
// buffer device addresses as typed pointers, no descriptor sets at all.

#include "vulkan_data.slang" // KernelParamsVulkan (frozen M4b slot layout)
#include "traverse.slang"    // ported traversal + Intersection + header

/* FNV-1a 32-bit, byte-wise, exactly like the host does over the dump's
 * raw .bin bytes (little-endian: byte order inside a word is identical on
 * GPU and host). */
static const uint FNV_BASIS = 2166136261u;
static const uint FNV_PRIME = 16777619u;

uint fnv_word(uint h, uint w)
{
    h = (h ^ (w & 0xFFu)) * FNV_PRIME;
    h = (h ^ ((w >> 8) & 0xFFu)) * FNV_PRIME;
    h = (h ^ ((w >> 16) & 0xFFu)) * FNV_PRIME;
    h = (h ^ ((w >> 24) & 0xFFu)) * FNV_PRIME;
    return h;
}

/* Sentinel written by the prepare kernel; proves the dispatch ran. */
static const uint PREPARE_SENTINEL = 0xB10C5EEDu;

/* -------------------------------------------------------------------------
 * Prepare pass. Push constants (56 bytes):
 *   hdr        typed view of the header buffer (the host filled counts,
 *              camera and flags; the pointer slots + root are OUR job)
 *   hdr_raw    raw uint64 view of the SAME buffer: bytes 0-71 hold exactly
 *              the nine pointer slots. Writing through the raw view and
 *              reading through the typed view keeps every value flowing as
 *              the same 8 bytes (no conversion anywhere)
 *   hdr_root   &hdr->root (a separate int* so writing root cannot clobber
 *              node_count at the next word)
 *   params     typed view of the LIVE KernelParamsVulkan buffer (device
 *              address from VulkanDevice::kernel_params_address())
 *   out        checksum block: [0] Slang sizeof(TraverseHeader),
 *              [1] fnv(bvh_nodes), [2] fnv(tri_verts), [3] root bits,
 *              [4] sentinel, [5..644] the whole live KernelData blob as
 *              raw words (M4c debugging aid: the host greps it for the
 *              KernelBVH signature so the root slot cannot hide)
 *   params_raw `params` viewed as plain uint32 words
 *   blob_root_word  word index of KernelData.bvh.root inside the KernelData
 *              blob; the host computed it with offsetof() and locked it
 *              with a static_assert (M4c decision Q3: root is read from
 *              the blob exactly where production kernels read it). */
struct PreparePush
{
    TraverseHeader* hdr;
    uint64_t* hdr_raw;
    int* hdr_root;
    KernelParamsVulkan* params;
    uint* out;
    uint* params_raw;      // same address as `params`, raw uint32 words
    uint blob_root_word;
};

[[vk::push_constant]]
ConstantBuffer<PreparePush> prepare_pc;

[shader("compute")]
[numthreads(1, 1, 1)]
void traverse_prepare_main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x != 0u)
        return;

    /* Slang-computed size of the shared header (host asserts 168). */
    prepare_pc.out[0] = uint(sizeof(TraverseHeader));

    /* The nine scene-array device addresses, straight from the live params
     * slots (offsets frozen in vulkan_data.slang:256-308, sourced from
     * kernel/data_arrays.h:15-37). */
    prepare_pc.hdr_raw[0] = prepare_pc.params->bvh_nodes;
    prepare_pc.hdr_raw[1] = prepare_pc.params->bvh_leaf_nodes;
    prepare_pc.hdr_raw[2] = prepare_pc.params->prim_index;
    prepare_pc.hdr_raw[3] = prepare_pc.params->prim_visibility;
    prepare_pc.hdr_raw[4] = prepare_pc.params->prim_object;
    prepare_pc.hdr_raw[5] = prepare_pc.params->tri_vindex;
    prepare_pc.hdr_raw[6] = prepare_pc.params->tri_verts;
    prepare_pc.hdr_raw[7] = prepare_pc.params->objects;
    prepare_pc.hdr_raw[8] = prepare_pc.params->object_node;

    /* BVH root: first int of KernelBVH inside the KernelData blob
     * (src/kernel/data_template.h:51-52). Reading it through the blob at
     * the host-computed word index also tests the blob upload.
     *
     * We read through the RAW uint32 view, not the typed
     * `params->data.words[]` view: Slang laid out the `uint words[640]`
     * member of KernelDataBlob with a stride that does not match the
     * host's flat uint32 array (the typed read returned 0x80000803 at
     * word 356, while the raw echo at the SAME index returned the dump's
     * 0x00000000). The raw view matched the dump byte-for-byte, so it is
     * the view that agrees with the C++ layout. */
    *prepare_pc.hdr_root = int(prepare_pc.params_raw[prepare_pc.blob_root_word]);
    prepare_pc.out[3] = uint(*prepare_pc.hdr_root);

    /* Checksum the live arrays the traversal will read. Counts come from
     * the header (dump meta, Q2); if the live scene is smaller than the
     * dump we would read a little past the array - the checksum would
     * differ anyway and the host reports INCONCLUSIVE. */
    uint h = FNV_BASIS;
    for (uint i = 0; i < prepare_pc.hdr->node_count; i++)
    {
        float4 v = prepare_pc.hdr->bvh_nodes[i];
        h = fnv_word(h, asuint(v.x));
        h = fnv_word(h, asuint(v.y));
        h = fnv_word(h, asuint(v.z));
        h = fnv_word(h, asuint(v.w));
    }
    prepare_pc.out[1] = h;

    h = FNV_BASIS;
    for (uint i = 0; i < prepare_pc.hdr->vert_count * 3u; i++)
        h = fnv_word(h, asuint(prepare_pc.hdr->tri_verts[i]));
    prepare_pc.out[2] = h;

    prepare_pc.out[4] = PREPARE_SENTINEL;

    /* Echo the whole live KernelData blob so the host can locate the
     * KernelBVH block empirically (640 words = 2560 bytes). */
    for (uint i = 0u; i < 640u; i++)
        prepare_pc.out[5u + i] = prepare_pc.params_raw[i];
}

/* -------------------------------------------------------------------------
 * Ray pass. Push constants (24 bytes): the header, the Intersection
 * output and the per-ray status word. */
struct ProbePush
{
    TraverseHeader* hdr;
    Intersection* out;
    int* status;
};

[[vk::push_constant]]
ConstantBuffer<ProbePush> probe_pc;

[shader("compute")]
[numthreads(8, 8, 1)]
void traverse_probe_main(uint3 tid : SV_DispatchThreadID)
{
    uint x = tid.x;
    uint y = tid.y;
    if (x >= probe_pc.hdr->width || y >= probe_pc.hdr->height)
        return;

    /* Pinhole ray through pixel (x, y). Same math as the M2c spike's main
     * (bvh_traverse.slang:301-307) and agentreference/raytri_cpu.h:85-91:
     * the image-plane point for pixel fractions (s, t) is l + s*u + t*v,
     * y = 0 is the TOP row. The host computes the reference rays from the
     * very same l/u/v floats it put in the header. */
    float s = ((float)x + 0.5f) / (float)probe_pc.hdr->width;
    float t_img = 1.0f - ((float)y + 0.5f) / (float)probe_pc.hdr->height;
    float3 o = float3(probe_pc.hdr->cam_ox, probe_pc.hdr->cam_oy, probe_pc.hdr->cam_oz);
    float3 l = float3(probe_pc.hdr->cam_lx, probe_pc.hdr->cam_ly, probe_pc.hdr->cam_lz);
    float3 u = float3(probe_pc.hdr->cam_ux, probe_pc.hdr->cam_uy, probe_pc.hdr->cam_uz);
    float3 v = float3(probe_pc.hdr->cam_vx, probe_pc.hdr->cam_vy, probe_pc.hdr->cam_vz);
    float3 d = normalize(l + s * u + t_img * v - o);

    float hit_t, hit_u, hit_v;
    uint hit_prim, hit_object, hit_type;
    /* tmin/tmax identical to the CPU reference (raytri_cpu.h:21-22). */
    uint status = traverse(probe_pc.hdr, o, d, 1e-4f, 1e30f,
                           probe_pc.hdr->visibility,
                           hit_t, hit_u, hit_v, hit_prim, hit_object, hit_type);

    uint idx = y * probe_pc.hdr->width + x;
    probe_pc.status[idx] = int(status);
    /* On an error the intersection words are still written (all-sentinel
     * values) so every output word is initialized and the host's
     * unwritten-word check stays meaningful. The status word - not the
     * intersection - is what tells hit/miss/error apart (Q7). */
    probe_pc.out[idx].t = (status == 0u) ? hit_t : -1.0f;
    probe_pc.out[idx].u = hit_u;
    probe_pc.out[idx].v = hit_v;
    probe_pc.out[idx].prim = int(hit_prim);
    probe_pc.out[idx].object = int(hit_object);
    probe_pc.out[idx].type = int(hit_type);
}
