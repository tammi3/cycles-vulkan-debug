// M2c: trace rays through the BVH that CYCLES built, and check the answer three ways.
// Usage:  m2c_bvh <bvh_traverse.spv | -> <dump_dir> [width] [height] [visibility_mask_hex] [cpu_threads]

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "bvh_cpu.h"
#include "bvh_dump.h"
#include "ppm.h"
#include "raytri_cpu.h"
#include "vk_util.h"

// NOTE: struct Hit is defined in raytri_cpu.h. Do NOT redefine it here.

struct WorldTri {
  float v[9];
  uint32_t obj;
  uint32_t prim;
};

// ---------------------------------------------------------------------------
// Structs shared with the shader. MUST match bvh_traverse.slang offset by offset.
// ---------------------------------------------------------------------------
struct SceneDataGpu {
  uint64_t bvh_nodes;        // 0
  uint64_t bvh_leaf_nodes;   // 8
  uint64_t prim_index;       // 16
  uint64_t prim_visibility;  // 24
  uint64_t prim_object;      // 32
  uint64_t tri_vindex;       // 40
  uint64_t tri_verts;        // 48
  uint64_t objects;          // 56
  uint64_t object_node;      // 64
  int32_t root;              // 72
  uint32_t node_count;       // 76
  uint32_t leaf_count;       // 80
  uint32_t prim_count;       // 84
  uint32_t tri_count;        // 88
  uint32_t vert_count;       // 92
  uint32_t object_count;     // 96
  uint32_t object_node_count;// 100
  uint32_t pad[2];           // 104
};
static_assert(offsetof(SceneDataGpu, bvh_nodes) == 0, "layout");
static_assert(offsetof(SceneDataGpu, bvh_leaf_nodes) == 8, "layout");
static_assert(offsetof(SceneDataGpu, prim_index) == 16, "layout");
static_assert(offsetof(SceneDataGpu, prim_visibility) == 24, "layout");
static_assert(offsetof(SceneDataGpu, prim_object) == 32, "layout");
static_assert(offsetof(SceneDataGpu, tri_vindex) == 40, "layout");
static_assert(offsetof(SceneDataGpu, tri_verts) == 48, "layout");
static_assert(offsetof(SceneDataGpu, objects) == 56, "layout");
static_assert(offsetof(SceneDataGpu, object_node) == 64, "layout");
static_assert(offsetof(SceneDataGpu, root) == 72, "layout");
static_assert(offsetof(SceneDataGpu, node_count) == 76, "layout");
static_assert(offsetof(SceneDataGpu, leaf_count) == 80, "layout");
static_assert(offsetof(SceneDataGpu, prim_count) == 84, "layout");
static_assert(offsetof(SceneDataGpu, tri_count) == 88, "layout");
static_assert(offsetof(SceneDataGpu, vert_count) == 92, "layout");
static_assert(offsetof(SceneDataGpu, object_count) == 96, "layout");
static_assert(offsetof(SceneDataGpu, object_node_count) == 100, "layout");
static_assert(offsetof(SceneDataGpu, pad) == 104, "layout");
static_assert(sizeof(SceneDataGpu) == 112, "layout");

struct TraversePush {
  uint64_t scene;       // 0
  uint64_t out;         // 8
  uint32_t width;       // 16
  uint32_t height;      // 20
  uint32_t visibility;  // 24
  float o[3];           // 28
  float l[3];           // 40
  float u[3];           // 52
  float v[3];           // 64
};
static_assert(offsetof(TraversePush, scene) == 0, "layout");
static_assert(offsetof(TraversePush, out) == 8, "layout");
static_assert(offsetof(TraversePush, width) == 16, "layout");
static_assert(offsetof(TraversePush, height) == 20, "layout");
static_assert(offsetof(TraversePush, visibility) == 24, "layout");
static_assert(offsetof(TraversePush, o) == 28, "layout");
static_assert(offsetof(TraversePush, l) == 40, "layout");
static_assert(offsetof(TraversePush, u) == 52, "layout");
static_assert(offsetof(TraversePush, v) == 64, "layout");
static_assert(sizeof(TraversePush) <= 128, "push constants are limited to 128 bytes on this GPU");

static constexpr uint32_t kCanaryHits = 16;
static constexpr uint32_t kCanaryWord = 0xCAFEBABEu;
static constexpr uint32_t kUnwrittenWord = 0xDEADBEEFu;

static uint32_t float_bits(float f) {
  uint32_t u; std::memcpy(&u, &f, sizeof(u)); return u;
}

// ---------------------------------------------------------------------------
// Compare two hit lists.
// ---------------------------------------------------------------------------
constexpr double kEdgeTol = 1e-5;

struct EdgeProbe {
  double margin = 1e30;
  int tri = -1;
};

static EdgeProbe probe_edge(const std::vector<WorldTri> &world_tris, const float *o, const float *d) {
  EdgeProbe best;
  for (size_t i = 0; i < world_tris.size(); i++) {
    const float *t = world_tris[i].v;
    double e1[3], e2[3], tv[3], dd[3];
    for (int a = 0; a < 3; a++) {
      e1[a] = (double)t[3 + a] - t[a];
      e2[a] = (double)t[6 + a] - t[a];
      tv[a] = (double)o[a] - t[a];
      dd[a] = d[a];
    }
    auto cross = [](const double *x, const double *y, double *r) {
      r[0] = x[1] * y[2] - x[2] * y[1]; r[1] = x[2] * y[0] - x[0] * y[2]; r[2] = x[0] * y[1] - x[1] * y[0];
    };
    auto dot = [](const double *x, const double *y) { return x[0] * y[0] + x[1] * y[1] + x[2] * y[2]; };
    double p[3], q[3];
    cross(dd, e2, p);
    const double det = dot(e1, p);
    if (std::fabs(det) < 1e-300) continue;
    const double u = dot(tv, p) / det;
    cross(tv, e1, q);
    const double v = dot(dd, q) / det;
    const double tt = dot(e2, q) / det;
    if (tt <= 0.0) continue;
    const double w = 1.0 - u - v;
    const double mn = std::min(u, std::min(v, w));
    if (mn < -kEdgeTol) continue;
    if (mn < best.margin) { best.margin = std::fabs(mn); best.tri = (int)i; }
  }
  return best;
}

struct Cmp {
  size_t same = 0, miss = 0, ties = 0, grazing = 0, bad = 0, errors = 0;
  std::vector<size_t> examples;
};

struct CmpContext {
  const std::vector<WorldTri> *world_tris;
  const Camera *cam;
  uint32_t W, H;
};

static Cmp compare_hits(const std::vector<Hit> &ref, const std::vector<Hit> &test, const CmpContext &cx) {
  Cmp c;
  for (size_t i = 0; i < ref.size(); i++) {
    const Hit &r = ref[i], &g = test[i];
    if (cyc::is_error_id(r.prim) || cyc::is_error_id(g.prim)) { c.errors++; continue; }
    bool disagree = false;
    if (r.prim == g.prim && r.object == g.object) {
      if (r.prim == 0xFFFFFFFFu) c.miss++;
      else if (std::fabs(g.t - r.t) <= 1e-4f * std::max(1.0f, r.t)) c.same++;
      else disagree = true;
    } else if (r.prim != 0xFFFFFFFFu && g.prim != 0xFFFFFFFFu && std::fabs(g.t - r.t) <= 1e-3f * std::max(1.0f, r.t)) {
      c.ties++;
    } else {
      disagree = true;
    }
    if (!disagree) continue;
    float d[3];
    primary_ray_dir(*cx.cam, (uint32_t)(i % cx.W), (uint32_t)(i / cx.W), cx.W, cx.H, d);
    const EdgeProbe e = probe_edge(*cx.world_tris, cx.cam->o, d);
    if (e.margin <= kEdgeTol) c.grazing++; else c.bad++;
    if (c.examples.size() < 5) c.examples.push_back(i);
  }
  return c;
}

static void print_tri_diag(const char *who, const std::vector<WorldTri> &world_tris, uint32_t id, const float *o, const float *d) {
  int tri_idx = -1;
  for (size_t i = 0; i < world_tris.size(); i++) { if (world_tris[i].prim == id) { tri_idx = (int)i; break; } }
  if (tri_idx == -1) { std::printf("        %s: prim %u not found in world_tris\n", who, id); return; }
  const float *t = world_tris[tri_idx].v;
  double e1[3], e2[3], v0o[3], dd[3];
  for (int a = 0; a < 3; a++) {
    e1[a] = (double)t[3 + a] - t[a]; e2[a] = (double)t[6 + a] - t[a];
    v0o[a] = (double)t[a] - (double)o[a]; dd[a] = d[a];
  }
  const double n[3] = {e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0]};
  const double nlen = std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
  const double dlen = std::sqrt(dd[0]*dd[0]+dd[1]*dd[1]+dd[2]*dd[2]);
  const double dn = n[0]*dd[0]+n[1]*dd[1]+n[2]*dd[2];
  if (nlen == 0.0 || dn == 0.0) return;
  const double t_exact = (n[0]*v0o[0]+n[1]*v0o[1]+n[2]*v0o[2]) / dn;
  const double sin_graz = std::fabs(dn) / (nlen * dlen);
  std::printf("        %s: tri %u  exact plane distance %.6f  sin(grazing angle) %.2e\n", who, id, t_exact, sin_graz);
}

static void print_cmp(const char *title, const Cmp &c, const std::vector<Hit> &ref, const std::vector<Hit> &test, const CmpContext &cx) {
  const size_t n = ref.size();
  std::printf("%s (%zu pixels)\n", title, n);
  std::printf("  both missed:                     %zu\n", c.miss);
  std::printf("  same triangle, same distance:    %zu\n", c.same);
  std::printf("  edge ties (acceptable):          %zu\n", c.ties);
  std::printf("  edge-grazing disagreements:      %zu\n", c.grazing);
  std::printf("  MISMATCHES (not near an edge):   %zu\n", c.bad);
  std::printf("  error codes:                     %zu\n", c.errors);
  for (size_t i : c.examples) {
    float d[3];
    primary_ray_dir(*cx.cam, (uint32_t)(i % cx.W), (uint32_t)(i / cx.W), cx.W, cx.H, d);
    const EdgeProbe e = probe_edge(*cx.world_tris, cx.cam->o, d);
    std::printf("    pixel (%zu,%zu): ref obj=%u prim=%u t=%.6f | test obj=%u prim=%u t=%.6f | nearest tri: %d, margin %.2e\n",
                i % cx.W, i / cx.W, ref[i].object, ref[i].prim, ref[i].t, test[i].object, test[i].prim, test[i].t, e.tri, e.margin);
    print_tri_diag("ref ", *cx.world_tris, ref[i].prim, cx.cam->o, d);
    print_tri_diag("test", *cx.world_tris, test[i].prim, cx.cam->o, d);
  }
}

static bool cmp_ok(const Cmp &c, size_t n) {
  const size_t grazing_limit = std::max<size_t>(8, n / 50000);
  return c.bad == 0 && c.errors == 0 && c.grazing <= grazing_limit;
}

static Buffer upload(const Context &ctx, const void *data, size_t bytes) {
  Buffer b = create_buffer(ctx, (VkDeviceSize)std::max<size_t>(bytes, 16));
  std::memset(b.mapped, 0, (size_t)b.size);
  if (bytes) std::memcpy(b.mapped, data, bytes);
  return b;
}

template <class Body>
static void parallel_rows(uint32_t rows, unsigned n_threads, const Body &body) {
  n_threads = std::max(1u, std::min<unsigned>(n_threads, rows));
  std::atomic<uint32_t> next_row{0};
  auto worker = [&] {
    for (;;) {
      const uint32_t y = next_row.fetch_add(1, std::memory_order_relaxed);
      if (y >= rows) break;
      body(y);
    }
  };
  std::vector<std::thread> pool;
  pool.reserve(n_threads - 1);
  for (unsigned i = 1; i < n_threads; i++) pool.emplace_back(worker);
  worker();
  for (std::thread &t : pool) t.join();
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <bvh_traverse.spv | -> <dump_dir> [width] [height] [visibility_hex] [cpu_threads]\n", argv[0]);
    return 1;
  }
  const char *spv_path = argv[1];
  const std::string dump_dir = argv[2];
  const uint32_t W = (argc >= 4) ? (uint32_t)std::atoi(argv[3]) : 256;
  const uint32_t H = (argc >= 5) ? (uint32_t)std::atoi(argv[4]) : 256;
  const uint32_t visibility = (argc >= 6) ? (uint32_t)std::strtoul(argv[5], nullptr, 16) : 0xFFFFFFFFu;
  const unsigned hw_threads = std::max(1u, std::thread::hardware_concurrency());
  const unsigned n_threads = (argc >= 7) ? (unsigned)std::max(1, std::atoi(argv[6])) : hw_threads;
  const bool gpu_wanted = std::strcmp(spv_path, "-") != 0;
  if (W < 1 || H < 1 || W > 4096 || H > 4096) fail("width and height must be between 1 and 4096");

  // ===== 1. load + validate the dump =======================================
  BvhDump dump;
  std::string err;
  if (!load_bvh_dump(dump_dir, dump, err)) {
    std::fprintf(stderr, "FAIL loading dump: %s\n", err.c_str());
    return 1;
  }
  std::printf("Dump: %u inner-node float4s (%u nodes), %u leaves, %u prims, %u triangles, %u vertices, %u objects\n",
              dump.node_count(), dump.node_count() / 4, dump.leaf_count(), dump.prim_count(), dump.tri_count(),
              dump.vert_count(), dump.object_count());

  // ===== 2. Build independent world-space reference triangles ==================
  std::vector<WorldTri> world_tris;
  for (uint32_t obj = 0; obj < dump.object_count(); obj++) {
    const uint32_t* obj_data = &dump.objects[obj * 64];
    float tfm[12];
    for (int i = 0; i < 12; i++) {
      uint32_t u; std::memcpy(&u, &obj_data[i], 4);
      tfm[i] = *reinterpret_cast<float*>(&u);
    }
    const uint32_t off = dump.position_offset(obj);
    for (uint32_t prim = 0; prim < dump.tri_count(); prim++) {
      WorldTri wt;
      wt.obj = obj;
      wt.prim = prim;
      for (int k = 0; k < 3; k++) {
        const uint32_t v_idx = dump.tri_vindex[prim * 3 + k] + off;
        if (v_idx >= dump.vert_count()) {
          std::fprintf(stderr, "FAIL: triangle %u uses vertex %u, but there are only %u vertices\n", prim, v_idx, dump.vert_count());
          return 3;
        }
        float vx = dump.tri_verts[v_idx * 3 + 0];
        float vy = dump.tri_verts[v_idx * 3 + 1];
        float vz = dump.tri_verts[v_idx * 3 + 2];
        wt.v[k * 3 + 0] = vx * tfm[0] + vy * tfm[1] + vz * tfm[2] + tfm[3];
        wt.v[k * 3 + 1] = vx * tfm[4] + vy * tfm[5] + vz * tfm[6] + tfm[7];
        wt.v[k * 3 + 2] = vx * tfm[8] + vy * tfm[9] + vz * tfm[10] + tfm[11];
      }
      world_tris.push_back(wt);
    }
  }
  std::printf("Built %zu world-space triangles for brute-force reference.\n", world_tris.size());

  float bmin[3] = {1e30f, 1e30f, 1e30f}, bmax[3] = {-1e30f, -1e30f, -1e30f};
  for (size_t i = 0; i < world_tris.size(); i++) {
    for (int p = 0; p < 3; p++) {
      for (int a = 0; a < 3; a++) {
        float v = world_tris[i].v[p * 3 + a];
        bmin[a] = std::min(bmin[a], v);
        bmax[a] = std::max(bmax[a], v);
      }
    }
  }
  const Camera cam = make_camera(bmin, bmax, 40.0f, (float)W / (float)H);
  const size_t pixel_count = (size_t)W * H;

  // ===== 3. CPU A: brute force ==============================================
  auto trace_cpu_inst = [&](const float *o, const float *d) {
    float best_t = 1e30f;
    uint32_t best_obj = 0xFFFFFFFFu;
    uint32_t best_prim = 0xFFFFFFFFu;
    for (size_t i = 0; i < world_tris.size(); i++) {
      const float *t = world_tris[i].v;
      float u, v, t_hit;
      if (cyc::ray_triangle_intersect({o[0], o[1], o[2]}, {d[0], d[1], d[2]}, 1e-4f, best_t,
                                      {t[0], t[1], t[2]}, {t[3], t[4], t[5]}, {t[6], t[7], t[8]},
                                      u, v, t_hit)) {
        best_t = t_hit;
        best_obj = world_tris[i].obj;
        best_prim = world_tris[i].prim;
      }
    }
    return Hit{best_t, best_obj, best_prim};
  };

  std::vector<Hit> hits_brute(pixel_count);
  parallel_rows(H, n_threads, [&](uint32_t y) {
    for (uint32_t x = 0; x < W; x++) {
      float d[3];
      primary_ray_dir(cam, x, y, W, H, d);
      hits_brute[(size_t)y * W + x] = trace_cpu_inst(cam.o, d);
    }
  });
  std::printf("\nCPU A (brute force, no BVH) done.\n");

  // ===== 4. CPU B: Cycles' traversal, in C++ ================================
  const cyc::SceneView view = cyc::make_view(dump);
  auto run_bvh_pass = [&](std::vector<Hit> &out, unsigned threads, size_t &errors) -> double {
    std::atomic<size_t> err_count{0};
    const auto start = std::chrono::steady_clock::now();
    parallel_rows(H, threads, [&](uint32_t y) {
      for (uint32_t x = 0; x < W; x++) {
        float d[3];
        primary_ray_dir(cam, x, y, W, H, d);
        float t;
        uint32_t prim;
        uint32_t hit_obj = 0xFFFFFFFFu;
        uint32_t status = cyc::traverse(view, {cam.o[0], cam.o[1], cam.o[2]}, {d[0], d[1], d[2]}, kRayTMin,
                                              kNoHitT, visibility, t, prim, hit_obj);
        Hit &h = out[(size_t)y * W + x];
        if (status != 0) {
          h = {-1.0f, 0xFFFFFFFFu, status};
          err_count.fetch_add(1, std::memory_order_relaxed);
        } else {
          h = {t, hit_obj, prim};
        }
      }
    });
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    errors = err_count.load();
    return ms;
  };

  std::vector<Hit> hits_cpu_bvh(pixel_count);
  size_t cpu_err = 0;
  const double cpu_ms = run_bvh_pass(hits_cpu_bvh, 1, cpu_err);
  std::printf("CPU B (Cycles traversal in C++, 1 thread) done: %.1f ms, %.2f Mrays/s, error codes: %zu\n", cpu_ms,
              (double)pixel_count / cpu_ms / 1000.0, cpu_err);

  std::vector<Hit> hits_cpu_bvh_mt(pixel_count);
  size_t cpu_mt_err = 0;
  const double cpu_mt_ms = run_bvh_pass(hits_cpu_bvh_mt, n_threads, cpu_mt_err);
  size_t mt_diffs = 0;
  for (size_t i = 0; i < pixel_count; i++)
    if (float_bits(hits_cpu_bvh[i].t) != float_bits(hits_cpu_bvh_mt[i].t) || hits_cpu_bvh[i].prim != hits_cpu_bvh_mt[i].prim)
      mt_diffs++;
  const bool mt_ok = (mt_diffs == 0) && (cpu_mt_err == cpu_err);
  std::printf("CPU B (same traversal, %u threads) done: %.1f ms, %.2f Mrays/s (%.1fx the 1-thread run), "
              "results identical to 1 thread: %s (%zu pixel differences)\n",
              n_threads, cpu_mt_ms, (double)pixel_count / cpu_mt_ms / 1000.0, cpu_ms / cpu_mt_ms,
              mt_ok ? "YES" : "NO !!!", mt_diffs);

  const CmpContext cx{&world_tris, &cam, W, H};
  const Cmp ab = compare_hits(hits_brute, hits_cpu_bvh, cx);
  std::printf("\n");
  print_cmp("A (brute force) vs B (Cycles BVH traversal, C++)", ab, hits_brute, hits_cpu_bvh, cx);
  const bool ab_ok = cmp_ok(ab, pixel_count);

  write_ppm("depth_cpu_brute.ppm", W, H, hits_to_depth_rgb(hits_brute));
  write_ppm("depth_cpu_bvh.ppm", W, H, hits_to_depth_rgb(hits_cpu_bvh));

  if (!gpu_wanted) {
    std::printf("\nGPU skipped (spv = \"-\").\n%s\n", (ab_ok && mt_ok) ? "CPU-ONLY PASS" : "CPU-ONLY FAIL: see the numbers above");
    return (ab_ok && mt_ok) ? 0 : 2;
  }

  // ===== 5. GPU =============================================================
  Context ctx = create_context();
  Kernel kernel = create_kernel(ctx, spv_path, (uint32_t)sizeof(TraversePush));
  print_kernel_stats(ctx, kernel);

  Buffer b_nodes = upload(ctx, dump.nodes.data(), dump.nodes.size() * 4);
  Buffer b_leaves = upload(ctx, dump.leaves.data(), dump.leaves.size() * 4);
  Buffer b_pindex = upload(ctx, dump.prim_index.data(), dump.prim_index.size() * 4);
  Buffer b_pvis = upload(ctx, dump.prim_visibility.data(), dump.prim_visibility.size() * 4);
  Buffer b_pobj = upload(ctx, dump.prim_object.data(), dump.prim_object.size() * 4);
  Buffer b_vindex = upload(ctx, dump.tri_vindex.data(), dump.tri_vindex.size() * 4);
  Buffer b_verts = upload(ctx, dump.tri_verts.data(), dump.tri_verts.size() * 4);
  Buffer b_objects = upload(ctx, dump.objects.data(), dump.objects.size() * 4);
  Buffer b_object_node = upload(ctx, dump.object_node.data(), dump.object_node.size() * 4);

  SceneDataGpu sd{};
  sd.bvh_nodes = b_nodes.address;
  sd.bvh_leaf_nodes = b_leaves.address;
  sd.prim_index = b_pindex.address;
  sd.prim_visibility = b_pvis.address;
  sd.prim_object = b_pobj.address;
  sd.tri_vindex = b_vindex.address;
  sd.tri_verts = b_verts.address;
  sd.objects = b_objects.address;
  sd.object_node = b_object_node.address;
  sd.root = dump.root();
  sd.node_count = dump.node_count();
  sd.leaf_count = dump.leaf_count();
  sd.prim_count = dump.prim_count();
  sd.tri_count = dump.tri_count();
  sd.vert_count = dump.vert_count();
  sd.object_count = dump.object_count();
  sd.object_node_count = dump.object_node_count();
  Buffer b_scene = upload(ctx, &sd, sizeof(sd));

  const VkDeviceSize out_bytes = (VkDeviceSize)(pixel_count + kCanaryHits) * sizeof(Hit);
  Buffer out_buf = create_buffer(ctx, out_bytes);
  {
    uint32_t *words = static_cast<uint32_t *>(out_buf.mapped);
    const size_t real_words = pixel_count * 3;
    for (size_t i = 0; i < real_words; i++) words[i] = kUnwrittenWord;
    for (size_t i = 0; i < (size_t)kCanaryHits * 3; i++) words[real_words + i] = kCanaryWord;
  }

  TraversePush pc{};
  pc.scene = b_scene.address;
  pc.out = out_buf.address;
  pc.width = W;
  pc.height = H;
  pc.visibility = visibility;
  for (int i = 0; i < 3; i++) {
    pc.o[i] = cam.o[i];
    pc.l[i] = cam.l[i];
    pc.u[i] = cam.u[i];
    pc.v[i] = cam.v[i];
  }

  const uint32_t groups_x = (W + 7) / 8, groups_y = (H + 7) / 8;
  const double ms = run_kernel(ctx, kernel, &pc, groups_x, groups_y);
  std::printf("GPU finished: %ux%u groups of 8x8 threads, %.3f ms, %.2f Mrays/s\n", groups_x, groups_y, ms,
              (double)pixel_count / ms / 1000.0);

  // ===== safety checks + comparisons =========================================
  const Hit *gpu_raw = static_cast<const Hit *>(out_buf.mapped);
  bool canaries_ok = true;
  for (uint32_t i = 0; i < kCanaryHits; i++) {
    const uint32_t *w = reinterpret_cast<const uint32_t *>(&gpu_raw[pixel_count + i]);
    if (w[0] != kCanaryWord || w[1] != kCanaryWord || w[2] != kCanaryWord) canaries_ok = false;
  }
  std::printf("\nCanaries after the output buffer: %s\n", canaries_ok ? "INTACT" : "!!! OVERWRITTEN !!!");

  std::vector<Hit> hits_gpu(gpu_raw, gpu_raw + pixel_count);
  size_t unwritten = 0, gpu_err = 0;
  for (const Hit &h : hits_gpu) {
    if (float_bits(h.t) == kUnwrittenWord && h.prim == kUnwrittenWord) unwritten++;
    else if (cyc::is_error_id(h.prim)) gpu_err++;
  }
  std::printf("Pixels the shader never wrote: %zu\n", unwritten);
  std::printf("Pixels where the shader returned an error code: %zu\n", gpu_err);

  std::printf("\n");
  const Cmp bg = compare_hits(hits_cpu_bvh, hits_gpu, cx);
  print_cmp("B (Cycles traversal, C++) vs GPU (Cycles traversal, Slang)", bg, hits_cpu_bvh, hits_gpu, cx);
  std::printf("\n");
  const Cmp ag = compare_hits(hits_brute, hits_gpu, cx);
  print_cmp("A (brute force) vs GPU", ag, hits_brute, hits_gpu, cx);

  write_ppm("depth_gpu_bvh.ppm", W, H, hits_to_depth_rgb(hits_gpu));
  write_ppm("ids_gpu_bvh.ppm", W, H, hits_to_id_rgb(hits_gpu));
  std::printf("Wrote depth_cpu_brute.ppm, depth_cpu_bvh.ppm, depth_gpu_bvh.ppm, ids_gpu_bvh.ppm\n");

  const bool ok = ab_ok && mt_ok && canaries_ok && unwritten == 0 && gpu_err == 0 && cmp_ok(bg, pixel_count) && cmp_ok(ag, pixel_count);
  std::printf("%s\n", ok ? "PASS" : "FAIL: see the numbers above");

  destroy_buffer(ctx, out_buf);
  destroy_buffer(ctx, b_scene);
  destroy_buffer(ctx, b_object_node);
  destroy_buffer(ctx, b_objects);
  destroy_buffer(ctx, b_verts);
  destroy_buffer(ctx, b_vindex);
  destroy_buffer(ctx, b_pobj);
  destroy_buffer(ctx, b_pvis);
  destroy_buffer(ctx, b_pindex);
  destroy_buffer(ctx, b_leaves);
  destroy_buffer(ctx, b_nodes);
  destroy_kernel(ctx, kernel);
  destroy_context(ctx);
  return ok ? 0 : 2;
}