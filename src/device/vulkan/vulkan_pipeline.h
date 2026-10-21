// SPDX-FileCopyrightText: 2011-2022 Blender Foundation
// SPDX-License-Identifier: Apache-2.0

/* Host-side mirror of VulkanKernelParams (src/kernel/device/vulkan/vulkan_data.slang).
 *
 * The Slang struct is compiled with std430-ish natural alignment; we replicate
 * it here with explicit types and static_asserts on offsets so any drift fails
 * the build instead of producing garbage images. */

#pragma once

#ifdef WITH_VULKAN

#include <cstdint>

CCL_NAMESPACE_BEGIN

struct alignas(8) VulkanKernelParams {
  /* --- film --- */
  uint32_t film_width;         // offset 0
  uint32_t film_height;        // offset 4
  uint32_t tile_offset_x;      // offset 8
  uint32_t tile_offset_y;      // offset 12
  uint32_t tile_width;         // offset 16
  uint32_t tile_height;        // offset 20
  uint32_t samples_per_pixel;  // offset 24
  uint32_t max_bounces;        // offset 28
  uint64_t film_buffer;        // offset 32

  /* --- camera --- */
  float cam_origin[3];         // offset 40 (float3, aligned to 4)
  float cam_aspect;            // offset 52
  float cam_forward[3];        // offset 56
  float cam_focal_length;      // offset 68
  float cam_right[3];          // offset 72
  float cam_sensor_width;      // offset 84
  float cam_up[3];             // offset 88
  float pad_cam;               // offset 100

  /* --- scene --- */
  uint64_t bvh_nodes;          // offset 104
  uint64_t bvh_prim_offsets;   // offset 112
  uint64_t bvh_prim_count;     // offset 120
  uint64_t tri_indices;        // offset 128
  uint64_t vert_co;            // offset 136
  uint64_t object_transforms;  // offset 144
  uint64_t light_prim_offsets; // offset 152
  uint64_t light_prim_counts;  // offset 160
  uint64_t light_emission;     // offset 168

  uint32_t num_triangles;      // offset 176
  uint32_t num_objects;        // offset 180
  uint32_t num_lights;         // offset 184
  uint32_t pad_scene;          // offset 188

  /* --- world --- */
  float bg_color[3];           // offset 192
  float pad_bg;                // offset 204

  /* --- rng --- */
  uint32_t rng_seed;           // offset 208
  uint32_t reserved_0;         // offset 212
  uint32_t reserved_1;         // offset 216
  uint32_t reserved_2;         // offset 220
};                           /* total 224 bytes = multiple of 16 */

static_assert(sizeof(VulkanKernelParams) == 224, "VulkanKernelParams size drift vs Slang");
static_assert(offsetof(VulkanKernelParams, film_buffer) == 32, "film_buffer offset");
static_assert(offsetof(VulkanKernelParams, cam_origin) == 40, "cam_origin offset");
static_assert(offsetof(VulkanKernelParams, bvh_nodes) == 104, "bvh_nodes offset");
static_assert(offsetof(VulkanKernelParams, bg_color) == 192, "bg_color offset");
static_assert(offsetof(VulkanKernelParams, rng_seed) == 208, "rng_seed offset");

constexpr int VULKAN_TILE_SIZE = 64;

CCL_NAMESPACE_END

#endif /* WITH_VULKAN */
