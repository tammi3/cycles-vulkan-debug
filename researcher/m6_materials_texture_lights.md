a) Principled BSDF: ONE closure (CLOSURE_BSDF_PRINCIPLED_ID) with lobes
Verified structure from E:\blender-git\blender\intern\cycles\kernel\svm\closure.h:210-457
The Principled BSDF is a single closure (CLOSURE_BSDF_PRINCIPLED_ID at types.h:562) that allocates multiple BSDF/BSSRDF lobes internally based on weight parameters. The svm_node_closure_bsdf() switch case at closure.h:210 shows the exact allocation order:
Order	Lobe/Condition	Closure Allocated	Weight Parameter	Closure Type
1	Emission	Emissive closure	emission	principled_bsdf_emission()
2	Metallic	MicrofacetBsdf + FresnelF82Tint	metallic	GGX conductor (ior=1)
3	Transmission	MicrofacetBsdf + FresnelGeneralizedSchlick	transmission_weight	GGX glass (thin_wall: thin_glass, else microfacet)
4	Specular (clearcoat-like)	MicrofacetBsdf + FresnelGeneralizedSchlick	remaining weight after metallic+transmission	GGX dielectric (eta from ior/specular_ior_level)
5	Subsurface (if __SUBSURFACE__)	Bssrdf (or thin_subsurface)	subsurface_weight	BSSRDF (random_walk/cubic/gaussian/burley)
6	Diffuse (base)	bsdf_diffuse_setup or bsdf_oren_nayar_setup	remaining weight	Lambertian/Oren-Nayar
Key finding: There is NO NODE_BSDF_PRINCIPLED_* SVM nodes, NO convert_bsdf_principled function. The Principled node is compiled by PrincipledBsdfNode::compile() (scene/shader_nodes.cpp:2822) into SVM nodes that set up the SVMNodePrincipledBsdfData struct, then svm_node_closure_bsdf() expands it into the above lobes at render time.
Phasing proposal by LOBE (USER DECISION)
Phase	Lobe(s)	Closures to port	Lines est.	Features unlocked
M6-P1	Diffuse (base)	bsdf_diffuse_setup, bsdf_oren_nayar_setup	~150	Matte surfaces (M4 done)
M6-P2	Metallic + Specular (GGX conductor + dielectric)	bsdf_microfacet_ggx_setup, bsdf_microfacet_setup_fresnel_f82_tint, bsdf_microfacet_setup_fresnel_generalized_schlick	~400	Metals, plastics, dielectrics
M6-P3	Transmission (GGX glass)	bsdf_microfacet_ggx_glass_setup, bsdf_thin_glass_setup	~300	Glass, liquids
M6-P4	Subsurface (requires __SUBSURFACE__)	bssrdf_setup, bsdf_thin_subsurface_setup + BSSRDF types	~500	Skin, wax, marble
M6-P5	Sheen (separate closure: CLOSURE_BSDF_SHEEN_ID)	bsdf_sheen_setup	~150	Cloth/fabric
M6-P6	Clearcoat (separate closure: CLOSURE_BSDF_CLEARCOAT_ID)	bsdf_clearcoat_setup	~150	Car paint, coated surfaces
M6-P7	Anisotropic (Beckmann/Ashikhmin-Shirley)	bsdf_ashikhmin_shirley_setup	~200	Brushed metal
Recommendation: M6 gate = P1+P2+P3 (Principled core: diffuse + metallic + specular + transmission). P4-P7 as M6b/M6c.
b) Image textures: Kernel uses KernelImageTexture + KernelImageInfo (NOT TextureInfo)
Verified structs from E:\blender-git\blender\intern\cycles\util\types_image.h:94-165
// KernelImageInfo (device image object)
struct KernelImageInfo {
    uint64_t data = 0;           // Pointer, offset or image/texture object depending on device
    uint data_type = IMAGE_DATA_NUM_TYPES;  // enum ImageDataType (float4, byte4, half4, float, byte, half, ushort4, ushort, nanovdb...)
    uint interpolation = INTERPOLATION_NONE;
    uint extension = EXTENSION_REPEAT;
    uint width = 0, height = 0;
    float inv_width = 0.0f, inv_height = 0.0f;
};

// KernelImageTexture (logical texture for shading, supports tile-based loading)
struct KernelImageTexture {
    uint image_info_id = KERNEL_IMAGE_NONE;
    uint tile_descriptor_offset = KERNEL_TILE_LOAD_NONE;
    int tile_size_shift = 0;
    int tile_levels = 0;
    int tile_num = 0;
    int width = 0, height = 0;
    uint interpolation = INTERPOLATION_NONE;
    uint extension = EXTENSION_REPEAT;
    uint use_transform_3d = false;
    Transform transform_3d = transform_zero();
    float4 average_color = zero_float4();
};
Texture fetch function (verified from E:\blender-git\blender\intern\cycles\kernel\util\image_2d.h)
No kernel_tex_image_interp with TextureInfo* — instead:
- kernel_image_tile_map() — maps UV to tile descriptor (handles mip levels, anisotropy, UDIM)
- Tile-based loading: image_texture_tile_descriptors array, image_texture_tile_access_state for cache misses
- On GPU: cache miss → mark tile as KERNEL_TILE_LOAD_REQUEST → cancel shader → host loads tile → resume
For Vulkan (no descriptor indexing, BDA-only):
- Option A (hardware): VkImageView + VkSampler per tile descriptor → needs VK_EXT_descriptor_indexing
- Option B (software): Raw BDA to tile buffers → implement kernel_image_tile_map in Slang with manual bilinear/trilinear, mip selection, wrap modes
Formats to support (from ImageDataType enum, types_image.h:35-52):
- IMAGE_DATA_TYPE_FLOAT4, BYTE4, HALF4 (RGBA)
- IMAGE_DATA_TYPE_FLOAT, BYTE, HALF (single channel)
- IMAGE_DATA_TYPE_USHORT4, USHORT
- Nanovdb types (deferred)
No TextureInfo struct in kernel — that was a CUDA/HIP host-side struct.
c) Light types and M5 gate scene
Verified light types from E:\blender-git\blender\intern\cycles\app\cycles_xml.cpp:646-667
if (light_type == "point")      → PointLight
else if (light_type == "sun")   → SunLight
else if (light_type == "background") → BackgroundLight
else if (light_type == "area")  → AreaLight
else                            → SpotLight (assert)
Area lights in XML: <light light_type="area"> creates an AreaLight object with geometry (rect/ellipse). Emissive meshes are triangle lights (light/triangle.h), NOT area lights.
Light sampling paths (verified from E:\blender-git\blender\intern\cycles\kernel\light\sample.h)
Function	Light types handled
light_sample_from_position (line 403)	All: calls light_sample<false> → dispatches to area_light_sample, point_light_sample, spot_light_sample, sun_light_sample, background_light_sample, triangle_light_sample
light_sample_from_volume_segment (line 371)	Same, light_sample<true>
light_distribution_sample (distribution.h:16)	Flat CDF over light_distribution array (used when use_light_tree=false)
light_tree_sample (tree.h:742)	Light tree traversal (used when use_light_tree=true)
use_light_tree is a runtime scene setting (scene/integrator.cpp:143, default true), NOT a CMake flag.
M5 Gate Scene — corrected
XML with real <light type="area"> syntax:
<cycles>
    <transform translate="0 -6 3">
        <camera type="perspective" fov="0.8" full_width="320" full_height="240" />
    </transform>

    <shader name="white_wall">
        <diffuse_bsdf name="bsdf" color="0.8 0.8 0.8" />
        <connect from="bsdf bsdf" to="output surface" />
    </shader>
    <shader name="light_shader">
        <emission_closure name="emit" color="1 1 1" strength="15" />
        <connect from="emit emission" to="output surface" />
    </shader>

    <state shader="white_wall">
        <object name="floor" mesh="box_floor" />
        <object name="ceiling" mesh="box_ceiling" />
        <object name="back_wall" mesh="box_back" />
        <object name="left_wall" mesh="box_left" />
        <object name="right_wall" mesh="box_right" />
        <object name="tall_box" mesh="box_tall" />
    </state>
    <transform translate="1.5 0 0">
        <state shader="white_wall">
            <object name="short_box" mesh="box_short" />
        </state>
    </transform>

    <!-- Area light on ceiling (rectangle) -->
    <transform translate="0 2.9 0">
        <state shader="light_shader">
            <light light_type="area" mesh="light_quad" />
        </state>
    </transform>

    <mesh name="box_floor" P="2 0 2 -2 0 2 -2 0 -2 2 0 -2" nverts="4" verts="0 1 2 3" />
    <mesh name="box_ceiling" P="2 3 2 2 3 -2 -2 3 -2 -2 3 2" nverts="4" verts="0 1 2 3" />
    <mesh name="box_back" P="2 0 -2 2 3 -2 -2 3 -2 -2 0 -2" nverts="4" verts="0 1 2 3" />
    <mesh name="box_left" P="-2 0 2 -2 3 2 -2 3 -2 -2 0 -2" nverts="4" verts="0 1 2 3" />
    <mesh name="box_right" P="2 0 2 2 0 -2 2 3 -2 2 3 2" nverts="4" verts="0 1 2 3" />
    <mesh name="light_quad" P="1 0 1 -1 0 1 -1 0 -1 1 0 -1" nverts="4" verts="0 1 2 3" />
    <mesh name="box_tall" P="0.5 0 0.5 -0.5 0 0.5 -0.5 1.5 0.5 0.5 1.5 0.5 -0.5 0 -0.5 0.5 0 -0.5 -0.5 1.5 -0.5 0.5 1.5 -0.5" nverts="4 4 4 4 4 4" verts="0 1 2 3 4 5 6 7 0 1 5 4 2 3 7 6 0 4 7 3 1 5 6 2" />
    <mesh name="box_short" P="0.3 0 0.3 -0.3 0 0.3 -0.3 0.6 0.3 0.3 0.6 0.3 -0.3 0 -0.3 0.3 0 -0.3 -0.3 0.6 -0.3 0.3 0.6 -0.3" nverts="4 4 4 4 4 4" verts="0 1 2 3 4 5 6 7 0 1 5 4 2 3 7 6 0 4 7 3 1 5 6 2" />
</cycles>
Integrator settings for CPU reference (disable light tree):
cycles.exe --device CPU --background cornell_area_light.xml \
  --set scene.integrator.use_light_tree=false \
  --samples 16 --resolution 320x240 \
  --set scene.cycles.filter_type=BOX \
  -o cornell_cpu_16spp.png
Tolerance: Per-pixel L2 in linear space. Propose: mean abs diff < 0.01 (1/255), max abs diff < 0.05 in penumbra. USER DECISION on exact threshold.