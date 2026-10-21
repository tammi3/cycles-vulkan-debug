# SPDX-FileCopyrightText: 2011-2026 Blender Foundation
# SPDX-License-Identifier: Apache-2.0

# Compile Slang kernels to SPIR-V using slangc from Vulkan SDK
# Output: build/spirv/path_trace.spv

if(WITH_CYCLES_DEVICE_VULKAN)

  # Find slangc - prefer Vulkan SDK, fall back to PATH
  if(DEFINED ENV{VULKAN_SDK})
    set(SLANG_CANDIDATE "$ENV{VULKAN_SDK}/Bin/slangc.exe")
    if(EXISTS "${SLANG_CANDIDATE}")
      set(SLANG_COMPILER "${SLANG_CANDIDATE}" CACHE FILEPATH "slangc compiler" FORCE)
    endif()
  endif()

  if(NOT SLANG_COMPILER)
    find_program(SLANG_COMPILER NAMES slangc slangc.exe
      DOC "Slang compiler for Vulkan SPIR-V generation")
  endif()

  if(NOT SLANG_COMPILER)
    message(WARNING "slangc not found - Vulkan kernels will not be compiled. Install Vulkan SDK or add slangc to PATH.")
    return()
  endif()

  message(STATUS "Found slangc: ${SLANG_COMPILER}")

  # Output directory for SPIR-V
  set(SPIRV_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/spirv")
  file(MAKE_DIRECTORY "${SPIRV_OUTPUT_DIR}")

  # Kernel source files - only compile the entry point, it includes the others
  set(VULKAN_KERNEL_ENTRY "${CMAKE_CURRENT_SOURCE_DIR}/path_trace.slang")

  # slangc command to compile path_trace.slang (entry point) to SPIR-V
  # -target spirv: compile to SPIR-V
  # -entry path_trace_kernel: the compute shader entry point
  # -I: include the current directory for #include resolution
  # -o: output file
  add_custom_command(
    OUTPUT "${SPIRV_OUTPUT_DIR}/path_trace.spv"
    COMMAND ${SLANG_COMPILER}
      -target spirv
      -entry path_trace_kernel
      -stage compute
      -I ${CMAKE_CURRENT_SOURCE_DIR}
      -o ${SPIRV_OUTPUT_DIR}/path_trace.spv
      ${VULKAN_KERNEL_ENTRY}
    DEPENDS ${VULKAN_KERNEL_ENTRY} ${SLANG_COMPILER}
    COMMENT "Compiling Vulkan Slang kernel to SPIR-V"
    VERBATIM
  )

  # Custom target to drive the compilation
  add_custom_target(cycles_kernel_vulkan_spirv
    DEPENDS "${SPIRV_OUTPUT_DIR}/path_trace.spv"
  )

  # Make the main cycles_kernel target depend on SPIR-V generation
  add_dependencies(cycles_kernel cycles_kernel_vulkan_spirv)

  # Export the SPIR-V directory for the device code to find at runtime
  set(VULKAN_SPIRV_DIR "${SPIRV_OUTPUT_DIR}" CACHE INTERNAL "Vulkan SPIR-V output directory")
  set(VULKAN_SPIRV_DIR "${SPIRV_OUTPUT_DIR}" PARENT_SCOPE)

endif()