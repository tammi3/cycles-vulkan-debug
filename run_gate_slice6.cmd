@echo off
set CYCLES_VULKAN_ALLOW_NO_KERNELS=1
set CYCLES_VULKAN_DISPATCH_GATE=1
set VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation
set VK_LOADER_DEBUG=layer
set CYCLES_VULKAN_RESET_SPV=C:\dev\cycles-vulkan-debug\src\kernel\device\vulkan\integrator_reset.spv
set CYCLES_VULKAN_INIT_FROM_CAMERA_KERNEL_SPV=C:\dev\cycles-vulkan-debug\src\kernel\device\vulkan\init_from_camera_kernel.spv
set CYCLES_VULKAN_INTERSECT_CLOSEST_SPV=C:\dev\cycles-vulkan-debug\src\kernel\device\vulkan\integrator_intersect_closest.spv
cd /d C:\dev\cycles-vulkan-debug
build\bin\Release\cycles.exe --device VULKAN --log-level info --samples 1 --background instance2.xml
