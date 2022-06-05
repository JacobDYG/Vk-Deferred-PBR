Third Party Software
====================

This code uses the following third party software/source code.

## Premake

- Where: https://premake.github.io/download
- What: Build configuration and generator
- License: BSD 3-Clause

Premake generates platform/toolchain specific build files, such as Visual
Studio project files or standard Makefiles. This project includes prebuilt
binaries for Linux and Windows.

## Volk

- Where: https://github.com/zeux/volk
- What: "Meta-loader for Vulkan"
- License: MIT License

Volk allows applications to dynamically load the Vulkan API without linking
against vulkan-1.dll or libvulkan.so. Furthermore, it enables the loading of
device-specific Vulkan functions, the use of which skips the overhead of
dispatching per-device.

Note: Only the source code (volk.{h,c}) and the license and readme files are
included here. The volk.c source is modified to include the volk.h header
correctly for this project.

## Vulkan Headers

- Where: Vulkan SDK (see https://vulkan.lunarg.com/sdk/home)
- What: Vulkan API header files
- License: Apache 2.0

This is a copy of the necessary Vulkan header files, extracted from the Vulkan
SDK (version 1.2.198.1). Only the LICENSE.txt, README.txt and a minimal amount
of header files are included.

## GLFW

- Where: https://www.glfw.org/
- What: Multi-platform lib for OpenGL & Vulkan development (desktop)
- License: Zlib/libpng

GLFW abstracts platform-specific functions, such as creating windows and
receiving/handling events. 

## STB libraries

- Where: https://github.com/nothings/stb
- What: single-header libraries
- License: Public Domain / MIT License

Collection of single-header libraries by Sean Barrett. Included are:
- stb_image.h         (image loading)
- stb_image_write.h   (image writing)

Note: the files in src/ are custom.

## Shaderc / glslc

- Where: https://github.com/google/shaderc
- What: GLSL to SpirV compiler
- License: Apache v2.0

Note: only pre-built glslc[.exe] compiler binaries for 64-bit Windows and
64-bit Linux are included here.

The glslc compiler is an offline compiler toolt that accepts (among others)
GLSL sources and compiles these to SpirV code that can be passed to Vulkan.

## VulkanMemoryAllocator

- Where: https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator
- What: Vulkan memory allocation library
- License: MIT License

An easy to integrate Vulkan memory allocation library developed by AMD. See
section "Problem" in their README.md for a concise description of the problems
it solves.

Note: the file in src/ is custom.

For compatibility with Volk, we need to define
  - VMA_STATIC_VULKAN_FUNCTIONS = 0
  - VMA_DYNAMIC_VULKAN_FUNCTIONS = 1
This way, we can pass just two function pointers to VMA when we're initializing
it, and it will load the rest itself. The definitions are only needed when
compiling the VMA implementation code, and are therefore only defined in the
custom vk_mem_alloc.cpp.

**MODIFIED**
- In the definition of vmaCreateAllocator(), the VMA_ASSERT() that checks the
  Vulkan version (<= 1.2) has been commented for compatibility with Vulkan 1.3.

## GLM

- Where: https://github.com/g-truc/glm
- What: C++ math library for OpenGL / GLSL
- License: The Happy Bunny License OR MIT License

GLM defines various math / linear algebra functions. Notes:

We want to make sure that GLM_FORCE_DEPTH_ZERO_TO_ONE is defined. GLM normally
targets OpenGL with a depth range of [-1, 1], whereas Vulkan uses [0, 1].
Additionally, we'll use GLM_FORCE_RADIANS, so angles are in radians.

GLM is a header only library, so we don't have to compile anything for it.
(Hence using project type "Utility" in premake.)

## Tinyobjloader

- Where: https://github.com/tinyobjloader/tinyobjloader
- What: Wavefront OBJ model loader
- License: MIT License

Tinyobjlloader is used to load Wavefront OBJ models (.obj + .mtl). This is one
of the common (but somewhat old) formats for exchanging 3D models. Most 3D
modelling software can deal with OBJ models to some degree.

Note: the version included here is somewhat old (1.0.6). However, it is 
battle-tested with the included models.
