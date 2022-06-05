workspace "COMP5822M-cw3"
	language "C++"
	cppdialect "C++17"

	platforms { "x64" }
	configurations { "debug", "release" }

	flags "NoPCH"
	flags "MultiProcessorCompile"

	startproject "cw3"

	debugdir "%{wks.location}"
	objdir "_build_/%{cfg.buildcfg}-%{cfg.platform}-%{cfg.toolset}"
	targetsuffix "-%{cfg.buildcfg}-%{cfg.platform}-%{cfg.toolset}"
	
	-- Default toolset options
	filter "toolset:gcc or toolset:clang"
		linkoptions { "-pthread" }
		buildoptions { "-march=native", "-Wall", "-pthread" }

	filter "toolset:msc-*"
		defines { "_CRT_SECURE_NO_WARNINGS=1" }
		defines { "_SCL_SECURE_NO_WARNINGS=1" }
		buildoptions { "/utf-8" }
	
	filter "*"

	-- default libraries
	filter "system:linux"
		links "dl"
	
	filter "system:windows"

	filter "*"

	-- default outputs
	filter "kind:StaticLib"
		targetdir "lib/"

	filter "kind:ConsoleApp"
		targetdir "bin/"
		targetextension ".exe"
	
	filter "*"

	--configurations
	filter "debug"
		symbols "On"
		defines { "_DEBUG=1" }

	filter "release"
		optimize "On"
		defines { "NDEBUG=1" }

	filter "*"

-- Third party dependencies
include "third_party" 

-- GLSLC helpers
dofile( "util/glslc.lua" )

-- Projects
project "cw3"
	local sources = { 
		"cw3/**.cpp",
		"cw3/**.hpp",
		"cw3/**.hxx"
	}

	kind "ConsoleApp"
	location "cw3"

	files( sources )

	dependson "cw3-shaders"

	links "labutils"
	links "x-volk"
	links "x-stb"
	links "x-glfw"
	links "x-vma"
	links "x-tinyobj"

	dependson "x-glm" 

project "cw3-shaders"
	local shaders = { 
		"cw3/shaders/*.vert",
		"cw3/shaders/*.frag"
	}

	kind "Utility"
	location "cw3/shaders"

	files( shaders )

	handle_glsl_files( "-O", "assets/cw3/shaders", {} )

project "labutils"
	local sources = { 
		"labutils/**.cpp",
		"labutils/**.hpp",
		"labutils/**.hxx"
	}

	kind "StaticLib"
	location "labutils"

	files( sources )

--EOF
