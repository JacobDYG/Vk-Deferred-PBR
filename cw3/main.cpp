#include <volk/volk.h>

#include <tuple>
#include <chrono>
#include <limits>
#include <vector>
#include <stdexcept>

#include <cstdio>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#if !defined(GLM_FORCE_RADIANS)
#	define GLM_FORCE_RADIANS
#endif
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "../labutils/to_string.hpp"
#include "../labutils/vulkan_window.hpp"

#include "../labutils/angle.hpp"
using namespace labutils::literals;

#include "../labutils/error.hpp"
#include "../labutils/vkutil.hpp"
#include "../labutils/vkimage.hpp"
#include "../labutils/vkobject.hpp"
#include "../labutils/vkbuffer.hpp"
#include "../labutils/allocator.hpp" 
namespace lut = labutils;

#include "model.hpp"
#include "camera.h"
#include <random>

namespace
{
	namespace cfg
	{
		// Compiled shader code for the graphics pipeline(s)
		// See sources in cw2/shaders/*. 
#		define SHADERDIR_ "assets/cw3/shaders/"
		//constexpr char const* kVertShaderPath = SHADERDIR_ "default.vert.spv";
		//constexpr char const* kFragShaderPath = SHADERDIR_ "default.frag.spv";

		//constexpr char const* kVertShaderPath = SHADERDIR_ "BlinnPhong.vert.spv";
		//constexpr char const* kFragShaderPath = SHADERDIR_ "BlinnPhong.frag.spv";

		constexpr char const* kVertShaderPath = SHADERDIR_ "PBR.vert.spv";
		constexpr char const* kFragShaderPath = SHADERDIR_ "PBR.frag.spv";
#		undef SHADERDIR_

		// Paths for the kModels
#		define ASSETPATH_ "assets/cw3/"
		constexpr char const* kAssetPath = ASSETPATH_;
		constexpr std::initializer_list<char const*> kModels = { ASSETPATH_ "materialtest.obj" };
#		undef ASSETPATH_

		// General rule: with a standard 24 bit or 32 bit float depth buffer,
		// you can support a 1:1000 ratio between the near and far plane with
		// minimal depth fighting. Larger ratios will introduce more depth
		// fighting problems; smaller ratios will increase the depth buffer's
		// resolution but will also limit the view distance.
		constexpr float kCameraNear  = 0.1f;
		constexpr float kCameraFar   = 100.f;

		constexpr auto kCameraFov    = 60.0_degf;
		constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;

		struct Sphere {
			glm::vec3 center;
			float radius;
		};
	}

	// Local types/structures:
	// For Blinn Phong (see BlinnPhong.frag):
	namespace glsl
	{
		struct SceneUniform
		{
			// Be careful about the packing/alignment!
			glm::mat4 camera;
			glm::mat4 projection;
			glm::mat4 projCam;
			glm::vec3 cameraPos;
		};
		static_assert(sizeof(SceneUniform) <= 65536, "SceneUniform must be less than 65536 bytes for vkCmdUpdateBuffer");
		static_assert(sizeof(SceneUniform) % 4 == 0, "SceneUniform must be a multiple of 4 bytes");
	}
	
	//namespace glsl
	//{
	//	struct MaterialUniform
	//	{
	//		glm::vec4 color;
	//	};
	//	static_assert(sizeof(MaterialUniform) <= 65536, "MaterialUniform must be less than 65536 bytes for vkCmdUpdateBuffer");
	//	static_assert(sizeof(MaterialUniform) % 4 == 0, "MaterialUniform must be a multiple of 4 bytes");
	//}
	
	//namespace glsl
	//{
	//	struct MaterialUniform
	//	{
	//		// Note: must map to the std140 uniform interface in the fragment
	//		// shader, so need to be careful about the packing/alignment here!
	//		glm::vec4 emissive;
	//		glm::vec4 diffuse;
	//		glm::vec4 specular;
	//		float shininess;
	//	};
	//	static_assert(sizeof(MaterialUniform) <= 65536, "MaterialUniform must be less than 65536 bytes for vkCmdUpdateBuffer");
	//	static_assert(sizeof(MaterialUniform) % 4 == 0, "MaterialUniform must be a multiple of 4 bytes");
	//}
	// For PBR (see PBR.frag):
	
	namespace glsl
	{
		struct MaterialUniform
		{
			// Note: must map to the std140 uniform interface in the fragment
			// shader, so need to be careful about the packing/alignment here!
			glm::vec4 emissive;
			glm::vec4 albedo;
			float shininess;
			float metalness;
		};
		static_assert(sizeof(MaterialUniform) <= 65536, "MaterialUniform must be less than 65536 bytes for vkCmdUpdateBuffer");
		static_assert(sizeof(MaterialUniform) % 4 == 0, "MaterialUniform must be a multiple of 4 bytes");
	}
	
	namespace glsl
	{
		constexpr unsigned int MAX_LIGHTS = 7;
		constexpr unsigned int PBR_PHONG = 1;
		constexpr unsigned int PBR_GGX = 2;
		struct PointLight
		{
			glm::vec4 position;
			glm::vec4 color;
		};
		struct LightUniform
		{
			PointLight lights[MAX_LIGHTS];
			glm::vec4 ambientColor;
			int numLights;
			int reflectionModel;
		};
	}

	// Local types/structures:
	struct GPUMaterialInfo
	{
		VkPipeline* pipe;
		VkPipelineLayout* pipeLayout;
		labutils::Buffer materialBuffer;
		VkDescriptorSet materialDescriptor;
	};

	struct GPUMesh
	{
		std::size_t materialInfoIdx;

		std::size_t vertexStartIndex;
		std::size_t numberOfVertices;
	};

	struct GPUModel
	{
		labutils::Buffer positions;
		labutils::Buffer normals;
		labutils::Buffer texcoords;

		std::vector<GPUMaterialInfo> materials;
		std::vector<GPUMesh> meshes;
	};

	struct LightState
	{
		bool animEnabled = true;
		int numLightsEnabled = glsl::MAX_LIGHTS;
		int reflectionModel = glsl::PBR_PHONG;
	};

	// Local functions:
	void glfw_callback_key_press(GLFWwindow*, int, int, int, int);
	void glfw_callback_mouse_click(GLFWwindow* window, int button, int action, int mods);
	void glfw_callback_mouse_move(GLFWwindow*, double, double);

	// Helpers:
	lut::RenderPass create_render_pass(lut::VulkanWindow const&);

	lut::DescriptorSetLayout create_scene_descriptor_layout(lut::VulkanWindow const&);
	lut::DescriptorSetLayout create_defaultobject_descriptor_layout(lut::VulkanWindow const&);
	lut::DescriptorSetLayout create_light_descriptor_layout(lut::VulkanWindow const&);

	GPUModel load_gpu_model(labutils::VulkanContext const& aContext, labutils::Allocator const& aAllocator, ModelData& model, std::vector<GPUMaterialInfo>&& materials);

	lut::PipelineLayout create_pipeline_layout(lut::VulkanContext const& aContext, std::vector<VkDescriptorSetLayout> descriptorSetlayouts);
	lut::Pipeline create_pipeline(lut::VulkanWindow const&, VkRenderPass, VkPipelineLayout, const char* aVertexShader, const char* aFragmentShader);

	std::tuple<lut::Image, lut::ImageView> create_depth_buffer(lut::VulkanWindow const&, lut::Allocator const&);

	void create_swapchain_framebuffers(
		lut::VulkanWindow const&,
		VkRenderPass,
		std::vector<lut::Framebuffer>&,
		VkImageView aDepthView
	);

	void update_scene_uniforms(
		glsl::SceneUniform&,
		std::uint32_t aFramebufferWidth,
		std::uint32_t aFramebufferHeight
	);

	void update_light_uniforms(
		glsl::LightUniform& aLightUniform,
		LightState aLightState,
		float aDt
	);

	void record_commands(VkCommandBuffer aCmdBuff,
		VkRenderPass aRenderPass, VkFramebuffer aFramebuffer, VkPipeline aGraphicsPipe, VkPipelineLayout aGraphicsPipelineLayout, VkExtent2D const& aImageExtent,
		VkBuffer aSceneUBO,
		glsl::SceneUniform aSceneUniform,
		VkDescriptorSet aSceneDescriptors,
		VkBuffer aLightUBO,
		glsl::LightUniform aLightUniform,
		VkDescriptorSet aLightDescriptors,
		std::vector<GPUModel>* models
	);

	void submit_commands(
		lut::VulkanContext const&,
		VkCommandBuffer,
		VkFence,
		VkSemaphore,
		VkSemaphore
	);
}

// Camera
camera::Camera mainCamera(glm::vec3(0.0f, 2.0f, 8.0f));
bool firstMouse = true;
float yaw = -90.0f;
float pitch = -0.0f;
float lastX = 1280 / 2;
float lastY = 720 / 2;

// Lights
LightState lightState{};

int main() try
{
	// Create our Vulkan Window
	lut::VulkanWindow window = lut::make_vulkan_window();

	// To track frametimes
	float lastTime = glfwGetTime();
	float currentTime = glfwGetTime() + 0.0001f;

	// Configure the GLFW window
	glfwSetKeyCallback(window.window, &glfw_callback_key_press);
	glfwSetMouseButtonCallback(window.window, &glfw_callback_mouse_click);
	glfwSetCursorPosCallback(window.window, &glfw_callback_mouse_move);

	// Create VMA allocator
	lut::Allocator allocator = lut::create_allocator(window);

	// Intialize resources
	lut::RenderPass renderPass = create_render_pass(window);

	// Create depth buffer
	auto [depthBuffer, depthBufferView] = create_depth_buffer(window, allocator);

	std::vector<lut::Framebuffer> framebuffers;
	create_swapchain_framebuffers(window, renderPass.handle, framebuffers, depthBufferView.handle);

	lut::CommandPool cpool = lut::create_command_pool(window, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

	std::vector<VkCommandBuffer> cbuffers;
	std::vector<lut::Fence> cbfences;

	for (std::size_t i = 0; i < framebuffers.size(); ++i)
	{
		cbuffers.emplace_back(lut::alloc_command_buffer(window, cpool.handle));
		cbfences.emplace_back(lut::create_fence(window, VK_FENCE_CREATE_SIGNALED_BIT));
	}

	lut::Semaphore imageAvailable = lut::create_semaphore(window);
	lut::Semaphore renderFinished = lut::create_semaphore(window);

	// Create a descriptor pool from which to allocate descriptor sets
	lut::DescriptorPool descPool = lut::create_descriptor_pool(window);

	// Copy descriptor sets into a vector for pipeline creation
	// It is imperative that the descriptor set order matches the binding order in the shaders
	std::vector<VkDescriptorSetLayout> descriptorSetLayouts;

	// Allocate a descriptor set for all vertex shaders
	// Hence the name 'scene' as it's global
	// This simply contains a camera/projection matrix, as we aren't dealing with
	// ...any other transformations for now, such as on the object itself.
	lut::DescriptorSetLayout sceneLayout = create_scene_descriptor_layout(window);
	descriptorSetLayouts.push_back(sceneLayout.handle);

	// Allocate two descriptor sets, one for materials, and one for lights
	// The light descriptor set will be the same across all objects
	// It is also convenient to keep them seperate as we will need to update the light descriptor set on each frame later, when animation is added
	lut::DescriptorSetLayout defaultObjectLayout = create_defaultobject_descriptor_layout(window);
	descriptorSetLayouts.push_back(defaultObjectLayout.handle);
	lut::DescriptorSetLayout lightLayout = create_light_descriptor_layout(window);
	descriptorSetLayouts.push_back(lightLayout.handle);

	// Create pipelines for each
	lut::PipelineLayout defaultPipeLayout = create_pipeline_layout(window, descriptorSetLayouts);
	lut::Pipeline defaultPipeline = create_pipeline(window, renderPass.handle, defaultPipeLayout.handle, cfg::kVertShaderPath, cfg::kFragShaderPath);

	// Allocate the set for scene and light, each mesh will have its own set
	VkDescriptorSet sceneDescriptors = lut::alloc_desc_set(window, descPool.handle, sceneLayout.handle);


	// Loading models
	std::vector<GPUModel> models;
	models.reserve(cfg::kModels.size());
	for (auto& modelPath : cfg::kModels)
	{
		auto model = load_obj_model(modelPath);
		// Populate materials
		std::vector<GPUMaterialInfo> materials;
		materials.reserve(model.materials.size());
		for (auto& materialInfo : model.materials)
		{
			// Get the material in suitable format for uniform
			// Color only, debug visualisations
			//glsl::MaterialUniform material = { glm::vec4(materialInfo.color, 1.0f) };

			// For Blinn-Phong
			//glsl::MaterialUniform material = {	glm::vec4(materialInfo.emissive, 1.0f),
			//									glm::vec4(materialInfo.diffuse, 1.0f),
			//									glm::vec4(materialInfo.specular, 1.0f),
			//									materialInfo.shininess };

			// For PBR
			glsl::MaterialUniform material = {	glm::vec4(materialInfo.emissive, 1.0f),
												glm::vec4(materialInfo.albedo, 1.0f),
												materialInfo.shininess,
												materialInfo.metalness };

			// Allocate a descriptor set for this material
			VkDescriptorSet matDescSet;
			lut::Buffer matUBO;
			VkPipeline* pipe;
			VkPipelineLayout* pipeLayout;

			matDescSet = lut::alloc_desc_set(window, descPool.handle, defaultObjectLayout.handle);

			// Set appropriate pipeline pointers for no textures
			pipe = &defaultPipeline.handle;
			pipeLayout = &defaultPipeLayout.handle;

			// Transfer material uniform
			{
				// Initialise it
				matUBO = lut::create_buffer(
					allocator,
					sizeof(glsl::MaterialUniform),
					VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
					VMA_MEMORY_USAGE_GPU_ONLY
				);
				// Staging buffer, which will be used to transfer data from the CPU to the GPU
				lut::Buffer matUBOStaging = lut::create_buffer(
					allocator,
					sizeof(glsl::MaterialUniform),
					VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
					VMA_MEMORY_USAGE_CPU_TO_GPU
				);
				VkWriteDescriptorSet writeDesc[1]{};

				VkDescriptorBufferInfo bufferInfo{};
				bufferInfo.buffer = matUBO.buffer;
				bufferInfo.range = VK_WHOLE_SIZE;

				writeDesc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				writeDesc[0].dstSet = matDescSet;
				writeDesc[0].dstBinding = 1;
				writeDesc[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
				writeDesc[0].descriptorCount = 1;
				writeDesc[0].pBufferInfo = &bufferInfo;

				constexpr auto numSets = sizeof(writeDesc) / sizeof(writeDesc[0]);
				vkUpdateDescriptorSets(window.device, numSets, writeDesc, 0, nullptr);

				// Write to the new uniform
				void* uniformPtr = nullptr;
				if (auto const res = vmaMapMemory(allocator.allocator, matUBOStaging.allocation, &uniformPtr); VK_SUCCESS != res)
				{
					throw lut::Error("Error mapping memory for writing\n"
						"vmaMapMemory() returned %s", lut::to_string(res).c_str()
					);
				}

				std::memcpy(uniformPtr, &material, sizeof(glsl::MaterialUniform));
				vmaUnmapMemory(allocator.allocator, matUBOStaging.allocation);

				// Now, we need to prepare to issue the transfer commands that will copy the data from the staging buffers to the GPU buffers
				// We need a fence to block here while the Vulkan commands are executed. We don't want to delete the resources while
				// ..the gpu is still using them...
				lut::Fence uploadComplete = create_fence(window);

				// Queue data uploads from staging buffers to the final buffers
				// This uses a seperate command pool for simplicity
				lut::CommandPool uploadPool = create_command_pool(window);
				VkCommandBuffer uploadCmd = alloc_command_buffer(window, uploadPool.handle);

				// Record the copy commands into the buffer
				VkCommandBufferBeginInfo beginInfo{};
				beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
				beginInfo.flags = 0;
				beginInfo.pInheritanceInfo = nullptr;

				if (auto const res = vkBeginCommandBuffer(uploadCmd, &beginInfo); VK_SUCCESS != res)
				{
					throw lut::Error("Beginning command buffer recording\n"
						"vkBeginCommandBuffer() returned %s", lut::to_string(res).c_str()
					);
				}

				VkBufferCopy matUBOCopy{};
				matUBOCopy.size = sizeof(glsl::MaterialUniform);

				lut::buffer_barrier(uploadCmd,
					matUBO.buffer,
					VK_ACCESS_UNIFORM_READ_BIT,
					VK_ACCESS_TRANSFER_WRITE_BIT,
					VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
					VK_PIPELINE_STAGE_TRANSFER_BIT
				);

				vkCmdCopyBuffer(uploadCmd, matUBOStaging.buffer, matUBO.buffer, 1, &matUBOCopy);

				lut::buffer_barrier(uploadCmd,
					matUBO.buffer,
					VK_ACCESS_TRANSFER_WRITE_BIT,
					VK_ACCESS_UNIFORM_READ_BIT,
					VK_PIPELINE_STAGE_TRANSFER_BIT,
					VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
				);

				if (auto const res = vkEndCommandBuffer(uploadCmd); VK_SUCCESS != res)
				{
					throw lut::Error("Ending command buffer recording\n"
						"vkEndCommandBuffer() returned %s", lut::to_string(res).c_str()
					);
				}

				VkSubmitInfo submitInfo{};
				submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
				submitInfo.commandBufferCount = 1;
				submitInfo.pCommandBuffers = &uploadCmd;

				if (auto const res = vkQueueSubmit(window.graphicsQueue, 1, &submitInfo, uploadComplete.handle); VK_SUCCESS != res)
				{
					throw lut::Error("SUbmitting commands\n"
						"vkQueueSubmit() returned %s", lut::to_string(res).c_str()
					);
				}

				// Wait for commands to finish before destroying the temp resources
				// We don't need to destroy them explicitly, they are handled by their respective destructors
				// But we do need to block here until we are ready for these to be called
				if (auto const res = vkWaitForFences(window.device, 1, &uploadComplete.handle, VK_TRUE, std::numeric_limits<std::uint64_t>::max()); VK_SUCCESS != res)
				{
					throw lut::Error("Waiting for upload to complete\n"
						"vkWaitForFences() returned %s", lut::to_string(res).c_str()
					);
				}
			}

			// Pass ownership to GPUMaterialInfo
			materials.emplace_back(
				GPUMaterialInfo{
					pipe,
					pipeLayout,
					std::move(matUBO),
					std::move(matDescSet)
				}
			);

			// materials.emplace_back(materialInfo.color);
		}

		// Send to gpu buffers
		models.push_back(load_gpu_model(window, allocator, model, std::move(materials)));
	}

	VkDescriptorSet lightDescriptors = lut::alloc_desc_set(window, descPool.handle, lightLayout.handle);

	// Create UBO for projection matrix
	lut::Buffer sceneUBO = lut::create_buffer(
		allocator,
		sizeof(glsl::SceneUniform),
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VMA_MEMORY_USAGE_CPU_TO_GPU
	);

	// Initialize scene descriptor set with vkUpdateDescriptorSets
	// Write to the descriptor set to specify its details
	{
		VkWriteDescriptorSet desc[1]{};

		VkDescriptorBufferInfo sceneUboInfo{};
		sceneUboInfo.buffer = sceneUBO.buffer;
		sceneUboInfo.range = VK_WHOLE_SIZE;

		desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc[0].dstSet = sceneDescriptors;
		desc[0].dstBinding = 0;
		desc[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		desc[0].descriptorCount = 1;
		desc[0].pBufferInfo = &sceneUboInfo;


		constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
		vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
	}

	// Create UBO for light matrix
	lut::Buffer lightUBO = lut::create_buffer(
		allocator,
		sizeof(glsl::LightUniform),
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VMA_MEMORY_USAGE_CPU_TO_GPU
	);

	// Initialise light descriptor set with vkUpdateDescriptorSets
	{
		VkWriteDescriptorSet desc[1]{};

		VkDescriptorBufferInfo lightUboInfo{};
		lightUboInfo.buffer = lightUBO.buffer;
		lightUboInfo.range = VK_WHOLE_SIZE;

		desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc[0].dstSet = lightDescriptors;
		desc[0].dstBinding = 2;
		desc[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		desc[0].descriptorCount = 1;
		desc[0].pBufferInfo = &lightUboInfo;


		constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
		vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
	}

	// Create a sphere from which to sample light positions
	cfg::Sphere sphere = {	glm::vec3(0.0f),
							7.0f };

	// Create random sampler
	std::random_device randomDevice;
	std::default_random_engine randomEngine(randomDevice());
	std::uniform_real_distribution<> uniformDistribution(0.0f, 360.0f);

	glsl::LightUniform lightUniforms{};
	lightUniforms.ambientColor = glm::vec4(0.02f, 0.02f, 0.02f, 1.0f);

	float theta = 0.0f, phi = 0.0f;

	for (size_t i = 0; i < glsl::MAX_LIGHTS; i++)
	{
		theta = uniformDistribution(randomEngine);
		phi = uniformDistribution(randomEngine);
		// Sample a new point from the sphere and make a light
		glm::vec3 cartesian =	sphere.center +
								glm::vec3(	sphere.radius * sin(phi) * cos(theta),
											sphere.radius * sin(phi) * sin(theta),
											sphere.radius * cos(phi)
		);

		// Add the light to the UBO
		lightUniforms.lights[i] = { glm::vec4(cartesian, 1.0f),
									glm::vec4((bool)(1 & (i + 1)), (bool)(2 & (i + 1)), (bool)(4 & (i + 1)), 1.0f)
		};

		lightUniforms.numLights++;
		lightState.numLightsEnabled = lightUniforms.numLights;
	}

	// Application main loop
	bool recreateSwapchain = false;

	while (!glfwWindowShouldClose(window.window))
	{
				// Let GLFW process events.
		// glfwPollEvents() checks for events, processes them. If there are no
		// events, it will return immediately. Alternatively, glfwWaitEvents()
		// will wait for any event to occur, process it, and only return at
		// that point. The former is useful for applications where you want to
		// render as fast as possible, whereas the latter is useful for
		// input-driven applications, where redrawing is only needed in
		// reaction to user input (or similar).
		glfwPollEvents(); // or: glfwWaitEvents()

		// Recreate swap chain?
		if (recreateSwapchain)
		{
			// Re-create swapchain and associated resources!
			vkDeviceWaitIdle(window.device);

			auto const changes = recreate_swapchain(window);

			if (changes.changedFormat)
			{
				renderPass = create_render_pass(window);
			}

			if (changes.changedSize)
			{
				std::tie(depthBuffer, depthBufferView) = create_depth_buffer(window, allocator);
				defaultPipeline = create_pipeline(window, renderPass.handle, defaultPipeLayout.handle, cfg::kVertShaderPath, cfg::kFragShaderPath);
			}

			framebuffers.clear();
			create_swapchain_framebuffers(window, renderPass.handle, framebuffers, depthBufferView.handle);

			recreateSwapchain = false;
			continue;
		}


		// Acquire swapchain image.
		std::uint32_t imageIndex = 0;
		auto const acquireRes = vkAcquireNextImageKHR(
			window.device,
			window.swapchain,
			std::numeric_limits<std::uint64_t>::max(),
			imageAvailable.handle,
			VK_NULL_HANDLE,
			&imageIndex
		);

		// Handle swapchain issues
		if (VK_SUBOPTIMAL_KHR == acquireRes || VK_ERROR_OUT_OF_DATE_KHR == acquireRes)
		{
			// This usually occurs when the window has been resized.
			// In any case, the latter flag means the swapchain needs to be recreated-
			//		the former we can technically continue, but it'mKeyS easier to recreate it now.
			// Hence, we set the recreate flag and skip to the top of the loop.
			recreateSwapchain = true;
			continue;
		}

		// If the above checks passed, check for general (unhandled) issues with the swapchain
		if (VK_SUCCESS != acquireRes)
		{
			throw lut::Error("Unable to acquire next swapchain image\n"
				"vkAcquireNextImageKHR() returned %s", lut::to_string(acquireRes).c_str()
			);
		}
		// Update times
		lastTime = currentTime;
		currentTime = glfwGetTime();
		float deltaTime = currentTime - lastTime;

		// Update camera
		mainCamera.stepPosition(deltaTime);

		// Prepare data for this frame
		glsl::SceneUniform sceneUniforms{};
		update_scene_uniforms(sceneUniforms, window.swapchainExtent.width, window.swapchainExtent.height);

		update_light_uniforms(lightUniforms, lightState, deltaTime * 2.5f);

		// Wait for command buffer to be available
		assert(std::size_t(imageIndex) < cbfences.size());

		if (auto const res = vkWaitForFences(window.device, 1, &cbfences[imageIndex].handle, VK_TRUE, std::numeric_limits<std::uint64_t>::max());
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to wait for command buffer fence %u\n"
				"vkWaitForFences() returned %s", imageIndex, lut::to_string(res).c_str()
			);
		}

		// Reset the fence to unsignalled, so it can be reused immediately
		if (auto const res = vkResetFences(window.device, 1, &cbfences[imageIndex].handle);
			VK_SUCCESS != res)
		{
			throw lut::Error("Inable to reset command buffer fence %u\n"
				"vkResetFences() returned %s", imageIndex, lut::to_string(res).c_str()
			);
		}

		// Record and submit commands
		assert(std::size_t(imageIndex) < cbuffers.size());
		assert(std::size_t(imageIndex) < framebuffers.size());

		record_commands(
			cbuffers[imageIndex],
			renderPass.handle,
			framebuffers[imageIndex].handle,
			defaultPipeline.handle,
			defaultPipeLayout.handle,
			window.swapchainExtent,
			sceneUBO.buffer,
			sceneUniforms,
			sceneDescriptors,
			lightUBO.buffer,
			lightUniforms,
			lightDescriptors,
			&models
		);

		submit_commands(
			window,
			cbuffers[imageIndex],
			cbfences[imageIndex].handle,
			imageAvailable.handle,
			renderFinished.handle
		);

		// Present rendered images.
		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = &renderFinished.handle;
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = &window.swapchain;
		presentInfo.pImageIndices = &imageIndex;
		presentInfo.pResults = nullptr;

		auto const presentRes = vkQueuePresentKHR(window.presentQueue, &presentInfo);

		if (VK_SUBOPTIMAL_KHR == presentRes || VK_ERROR_OUT_OF_DATE_KHR == presentRes)
		{
			recreateSwapchain = true;
			continue;
		}
		else if (VK_SUCCESS != presentRes)
		{
			throw lut::Error("Unable to present swapchain image %u\n"
				"vkQueuePresentKHR() returned %s", imageIndex, lut::to_string(presentRes).c_str()
			);
		}
	}

	// Cleanup takes place automatically in the destructors, but we sill need
	// to ensure that all Vulkan commands have finished before that.
	vkDeviceWaitIdle(window.device);

	return 0;
}
catch (std::exception const& eErr)
{
	std::fprintf(stderr, "\n");
	std::fprintf(stderr, "Error: %s\n", eErr.what());
	return 1;
}

namespace
{
	void glfw_callback_key_press(GLFWwindow* aWindow, int aKey, int /*aScanCode*/, int aAction, int /*aModifierFlags*/)
	{
		if (GLFW_KEY_ESCAPE == aKey && GLFW_PRESS == aAction)
		{
			glfwSetWindowShouldClose(aWindow, GLFW_TRUE);
			return;
		}

		// Update the camera
		bool state;
		aAction == GLFW_PRESS || aAction == GLFW_REPEAT ? state = true : state = false;
		switch (aKey)
		{
		case GLFW_KEY_W:
			mainCamera.setKeyState(camera::KEY_W, state);
			break;
		case GLFW_KEY_S:
			mainCamera.setKeyState(camera::KEY_S, state);
			break;
		case GLFW_KEY_A:
			mainCamera.setKeyState(camera::KEY_A, state);
			break;
		case GLFW_KEY_D:
			mainCamera.setKeyState(camera::KEY_D, state);
			break;
		case GLFW_KEY_Q:
			mainCamera.setKeyState(camera::KEY_Q, state);
			break;
		case GLFW_KEY_E:
			mainCamera.setKeyState(camera::KEY_E, state);
			break;
		case GLFW_KEY_LEFT_SHIFT:
			mainCamera.setKeyState(camera::KEY_SHIFT, state);
			break;
		case GLFW_KEY_LEFT_CONTROL:
			mainCamera.setKeyState(camera::KEY_CTRL, state);
			break;
		case GLFW_KEY_SPACE:
			if (aAction == GLFW_PRESS)
			{
				lightState.animEnabled = !lightState.animEnabled;
			}
			break;
		case GLFW_KEY_0:
			if (aAction == GLFW_PRESS)
			{
				lightState.numLightsEnabled = 0;
			}
			break;
		case GLFW_KEY_1:
			if (aAction == GLFW_PRESS)
			{
				lightState.numLightsEnabled = 1;
			}
			break;
		case GLFW_KEY_2:
			if (aAction == GLFW_PRESS)
			{
				lightState.numLightsEnabled = 2;
			}
			break;
		case GLFW_KEY_3:
			if (aAction == GLFW_PRESS)
			{
				lightState.numLightsEnabled = 3;
			}
			break;
		case GLFW_KEY_4:
			if (aAction == GLFW_PRESS)
			{
				lightState.numLightsEnabled = 4;
			}
			break;
		case GLFW_KEY_5:
			if (aAction == GLFW_PRESS)
			{
				lightState.numLightsEnabled = 5;
			}
			break;
		case GLFW_KEY_6:
			if (aAction == GLFW_PRESS)
			{
				lightState.numLightsEnabled = 6;
			}
			break;
		case GLFW_KEY_7:
			if (aAction == GLFW_PRESS)
			{
				lightState.numLightsEnabled = 7;
			}
			break;
		case GLFW_KEY_M:
			if (aAction == GLFW_PRESS)
			{
				if (lightState.reflectionModel == glsl::PBR_PHONG)
				{
					lightState.reflectionModel = glsl::PBR_GGX;
				}
				else
				{
					lightState.reflectionModel = glsl::PBR_PHONG;
				}
			}
			break;
		default:
			break;
		}

	}

	void glfw_callback_mouse_click(GLFWwindow* aWindow, int aButton, int aAction, int aMods)
	{
		if (GLFW_MOUSE_BUTTON_2 == aButton && GLFW_PRESS == aAction)
		{
			bool mouseCaptured = mainCamera.getKeyState(camera::MOUSE_2);
			double xPos, yPos;
			mouseCaptured ? glfwSetInputMode(aWindow, GLFW_CURSOR, GLFW_CURSOR_NORMAL) : glfwSetInputMode(aWindow, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
			glfwGetCursorPos(aWindow, &xPos, &yPos);
			lastX = xPos;
			lastY = yPos;
			mainCamera.setKeyState(camera::MOUSE_2, !mouseCaptured);
			firstMouse = true;
		}
	}

	void glfw_callback_mouse_move(GLFWwindow* aWindow, double aXPos, double aYPos)
	{
		if (!mainCamera.getKeyState(camera::MOUSE_2))
		{
			return;
		}
		if (firstMouse)
		{
			lastX = aXPos;
			lastY = aYPos;
			firstMouse = false;
		}
		float xOffset = aXPos - lastX;
		float yOffset = lastY - aYPos;
		lastX = aXPos;
		lastY = aYPos;

		const float sensitivity = 0.1f;
		xOffset *= sensitivity;
		yOffset *= sensitivity;

		yaw = yaw + xOffset;
		float newPitch = pitch + yOffset;
		//Clip pitch if user tries to look too far up
		if (!(newPitch < 90.0f))
			pitch = 89.999;
		else if (!(newPitch > -90.0f))
			pitch = -89.999f;
		else
			pitch = newPitch;

		glm::vec3 direction;
		direction.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
		direction.y = sin(glm::radians(pitch));
		direction.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
		mainCamera.mCameraFront = glm::normalize(direction);
	}

	void update_scene_uniforms(glsl::SceneUniform& aSceneUniforms, std::uint32_t aFramebufferWidth, std::uint32_t aFramebufferHeight)
	{
		float const aspect = float(aFramebufferWidth) / float(aFramebufferHeight);

		aSceneUniforms.projection = glm::perspectiveRH_ZO(
			lut::Radians(cfg::kCameraFov).value(),
			aspect,
			cfg::kCameraNear,
			cfg::kCameraFar
		);
		aSceneUniforms.projection[1][1] *= -1.0f; // Mirror Y axis

		aSceneUniforms.camera = glm::lookAt(mainCamera.mCameraPosition, mainCamera.mCameraPosition + mainCamera.mCameraFront, mainCamera.mCameraUp);

		aSceneUniforms.projCam = aSceneUniforms.projection * aSceneUniforms.camera;

		aSceneUniforms.cameraPos = glm::vec4(mainCamera.mCameraPosition, 1.0f);
	}

	void update_light_uniforms(glsl::LightUniform& aLightUniform, LightState aLightState, float aDt)
	{
		if (aLightState.animEnabled)
		{
			glm::mat4 rotation = glm::mat4(1.0f);
			rotation = glm::rotate(rotation, aDt, glm::vec3(0.0f, 1.0f, 0.0f));

			for (size_t i = 0; i < aLightUniform.numLights; i++)
			{
				glm::vec4 newPos = aLightUniform.lights[i].position * rotation;
				aLightUniform.lights[i].position = newPos;
			}
		}

		aLightUniform.numLights = aLightState.numLightsEnabled;
		aLightUniform.reflectionModel = aLightState.reflectionModel;
	}

	lut::RenderPass create_render_pass(lut::VulkanWindow const& aWindow)
	{
		// Render Pass Attachments
		VkAttachmentDescription attachments[2]{};
		attachments[0].format = aWindow.swapchainFormat;
		attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		attachments[1].format = cfg::kDepthFormat;
		attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		// The one and only subpass
		// Attachments first
		VkAttachmentReference subpassAttachments[1]{};
		subpassAttachments[0].attachment = 0; // 0 Refers to attachments[0] declared earlier
		subpassAttachments[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentReference depthAttachment{};
		depthAttachment.attachment = 1;
		depthAttachment.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		// Now the description
		VkSubpassDescription subpasses[1]{};
		subpasses[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpasses[0].colorAttachmentCount = 1;
		subpasses[0].pColorAttachments = subpassAttachments;
		subpasses[0].pDepthStencilAttachment = &depthAttachment;
		// Many more members of subpass exist and are left at null, as we're only using the colour attachment

		// No explicit subpass dependencies, as we don't need to copy the image as in the previous exercise
		// Vulkan does set up some implicit dependencies for us

		// Can finally create the actual render pass
		// The info to pass
		VkRenderPassCreateInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		passInfo.attachmentCount = 2;
		passInfo.pAttachments = attachments;
		passInfo.subpassCount = 1;
		passInfo.pSubpasses = subpasses;
		passInfo.dependencyCount = 0;
		passInfo.pDependencies = nullptr;

		// The pass!!
		VkRenderPass rpass = VK_NULL_HANDLE;
		if (auto const res = vkCreateRenderPass(aWindow.device, &passInfo, nullptr, &rpass); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create renderpass\n", "vkCreateRenderPass() returned %s", lut::to_string(res).c_str());
		}

		return lut::RenderPass(aWindow.device, rpass);
	}

	lut::PipelineLayout create_pipeline_layout(lut::VulkanContext const& aContext, std::vector<VkDescriptorSetLayout> descriptorSetlayouts)
	{
		VkPipelineLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		layoutInfo.setLayoutCount = descriptorSetlayouts.size();
		layoutInfo.pSetLayouts = descriptorSetlayouts.data();
		layoutInfo.pushConstantRangeCount = 0;
		layoutInfo.pPushConstantRanges = nullptr;

		VkPipelineLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreatePipelineLayout(aContext.device, &layoutInfo, nullptr, &layout);
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create pipeline layout\n"
				"vkCreatePipelineLayout() returned %s", lut::to_string(res).c_str()
			);
		}

		return lut::PipelineLayout(aContext.device, layout);
	}

	lut::Pipeline create_pipeline(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout, const char* aVertexShader, const char* aFragmentShader)
	{
		// Load the shader modules
		lut::ShaderModule vertexShader = lut::load_shader_module(aWindow, aVertexShader);
		lut::ShaderModule fragmentShader = lut::load_shader_module(aWindow, aFragmentShader);

		// Filling out the required parts for VkGraphicsPipelineCreateInfo, in order.
		// Define the shader stages
		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vertexShader.handle;
		stages[0].pName = "main";

		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = fragmentShader.handle;
		stages[1].pName = "main";

		// We need to describe the vertex inputs
		// Positions description
		VkVertexInputBindingDescription vertexInputs[3]{};
		vertexInputs[0].binding = 0;
		vertexInputs[0].stride = sizeof(float) * 3;
		vertexInputs[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		vertexInputs[1].binding = 1;
		vertexInputs[1].stride = sizeof(float) * 3;
		vertexInputs[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		vertexInputs[2].binding = 2;
		vertexInputs[2].stride = sizeof(float) * 2;
		vertexInputs[2].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		// Attribute description
		VkVertexInputAttributeDescription vertexAttributes[3]{};
		vertexAttributes[0].binding = 0;
		vertexAttributes[0].location = 0;
		vertexAttributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[0].offset = 0;

		vertexAttributes[1].binding = 1;
		vertexAttributes[1].location = 1;
		vertexAttributes[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[1].offset = 0;

		vertexAttributes[2].binding = 2;
		vertexAttributes[2].location = 2;
		vertexAttributes[2].format = VK_FORMAT_R32G32_SFLOAT;
		vertexAttributes[2].offset = 0;

		VkPipelineVertexInputStateCreateInfo inputInfo{};
		inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		inputInfo.vertexBindingDescriptionCount = 3;
		inputInfo.pVertexBindingDescriptions = vertexInputs;
		inputInfo.vertexAttributeDescriptionCount = 3;
		inputInfo.pVertexAttributeDescriptions = vertexAttributes;

		// Define which primitive the input is. We are using triangles.
		VkPipelineInputAssemblyStateCreateInfo assemblyInfo{};
		assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assemblyInfo.primitiveRestartEnable = VK_FALSE;

		// Not using the tessalation stage, hence leave at nullptr.

		// Creating the viewport
		VkViewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = float(aWindow.swapchainExtent.width);
		viewport.height = float(aWindow.swapchainExtent.height);
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;

		VkRect2D scissor{};
		scissor.offset = VkOffset2D{ 0, 0 };
		scissor.extent = VkExtent2D{ aWindow.swapchainExtent.width, aWindow.swapchainExtent.height };

		VkPipelineViewportStateCreateInfo viewportInfo{};
		viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportInfo.viewportCount = 1;
		viewportInfo.pViewports = &viewport;
		viewportInfo.scissorCount = 1;
		viewportInfo.pScissors = &scissor;

		// Rasterisation pipeline
		VkPipelineRasterizationStateCreateInfo rasterInfo{};
		rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterInfo.depthClampEnable = VK_FALSE;
		rasterInfo.rasterizerDiscardEnable = VK_FALSE;
		rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
		rasterInfo.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterInfo.depthBiasEnable = VK_FALSE;
		rasterInfo.lineWidth = 1.0f;

		// Multisampling state: not using here.
		VkPipelineMultisampleStateCreateInfo samplingInfo{};
		samplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		samplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		// Depth/stencil state
		VkPipelineDepthStencilStateCreateInfo depthInfo{};
		depthInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthInfo.depthTestEnable = VK_TRUE;
		depthInfo.depthWriteEnable = VK_TRUE;
		depthInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthInfo.minDepthBounds = 0.0f;
		depthInfo.maxDepthBounds = 1.0f;

		// Colour blend state. These are defined per colour attachement, of which we have one, so we only need one state:
		VkPipelineColorBlendAttachmentState blendStates[1]{};
		blendStates[0].blendEnable = VK_FALSE;
		blendStates[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
			| VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

		VkPipelineColorBlendStateCreateInfo blendInfo{};
		blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blendInfo.logicOpEnable = VK_FALSE;
		blendInfo.attachmentCount = 1;
		blendInfo.pAttachments = blendStates;

		// Dynamic state: not used here, can also be left as nullptr

		// Create pipeline
		VkGraphicsPipelineCreateInfo pipelineInfo{};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

		pipelineInfo.stageCount = 2;
		pipelineInfo.pStages = stages;

		pipelineInfo.pVertexInputState = &inputInfo;
		pipelineInfo.pInputAssemblyState = &assemblyInfo;
		pipelineInfo.pTessellationState = nullptr;
		pipelineInfo.pViewportState = &viewportInfo;
		pipelineInfo.pRasterizationState = &rasterInfo;
		pipelineInfo.pMultisampleState = &samplingInfo;
		pipelineInfo.pDepthStencilState = &depthInfo;
		pipelineInfo.pColorBlendState = &blendInfo;
		pipelineInfo.pDynamicState = nullptr;
		pipelineInfo.layout = aPipelineLayout;
		pipelineInfo.renderPass = aRenderPass;
		pipelineInfo.subpass = 0;

		VkPipeline pipeline = VK_NULL_HANDLE;
		auto const res = vkCreateGraphicsPipelines(aWindow.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline);
		if (VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create graphics pipeline\n"
				"vkCreateGraphicsPipelines() returned %s", lut::to_string(res).c_str());
		}

		return lut::Pipeline(aWindow.device, pipeline);
	}

	// Create framebuffers for the swapchain, storing them in aFramebuffers
	void create_swapchain_framebuffers(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, std::vector<lut::Framebuffer>& aFramebuffers, VkImageView aDepthView)
	{
		assert(aFramebuffers.empty());

		for (std::size_t i = 0; i < aWindow.swapViews.size(); ++i)
		{
			VkImageView attachments[2] = {
				aWindow.swapViews[i],
				aDepthView
			};

			VkFramebufferCreateInfo fbInfo{};
			fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			fbInfo.flags = 0;
			fbInfo.renderPass = aRenderPass;
			fbInfo.attachmentCount = 2;
			fbInfo.pAttachments = attachments;
			fbInfo.width = aWindow.swapchainExtent.width;
			fbInfo.height = aWindow.swapchainExtent.height;
			fbInfo.layers = 1;

			VkFramebuffer fb = VK_NULL_HANDLE;
			if (auto const res = vkCreateFramebuffer(aWindow.device, &fbInfo, nullptr, &fb);
				VK_SUCCESS != res)
			{
				throw lut::Error("Unable to create framebuffer for swap chain image %zu\n"
					"vkCreateFramebuffer() returned %s", i, lut::to_string(res).c_str()
				);
			}

			aFramebuffers.emplace_back(lut::Framebuffer(aWindow.device, fb));
		}

		assert(aWindow.swapViews.size() == aFramebuffers.size());
	}

	lut::DescriptorSetLayout create_scene_descriptor_layout(lut::VulkanWindow const& aWindow)
	{
		VkDescriptorSetLayoutBinding bindings[1]{};
		bindings[0].binding = 0; // To batch binding = N in the shader

		bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		bindings[0].descriptorCount = 1;
		bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT; // Defines the stages where this buffer is used

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = sizeof(bindings) / sizeof(bindings[0]);
		layoutInfo.pBindings = bindings;

		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreateDescriptorSetLayout(aWindow.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create descriptor set layout\n"
				"vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str()
			);
		}

		return lut::DescriptorSetLayout(aWindow.device, layout);
	}

	lut::DescriptorSetLayout create_defaultobject_descriptor_layout(lut::VulkanWindow const& aWindow)
	{
		VkDescriptorSetLayoutBinding bindings[1]{};
		bindings[0].binding = 1;
		bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		bindings[0].descriptorCount = 1;
		bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT; // used ONLY in the fragment shader!

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = sizeof(bindings) / sizeof(bindings[0]);
		layoutInfo.pBindings = bindings;

		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreateDescriptorSetLayout(aWindow.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create descriptor set layout\n"
				"vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str()
			);
		}

		return lut::DescriptorSetLayout(aWindow.device, layout);
	}

	lut::DescriptorSetLayout create_light_descriptor_layout(lut::VulkanWindow const& aWindow)
	{
		VkDescriptorSetLayoutBinding bindings[1]{};
		bindings[0].binding = 2;
		bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		bindings[0].descriptorCount = 1;
		bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT; // used ONLY in the fragment shader!

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = sizeof(bindings) / sizeof(bindings[0]);
		layoutInfo.pBindings = bindings;

		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreateDescriptorSetLayout(aWindow.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create descriptor set layout\n"
				"vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str()
			);
		}

		return lut::DescriptorSetLayout(aWindow.device, layout);
	}

	GPUModel load_gpu_model(labutils::VulkanContext const& aContext, labutils::Allocator const& aAllocator, ModelData& model, std::vector<GPUMaterialInfo>&& materials)
	{
		// Get sizes
		size_t posBuffSize, normBuffSize, texcoordBuffSize;
		posBuffSize = model.vertexPositions.size() * sizeof(float) * 3;
		normBuffSize = model.vertexNormals.size() * sizeof(float) * 3;
		texcoordBuffSize = model.vertexTextureCoords.size() * sizeof(float) * 2;

		// Create the buffers on GPU
		// Destination buffers, hence GPU only
		lut::Buffer vertexPosGPU = lut::create_buffer(
			aAllocator,
			posBuffSize,
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VMA_MEMORY_USAGE_GPU_ONLY
		);
		lut::Buffer vertexNormGPU = lut::create_buffer(
			aAllocator,
			normBuffSize,
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VMA_MEMORY_USAGE_GPU_ONLY
		);
		lut::Buffer vertexTexcoordGPU = lut::create_buffer(
			aAllocator,
			texcoordBuffSize,
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VMA_MEMORY_USAGE_GPU_ONLY
		);
		// Staging buffers, which will be used to transfer data from the CPU to the GPU
		lut::Buffer posStaging = lut::create_buffer(
			aAllocator,
			posBuffSize,
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_MEMORY_USAGE_CPU_TO_GPU
		);
		lut::Buffer normStaging = lut::create_buffer(
			aAllocator,
			normBuffSize,
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_MEMORY_USAGE_CPU_TO_GPU
		);
		lut::Buffer texcoordStaging = lut::create_buffer(
			aAllocator,
			texcoordBuffSize,
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VMA_MEMORY_USAGE_CPU_TO_GPU
		);

		// We must now access the staging buffers
		// We use VMA functions to map memory to mKeyA normal C++ pointer, and copy the data there
		// We can then unmap the memory
		void* posPtr = nullptr;
		if (auto const res = vmaMapMemory(aAllocator.allocator, posStaging.allocation, &posPtr); VK_SUCCESS != res)
		{
			throw lut::Error("Error mapping memory for writing\n"
				"vmaMapMemory() returned %s", lut::to_string(res).c_str()
			);
		}
		std::memcpy(posPtr, model.vertexPositions.data(), posBuffSize);
		vmaUnmapMemory(aAllocator.allocator, posStaging.allocation);

		void* normPtr = nullptr;
		if (auto const res = vmaMapMemory(aAllocator.allocator, normStaging.allocation, &normPtr); VK_SUCCESS != res)
		{
			throw lut::Error("Error mapping memory for writing\n"
				"vmaMapMemory() returned %s", lut::to_string(res).c_str()
			);
		}
		std::memcpy(normPtr, model.vertexNormals.data(), normBuffSize);
		vmaUnmapMemory(aAllocator.allocator, normStaging.allocation);

		void* texCoordPtr = nullptr;
		if (auto const res = vmaMapMemory(aAllocator.allocator, texcoordStaging.allocation, &texCoordPtr); VK_SUCCESS != res)
		{
			throw lut::Error("Error mapping memory for writing\n"
				"vmaMapMemory() returned %s", lut::to_string(res).c_str()
			);
		}
		std::memcpy(texCoordPtr, model.vertexTextureCoords.data(), texcoordBuffSize);
		vmaUnmapMemory(aAllocator.allocator, texcoordStaging.allocation);

		// Now, we need to prepare to issue the transfer commands that will copy the data from the staging buffers to the GPU buffers
		// We need mKeyA fence to block here while the Vulkan commands are executed. We don't want to delete the resources while
		// ..the gpu is still using them...
		lut::Fence uploadComplete = create_fence(aContext);

		// Queue data uploads from staging buffers to the final buffers
		// This uses mKeyA seperate command pool for simplicity
		lut::CommandPool uploadPool = create_command_pool(aContext);
		VkCommandBuffer uploadCmd = alloc_command_buffer(aContext, uploadPool.handle);

		// Record the copy commands into the buffer
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = 0;
		beginInfo.pInheritanceInfo = nullptr;

		if (auto const res = vkBeginCommandBuffer(uploadCmd, &beginInfo); VK_SUCCESS != res)
		{
			throw lut::Error("Beginning command buffer recording\n"
				"vkBeginCommandBuffer() returned %s", lut::to_string(res).c_str()
			);
		}

		VkBufferCopy posCopy{};
		posCopy.size = posBuffSize;

		lut::buffer_barrier(uploadCmd,
			vertexPosGPU.buffer,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT
		);

		vkCmdCopyBuffer(uploadCmd, posStaging.buffer, vertexPosGPU.buffer, 1, &posCopy);

		lut::buffer_barrier(uploadCmd,
			vertexPosGPU.buffer,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
		);

		VkBufferCopy normCopy{};
		normCopy.size = normBuffSize;

		lut::buffer_barrier(uploadCmd,
			vertexNormGPU.buffer,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT
		);

		vkCmdCopyBuffer(uploadCmd, normStaging.buffer, vertexNormGPU.buffer, 1, &normCopy);

		lut::buffer_barrier(uploadCmd,
			vertexNormGPU.buffer,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
		);

		VkBufferCopy texcoordCopy{};
		texcoordCopy.size = texcoordBuffSize;

		lut::buffer_barrier(uploadCmd,
			vertexTexcoordGPU.buffer,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT
		);

		vkCmdCopyBuffer(uploadCmd, texcoordStaging.buffer, vertexTexcoordGPU.buffer, 1, &texcoordCopy);

		lut::buffer_barrier(uploadCmd,
			vertexTexcoordGPU.buffer,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
		);

		if (auto const res = vkEndCommandBuffer(uploadCmd); VK_SUCCESS != res)
		{
			throw lut::Error("Ending command buffer recording\n"
				"vkEndCommandBuffer() returned %s", lut::to_string(res).c_str()
			);
		}

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &uploadCmd;

		if (auto const res = vkQueueSubmit(aContext.graphicsQueue, 1, &submitInfo, uploadComplete.handle); VK_SUCCESS != res)
		{
			throw lut::Error("SUbmitting commands\n"
				"vkQueueSubmit() returned %s", lut::to_string(res).c_str()
			);
		}

		// Wait for commands to finish before destroying the temp resources
		// We don't need to destroy them explicitly, they are handled by their respective destructors
		// But we do need to block here until we are ready for these to be called
		if (auto const res = vkWaitForFences(aContext.device, 1, &uploadComplete.handle, VK_TRUE, std::numeric_limits<std::uint64_t>::max()); VK_SUCCESS != res)
		{
			throw lut::Error("Waiting for upload to complete\n"
				"vkWaitForFences() returned %s", lut::to_string(res).c_str()
			);
		}

		// Convert meshes into GPUMesh structures
		std::vector<GPUMesh> meshes;
		meshes.reserve(model.meshes.size());
		for (auto& mesh : model.meshes)
		{
			meshes.emplace_back(GPUMesh{ mesh.materialIndex, mesh.vertexStartIndex, mesh.numberOfVertices });
		}

		return GPUModel{
			std::move(vertexPosGPU),
			std::move(vertexNormGPU),
			std::move(vertexTexcoordGPU),
			std::move(materials),
			meshes
		};
	}

	void record_commands(VkCommandBuffer aCmdBuff, VkRenderPass aRenderPass, VkFramebuffer aFramebuffer, VkPipeline aGraphicsPipe, VkPipelineLayout aGraphicsPipelineLayout, VkExtent2D const& aImageExtent,
		VkBuffer aSceneUBO, glsl::SceneUniform aSceneUniform, VkDescriptorSet aSceneDescriptors, VkBuffer aLightUBO, glsl::LightUniform aLightUniform, VkDescriptorSet aLightDescriptors, std::vector<GPUModel>* models)
	{
		// Begin recording commands
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		beginInfo.pInheritanceInfo = nullptr;

		if (auto const res = vkBeginCommandBuffer(aCmdBuff, &beginInfo); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to begin recording command buffer\n"
				"vkBeginCommandBuffer() returned %s", lut::to_string(res).c_str());
		}

		// Upload scene uniforms
		// Use two barriers, both of which are only protecting the vertex shader behaivour.
		// We use the first one to ensure that previous ops are complete before updating the uniforms
		// ...and the second to make sure the uniform update is complete.
		lut::buffer_barrier(aCmdBuff,
			aSceneUBO,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT
		);

		vkCmdUpdateBuffer(aCmdBuff, aSceneUBO, 0, sizeof(glsl::SceneUniform), &aSceneUniform);

		lut::buffer_barrier(aCmdBuff,
			aSceneUBO,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_SHADER_BIT
		);

		// Update the light uniforms, same as above
		lut::buffer_barrier(aCmdBuff,
			aLightUBO,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT
		);

		vkCmdUpdateBuffer(aCmdBuff, aLightUBO, 0, sizeof(glsl::LightUniform), &aLightUniform);

		lut::buffer_barrier(aCmdBuff,
			aLightUBO,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_SHADER_BIT
		);

		// Begin render pass
		VkClearValue clearValues[2]{};
		clearValues[0].color.float32[0] = 0.1f;
		clearValues[0].color.float32[1] = 0.1f;
		clearValues[0].color.float32[2] = 0.1f;
		clearValues[0].color.float32[3] = 0.1f;

		clearValues[1].depthStencil.depth = 1.0f;

		VkRenderPassBeginInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		passInfo.renderPass = aRenderPass;
		passInfo.framebuffer = aFramebuffer;
		passInfo.renderArea.offset = VkOffset2D{ 0, 0 };
		passInfo.renderArea.extent = aImageExtent;
		passInfo.clearValueCount = 2;
		passInfo.pClearValues = clearValues;

		vkCmdBeginRenderPass(aCmdBuff, &passInfo, VK_SUBPASS_CONTENTS_INLINE);

		// Draw commands for all models. Models contain, or contain references to the relevant structures e.g. descriptor sets to draw themselves correctly.
		for (auto& model : *models)
		{
			// Bind vertex inputs
			VkBuffer buffers[3] = { model.positions.buffer, model.normals.buffer, model.texcoords.buffer };
			VkDeviceSize offsets[3]{};
			for (auto& mesh : model.meshes)
			{
				auto material = &model.materials[mesh.materialInfoIdx];
				vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, *(material->pipe));
				// Bind the global desc set
				vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, *(material->pipeLayout), 0, 1, &aSceneDescriptors, 0, nullptr);
				vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, *(material->pipeLayout), 1, 1, &(material->materialDescriptor), 0, nullptr);
				vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, *(material->pipeLayout), 2, 1, &aLightDescriptors, 0, nullptr);
				vkCmdBindVertexBuffers(aCmdBuff, 0, 3, buffers, offsets);
				vkCmdDraw(aCmdBuff, mesh.numberOfVertices, 1, mesh.vertexStartIndex, 0);
			}
		}

		// End the render pass
		vkCmdEndRenderPass(aCmdBuff);

		// End command recording
		if (auto const res = vkEndCommandBuffer(aCmdBuff); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to end recording buffer\n"
				"vkEndCommandBuffer() returned %s", lut::to_string(res).c_str());
		}
	}

	void submit_commands(lut::VulkanContext const& aContext, VkCommandBuffer aCmdBuff, VkFence aFence, VkSemaphore aWaitSemaphore, VkSemaphore aSignalSemaphore)
	{
		// This allows us to specify where exactly in the pipeline we must wait
		// In this case, we are waiting for mKeyA framebuffer to become available so
		//		we can write mKeyA new image to it. Hence, stages that take place
		//		before writing the image, such as the vertex shader, can still run.
		VkPipelineStageFlags waitPipelineStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

		// The above record_commands function leaves the command buffer in the executable state, but does not automatically submit it
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &aCmdBuff;

		// Extensions above exercise 2 to the info structure, describing the waits
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = &aWaitSemaphore;
		submitInfo.pWaitDstStageMask = &waitPipelineStages;

		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = &aSignalSemaphore;

		if (auto const res = vkQueueSubmit(aContext.graphicsQueue, 1, &submitInfo, aFence); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to submit command buffer to queue\n"
				"vkQueueSubmit() returned %s", lut::to_string(res).c_str());
		}
	}

	std::tuple<lut::Image, lut::ImageView> create_depth_buffer(lut::VulkanWindow const& aWindow, lut::Allocator const& aAllocator)
	{
		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.format = cfg::kDepthFormat;
		imageInfo.extent.width = aWindow.swapchainExtent.width;
		imageInfo.extent.height = aWindow.swapchainExtent.height;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

		VmaAllocationCreateInfo allocInfo{};
		allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

		VkImage image = VK_NULL_HANDLE;
		VmaAllocation allocation = VK_NULL_HANDLE;

		if (auto const res = vmaCreateImage(aAllocator.allocator, &imageInfo, &allocInfo, &image, &allocation, nullptr); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to allocate image.\n"
				"vmaCreateImage() returned %s", lut::to_string(res).c_str()
			);
		}

		lut::Image depthImage(aAllocator.allocator, image, allocation);

		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = depthImage.image;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = cfg::kDepthFormat;
		viewInfo.components = VkComponentMapping{}; // == identity
		//viewInfo.subresourceRange = range;
		viewInfo.subresourceRange = VkImageSubresourceRange{
			VK_IMAGE_ASPECT_DEPTH_BIT,
			0, 1,
			0, 1
		};

		// Finally, create the view
		VkImageView view = VK_NULL_HANDLE;
		if (auto const res = vkCreateImageView(aWindow.device, &viewInfo, nullptr, &view); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create image view\n"
				"vkCreateImageView() returned %s", lut::to_string(res).c_str());
		}

		return { std::move(depthImage), lut::ImageView(aWindow.device, view) };
	}

}

//EOF vim:syntax=cpp:foldmethod=marker:ts=4:noexpandtab: 
