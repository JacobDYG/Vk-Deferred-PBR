#include "vkutil.hpp"

#include <vector>

#include <fstream>
#include <cassert>

#include "error.hpp"
#include "to_string.hpp"

namespace labutils
{
	ShaderModule load_shader_module(VulkanContext const& aContext, char const* aSpirvPath)
	{
		assert(aSpirvPath);

		// Open the file for reading in binary mode, and start at the end of the file (ios::ate)
		std::ifstream inFile(aSpirvPath, std::ios::binary | std::ios::ate);

		// Check the stream is good
		if (!inFile)
		{
			throw Error("Cannot open '%s' for reading", aSpirvPath);
		}

		// Get file size in bytes
		size_t const bytes = (size_t)inFile.tellg();
		// Seek to start
		inFile.seekg(0, inFile.beg);

		// SPIR-V consists of 4 byte words, so check that the number of bytes is a multiple of 4
		assert(0 == bytes % 4);

		// Copy the file into a vector
		std::vector<char> code((std::istreambuf_iterator<char>(inFile)), (std::istreambuf_iterator<char>()));

		// Error checking
		if (inFile.fail() || bytes != inFile.tellg())
		{
			inFile.close();

			throw Error("Error reading '%s': Error: '%s', EOF: '%d'", aSpirvPath, strerror(errno), inFile.eof());
		}

		inFile.close();

		// Create the shader module
		VkShaderModuleCreateInfo moduleInfo{};
		moduleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		moduleInfo.codeSize = bytes;
		moduleInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

		VkShaderModule shaderModule = VK_NULL_HANDLE;
		if (auto const res = vkCreateShaderModule(aContext.device, &moduleInfo, nullptr, &shaderModule);
			VK_SUCCESS != res)
		{
			throw Error("Unable to create shader module from %s\n"
				"vkCreateShaderModule() returned %s", aSpirvPath, to_string(res).c_str());
		}

		return ShaderModule(aContext.device, shaderModule);
	}


	CommandPool create_command_pool(VulkanContext const& aContext, VkCommandPoolCreateFlags aFlags)
	{
		// Create a command pool to be able to send commands
		// Commands are not sent one by one, but as groups.
		//		this is advantageous because the implementation can optimise groups of commands
		//		it also makes it possible to submit commands from multiple threads
		VkCommandPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		poolInfo.queueFamilyIndex = aContext.graphicsFamilyIndex;
		poolInfo.flags = aFlags;

		VkCommandPool commandPool = VK_NULL_HANDLE;
		if (auto const res = vkCreateCommandPool(aContext.device, &poolInfo, nullptr, &commandPool); VK_SUCCESS != res)
		{
			throw Error("Unable to create command pool\n"
				"vkCreateCommandPool() returned %s", to_string(res).c_str());
		}

		return CommandPool(aContext.device, commandPool);
	}

	VkCommandBuffer alloc_command_buffer(VulkanContext const& aContext, VkCommandPool aCmdPool)
	{
		VkCommandBufferAllocateInfo commandBufferInfo{};
		commandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		commandBufferInfo.commandPool = aCmdPool;
		commandBufferInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		commandBufferInfo.commandBufferCount = 1;

		VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
		if (auto const res = vkAllocateCommandBuffers(aContext.device, &commandBufferInfo, &commandBuffer); VK_SUCCESS != res)
		{
			throw Error("Unable to allocate command buffer\n"
				"vkAllocateCommandBuffers() returned %s", to_string(res).c_str());
		}

		// Note that the command buffer is not wrapped at all
		// This is because we are not directly responsible for their destruction
		//		they are automatically destroyed when their parent command pool is destroyed
		return commandBuffer;
	}


	Fence create_fence(VulkanContext const& aContext, VkFenceCreateFlags aFlags)
	{
		// A fence is similar to a semaphore (for synchronisation), except it is for the CPU, not GPU.
		//		it is used to signal to the CPU that the GPU is done with some work.
		VkFenceCreateInfo fenceInfo{};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceInfo.flags = aFlags;

		VkFence fence = VK_NULL_HANDLE;
		if (auto const res = vkCreateFence(aContext.device, &fenceInfo, nullptr, &fence); VK_SUCCESS != res)
		{
			throw Error("Unable to create fence\n"
				"vkCreateFence() returned %s", to_string(res).c_str());
		}

		return Fence(aContext.device, fence);
	}

	Semaphore create_semaphore(VulkanContext const& aContext)
	{
		VkSemaphoreCreateInfo semaphoreInfo{};
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		VkSemaphore semaphore = VK_NULL_HANDLE;
		if (auto const res = vkCreateSemaphore(aContext.device, &semaphoreInfo, nullptr, &semaphore);
			VK_SUCCESS != res)
		{
			throw Error("Unable to create sempahore\n"
				"vkCreateSemaphore() returned %s", to_string(res).c_str()
			);
		}

		return Semaphore(aContext.device, semaphore);
	}

	DescriptorPool create_descriptor_pool(VulkanContext const& aContext, std::uint32_t aMaxDescriptors, std::uint32_t aMaxSets)
	{
		VkDescriptorPoolSize const pools[] = {
			{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, aMaxDescriptors},
			{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, aMaxDescriptors}
		};

		VkDescriptorPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.maxSets = aMaxSets;
		poolInfo.poolSizeCount = sizeof(pools) / sizeof(pools[0]);
		poolInfo.pPoolSizes = pools;

		VkDescriptorPool pool = VK_NULL_HANDLE;
		if (auto const res = vkCreateDescriptorPool(aContext.device, &poolInfo, nullptr, &pool); VK_SUCCESS != res)
		{
			throw Error("Unable to create descriptor pool\n"
				"vkCreateDescriptorPool() returned %s", to_string(res).c_str()
			);
		}

		return DescriptorPool(aContext.device, pool);
	}

	VkDescriptorSet alloc_desc_set(VulkanContext const& aContext, VkDescriptorPool aPool, VkDescriptorSetLayout aSetLayout)
	{
		// Note that Vulkan is capable of allocating many descriptor sets at once (as shown by the plural in vkAllocateDescriptorSets).
		// This function allocates descriptor sets one by one; this may not be the best approach generally, but we use it here as we are using so few sets.
		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = aPool;
		allocInfo.descriptorSetCount = 1;
		allocInfo.pSetLayouts = &aSetLayout;

		VkDescriptorSet descSet = VK_NULL_HANDLE;
		if (auto const res = vkAllocateDescriptorSets(aContext.device, &allocInfo, &descSet); VK_SUCCESS != res)
		{
			throw Error("Unable to allocate descriptor set\n"
				"vkAllocateDescriptorSets() returned %s", to_string(res).c_str()
			);
		}

		return descSet;
	}

	void buffer_barrier(VkCommandBuffer aCmdBuff, VkBuffer aBuffer, VkAccessFlags aSrcAccessMask,
		VkAccessFlags aDstAccessMask, VkPipelineStageFlags aSrcStageMask, VkPipelineStageFlags aDstStageMask,
		VkDeviceSize aSize, VkDeviceSize aOffset, uint32_t aSrcQueueFamilyIndex, uint32_t aDstQueueFamilyIndex)
	{
		VkBufferMemoryBarrier bufferBarrier{};
		bufferBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
		bufferBarrier.srcAccessMask = aSrcAccessMask;
		bufferBarrier.dstAccessMask = aDstAccessMask;
		bufferBarrier.buffer = aBuffer;
		bufferBarrier.size = aSize;
		bufferBarrier.offset = aOffset;
		bufferBarrier.srcQueueFamilyIndex = aSrcQueueFamilyIndex;
		bufferBarrier.dstQueueFamilyIndex = aDstQueueFamilyIndex;

		vkCmdPipelineBarrier(
			aCmdBuff,
			aSrcStageMask, aDstStageMask,
			0,
			0, nullptr,
			1, &bufferBarrier,
			0, nullptr
		);
	}
	void image_barrier(VkCommandBuffer aCmdBuff, VkImage aImage, VkAccessFlags aSrcAccessMask, VkAccessFlags aDstAccessMask,
		VkImageLayout aSrcLayout, VkImageLayout aDstLayout, VkPipelineStageFlags aSrcStageMask, VkPipelineStageFlags aDstStageMask,
		VkImageSubresourceRange aRange, std::uint32_t aSrcQueueFamilyIndex, std::uint32_t aDstQueueFamilyIndex)
	{
		VkImageMemoryBarrier iBarrier{};
		iBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		iBarrier.image = aImage;
		iBarrier.srcAccessMask = aSrcAccessMask;
		iBarrier.dstAccessMask = aDstAccessMask;
		iBarrier.srcQueueFamilyIndex = aSrcQueueFamilyIndex;
		iBarrier.dstQueueFamilyIndex = aDstQueueFamilyIndex;
		iBarrier.oldLayout = aSrcLayout;
		iBarrier.newLayout = aDstLayout;
		iBarrier.subresourceRange = aRange;

		vkCmdPipelineBarrier(aCmdBuff, aSrcStageMask, aDstStageMask, 0, 0, nullptr, 0, nullptr, 1, &iBarrier);
	}


	Sampler create_default_sampler(VulkanContext const& aContext)
	{
		// Check anisotropy is available
		VkPhysicalDeviceFeatures supportedFeatures;
		vkGetPhysicalDeviceFeatures(aContext.physicalDevice, &supportedFeatures);
		float maxAnisotropy = 1.0f;
		auto enableAnisotropy = VK_FALSE;
		if (supportedFeatures.samplerAnisotropy)
		{
			enableAnisotropy = VK_TRUE;
			// Find supported anisotropy
			VkPhysicalDeviceProperties props{};
			vkGetPhysicalDeviceProperties(aContext.physicalDevice, &props);
			maxAnisotropy = props.limits.maxSamplerAnisotropy;
		}

		VkSamplerCreateInfo samplerinfo{};
		samplerinfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerinfo.magFilter = VK_FILTER_LINEAR;
		samplerinfo.minFilter = VK_FILTER_LINEAR;
		samplerinfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		samplerinfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerinfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerinfo.minLod = 0.0f;
		samplerinfo.maxLod = VK_LOD_CLAMP_NONE;
		samplerinfo.mipLodBias = 0;
		// Anisotropic filtering
		samplerinfo.anisotropyEnable = enableAnisotropy;
		samplerinfo.maxAnisotropy = maxAnisotropy;

		VkSampler sampler = VK_NULL_HANDLE;

		if (auto const res = vkCreateSampler(aContext.device, &samplerinfo, nullptr, &sampler); VK_SUCCESS != res)
		{
			throw Error("Unable to create sampler\n"
				"vkCreateSampler() returned %s", to_string(res).c_str()
			);
		}

		return Sampler(aContext.device, sampler);
	}
}
