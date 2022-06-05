#pragma once

#include <volk/volk.h>

#include "vkobject.hpp"
#include "vulkan_context.hpp"

namespace labutils
{
	ShaderModule load_shader_module( VulkanContext const&, char const* aSpirvPath );

	CommandPool create_command_pool( VulkanContext const&, VkCommandPoolCreateFlags = 0 );
	VkCommandBuffer alloc_command_buffer( VulkanContext const&, VkCommandPool );

	Fence create_fence( VulkanContext const&, VkFenceCreateFlags = 0 );
	Semaphore create_semaphore( VulkanContext const& );

	// Descriptor pool requirements. Declared here as we are using generic descriptor pools.
	// Descriptor sets must be allocated from a pool.
	DescriptorPool create_descriptor_pool(VulkanContext const&, std::uint32_t aMaxDescriptors = 2048, std::uint32_t aMaxSets = 1024);
	VkDescriptorSet alloc_desc_set(VulkanContext const&, VkDescriptorPool, VkDescriptorSetLayout);

	// We will need to create many barriers for buffers
	// These will provide synchronisation, for example where a host-visible buffer is being
	// ..copied into a device-only buffer, a barrier will be used to ensure this copy is complete.
	void buffer_barrier(
		VkCommandBuffer,
		VkBuffer,
		VkAccessFlags aSrcAccessMask,
		VkAccessFlags aDstAccessMask,
		VkPipelineStageFlags aSrcStageMask,
		VkPipelineStageFlags aDstStageMask,
		VkDeviceSize aSize = VK_WHOLE_SIZE,
		VkDeviceSize aOffset = 0,
		uint32_t aSrcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		uint32_t aDstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED
	);

	// Images need barriers as well
	// These barriers will be used for recording into a command buffer
	// ...as the synchronisation needs to happen on device
	void image_barrier(
		VkCommandBuffer,
		VkImage,
		VkAccessFlags aSrcAccessMask,
		VkAccessFlags aDstAccessMask,
		VkImageLayout aSrcLayout,
		VkImageLayout aDstLayout,
		VkPipelineStageFlags aSrcStageMask,
		VkPipelineStageFlags aDstStageMask,
		VkImageSubresourceRange = VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
		std::uint32_t aSrcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		std::uint32_t aDstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED
	);

	Sampler create_default_sampler(VulkanContext const& aContext);
}
