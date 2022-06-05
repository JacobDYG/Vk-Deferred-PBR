#include "vkimage.hpp"

#include <vector>
#include <utility>
#include <algorithm>

#include <cstdio>
#include <cassert>
#include <cstring> // for std::memcpy()

#include <stb_image.h>

#include "error.hpp"
#include "vkutil.hpp"
#include "vkbuffer.hpp"
#include "to_string.hpp"

namespace
{
	// Unfortunately, std::countl_zero() isn't available in C++17; it was added
	// in C++20. This provides a fallback implementation. Unlike C++20, this
	// returns a std::uint32_t and not a signed int.
	//
	// See https://graphics.stanford.edu/~seander/bithacks.html for this and
	// other methods like it.
	//
	// Note: that this is unlikely to be the most efficient implementation on
	// most processors. Many instruction sets have dedicated instructions for
	// this operation. E.g., lzcnt (x86 ABM/BMI), bsr (x86).
	inline 
	std::uint32_t countl_zero_( std::uint32_t aX )
	{
		if( !aX ) return 32;

		uint32_t res = 0;

		if( !(aX & 0xffff0000) ) (res += 16), (aX <<= 16);
		if( !(aX & 0xff000000) ) (res +=  8), (aX <<=  8);
		if( !(aX & 0xf0000000) ) (res +=  4), (aX <<=  4);
		if( !(aX & 0xc0000000) ) (res +=  2), (aX <<=  2);
		if( !(aX & 0x80000000) ) (res +=  1);

		return res;
	}
}

namespace labutils
{
	Image::Image() noexcept = default;

	Image::~Image()
	{
		if( VK_NULL_HANDLE != image )
		{
			assert( VK_NULL_HANDLE != mAllocator );
			assert( VK_NULL_HANDLE != allocation );
			vmaDestroyImage( mAllocator, image, allocation );
		}
	}

	Image::Image( VmaAllocator aAllocator, VkImage aImage, VmaAllocation aAllocation ) noexcept
		: image( aImage )
		, allocation( aAllocation )
		, mAllocator( aAllocator )
	{}

	Image::Image( Image&& aOther ) noexcept
		: image( std::exchange( aOther.image, VK_NULL_HANDLE ) )
		, allocation( std::exchange( aOther.allocation, VK_NULL_HANDLE ) )
		, mAllocator( std::exchange( aOther.mAllocator, VK_NULL_HANDLE ) )
	{}
	Image& Image::operator=( Image&& aOther ) noexcept
	{
		std::swap( image, aOther.image );
		std::swap( allocation, aOther.allocation );
		std::swap( mAllocator, aOther.mAllocator );
		return *this;
	}
}

namespace labutils
{
	Image load_image_texture2d(char const* aPattern, VulkanContext const& aContext, VkCommandPool aCmdPool, Allocator const& aAllocator)
	{
		// Determine base image size
		int baseWidthi, baseHeighti, baseChannelsi;
		if (1 != stbi_info(aPattern, &baseWidthi, &baseHeighti, &baseChannelsi))
		{
			throw Error("%s: unable to get image information (%s)", aPattern, stbi_failure_reason());
		}

		assert(baseWidthi > 0 && baseHeighti > 0);

		auto const baseWidth = std::uint32_t(baseWidthi);
		auto const baseHeight = std::uint32_t(baseHeighti);

		// Image creation info
		VkFormat imgFormat = VK_FORMAT_R8G8B8A8_SRGB;
		auto const mipLevels = compute_mip_level_count(baseWidth, baseHeight);

		// Check blit is available
		VkFormatProperties formatProperties;
		vkGetPhysicalDeviceFormatProperties(aContext.physicalDevice, imgFormat, &formatProperties);

		if (!(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT))
		{
			throw Error("Image format does not support blitting!\n");
		}

		Image ret = create_image_texture2d(aAllocator, baseWidth, baseHeight,
			imgFormat, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

		// Create command buffer to copy image into buff
		VkCommandBuffer cbuff = alloc_command_buffer(aContext, aCmdPool);

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = 0;
		beginInfo.pInheritanceInfo = nullptr;

		if (auto const res = vkBeginCommandBuffer(cbuff, &beginInfo); VK_SUCCESS != res)
		{
			throw Error("Beginning command buffer recording\n"
				"vkBeginCommandBuffer() returned %s", to_string(res).c_str()
			);
		}

		// Load the base image
		int widthi, heighti, channelsi;
		stbi_uc* data = stbi_load(aPattern, &widthi, &heighti, &channelsi, 4 /*4 to get RGBA channels*/);

		assert(widthi > 0 && widthi == baseWidth);
		assert(heighti > 0 && heighti == baseHeight);

		auto const sizeInBytes = baseWidth * baseHeight * 4;

		// Create staging buffer and copy image data to it
		auto& staging = create_buffer(aAllocator, sizeInBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

		void* sptr = nullptr;
		if (auto const res = vmaMapMemory(aAllocator.allocator, staging.allocation, &sptr); VK_SUCCESS != res)
		{
			throw Error("Mapping memory for writing\n"
				"vmaMapMemory() returned %s", to_string(res).c_str()
			);
		}

		std::memcpy(sptr, data, sizeInBytes);
		vmaUnmapMemory(aAllocator.allocator, staging.allocation);

		// Free STB image data
		stbi_image_free(data);

		// Upload data from staging buffer to image
		VkBufferImageCopy copy;
		copy.bufferOffset = 0;
		copy.bufferRowLength = 0;
		copy.bufferImageHeight = 0;
		copy.imageSubresource = VkImageSubresourceLayers{
			VK_IMAGE_ASPECT_COLOR_BIT,
			0,
			0, 1
		};
		copy.imageOffset = VkOffset3D{ 0, 0, 0 };
		copy.imageExtent = VkExtent3D{ baseWidth, baseHeight, 1 };

		image_barrier(cbuff, ret.image,
			0,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_IMAGE_LAYOUT_UNDEFINED,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VkImageSubresourceRange {
				VK_IMAGE_ASPECT_COLOR_BIT,
				0, mipLevels,
				0, 1
			}
		);

		// Record the command
		vkCmdCopyBufferToImage(cbuff, staging.buffer, ret.image,	// The cmd that performs the copy. This struct accepts multiple
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);		// ...image copy structures at once; we could have prepared
																	// ...the info and issued a single copy command, if we wanted.

		// Upload mip levels
		int mipWidth = baseWidth, mipHeight = baseHeight;
		if (mipWidth < 1)
			mipWidth = 1;
		if (mipHeight < 1)
			mipHeight = 1;

		// Starting from the second mip level, the first level is the original
		for (std::uint32_t i = 1; i < mipLevels; i++)
		{
			image_barrier(cbuff, ret.image,
				VK_ACCESS_TRANSFER_WRITE_BIT,
				VK_ACCESS_TRANSFER_READ_BIT,
				VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VkImageSubresourceRange{
					VK_IMAGE_ASPECT_COLOR_BIT,
					i - 1, 1,
					0, 1
				}
			);

			// Describe how to generate mips
			VkImageBlit blit{};
			blit.srcOffsets[0] = { 0,0,0 };
			blit.srcOffsets[1] = { mipWidth, mipHeight, 1 };	// range of image from source
			blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			blit.srcSubresource.mipLevel = i - 1;
			blit.srcSubresource.baseArrayLayer = 0;
			blit.srcSubresource.layerCount = 1;
			blit.dstOffsets[0] = { 0,0,0 };
			blit.dstOffsets[1] = { mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1 };
			blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			blit.dstSubresource.mipLevel = i;
			blit.dstSubresource.baseArrayLayer = 0;
			blit.dstSubresource.layerCount = 1;

			// Submit the command that generates mips
			vkCmdBlitImage(cbuff,
				ret.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				ret.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				1, &blit,
				VK_FILTER_LINEAR
			);

			image_barrier(cbuff, ret.image,
				VK_ACCESS_TRANSFER_READ_BIT,
				VK_ACCESS_SHADER_READ_BIT,
				VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
				VkImageSubresourceRange{
					VK_IMAGE_ASPECT_COLOR_BIT,
					i - 1, 1,
					0, 1
				}
			);

			// Next mip level
			mipWidth >>= 1;
			mipHeight >>= 1;
			if (mipWidth < 1)
				mipWidth = 1;
			if (mipHeight < 1)
				mipHeight = 1;
		}

		image_barrier(cbuff, ret.image,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_SHADER_READ_BIT,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			VkImageSubresourceRange{
				VK_IMAGE_ASPECT_COLOR_BIT,
				mipLevels - 1, 1,
				0, 1
			}
		);

		// The neccasary commands are recorded, so submit the cbuffer for execution, and wait for the commands to complete before returning
		if (auto const res = vkEndCommandBuffer(cbuff); VK_SUCCESS != res)
		{
			throw Error("Ending command buffer recording\n"
				"vkEndCommandBuffer() returned %s", to_string(res).c_str()
			);
		}

		// Submit the buffer and wait for the commands to complete
		Fence uploadComplete = create_fence(aContext);

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &cbuff;

		if (auto const res = vkQueueSubmit(aContext.graphicsQueue, 1, &submitInfo, uploadComplete.handle); VK_SUCCESS != res)
		{
			throw Error("Submitting commands\n"
				"vkQueueSubmit() returned %s", to_string(res).c_str()
			);
		}

		if (auto const res = vkWaitForFences(aContext.device, 1, &uploadComplete.handle, VK_TRUE, std::numeric_limits<std::uint64_t>::max()); VK_SUCCESS != res)
		{
			throw Error("Waiting for upload to complete\n"
				"vkWaitForFences() returned %s", to_string(res).c_str()
			);
		}

		// Return the resulting image
		// Most resources are automatically cleaned up, but not the command buffer
		// ... so we do it here.
		vkFreeCommandBuffers(aContext.device, aCmdPool, 1, &cbuff);

		return ret;
	}

	Image create_image_texture2d(Allocator const& aAllocator, std::uint32_t aWidth, std::uint32_t aHeight, VkFormat aFormat, VkImageUsageFlags aUsage)
	{
		auto const mipLevels = compute_mip_level_count(aWidth, aHeight);
		//auto const mipLevels = 1;

		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.format = aFormat;
		imageInfo.extent.width = aWidth;
		imageInfo.extent.height = aHeight;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = mipLevels;
		imageInfo.arrayLayers = 1;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.usage = aUsage;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

		VmaAllocationCreateInfo allocInfo{};
		allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

		VkImage image = VK_NULL_HANDLE;
		VmaAllocation allocation = VK_NULL_HANDLE;

		if (auto const res = vmaCreateImage(aAllocator.allocator, &imageInfo, &allocInfo, &image, &allocation, nullptr); VK_SUCCESS != res)
		{
			throw Error("Unable to allocate image.\n"
				"vmaCreateImage() returned %s", to_string(res).c_str()
			);
		}

		return Image(aAllocator.allocator, image, allocation);
	}

	ImageView create_image_view_texture2d(VulkanContext const& aContext, VkImage aImage, VkFormat aFormat)
	{
		/*VkImageSubresourceRange range;
		range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		range.baseMipLevel = 0;
		range.levelCount = VK_REMAINING_MIP_LEVELS;
		range.baseArrayLayer = 0;
		range.layerCount = 1;*/

		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = aImage;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = aFormat;
		viewInfo.components = VkComponentMapping{}; // == identity
		//viewInfo.subresourceRange = range;
		viewInfo.subresourceRange = VkImageSubresourceRange{
			VK_IMAGE_ASPECT_COLOR_BIT,
			0, VK_REMAINING_MIP_LEVELS,
			0, 1
		};

		// Finally, create the view
		VkImageView view = VK_NULL_HANDLE;
		if (auto const res = vkCreateImageView(aContext.device, &viewInfo, nullptr, &view); VK_SUCCESS != res)
		{
			throw Error("Unable to create image view\n"
				"vkCreateImageView() returned %s", to_string(res).c_str());
		}

		return ImageView(aContext.device, view);
	}

	std::uint32_t compute_mip_level_count( std::uint32_t aWidth, std::uint32_t aHeight )
	{
		std::uint32_t const bits = aWidth | aHeight;
		std::uint32_t const leadingZeros = countl_zero_( bits );
		return 32-leadingZeros;
	}
}
