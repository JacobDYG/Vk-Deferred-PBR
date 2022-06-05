#pragma once

// The model.hpp and model.cpp are based on the Model.{hpp,cpp} from 
//  https://gitlab.com/chalmerscg/tda362-labs-2020
// and are used with permission.

#include <string>
#include <vector>
#include <string_view>

#include <cstdint>

#include <glm/glm.hpp>
#include <tiny_obj_loader.h>

/* The structures here are intended to be used during loading only. At runtime,
 * you probably want to use a different set of structures that instead hold e.g.
 * references to the Vulkan resources in which a subset of the data resides. At
 * that point, there is no reason to keep CPU copies of the data.
 *
 * Whether you want to keep meta-data (such as mesh/material names) is up to
 * you. They can -occasionally- be useful when debugging.
 *
 * If you wish to organize your model data differently, you may do so.
 */
struct MaterialInfo
{
	/* Combined material for CW3.
	 */

	std::string materialName;

	// For CW1 compatibility:
	glm::vec3 color;

	// For Blinn-Phong:
	glm::vec3 emissive;
	glm::vec3 diffuse;
	glm::vec3 specular;
	float shininess;

	// For PBR material
	//glm::vec3 emissive; // See above
	glm::vec3 albedo;
	//float shininess; // See above
	float metalness;

	// If you want to play around with the Sponza model (optional):
	// NOTE: Implementing normal mapping will require computing tangents and/or
	// binormals, such that you can transform to/from tangent space. You can
	// just stick to a subset of these, though.
	std::string mapDiffuse;
	std::string mapSpecular;
	std::string mapAlpha;
	std::string mapNormals; // In some models, this is a bump map instead.
};

struct MeshInfo
{
	std::string meshName;

	// The materialIndex is a index into the ModelData::materials vector, 
	// specifying the corresponding MaterialInfo in it.
	std::uint32_t materialIndex;

	// The mesh consists of numberOfVertices vertices, starting at the index
	// vertexStartIndex. The actual vertex data is then found in the 
	// vertexPositions, vertexNormals and vertexTextureCoords members of
	// ModelData.
	std::size_t vertexStartIndex;
	std::size_t numberOfVertices;
};


struct ModelData
{
	ModelData() noexcept;

	ModelData( ModelData const& ) = delete;
	ModelData& operator= (ModelData const&) = delete;

	ModelData( ModelData&& ) noexcept;
	ModelData& operator= (ModelData&&) noexcept;


	std::string modelName;
	std::string modelSourcePath;

	std::vector<MaterialInfo> materials;
	std::vector<MeshInfo> meshes;

	std::vector<glm::vec3> vertexPositions;
	std::vector<glm::vec3> vertexNormals;
	std::vector<glm::vec2> vertexTextureCoords;
};

ModelData load_obj_model( std::string_view const& aOBJPath );
