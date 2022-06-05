#include "model.hpp"

#include <utility>

#include <cstdio>
#include <cassert>

#include "../labutils/error.hpp"
namespace lut = labutils;

// ModelData
ModelData::ModelData() noexcept = default;

ModelData::ModelData( ModelData&& aOther ) noexcept
	: modelName( std::exchange( aOther.modelName, {} ) )
	, modelSourcePath( std::exchange( aOther.modelSourcePath, {} ) )
	, materials( std::move( aOther.materials ) )
	, meshes( std::move( aOther.meshes ) )
	, vertexPositions( std::move( aOther.vertexPositions ) )
	, vertexNormals( std::move( aOther.vertexNormals ) )
	, vertexTextureCoords( std::move( aOther.vertexTextureCoords ) )
{}

ModelData& ModelData::operator=( ModelData&& aOther ) noexcept
{
	std::swap( modelName, aOther.modelName );
	std::swap( modelSourcePath, aOther.modelSourcePath );
	std::swap( materials, aOther.materials );
	std::swap( meshes, aOther.meshes );
	std::swap( vertexPositions, aOther.vertexPositions );
	std::swap( vertexNormals, aOther.vertexNormals );
	std::swap( vertexTextureCoords, aOther.vertexTextureCoords );
	return *this;
}


// load_obj_model()
ModelData load_obj_model( std::string_view const& aOBJPath )
{
	// "Decode" path
	std::string fileName, directory;

	if( auto const separator = aOBJPath.find_last_of( "/\\" ); std::string_view::npos != separator )
	{
		fileName = aOBJPath.substr( separator+1 );
		directory = aOBJPath.substr( 0, separator+1 );
	}
	else
	{
		fileName = aOBJPath;
		directory = "./";
	}

	std::string const normalizedPath = directory + fileName;

	// Load model
	std::printf( "Loading: '%s' ...", normalizedPath.c_str() );
	std::fflush( stdout );

	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string err;

	if( !tinyobj::LoadObj( &attrib, &shapes, &materials, &err, normalizedPath.c_str(), directory.c_str(), true ) )
	{
		throw lut::Error( "Unable to load OBJ '%s':\n%s", normalizedPath.c_str(), err.c_str() );
	}

	// Apparently this can include some warnings:
	if( !err.empty() )
		std::printf( "\n%s\n... OK", err.c_str() );
	else
		std::printf( " OK\n" );

	// Transfer into our ModelData structures
	ModelData model;
	model.modelName        = aOBJPath;
	model.modelSourcePath  = normalizedPath;

	// ... copy over material data ...
	for( auto const& m : materials )
	{
		MaterialInfo info{};
		info.materialName      = m.name;

		info.color  = glm::vec3( m.diffuse[0], m.diffuse[1], m.diffuse[2] );

		info.emissive  = glm::vec3( m.emission[0], m.emission[1], m.emission[2] );
		info.diffuse  = glm::vec3( m.diffuse[0], m.diffuse[1], m.diffuse[2] );
		info.specular  = glm::vec3( m.specular[0], m.specular[1], m.specular[2] );
		info.shininess  = m.roughness;

		info.albedo  = glm::vec3( m.diffuse[0], m.diffuse[1], m.diffuse[2] );
		info.metalness  = m.metallic;

		if( !m.diffuse_texname.empty() ) info.mapDiffuse = directory + m.diffuse_texname;
		if( !m.specular_texname.empty() ) info.mapSpecular = directory + m.specular_texname;
		if( !m.alpha_texname.empty() ) info.mapAlpha = directory + m.alpha_texname;
		if( !m.normal_texname.empty() ) info.mapNormals = directory + m.normal_texname;

		model.materials.emplace_back( info );
	}

	// ... copy over mesh data ...
	// Note: this converts the mesh into a triangle soup. OBJ meshes use
	// separate indices for vertex positions, texture coordinates and normals.
	// This is not compatible with the default draw modes of OpenGL or Vulkan,
	// where each vertex has a single index that refers to all attributes.
	//
	// tinyobjloader additionally complicates the situation by specifying a
	// per-face material indices, which is rather impractical.
	//
	// In short- The OBJ format isn't exactly a great format (in a modern
	// context), and tinyobjloader is not making the situation a lot better.
	std::size_t totalVertices = 0;
	for( auto const& s : shapes )
	{
		totalVertices += s.mesh.indices.size();
	}

	model.vertexPositions.reserve( totalVertices );
	model.vertexNormals.reserve( totalVertices );
	model.vertexTextureCoords.reserve( totalVertices );

	std::size_t currentIndex = 0;
	for( auto const& s : shapes )
	{
		auto const& objMesh = s.mesh;

		if( objMesh.indices.empty() )
			continue;

		assert( !objMesh.material_ids.empty() );

		// The OBJ format allows for general polygons and not just triangles.
		// However, this code only deals with triangles. Check that we only
		// have triangles (debug mode only).
		assert( objMesh.indices.size() % 3 == 0 );
#		ifndef NDEBUG
		for( auto faceVerts : objMesh.num_face_vertices )
			assert( 3 == faceVerts );
#		endif // ~ NDEBUG

		// Each of our rendered objMeshes can only have a single material. 
		// Split the OBJ objMesh into multiple objMeshes if there are multiple 
		// materials. 
		//
		// Note: if a single material is repeated multiple times, this will
		// generate a new objMesh for each time the material is encountered.
		int currentMaterial = -1; // start a new material!

		std::size_t indices = 0;

		std::size_t face = 0, vert = 0;
		for( auto const& objIdx : objMesh.indices )
		{
			// check if material changed
			auto const matId = objMesh.material_ids[face];
			if( matId != currentMaterial )
			{
				if( indices )
				{
					assert( currentMaterial >= 0 ); 

					MeshInfo mesh{};
					mesh.materialIndex     = currentMaterial;
					mesh.meshName          = s.name + "::" + model.materials[currentMaterial].materialName;
					mesh.vertexStartIndex  = currentIndex;
					mesh.numberOfVertices  = indices;

					model.meshes.emplace_back( mesh );
				}

				currentIndex += indices;
				currentMaterial = matId;
				indices = 0;
			}

			// copy over data
			model.vertexPositions.emplace_back( glm::vec3(
				attrib.vertices[ objIdx.vertex_index * 3 + 0 ],
				attrib.vertices[ objIdx.vertex_index * 3 + 1 ],
				attrib.vertices[ objIdx.vertex_index * 3 + 2 ]
			) );

			assert( objIdx.normal_index >= 0 ); // must have a normal!
			model.vertexNormals.emplace_back( glm::vec3(
				attrib.normals[ objIdx.normal_index * 3 + 0 ],
				attrib.normals[ objIdx.normal_index * 3 + 1 ],
				attrib.normals[ objIdx.normal_index * 3 + 2 ]
			) );

			if( objIdx.texcoord_index >= 0 )
			{
				model.vertexTextureCoords.emplace_back( glm::vec2(
					attrib.texcoords[ objIdx.texcoord_index * 2 + 0 ],
					attrib.texcoords[ objIdx.texcoord_index * 2 + 1 ]
				) );
			}
			else
			{
				model.vertexTextureCoords.emplace_back( glm::vec2( 0.f, 0.f ) );
			}

			++indices;

			// accounting: next vertex
			++vert;
			if( 3 == vert )
			{
				++face;
				vert = 0;
			}
		}

		if( indices )
		{
			assert( currentMaterial >= 0 ); 

			MeshInfo mesh{};
			mesh.materialIndex     = currentMaterial;
			mesh.meshName          = s.name + "::" + model.materials[currentMaterial].materialName;
			mesh.vertexStartIndex  = currentIndex;
			mesh.numberOfVertices  = indices;

			currentIndex += indices;

			model.meshes.emplace_back( mesh );
		}
	}

	assert( model.vertexPositions.size() == totalVertices );
	assert( model.vertexNormals.size() == totalVertices );
	assert( model.vertexTextureCoords.size() == totalVertices );
	
	return model;
}
