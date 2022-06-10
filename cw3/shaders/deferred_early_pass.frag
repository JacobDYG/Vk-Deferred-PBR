#version 450

layout(location = 0) in vec3 inV3fFragPosition; 
layout(location = 1) in vec3 inV3fNormal;
layout(location = 2) in vec2 inV2fTexCoord;

// PBR material
layout(set = 1, binding = 1) uniform UMaterial
{
	vec4 emissive;
	vec4 albedo;
	float shininess;
	float metalness;
} uMaterial;

layout (location = 0) out vec4 outPosition;
layout (location = 1) out vec4 outNormal;
layout (location = 2) out vec4 outAlbedoShininess;
layout (location = 3) out vec4 outEmissiveMetalness;

void main()
{
	outPosition = vec4(inV3fFragPosition, 1.0f);
	outNormal = vec4(inV3fNormal, 1.0f);
	outAlbedoShininess = vec4(uMaterial.albedo.x, uMaterial.albedo.y, uMaterial.albedo.z, uMaterial.shininess);
	outEmissiveMetalness = vec4(uMaterial.emissive.x, uMaterial.emissive.y, uMaterial.emissive.z, uMaterial.metalness);
}