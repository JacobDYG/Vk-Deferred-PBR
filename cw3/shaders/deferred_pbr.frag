#version 450

layout (binding = 0) uniform sampler2D position;
layout (binding = 1) uniform sampler2D normal;
layout (binding = 2) uniform sampler2D albedo_specular;
layout (binding = 3) uniform sampler2D emissive_metalness;

layout (location = 0) in vec2 inUV;

layout (location = 0) out vec4 outFragColor;

void main()
{
	// Visualise position
	//outFragColor = texture(position, inUV);
	// Visualise normal
	outFragColor = texture(normal, inUV);
	// Visualise albedo
	//outFragColor = vec4(vec3(texture(albedo_specular, inUV)), 1.0f);
	// Visualise emissive
	//outFragColor = vec4(vec3(texture(emissive_metalness, inUV)), 1.0f);
}