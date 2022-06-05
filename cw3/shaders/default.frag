#version 450

#define MAX_LIGHTS 7

struct PointLight {
	vec4 lightPosition;
	vec4 lightColor;
};

layout(location = 0) in vec3 inV3fFragPosition; 
layout(location = 1) in vec3 inV3fNormal;
layout(location = 2) in vec2 inV2fTexCoord;

layout(location = 0) out vec4 oColor;

layout(set = 0, binding = 0) uniform UScene
{
	mat4 camera;
	mat4 projection;
	mat4 projCam;
	vec3 cameraPosition;
} uScene;

layout(set = 1, binding = 1) uniform UMaterial

{
	vec4 color;
} uMaterial;

layout(set = 2, binding = 2) uniform ULight
{
	PointLight lights[MAX_LIGHTS];
	vec4 ambientLight;
	int numLights;
} uLight;

void main()
{
	// Basic color ala CW1
	//oColor = vec4(uMaterial.color, 1.0f);

	// Uncomment to visualise normal
	// vec3 debugNorm = inV3fNormal;						// Negative normals will be displayed as black, and positive normals will be solid colors
	// vec3 debugNorm = (inV3fNormal + vec3(1)) / 2;		// Normals will be remapped to the range [0...1]
	// oColor = vec4(debugNorm, 1.0f);

	// Uncomment to visualise view direction
	// oColor = vec4(normalize(uScene.cameraPosition - inV3fFragPosition), 1.0f);

	// Uncomment to visualise light direction
	int numLights = uLight.numLights;

	vec4 totalLight = vec4(0.0f);
	for (int i = 0; i < numLights; i++)
	{
		totalLight += normalize(uLight.lights[i].lightPosition - vec4(inV3fFragPosition, 1.0f)) / numLights;
	}

	oColor = totalLight;
}
