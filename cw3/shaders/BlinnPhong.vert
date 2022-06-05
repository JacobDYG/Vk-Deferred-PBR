#version 450

// Inputs
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexcoord;

layout(set = 0, binding = 0) uniform UScene
{
	mat4 camera;
	mat4 projection;
	mat4 projCam;
	vec3 cameraPosition;
} uScene;

// Outputs
layout(location = 0) out vec3 outV3fFragPosition;
layout(location = 1) out vec3 outV3fNormal;
layout(location = 2) out vec2 outV2fTexCoord;


void main()
{
	outV3fFragPosition = inPosition;
	outV3fNormal = inNormal;
	outV2fTexCoord = inTexcoord;

	gl_Position = uScene.projCam * vec4(inPosition, 1.0f);
}
