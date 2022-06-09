#version 450

layout (binding = 0) uniform sampler2D passthru;

layout (location = 0) in vec2 inUV;

layout (location = 0) out vec4 outFragcolor;

void main()
{
	outFragcolor = texture(passthru, inUV);
}