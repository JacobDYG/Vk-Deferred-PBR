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
	vec4 emissive;
	vec4 diffuse;
	vec4 specular;
	float shininess; // Last to make std140 alignment easier.
} uMaterial;

layout(set = 2, binding = 2) uniform ULight
{
	PointLight lights[MAX_LIGHTS];
	vec4 ambientLight;
	int numLights;
} uLight;

#define M_PI 3.1415926535897932384626433832795

void main()
{
	// Basic color ala CW1
	//oColor = vec4(uMaterial.color, 1.0f);
	// Diffuse color only ala CW1
	// oColor = uMaterial.diffuse;

	// Uncomment to visualise normal
	// vec3 debugNorm = inV2fNormal;						// Negative normals will be displayed as black, and positive normals will be solid colors
	// vec3 debugNorm = (inV2fNormal + vec3(1)) / 2;		// Normals will be remapped to the range [0...1]
	// oColor = vec4(debugNorm, 1.0f);

	// Uncomment to visualise view direction
	// oColor = vec4(normalize(uScene.cameraPosition - inV3fFragPosition), 1.0f);

	// Uncomment to visualise light direction
	// oColor = vec4(normalize(uLight.light.lightPosition - inV3fFragPosition), 1.0f);
	int numLights = min(uLight.numLights, MAX_LIGHTS);
	
	vec3 viewDirection = normalize(uScene.cameraPosition - inV3fFragPosition);
	vec4 ambientPart = uMaterial.emissive + uLight.ambientLight * uMaterial.diffuse;
	
	vec4 lightPart = vec4(0.0f);

	for (int i = 0; i < numLights; ++i)
	{
		vec3 lightDirection = normalize(vec3(uLight.lights[i].lightPosition.x, uLight.lights[i].lightPosition.y, uLight.lights[i].lightPosition.z) - inV3fFragPosition);
		// Obtain the half vector: this should be exactly halfway between the light and view vectors
		// ...hence we can add these two together and normalise to get the direction
		vec3 halfVector = normalize(viewDirection + lightDirection);


		vec4 specularTerm = (uMaterial.diffuse / M_PI + ((uMaterial.shininess + 2) / 8 * pow(max(0, dot(normalize(inV3fNormal), halfVector)), uMaterial.shininess) * uMaterial.specular));
		lightPart +=	specularTerm
						* max(0, dot(normalize(inV3fNormal), lightDirection))
						* uLight.lights[i].lightColor;
	}


	oColor = ambientPart + lightPart;
	
	//oColor = vec4(vec3(max(0, dot(normalize(inV3fNormal), normalize(lightDirection)))), 1.0f);	// norm dot lightvec
	//oColor = vec4(vec3(max(0, dot(normalize(inV3fNormal), normalize(halfVector)))), 1.0f);		// norm dot halfvec
	//oColor = vec4(vec3((uMaterial.shininess + 2) / 8 * pow(max(0, dot(inV3fNormal, halfVector)), uMaterial.shininess)), 1.0f);	// Specular component (without color)
}