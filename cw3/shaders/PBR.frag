#version 450

#define MAX_LIGHTS 7
#define PBR_PHONG 1
#define PBR_GGX 2

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

// PBR material
layout(set = 1, binding = 1) uniform UMaterial
{
	vec4 emissive;
	vec4 albedo;
	float shininess;
	float metalness;
} uMaterial;

layout(set = 2, binding = 2) uniform ULight
{
	PointLight lights[MAX_LIGHTS];
	vec4 ambientLight;
	int numLights;
	int reflectionModel;	// Possibly not best practice to introduce an if statement in the shader here.... 
							// ...doing so in interest of being able to easily visualise different PBR models!
} uLight;

#define M_PI 3.1415926535897932384626433832795
#define M_EPS 0.0000001

float blinnPhongNormalDist(float shininess, vec3 normalDirection, vec3 halfVector)
{
	return ((shininess + 2) / (2 * M_PI)) * pow(max(0, dot(normalDirection, halfVector)), shininess);
}

float ggxNormalDist(float scalingConst, float metalness, vec3 normalDirection, vec3 halfVector)
{
	float roughness = 1.0f - metalness;
	float cosThetaH = dot(normalDirection, halfVector);
	return scalingConst / pow(
	(pow(roughness, 2) * pow(cosThetaH, 2) + pow(sin(acos(cosThetaH)), 2))
	, 2);
}

vec3 burleyBRDF(vec3 lightDirection, vec3 viewDirection, vec3 normalDirection)
{
	// Obtain the half vector: this should be exactly halfway between the light and view vectors
	// ...hence we can add these two together and normalise to get the direction
	vec3 halfVector = normalize(viewDirection + lightDirection);

	// normal distribution
	float normalDistribution = 0.0f;
	if (uLight.reflectionModel == PBR_GGX)
	{
		normalDistribution = ggxNormalDist(0.1, uMaterial.metalness, normalDirection, halfVector);
	}
	else
	{
		normalDistribution = blinnPhongNormalDist(uMaterial.shininess, normalDirection, halfVector);
	}

	// Cook-Torrance masking term
	float masking = min(1, min(
		2 * max(0, dot(normalDirection, halfVector)) * max(0, dot(normalDirection, viewDirection)) / (dot(viewDirection, halfVector) + M_EPS),
		2 * max(0, dot(normalDirection, halfVector)) * max(0, dot(normalDirection, lightDirection)) / (dot(viewDirection, halfVector) + M_EPS)
	));
	
	// Fresnel by Schlick approximation
	vec3 specularBaseReflectivity = (1 - uMaterial.metalness) * vec3(0.04) + uMaterial.metalness * vec3(uMaterial.albedo.x, uMaterial.albedo.y, uMaterial.albedo.z);
	vec3 fresnel = specularBaseReflectivity + (vec3(1.0f) - specularBaseReflectivity) * pow((1 - dot(halfVector, viewDirection)), 5);

	// Calculate diffuse with simple lambertian
	vec3 diffuse = vec3(uMaterial.albedo.x, uMaterial.albedo.y, uMaterial.albedo.z) / M_PI * (vec3(1.0f) - fresnel) * (1 - uMaterial.metalness);
	
	return diffuse + normalDistribution * fresnel * masking / (4 * dot(normalDirection, viewDirection) * dot(normalDirection, lightDirection) + M_EPS);
}

void main()
{
	// Calculate directions
	vec3 viewDirection = normalize(uScene.cameraPosition - inV3fFragPosition);
	// Normalise the normal, just in case...
	vec3 normalDirection = normalize(inV3fNormal);
	
	// Convert to vec3 to avoid adding loads of ws
	vec3 lightEmitted = vec3(uMaterial.emissive.x, uMaterial.emissive.y, uMaterial.emissive.z);
	vec3 lightAmbient = vec3(0.002f);

	int numLights = min(uLight.numLights, MAX_LIGHTS);
	vec3 totalLight = vec3(0.0f);
	
	for (int i = 0; i < numLights; i++)
	{
		vec3 lightPosition = vec3(uLight.lights[i].lightPosition.x, uLight.lights[i].lightPosition.y, uLight.lights[i].lightPosition.z);
		vec3 lightDirection = normalize(lightPosition - inV3fFragPosition);

		vec3 brdf = burleyBRDF(lightDirection, viewDirection, normalDirection);

		vec3 lightColor = vec3(uLight.lights[i].lightColor.x, uLight.lights[i].lightColor.y, uLight.lights[i].lightColor.z);
		totalLight += brdf * lightColor * max(0, dot(normalDirection, lightDirection));
	}


	oColor = vec4(totalLight + lightEmitted + lightAmbient, 1.0f);
}