#version 450

#define MAX_LIGHTS 7
#define PBR_PHONG 1
#define PBR_GGX 2

struct PointLight {
	vec4 lightPosition;
	vec4 lightColor;
};

layout(set = 0, binding = 0) uniform UScene
{
	mat4 camera;
	mat4 projection;
	mat4 projCam;
	vec3 cameraPosition;
} uScene;

layout(set = 1, binding = 1) uniform ULight
{
	PointLight lights[MAX_LIGHTS];
	vec4 ambientLight;
	int numLights;
	int reflectionModel;	// Possibly not best practice to introduce an if statement in the shader here.... 
							// ...doing so in interest of being able to easily visualise different PBR models!
} uLight;

#define M_PI 3.1415926535897932384626433832795
#define M_EPS 0.0000001

layout (set = 2, binding = 2) uniform sampler2D position;
layout (set = 2, binding = 3) uniform sampler2D normal;
layout (set = 2, binding = 4) uniform sampler2D albedo_specular;
layout (set = 2, binding = 5) uniform sampler2D emissive_metalness;

layout (location = 0) in vec2 inUV;

layout (location = 0) out vec4 outFragColor;

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
		normalDistribution = ggxNormalDist(0.1, texture(emissive_metalness, inUV).w, normalDirection, halfVector);
	}
	else
	{
		normalDistribution = blinnPhongNormalDist(texture(albedo_specular, inUV).w, normalDirection, halfVector);
	}

	// Cook-Torrance masking term
	float masking = min(1, min(
		2 * max(0, dot(normalDirection, halfVector)) * max(0, dot(normalDirection, viewDirection)) / (dot(viewDirection, halfVector) + M_EPS),
		2 * max(0, dot(normalDirection, halfVector)) * max(0, dot(normalDirection, lightDirection)) / (dot(viewDirection, halfVector) + M_EPS)
	));
	
	// Fresnel by Schlick approximation
	vec3 specularBaseReflectivity = (1 - texture(emissive_metalness, inUV).w) * vec3(0.04) + texture(emissive_metalness, inUV).w * vec3(texture(albedo_specular, inUV).x, texture(albedo_specular, inUV).y, texture(albedo_specular, inUV).z);
	vec3 fresnel = specularBaseReflectivity + (vec3(1.0f) - specularBaseReflectivity) * pow((1 - dot(halfVector, viewDirection)), 5);

	// Calculate diffuse with simple lambertian
	vec3 diffuse = vec3(texture(albedo_specular, inUV).x, texture(albedo_specular, inUV).y, texture(albedo_specular, inUV).z) / M_PI * (vec3(1.0f) - fresnel) * (1 - texture(emissive_metalness, inUV).w);
	
	return diffuse + normalDistribution * fresnel * masking / (4 * dot(normalDirection, viewDirection) * dot(normalDirection, lightDirection) + M_EPS);
}

void main()
{
	// Don't write fragments to areas where no geometry was recorded
	// Checking W in pos, as it'll be 0 when nothing exists there, and if there is for some reason, it wouldnt be rendered anyway as it's at infinity.
	if (texture(position, inUV).w == 0)
	{
		discard;
	}
	// Visualise position
	//outFragColor = texture(position, inUV);
	// Visualise normal
	//outFragColor = texture(normal, inUV);
	// Visualise albedo
	//outFragColor = vec4(vec3(texture(albedo_specular, inUV)), 1.0f);
	// Visualise emissive
	//outFragColor = vec4(vec3(texture(emissive_metalness, inUV)), 1.0f);

	// Calculate directions
	vec3 viewDirection = normalize(uScene.cameraPosition - vec3(texture(position, inUV)));
	// Normalise the normal, just in case...
	vec3 normalDirection = normalize(vec3(texture(normal, inUV)));
	
	// Convert to vec3 to avoid adding loads of ws
	vec3 lightEmitted = vec3(texture(emissive_metalness, inUV));
	vec3 lightAmbient = vec3(0.002f);

	int numLights = min(uLight.numLights, MAX_LIGHTS);
	vec3 totalLight = vec3(0.0f);
	
	for (int i = 0; i < numLights; i++)
	{
		vec3 lightPosition = vec3(uLight.lights[i].lightPosition.x, uLight.lights[i].lightPosition.y, uLight.lights[i].lightPosition.z);
		vec3 lightDirection = normalize(lightPosition - vec3(texture(position, inUV)));

		vec3 brdf = burleyBRDF(lightDirection, viewDirection, normalDirection);

		vec3 lightColor = vec3(uLight.lights[i].lightColor.x, uLight.lights[i].lightColor.y, uLight.lights[i].lightColor.z);
		totalLight += brdf * lightColor * max(0, dot(normalDirection, lightDirection));
	}


	outFragColor = vec4(totalLight + lightEmitted + lightAmbient, 1.0f);
}