# Vulkan Deferred Physically-Based Renderer
## Overview
Renderer implementing a customised Physically Based (PBR) lighting model, utilising deferred shading.

Customised PBR Model       |  Blinn-Phong Reference
:-------------------------:|:-------------------------:
![PBR Screenshot](https://user-images.githubusercontent.com/32328378/177782187-307a72b8-5452-4041-867b-d65d9da3dcb5.png)  |  ![Blinn-Phong Screenshot](https://user-images.githubusercontent.com/32328378/177782232-512afae1-1808-42c5-877d-f6c334489b7a.png)

### Controls
The number of active lights in the scene can be changed using numbers 0-8. Animation can be enabled and disabled with the spacebar. All lights will orbit the Y axis when animation is enabled.
Switch between the PBR model and a Blinn-Phong reference with the M key. Note that the default lighting state is Blinn-Phong.

## Physically Based Rendering
### Custom PBR Model
In the custom PBR Model, the Blinn-Phong normal distribution has been replaced by the GGX normal distribution first introduced by Walter et al. (2007) [1]. 

While using the PBR shader, it is possible to switch between the original and customised PBR model by pressing the M key. This is achieved by a conditional in the shader- this may not be ideal performance wise, but is sufficient for comparing the two models at runtime, which is neccasary for good comparisons when the lights are randomised on each launch.

Burley (2012) [2] describes this model among others in the course of explaining Disney's choice of normal distribution. The motivation for introducing GGX was the need for a longer 'tail'- the responsivness of the distribution to the cosine of oblique angles between the half-vector and the normal vector. This longer tail creates a more realistic, 'glowy' appearance compared to something more of a spec seen with other models such as Blinn-Phong.

The new normal distribution function is as follows: 

**c / (α<sup>2</sup> cos<sup>2</sup> θ<sub>h</sub> + sin<sup>2</sup> θ<sub>h</sub>)<sup>2</sup>**

### Descriptor Set Layout & Bindings
Shading is performed in world space.

Seperate descriptor sets are used for camera transformations, material information and lights. These are fixed descriptor sets with UBO blocks (no SSBOs, dynamic uniforms etc). The reason for using 3 sets is to reduce the amount of data that needs to be modified on each frame, i.e the camera and lights are updated every frame, whereas the materials remain static. The material descriptors could have benefitted from a dynamic allocation, as different buffers are currently bound for each material. While the individual materials are seldom updated, the bound material descriptor set must be changed whenever we attempt to render a model using a new material; with fixed descriptor sets, this also involves binding a new buffer, which is more expensive than changing the index into an already bound buffer. However, the number of materials in this example did not justify the additional complexity.

## Deferred Shading
### Synchronisation Overview
A semaphore was required to signal completion of the new, intermediate render pass. Combined with a wait in the second render pass, this prevents the deferred render pass from reading the G-Buffer before the intermediate render pass has completed. The only barriers present are for updating UBOs; these have been moved to the deferred render pass, as camera and lighting data is now required there, instead of in the first render pass. Subpass dependencies were also used to protect the G-Buffer, defined so that the second render pass does not run until all the G-Buffer images have transitioned past fragment and colour writing.

### G-Buffer Overview
The G-Buffer is somewhat naive, in the interest of saving time, storing some redundant data.
In a future revision, at minimum, positions would be calculated by inversing the transformation matrices from clip-space co-ordinates, and normals would be stored using octahedral co-ordinates.
* Positions: Float32
* Normals: Float32
* Albedo + Shininess encoded in alpha: Float32
* Emissive + metalness encoded in alpha: Float32

## References
[1] Bruce Walter, Stephen R. Marschner, Hongsong Li, and Kenneth E. Torrance. Microfacet models
for refraction through rough surfaces. In Proceedings of the Eurographics Symposium on Rendering,
2007.

[2] Brent Burley. Physically-based shading at Disney, course notes, revised 2014. In ACM
SIGGRAPH, Practical physically-based shading in film and game production, 2012.
