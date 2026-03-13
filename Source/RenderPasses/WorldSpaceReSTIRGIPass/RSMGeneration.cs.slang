/***************************************************************************
 # RSM Generation Pass (Compute Shader with Ray Tracing)
 # Renders the scene from light's perspective to generate Reflective Shadow Map
 # Output: position, normal, flux textures
 **************************************************************************/

import Scene.Raytracing;
import Scene.Intersection;
import Scene.Shading;
import Utils.Math.MathHelpers;
import Utils.Sampling.SampleGenerator;
import Rendering.Materials.MaterialShading;
import RSMData;

// RSM generation parameters
cbuffer RSMGenerationCB
{
    float4x4 gLightViewProj;        // Light view-projection matrix
    float4x4 gLightView;            // Light view matrix
    float3 gLightPosition;          // Light position
    float gLightIntensity;          // Light intensity
    float3 gLightColor;             // Light color
    uint gRSMResolution;            // RSM resolution
    float3 gLightDirection;         // Light direction (normalized)
    float gLightArea;               // Light source area
    uint gFrameCount;
    float3 _pad;
};

// Output textures
RWTexture2D<float4> gRSMPosition;   // World position + depth
RWTexture2D<float4> gRSMNormal;     // World normal + material ID
RWTexture2D<float4> gRSMFlux;       // Flux (RGB) + valid flag

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadId.xy;
    if (pixel.x >= gRSMResolution || pixel.y >= gRSMResolution)
        return;
    
    // Initialize outputs as invalid
    gRSMPosition[pixel] = float4(0, 0, 0, -1);
    gRSMNormal[pixel] = float4(0, 0, 0, 0);
    gRSMFlux[pixel] = float4(0, 0, 0, 0);
    
    // Convert pixel to normalized device coordinates [-1, 1]
    float2 ndc = (float2(pixel) + 0.5f) / float(gRSMResolution) * 2.0f - 1.0f;
    ndc.y = -ndc.y; // Flip Y for correct orientation
    
    // For area light: sample position on light surface with jitter
    SampleGenerator sg = SampleGenerator(pixel, gFrameCount);
    float2 lightJitter = (sampleNext2D(sg) - 0.5f) * 0.13f; // Light size is 0.13
    float3 lightSamplePos = gLightPosition + lightJitter.x * float3(1, 0, 0) + lightJitter.y * float3(0, 0, 1);
    
    // Build ray direction from light through this RSM pixel
    // Using a cone distribution from the area light
    float3 tangent = float3(1, 0, 0);
    float3 bitangent = float3(0, 0, 1);
    float halfAngle = 0.7f; // ~40 degree half-angle cone
    float3 rayDir = normalize(gLightDirection + ndc.x * halfAngle * tangent + ndc.y * halfAngle * bitangent);
    
    // Create ray from light
    RayDesc ray;
    ray.Origin = lightSamplePos;
    ray.Direction = rayDir;
    ray.TMin = 0.001f;
    ray.TMax = 100.0f;
    
    // Trace ray from light into scene
    HitInfo hit;
    float hitT;
    
    if (traceRay(ray, hit, hitT, RAY_FLAG_NONE, 0xFF))
    {
        // Get vertex data at hit point
        VertexData v = gScene.getVertexData(hit);
        uint materialID = gScene.getMaterialID(hit.getInstanceID());
        
        // Prepare shading data
        ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -rayDir, 0);
        
        // Calculate flux at this point
        // For Lambertian area light: Le * cos(theta_light) * cos(theta_surface) / distance^2
        // But for RSM, we store the flux that would be reflected/transmitted
        float cosLightAngle = max(dot(-rayDir, gLightDirection), 0.0f);
        float cosSurfaceAngle = max(dot(-rayDir, sd.N), 0.0f);
        
        // Light emission
        float3 Le = gLightColor * gLightIntensity;
        
        // Flux = Le * cos(theta) * area / (num_pixels)
        // Each RSM pixel represents a portion of the light's solid angle
        float solidAngleFraction = 4.0f * halfAngle * halfAngle / float(gRSMResolution * gRSMResolution);
        float3 flux = Le * cosLightAngle * gLightArea * cosSurfaceAngle;
        
        // Store RSM data
        gRSMPosition[pixel] = float4(sd.posW, hitT);
        gRSMNormal[pixel] = float4(sd.N, float(materialID));
        gRSMFlux[pixel] = float4(flux, 1.0f); // w=1 indicates valid pixel
    }
}
