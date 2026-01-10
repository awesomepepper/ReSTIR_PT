/***************************************************************************
 # Light Tracing Pass for Caustic Photon Mapping
 # Traces photons from light sources through specular surfaces to diffuse surfaces
 **************************************************************************/

import Scene.Raytracing;
import Scene.Intersection;
import Scene.Shading;
import Utils.Math.MathHelpers;
import Utils.Math.HashUtils;
import Utils.Geometry.GeometryHelpers;
import Utils.Sampling.SampleGenerator;
import Rendering.Lights.LightHelpers;
import Rendering.Lights.EmissiveLightSampler;
import Rendering.Lights.EmissiveLightSamplerHelpers;
import Rendering.Materials.MaterialShading;
import Rendering.Materials.IBxDF;
import Rendering.Materials.BxDF;  // For FalcorBSDF::getLobes
import CausticPhoton;
import Params;

#include "Utils/Math/MathConstants.slangh"

// Photon tracing parameters
cbuffer PhotonTracingCB
{
    uint gPhotonsPerLight;      // Number of photons per light
    uint gMaxPhotonBounces;     // Maximum bounces for photon
    uint gFrameCount;
    uint gTotalPhotons;         // Total photons to trace
    float3 gSceneBBMin;
    float gCellSize;
    uint gHashTableSize;        // Hash table size for photon grid
};

// Output buffers
RWStructuredBuffer<PackedCausticPhoton> gPhotonBuffer;
RWStructuredBuffer<PhotonAppendData> gPhotonAppendBuffer;
RWByteAddressBuffer gPhotonCheckSum;
RWByteAddressBuffer gPhotonCellCounters;

// Scene resources
EmissiveLightSampler gEmissiveSampler;

// PCG hash for photon grid (defined before use)
uint photonPcg32(uint input)
{
    uint state = input * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Photon payload
struct PhotonPayload
{
    float3 flux;        // Current photon flux
    float3 origin;      // Ray origin
    float3 direction;   // Ray direction
    uint bounceCount;   // Number of bounces
    uint photonIndex;   // Index for storing
    bool hasPassedSpecular; // Has passed through specular surface
    bool terminated;
    bool stored;        // Whether photon was stored
    SampleGenerator sg;
};

/// Sample a light source and get initial photon direction
bool sampleLightPhoton(inout SampleGenerator sg, out float3 origin, out float3 direction, out float3 flux)
{
    origin = float3(0);
    direction = float3(0);
    flux = float3(0);
    
    // Hardcoded Cornell Box light source (top area light)
    // Light scaling is 0.13, so actual size is 0.13 x 0.13 centered at (0, 0.549, 0)
    float halfSize = 0.065f;  // 0.13 / 2
    float2 uPos = sampleNext2D(sg);
    float lx = lerp(-halfSize, halfSize, uPos.x);
    float lz = lerp(-halfSize, halfSize, uPos.y);
    float3 lightPos = float3(lx, 0.548f, lz);
    
    // Sample direction on hemisphere (cosine-weighted, pointing down into the scene)
    float2 uDir = sampleNext2D(sg);
    float pdf;
    float3 localDir = sample_cosine_hemisphere_concentric(uDir, pdf);
    
    // Transform to world space
    // Light emits downward, so light normal is (0, -1, 0)
    // localDir.z is the "up" direction in local space (along normal)
    // We need to map this to world space where "up" is (0, -1, 0)
    float3 tangent = float3(1, 0, 0);
    float3 bitangent = float3(0, 0, 1);
    float3 lightNormal = float3(0, -1, 0);
    
    // localDir = (x, y, z) where z is along normal
    direction = normalize(localDir.x * tangent + localDir.y * bitangent + localDir.z * lightNormal);
    
    // Offset origin slightly in the emission direction to avoid self-intersection
    origin = lightPos + direction * 0.001f;
    
    // Physically correct flux calculation:
    // For a Lambertian emitter with radiance Le:
    // - Total power (Phi) = Le * A * pi
    // - Each photon carries: Phi / N (where N = total photons, divided later in RayGen)
    // - Cosine-weighted sampling PDF = cos(theta) / pi
    // - The weight = Le * cos(theta) / PDF = Le * cos(theta) / (cos(theta)/pi) = Le * pi
    // - So flux per photon = Le * pi * A (area is for uniform position sampling)
    float lightArea = 0.13f * 0.13f;
    float3 Le = float3(17.0f, 12.0f, 4.0f) * 5.0f;  // emissiveColor * emissiveFactor
    flux = Le * M_PI * lightArea;
    
    return true;
}

/// Check if material is specular/delta (for caustic tracing)
bool isSpecularMaterial(ShadingData sd)
{
    // Glass material check - most important for caustics
    if (sd.specularTransmission > 0.01f) return true;
    
    // Low roughness is specular
    if (sd.linearRoughness < 0.2f) return true;
    
    // Check for specular lobes
    uint lobeTypes = FalcorBSDF::getLobes(sd);
    if ((lobeTypes & (uint)LobeType::SpecularOrDelta) != 0) return true;
    
    return false;
}

/// Check if material is diffuse (for storing caustic photons)
bool isDiffuseMaterial(ShadingData sd)
{
    // Not glass
    if (sd.specularTransmission > 0.01f) return false;
    
    // High roughness is diffuse
    if (sd.linearRoughness >= 0.3f) return true;
    
    // Check for diffuse lobes
    uint lobeTypes = FalcorBSDF::getLobes(sd);
    if ((lobeTypes & (uint)LobeType::Diffuse) != 0) return true;
    
    return false;
}

/// Store photon at current location
void storePhoton(uint photonIndex, float3 position, float3 normal, float3 flux, float3 incidentDir)
{
    CausticPhoton photon = CausticPhoton.create(position, normal, flux, incidentDir);
    gPhotonBuffer[photonIndex] = PackedCausticPhoton.pack(photon);
    
    // Add to hash grid
    int3 cell = int3(floor((position - gSceneBBMin) / gCellSize));
    uint cellHash = photonPcg32(uint(cell.x) + photonPcg32(uint(cell.y) + photonPcg32(uint(cell.z))));
    uint checkSum = max(jenkinsHash(uint(cell.x) + jenkinsHash(uint(cell.y) + jenkinsHash(uint(cell.z)))), 1u);
    
    // Find or insert cell using linear probing within hash table
    int foundCell = -1;
    uint maxSlots = gHashTableSize * 32;
    
    for (uint i = 0; i < 32; i++)
    {
        // Use modulo to wrap around within valid range
        uint idx = (cellHash * 32 + i) % maxSlots;
        
        uint checkSumPre;
        gPhotonCheckSum.InterlockedCompareExchange(idx * 4, 0, checkSum, checkSumPre);
        if (checkSumPre == 0 || checkSumPre == checkSum)
        {
            foundCell = int(idx);
            break;
        }
    }
    
    if (foundCell >= 0)
    {
        uint inCellIdx;
        gPhotonCellCounters.InterlockedAdd(foundCell * 4, 1, inCellIdx);
        
        PhotonAppendData appendData;
        appendData.isValid = 1;
        appendData.photonIndex = photonIndex;
        appendData.cellIdx = uint(foundCell);
        appendData.inCellIdx = inCellIdx;
        gPhotonAppendBuffer[photonIndex] = appendData;
    }
}

//
// Ray tracing shaders
//

[shader("raygeneration")]
void PhotonRayGen()
{
    uint photonIndex = DispatchRaysIndex().x;
    if (photonIndex >= gTotalPhotons) return;
    
    // Initialize random generator
    SampleGenerator sg = SampleGenerator(uint2(photonIndex, 0), gFrameCount);
    
    // Sample light source
    float3 origin, direction, flux;
    if (!sampleLightPhoton(sg, origin, direction, flux))
    {
        gPhotonBuffer[photonIndex] = PackedCausticPhoton.pack(CausticPhoton.createInvalid());
        return;
    }
    
    // Scale flux by number of photons
    flux /= float(gTotalPhotons);
    
    // Initialize payload
    PhotonPayload payload;
    payload.flux = flux;
    payload.origin = origin;
    payload.direction = direction;
    payload.bounceCount = 0;
    payload.photonIndex = photonIndex;
    payload.hasPassedSpecular = false;
    payload.terminated = false;
    payload.stored = false;
    payload.sg = sg;
    
    // Trace photon
    for (uint bounce = 0; bounce < gMaxPhotonBounces && !payload.terminated; bounce++)
    {
        RayDesc ray;
        ray.Origin = payload.origin;
        ray.Direction = payload.direction;
        ray.TMin = 0.001f;
        ray.TMax = 1e30f;
        
        TraceRay(
            gScene.rtAccel,
            RAY_FLAG_NONE,
            0xFF,
            0, // hitIndex
            rayTypeCount,
            0, // missIndex
            ray,
            payload
        );
        
        payload.bounceCount++;
    }
    
    // If photon didn't store (no valid caustic path), mark as invalid
    if (!payload.stored)
    {
        gPhotonBuffer[photonIndex] = PackedCausticPhoton.pack(CausticPhoton.createInvalid());
    }
}

[shader("miss")]
void PhotonMiss(inout PhotonPayload payload)
{
    payload.terminated = true;
}

[shader("anyhit")]
void PhotonAnyHit(inout PhotonPayload payload, BuiltInTriangleIntersectionAttributes attribs)
{
    // Alpha test
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    uint materialID = gScene.getMaterialID(instanceID);
    
    if (alphaTest(v, gScene.materials[materialID], gScene.materialResources[materialID], 0.f))
        IgnoreHit();
}

[shader("closesthit")]
void PhotonClosestHit(inout PhotonPayload payload, BuiltInTriangleIntersectionAttributes attribs)
{
    // Get hit info
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    uint materialID = gScene.getMaterialID(instanceID);
    
    // Load shading data
    ShadingData sd = prepareShadingData(v, materialID, gScene.materials[materialID], gScene.materialResources[materialID], -payload.direction, 0.f);
    
    // Check if this is glass material
    bool isGlass = sd.specularTransmission > 0.01f;
    
    // For glass, always continue tracing (don't store on glass surface)
    if (isGlass)
    {
        payload.hasPassedSpecular = true;
        
        // Enable all lobes for glass
        sd.setActiveLobes((uint)LobeType::All);
        
        // Sample BSDF for next direction
        BSDFSample_ bsdfSample;
        if (sampleBSDF(sd, payload.sg, bsdfSample))
        {
            if (any(isnan(bsdfSample.weight)) || all(bsdfSample.weight <= 0))
            {
                payload.terminated = true;
                return;
            }
            
            payload.flux *= bsdfSample.weight;
            
            float maxFlux = max(payload.flux.r, max(payload.flux.g, payload.flux.b));
            if (maxFlux <= 0 || isnan(maxFlux) || isinf(maxFlux))
            {
                payload.terminated = true;
                return;
            }
            
            payload.direction = bsdfSample.wi;
            
            // Determine transmission by checking if ray goes to opposite side
            float dotOut = dot(bsdfSample.wi, sd.faceN);
            float dotIn = dot(-payload.direction, sd.faceN);
            bool isTransmission = (dotOut * dotIn) < 0;
            
            payload.origin = sd.computeNewRayOrigin(!isTransmission);
        }
        else
        {
            payload.terminated = true;
        }
    }
    else
    {
        // Non-glass surface
        bool isSpecular = sd.linearRoughness < 0.2f;
        
        if (isSpecular)
        {
            // Specular non-glass (mirror) - continue tracing
            payload.hasPassedSpecular = true;
            
            BSDFSample_ bsdfSample;
            if (sampleBSDF(sd, payload.sg, bsdfSample))
            {
                if (any(isnan(bsdfSample.weight)) || all(bsdfSample.weight <= 0))
                {
                    payload.terminated = true;
                    return;
                }
                
                payload.flux *= bsdfSample.weight;
                payload.direction = bsdfSample.wi;
                payload.origin = sd.computeNewRayOrigin(true);  // Reflection
            }
            else
            {
                payload.terminated = true;
            }
        }
        else if (payload.hasPassedSpecular)
        {
            // Diffuse surface after passing through specular - store caustic photon
            storePhoton(payload.photonIndex, sd.posW, sd.N, payload.flux, payload.direction);
            payload.stored = true;
            payload.terminated = true;
        }
        else
        {
            // Diffuse surface without passing through specular - not a caustic
            payload.terminated = true;
        }
    }
}
