/***************************************************************************
 # Light Tracing Pass for Caustic Photon Mapping
 # Traces photons from light sources through specular surfaces to diffuse surfaces
 **************************************************************************/

import Scene.Raytracing;
import Scene.Intersection;
import Scene.Shading;
import Utils.Math.MathHelpers;
import Utils.Math.HashUtils;
import Utils.Sampling.SampleGenerator;
import Rendering.Lights.LightHelpers;
import Rendering.Lights.EmissiveLightSampler;
import Rendering.Lights.EmissiveLightSamplerHelpers;
import Rendering.Materials.MaterialShading;
import Rendering.Materials.IBxDF;
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
    
#if USE_EMISSIVE_LIGHTS
    // Check if there are any active triangles
    if (gScene.lightCollection.getActiveTriangleCount() == 0) return false;
    
    // Randomly pick a triangle
    float uLight = sampleNext1D(sg);
    uint triangleCount = gScene.lightCollection.getActiveTriangleCount();
    uint idx = min((uint)(uLight * triangleCount), triangleCount - 1);
    uint triangleIndex = gScene.lightCollection.activeTriangles[idx];
    
    // Sample position on triangle
    float2 u = sampleNext2D(sg);
    float3 barycentrics = sample_triangle(u);
    
    // Get triangle data
    EmissiveTriangle tri = gScene.lightCollection.getTriangle(triangleIndex);
    origin = barycentrics.x * tri.posW[0] + barycentrics.y * tri.posW[1] + barycentrics.z * tri.posW[2];
    float3 normalW = normalize(cross(tri.posW[1] - tri.posW[0], tri.posW[2] - tri.posW[0]));
    
    // Sample direction on hemisphere (cosine-weighted)
    float2 uDir = sampleNext2D(sg);
    float pdf;
    float3 localDir = sample_cosine_hemisphere_concentric(uDir, pdf);
    
    // Transform to world space using triangle normal
    float3 tangent = perp_stark(normalW);
    float3 bitangent = cross(normalW, tangent);
    direction = localDir.x * tangent + localDir.y * bitangent + localDir.z * normalW;
    
    // Photon flux = emitted radiance * area * pi / (numPhotons * triangleSelectionPdf)
    // Simplified: Le * area * pi * numTriangles
    float3 Le = gScene.lightCollection.getAverageRadiance(triangleIndex);
    float area = tri.area;
    flux = Le * area * M_PI * float(triangleCount);
    
    return true;
#else
    return false;
#endif
}

/// Check if material is specular/delta
bool isSpecularMaterial(ShadingData sd)
{
    return sd.linearRoughness < 0.2f; // Threshold for specular
}

/// Check if material is diffuse
bool isDiffuseMaterial(ShadingData sd)
{
    return sd.linearRoughness >= 0.2f;
}

/// Store photon at current location
void storePhoton(uint photonIndex, float3 position, float3 normal, float3 flux, float3 incidentDir)
{
    CausticPhoton photon = CausticPhoton.create(position, normal, flux, incidentDir);
    gPhotonBuffer[photonIndex] = PackedCausticPhoton.pack(photon);
    
    // Add to hash grid
    int3 cell = int3(floor((position - gSceneBBMin) / gCellSize));
    uint cellIndex = (photonPcg32(uint(cell.x) + photonPcg32(uint(cell.y) + photonPcg32(uint(cell.z))))) % 100000;
    uint checkSum = max(jenkinsHash(uint(cell.x) + jenkinsHash(uint(cell.y) + jenkinsHash(uint(cell.z)))), 1u);
    
    // Find or insert cell
    int foundCell = -1;
    for (uint i = 0; i < 32; i++)
    {
        uint idx = cellIndex * 32 + i;
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
        // Store invalid photon
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
    
    // Check material type
    bool isSpecular = isSpecularMaterial(sd);
    bool isDiffuse = isDiffuseMaterial(sd);
    
    if (isSpecular)
    {
        // Specular surface - continue tracing
        payload.hasPassedSpecular = true;
        
        // Sample BSDF for next direction
        BSDFSample_ bsdfSample;
        if (sampleBSDF(sd, payload.sg, bsdfSample))
        {
            // Update flux with BSDF weight
            payload.flux *= bsdfSample.weight;
            
            // Setup next ray
            payload.origin = sd.posW;
            payload.direction = bsdfSample.wi;
            
            // Russian roulette
            float maxFlux = max(payload.flux.r, max(payload.flux.g, payload.flux.b));
            if (maxFlux < 0.01f)
            {
                float q = max(0.05f, 1.0f - maxFlux);
                if (sampleNext1D(payload.sg) < q)
                {
                    payload.terminated = true;
                    return;
                }
                payload.flux /= (1.0f - q);
            }
        }
        else
        {
            payload.terminated = true;
        }
    }
    else if (isDiffuse && payload.hasPassedSpecular)
    {
        // Diffuse surface after passing through specular - store photon (caustic)
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
