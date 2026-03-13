/***************************************************************************
 # RSM-based Photon Tracing Pass
 # Uses RSM pixels as virtual photon sources for efficient caustic generation
 # Reference: "Real-Time Caustics Using Cascaded Image-Space Photon Tracing"
 **************************************************************************/

import Scene.Raytracing;
import Scene.Intersection;
import Scene.Shading;
import Utils.Math.MathHelpers;
import Utils.Math.HashUtils;
import Utils.Geometry.GeometryHelpers;
import Utils.Sampling.SampleGenerator;
import Rendering.Lights.LightHelpers;
import Rendering.Materials.MaterialShading;
import Rendering.Materials.IBxDF;
import Rendering.Materials.BxDF;
import CausticPhoton;
import RSMData;
import Params;

#include "Utils/Math/MathConstants.slangh"

// RSM Photon tracing parameters
cbuffer RSMPhotonTracingCB
{
    uint gRSMResolution;        // RSM texture resolution
    uint gMaxPhotonBounces;     // Maximum bounces for photon
    uint gFrameCount;
    uint gTotalPhotons;         // Total photons to trace (RSM_res * RSM_res)
    float3 gSceneBBMin;
    float gCellSize;
    uint gHashTableSize;
    float gPhotonFluxScale;     // Scale factor for photon flux
    float2 _pad;
};

// RSM textures (read-only)
Texture2D<float4> gRSMPosition;     // World position + depth
Texture2D<float4> gRSMNormal;       // World normal + material ID
Texture2D<float4> gRSMFlux;         // Flux (RGB) + valid flag

// Output buffers
RWStructuredBuffer<PackedCausticPhoton> gPhotonBuffer;
RWStructuredBuffer<PhotonAppendData> gPhotonAppendBuffer;
RWByteAddressBuffer gPhotonCheckSum;
RWByteAddressBuffer gPhotonCellCounters;

// PCG hash for photon grid
uint photonPcg32(uint input)
{
    uint state = input * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Photon payload
struct RSMPhotonPayload
{
    float3 flux;
    float3 origin;
    float3 direction;
    uint bounceCount;
    uint photonIndex;
    bool hasPassedSpecular;
    bool terminated;
    bool stored;
    SampleGenerator sg;
};

/// Store photon at current location
void storePhoton(uint photonIndex, float3 position, float3 normal, float3 flux, float3 incidentDir)
{
    CausticPhoton photon = CausticPhoton.create(position, normal, flux, incidentDir);
    gPhotonBuffer[photonIndex] = PackedCausticPhoton.pack(photon);
    
    // Add to hash grid
    int3 cell = int3(floor((position - gSceneBBMin) / gCellSize));
    uint cellHash = photonPcg32(uint(cell.x) + photonPcg32(uint(cell.y) + photonPcg32(uint(cell.z))));
    uint checkSum = max(jenkinsHash(uint(cell.x) + jenkinsHash(uint(cell.y) + jenkinsHash(uint(cell.z)))), 1u);
    
    int foundCell = -1;
    uint maxSlots = gHashTableSize * 32;
    
    for (uint i = 0; i < 32; i++)
    {
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
void RSMPhotonRayGen()
{
    uint photonIndex = DispatchRaysIndex().x;
    
    // Convert linear index to RSM pixel coordinates
    uint2 rsmPixel = uint2(photonIndex % gRSMResolution, photonIndex / gRSMResolution);
    
    if (rsmPixel.x >= gRSMResolution || rsmPixel.y >= gRSMResolution)
    {
        gPhotonBuffer[photonIndex] = PackedCausticPhoton.pack(CausticPhoton.createInvalid());
        return;
    }
    
    // Read RSM data
    float4 posDepth = gRSMPosition[rsmPixel];
    float4 normMat = gRSMNormal[rsmPixel];
    float4 fluxValid = gRSMFlux[rsmPixel];
    
    // Check if RSM pixel is valid
    if (posDepth.w < 0 || fluxValid.w < 0.5f)
    {
        gPhotonBuffer[photonIndex] = PackedCausticPhoton.pack(CausticPhoton.createInvalid());
        return;
    }
    
    float3 rsmPosition = posDepth.xyz;
    float3 rsmNormal = normMat.xyz;
    float3 rsmFlux = fluxValid.xyz;
    
    // Initialize random generator
    SampleGenerator sg = SampleGenerator(rsmPixel, gFrameCount);
    
    // Sample direction from RSM point (cosine-weighted hemisphere)
    float2 uDir = sampleNext2D(sg);
    float pdf;
    float3 localDir = sample_cosine_hemisphere_concentric(uDir, pdf);
    
    // Build tangent frame from RSM normal
    float3 tangent, bitangent;
    if (abs(rsmNormal.y) < 0.999f)
    {
        tangent = normalize(cross(float3(0, 1, 0), rsmNormal));
    }
    else
    {
        tangent = normalize(cross(float3(1, 0, 0), rsmNormal));
    }
    bitangent = cross(rsmNormal, tangent);
    
    // Transform to world space
    float3 direction = normalize(localDir.x * tangent + localDir.y * bitangent + localDir.z * rsmNormal);
    
    // Calculate photon flux
    // Flux = RSM_flux * cos(theta) / pdf / num_photons
    // With cosine-weighted sampling: pdf = cos(theta) / pi
    // So: Flux = RSM_flux * pi / num_photons
    float3 flux = rsmFlux * M_PI * gPhotonFluxScale / float(gTotalPhotons);
    
    // Initialize payload
    RSMPhotonPayload payload;
    payload.flux = flux;
    payload.origin = rsmPosition + direction * 0.001f;
    payload.direction = direction;
    payload.bounceCount = 0;
    payload.photonIndex = photonIndex;
    payload.hasPassedSpecular = false;
    payload.terminated = false;
    payload.stored = false;
    payload.sg = sg;
    
    // Check if RSM hit point is on specular surface
    // If so, mark as already passed specular (RSM captured the first specular hit)
    uint materialID = uint(normMat.w);
    // We'll determine this during tracing
    
    // Trace photon through scene
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
            0,
            rayTypeCount,
            0,
            ray,
            payload
        );
        
        payload.bounceCount++;
    }
    
    // If photon didn't store, mark as invalid
    if (!payload.stored)
    {
        gPhotonBuffer[photonIndex] = PackedCausticPhoton.pack(CausticPhoton.createInvalid());
    }
}

[shader("miss")]
void RSMPhotonMiss(inout RSMPhotonPayload payload)
{
    payload.terminated = true;
}

[shader("anyhit")]
void RSMPhotonAnyHit(inout RSMPhotonPayload payload, BuiltInTriangleIntersectionAttributes attribs)
{
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    uint materialID = gScene.getMaterialID(instanceID);
    
    if (alphaTest(v, gScene.materials[materialID], gScene.materialResources[materialID], 0.f))
        IgnoreHit();
}

[shader("closesthit")]
void RSMPhotonClosestHit(inout RSMPhotonPayload payload, BuiltInTriangleIntersectionAttributes attribs)
{
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    uint materialID = gScene.getMaterialID(instanceID);
    
    ShadingData sd = prepareShadingData(v, materialID, gScene.materials[materialID], gScene.materialResources[materialID], -payload.direction, 0.f);
    
    bool isGlass = sd.specularTransmission > 0.01f;
    
    if (isGlass)
    {
        // Glass surface - continue tracing
        payload.hasPassedSpecular = true;
        
        sd.setActiveLobes((uint)LobeType::All);
        
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
        bool isSpecular = sd.linearRoughness < 0.2f;
        
        if (isSpecular)
        {
            // Specular non-glass - continue
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
                payload.origin = sd.computeNewRayOrigin(true);
            }
            else
            {
                payload.terminated = true;
            }
        }
        else if (payload.hasPassedSpecular)
        {
            // Diffuse after specular - store caustic
            storePhoton(payload.photonIndex, sd.posW, sd.N, payload.flux, payload.direction);
            payload.stored = true;
            payload.terminated = true;
        }
        else
        {
            // Diffuse without specular - not caustic
            payload.terminated = true;
        }
    }
}
