import Scene.Raytracing;
import Scene.Intersection;
import Scene.RaytracingInline;
import Scene.Shading;
import Utils.Math.MathHelpers;
import Utils.Math.HashUtils;
import Utils.Geometry.GeometryHelpers;
import Utils.Sampling.SampleGenerator;
import Rendering.Lights.LightHelpers;
import Rendering.Lights.EnvMapSampler;
import Rendering.Lights.EmissiveLightSampler;
import Rendering.Lights.EmissiveLightSamplerHelpers;
import Rendering.Materials.MaterialShading;
import Experimental.WorldSpaceReSTIRGI.GIFinalSample;
import Experimental.WorldSpaceReSTIRGI.ReconnectionData;
import Params;
import LoadShadingData;
import PathTracer;
import CausticPhoton;

#include "Utils/Math/MathConstants.slangh"

// Caustic photon mapping parameters
// Note: Careful with alignment! float3 requires 16-byte alignment in cbuffer
cbuffer CausticPhotonCB
{
    float3 gSceneBBMin;         // 0-11 (12 bytes)
    float gCellSize;            // 12-15 (4 bytes) - packs with float3
    float gGatherRadius;        // 16-19 (4 bytes)
    uint gMaxGatherPhotons;     // 20-23 (4 bytes)
    uint gHashTableSize;        // 24-27 (4 bytes)
    uint gTotalPhotons;         // 28-31 (4 bytes)
    uint gUseCausticPhotonMapping; // 32-35 (4 bytes) - use uint instead of bool
};

// Caustic photon buffers
StructuredBuffer<PackedCausticPhoton> gPhotonBuffer;
StructuredBuffer<uint> gPhotonCellStorage;
ByteAddressBuffer gPhotonIndexBuffer;
ByteAddressBuffer gPhotonCheckSumBuffer;

// Helper functions for photon gathering
uint causticPcg32(uint input)
{
    uint state = input * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

int3 positionToPhotonCell(float3 pos)
{
    return int3(floor((pos - gSceneBBMin) / gCellSize));
}

uint cellToPhotonHash(int3 cell)
{
    uint h = causticPcg32(uint(cell.x) + causticPcg32(uint(cell.y) + causticPcg32(uint(cell.z))));
    return h % gHashTableSize;
}

uint cellToPhotonChecksum(int3 cell)
{
    return max(jenkinsHash(uint(cell.x) + jenkinsHash(uint(cell.y) + jenkinsHash(uint(cell.z)))), 1u);
}

int findPhotonCellByCoord(int3 cell)
{
    uint cellHash = causticPcg32(uint(cell.x) + causticPcg32(uint(cell.y) + causticPcg32(uint(cell.z))));
    uint checkSum = cellToPhotonChecksum(cell);
    uint maxSlots = gHashTableSize * 32;
    
    for (uint i = 0; i < 32; i++)
    {
        uint idx = (cellHash * 32 + i) % maxSlots;
        uint storedChecksum = gPhotonCheckSumBuffer.Load(idx * 4);
        if (storedChecksum == checkSum)
        {
            return int(idx);
        }
        if (storedChecksum == 0)
        {
            // Empty slot means cell not found
            return -1;
        }
    }
    
    return -1;
}

/// Debug: Count valid photons and check their Y position distribution
/// Returns: x = valid count / 1000, y = avg Y position, z = photons near floor (y < 0.05)
float3 debugPhotonStats()
{
    uint validCount = 0;
    float sumY = 0;
    uint nearFloorCount = 0;
    
    uint maxCheck = min(gTotalPhotons, 50000u);
    for (uint i = 0; i < maxCheck; i++)
    {
        PackedCausticPhoton packed = gPhotonBuffer[i];
        if (!packed.isValid()) continue;
        
        CausticPhoton photon = packed.unpack();
        validCount++;
        sumY += photon.position.y;
        
        // Check if photon is near floor (y close to 0)
        if (photon.position.y < 0.05f && photon.position.y > -0.1f)
        {
            nearFloorCount++;
        }
    }
    
    float avgY = validCount > 0 ? sumY / float(validCount) : 0;
    return float3(float(validCount) / 1000.0, avgY, float(nearFloorCount) / 100.0);
}

/// Gather caustic contribution at a point
float3 gatherCaustics(float3 pos, float3 normal, float3 viewDir)
{
    if (gUseCausticPhotonMapping == 0) return float3(0);
    if (gTotalPhotons == 0) return float3(0);
    
    float3 causticRadiance = float3(0);
    float radiusSq = gGatherRadius * gGatherRadius;
    uint photonCount = 0;
    
    // Limit search to avoid GPU timeout
    uint maxCheck = min(gTotalPhotons, 200000u);
    
    for (uint i = 0; i < maxCheck; i++)
    {
        PackedCausticPhoton packed = gPhotonBuffer[i];
        if (!packed.isValid()) continue;
        
        CausticPhoton photon = packed.unpack();
        
        // Distance check
        float3 diff = photon.position - pos;
        float distSq = dot(diff, diff);
        if (distSq > radiusSq) continue;
        
        // Cone filter kernel
        float dist = sqrt(distSq);
        float k = 1.0 - dist / gGatherRadius;
        float weight = k * k;
        
        // BRDF evaluation (diffuse)
        float cosTheta = abs(dot(photon.incidentDir, normal));
        
        // Accumulate contribution
        causticRadiance += photon.flux * weight * cosTheta;
        photonCount++;
        
        if (photonCount >= gMaxGatherPhotons) break;
    }
    
    // Normalize by disk area
    if (photonCount > 0)
    {
        float area = M_PI * radiusSq;
        causticRadiance /= area;
    }
    
    return causticRadiance;
}

struct FinalShading
{
    Texture2D<PackedHitInfo> vbuffer;
    StructuredBuffer<FinalSample> finalSample;
    StructuredBuffer<ReconnectionData> reconnectionDataBuffer;

    RWTexture2D<float3> outputColor;

    PTRuntimeParams params;

    void execute(uint2 pixel)
    {
        uint linearID = pixel.y * params.frameDim.x + pixel.x;
        ReconnectionData data = reconnectionDataBuffer[linearID];

        HitInfo hit = HitInfo(data.preRcVertexHitInfo);
        if (!hit.isValid())
            return;
        float flag = 1.f;
        float lod = 0.f;
        bool adjustShadingNormal = data.pathLength <= 1 ? true : false;
        ShadingData sd = LoadShadingData(hit, data.preRcVertexWo, lod, adjustShadingNormal);
                 
       
        FinalSample sample = finalSample[linearID];
        if (sd.linearRoughness < 0.2 && data.pathLength == 1)
        {
            
            //sample.Li = 0.f;
            //flag = sd.p
            FalcorBSDF bsdf;
            bsdf.setup(sd);
            flag = bsdf.pDiffuseReflection;
            sd.setActiveLobes((uint) LobeType::Diffuse);
        }

        // Photon Mapping 焦散贡献
        // 焦散应该应用在所有漫反射表面上，不管pathLength是多少
        // 因为透过玻璃看到的地板也应该显示焦散
        float3 causticContribution = float3(0);
        if (gUseCausticPhotonMapping)
        {
            // Only apply caustics to diffuse surfaces (not glass/specular)
            if (sd.linearRoughness >= 0.3f && sd.specularTransmission < 0.01f)
            {
                causticContribution = gatherCaustics(sd.posW, sd.N, -data.preRcVertexWo);
                // Increase intensity for better visibility
                causticContribution *= 20.0;
            }
        }
        
        // Note: causticContribution is added separately and NOT divided by numGIInstance
        // because it represents direct caustic illumination, not a GI sample
        float3 giContribution = (data.pathPreRadiance + data.pathPreThp * evalBSDFCosine(sd, sample.dir) * sample.Li / flag) / params.numGIInstance;
        outputColor[pixel] += giContribution + causticContribution;

    }
};

ParameterBlock<FinalShading> finalShading;

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    finalShading.execute(dispatchThreadId.xy);
}
             