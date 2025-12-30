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
cbuffer CausticPhotonCB
{
    bool gUseCausticPhotonMapping;
    float3 gSceneBBMin;
    float gCellSize;
    float gGatherRadius;
    uint gMaxGatherPhotons;
    uint gHashTableSize;
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
    uint cellIndex = cellToPhotonHash(cell);
    uint checkSum = cellToPhotonChecksum(cell);
    
    for (uint i = 0; i < 32; i++)
    {
        uint idx = (cellIndex * 32 + i) * 4;
        if (gPhotonCheckSumBuffer.Load(idx) == checkSum)
        {
            return int(cellIndex * 32 + i);
        }
    }
    
    return -1;
}

/// Gather caustic contribution at a point
float3 gatherCaustics(float3 pos, float3 normal, float3 viewDir)
{
    if (!gUseCausticPhotonMapping) return float3(0);
    
    float3 causticRadiance = float3(0);
    float radiusSq = gGatherRadius * gGatherRadius;
    uint photonCount = 0;
    
    // Search neighboring cells
    int3 baseCell = positionToPhotonCell(pos);
    int searchRadius = int(ceil(gGatherRadius / gCellSize));
    
    for (int dx = -searchRadius; dx <= searchRadius; dx++)
    {
        for (int dy = -searchRadius; dy <= searchRadius; dy++)
        {
            for (int dz = -searchRadius; dz <= searchRadius; dz++)
            {
                int3 neighborCell = baseCell + int3(dx, dy, dz);
                
                int cellIdx = findPhotonCellByCoord(neighborCell);
                if (cellIdx < 0) continue;
                
                // Get photon range for this cell
                uint startIdx = cellIdx > 0 ? gPhotonIndexBuffer.Load((cellIdx - 1) * 4) : 0;
                uint endIdx = gPhotonIndexBuffer.Load(cellIdx * 4);
                
                for (uint i = startIdx; i < endIdx && photonCount < gMaxGatherPhotons; i++)
                {
                    uint photonIdx = gPhotonCellStorage[i];
                    PackedCausticPhoton packed = gPhotonBuffer[photonIdx];
                    
                    if (!packed.isValid()) continue;
                    
                    CausticPhoton photon = packed.unpack();
                    
                    // Distance check
                    float3 diff = photon.position - pos;
                    float distSq = dot(diff, diff);
                    if (distSq > radiusSq) continue;
                    
                    // Normal check - photon should be on same surface
                    if (dot(photon.normal, normal) < 0.8) continue;
                    
                    // Cone filter kernel
                    float dist = sqrt(distSq);
                    float k = 1.0 - dist / gGatherRadius;
                    float weight = k * k;
                    
                    // BRDF evaluation (diffuse)
                    float cosTheta = max(0, -dot(photon.incidentDir, normal));
                    
                    // Accumulate contribution
                    causticRadiance += photon.flux * weight * cosTheta;
                    photonCount++;
                }
            }
        }
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

        // Photon Mapping 焦散贡献（仅在开关打开时启用）
        float3 causticContribution = float3(0);
        if (gUseCausticPhotonMapping && data.pathLength == 1)
        {
            causticContribution = gatherCaustics(sd.posW, sd.N, -data.preRcVertexWo);
        }

        outputColor[pixel] += (data.pathPreRadiance + data.pathPreThp * evalBSDFCosine(sd, sample.dir) * sample.Li / flag + causticContribution) / params.numGIInstance;
        //outputColor[pixel] = sample.Li;
        //outputColor[pixel] += sample.Li;

    }
};

ParameterBlock<FinalShading> finalShading;

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    finalShading.execute(dispatchThreadId.xy);
}
             