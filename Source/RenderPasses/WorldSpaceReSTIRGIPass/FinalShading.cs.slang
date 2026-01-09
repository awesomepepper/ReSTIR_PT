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

// PPM Caustic photon mapping parameters
cbuffer CausticPhotonCB
{
    float3 gSceneBBMin;         // 0-11 (12 bytes)
    float gCellSize;            // 12-15 (4 bytes)
    float gInitialRadius;       // 16-19 (4 bytes) - PPM initial radius
    uint gMaxGatherPhotons;     // 20-23 (4 bytes)
    uint gHashTableSize;        // 24-27 (4 bytes)
    uint gTotalPhotons;         // 28-31 (4 bytes)
    uint gUseCausticPhotonMapping; // 32-35 (4 bytes)
    uint gFrameIndex;           // 36-39 (4 bytes) - current frame
    float gPPMAlpha;            // 40-43 (4 bytes) - PPM alpha (typically 0.7)
    uint gPhotonsPerFrame;      // 44-47 (4 bytes)
};

// Caustic photon buffers
StructuredBuffer<PackedCausticPhoton> gPhotonBuffer;
StructuredBuffer<uint> gPhotonCellStorage;
ByteAddressBuffer gPhotonIndexBuffer;
ByteAddressBuffer gPhotonCheckSumBuffer;

// PPM per-pixel statistics
// gPPMStats: R = accumulated photon count (N), G = current radius squared (R²), B = unused, A = unused
// gPPMFlux: RGB = accumulated tau (flux * BRDF), A = total emitted photons so far
RWTexture2D<float4> gPPMStats;
RWTexture2D<float4> gPPMFlux;

/// Gather photons using brute force search - simple single-frame version for debugging
/// Returns the current estimate of caustic radiance
float3 gatherAndUpdatePPM(float3 pos, float3 normal, float3 diffuseAlbedo, uint2 pixel)
{
    if (gUseCausticPhotonMapping == 0) return float3(0);
    if (gTotalPhotons == 0) return float3(0);
    
    // Use larger radius for gathering
    float radiusSq = gInitialRadius * gInitialRadius;
    float3 fluxSum = float3(0);
    uint M = 0; // Photons found
    
    // Limit search to avoid GPU timeout
    uint maxSearch = min(gTotalPhotons, 100000u);
    
    // Brute force search through photons
    for (uint i = 0; i < maxSearch; i++)
    {
        // Early exit if we found enough photons
        if (M >= gMaxGatherPhotons) break;
        
        PackedCausticPhoton packed = gPhotonBuffer[i];
        if (!packed.isValid()) continue;
        
        CausticPhoton photon = packed.unpack();
        
        // Distance check
        float3 diff = photon.position - pos;
        float distSq = dot(diff, diff);
        if (distSq > radiusSq) continue;
        
        // Normal consistency check - photon should hit front face
        float3 lightDir = -photon.incidentDir;
        float NdotL = dot(normal, lightDir);
        if (NdotL <= 0) continue;
        
        // Accumulate flux
        fluxSum += photon.flux;
        M++;
    }
    
    // Simplified density estimation:
    // L = Σ(flux × BRDF) / (π × R²)
    // For Lambertian: BRDF = albedo / π
    // L = Σ(flux) × albedo / (π² × R²)
    // But this seems too dim, so let's try just: L = Σ(flux) × albedo / (π × R²)
    float3 radiance = float3(0);
    if (M > 0 && radiusSq > 0)
    {
        // Use simpler formula - just divide by disk area
        radiance = fluxSum * diffuseAlbedo / (M_PI * radiusSq);
    }
    
    return radiance;
}

/// Trace through glass and gather caustics on the diffuse surface behind
float3 traceGlassCaustics(float3 origin, float3 viewDir, float3 normal, ShadingData glassSd, uint2 pixel)
{
    if (gUseCausticPhotonMapping == 0) return float3(0);
    
    // For glass, trace refracted ray to find diffuse surface behind
    // Use Snell's law for refraction
    float eta = glassSd.eta;  // IOR ratio
    float cosI = dot(-viewDir, normal);
    bool entering = cosI > 0;
    if (!entering) {
        normal = -normal;
        cosI = -cosI;
        eta = 1.0f / eta;
    }
    
    float sinT2 = eta * eta * (1.0f - cosI * cosI);
    if (sinT2 > 1.0f) return float3(0);  // Total internal reflection
    
    float cosT = sqrt(1.0f - sinT2);
    float3 refractDir = eta * viewDir + (eta * cosI - cosT) * normal;
    refractDir = normalize(refractDir);
    
    // Trace refracted ray
    RayDesc ray;
    ray.Origin = origin + refractDir * 0.002f;
    ray.Direction = refractDir;
    ray.TMin = 0.001f;
    ray.TMax = 10.0f;
    
    float3 causticSum = float3(0);
    float3 throughput = float3(1.0f);  // Fresnel transmission would go here
    
    // Simple trace - up to 4 bounces through glass
    for (int bounce = 0; bounce < 4; bounce++)
    {
        SceneRayQuery<0> rayQuery;
        HitInfo hit;
        float hitT;
        if (!rayQuery.traceRay(ray, hit, hitT, RAY_FLAG_NONE, 0xff))
            break;
        
        // Only handle triangle hits
        if (hit.getType() != HitType::Triangle)
            break;
            
        const TriangleHit triangleHit = hit.getTriangleHit();
        VertexData v = gScene.getVertexData(triangleHit);
        uint materialID = gScene.getMaterialID(triangleHit.instanceID);
        ShadingData hitSd = prepareShadingData(v, materialID, gScene.materials[materialID], gScene.materialResources[materialID], -ray.Direction, 0.f);
        
        // If we hit a diffuse surface, gather caustics
        if (hitSd.specularTransmission < 0.01f && hitSd.linearRoughness >= 0.3f)
        {
            causticSum += throughput * gatherAndUpdatePPM(hitSd.posW, hitSd.N, hitSd.diffuse, pixel);
            break;
        }
        
        // If still glass, continue tracing
        if (hitSd.specularTransmission > 0.01f)
        {
            // Simplified: just continue in same direction with some attenuation
            throughput *= 0.95f;
            ray.Origin = hitSd.posW + ray.Direction * 0.002f;
        }
        else
        {
            break;
        }
    }
    
    return causticSum;
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
        // NOTE: 对于玻璃材质（有透射），不要禁用透射lobe，否则玻璃会变暗
        if (sd.linearRoughness < 0.2 && data.pathLength == 1 && sd.specularTransmission < 0.01f)
        {
            
            //sample.Li = 0.f;
            //flag = sd.p
            FalcorBSDF bsdf;
            bsdf.setup(sd);
            flag = bsdf.pDiffuseReflection;
            sd.setActiveLobes((uint) LobeType::Diffuse);
        }

        // PPM Caustic contribution
        // Use Progressive Photon Mapping for caustics on diffuse surfaces
        float3 causticContribution = float3(0);
        if (gUseCausticPhotonMapping)
        {
            // For diffuse surfaces, directly gather caustics
            if (sd.linearRoughness >= 0.3f && sd.specularTransmission < 0.01f)
            {
                causticContribution = gatherAndUpdatePPM(sd.posW, sd.N, sd.diffuse, pixel);
            }
            // NOTE: traceGlassCaustics disabled for now - causes GPU hang
            // For glass surfaces, the caustics are handled by path tracing
        }
        
        // GI contribution (from ReSTIR)
        float3 giContribution = (data.pathPreRadiance + data.pathPreThp * evalBSDFCosine(sd, sample.dir) * sample.Li / flag) / params.numGIInstance;
        
        // Combine GI and caustic contributions
        outputColor[pixel] += giContribution + causticContribution;
    }
};

ParameterBlock<FinalShading> finalShading;

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    finalShading.execute(dispatchThreadId.xy);
}
             