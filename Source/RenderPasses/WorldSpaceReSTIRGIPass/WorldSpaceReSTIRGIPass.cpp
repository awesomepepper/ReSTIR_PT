/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#include "WorldSpaceReSTIRGIPass.h"
#include <glm/gtc/matrix_transform.hpp>


namespace
{
    const char kDesc[] = "Insert pass description here";    

    const std::string& kShaderMode = "6_5";
    const std::string& kTracePassFilePath = "RenderPasses/WorldSpaceReSTIRGIPass/TracePass.rt.slang";
    const std::string& kFinalShadingFilePath = "RenderPasses/WorldSpaceReSTIRGIPass/FinalShading.cs.slang";
    const std::string& kReflectTypeFilePath = "RenderPasses/WorldSpaceReSTIRGIPass/ReflectTypes.cs.slang";
    const std::string& kLightTracingPassFilePath = "RenderPasses/WorldSpaceReSTIRGIPass/LightTracingPass.rt.slang";
    const std::string& kBuildPhotonHashGridFilePath = "RenderPasses/WorldSpaceReSTIRGIPass/BuildPhotonHashGrid.cs.slang";
    const std::string& kRSMGenerationFilePath = "RenderPasses/WorldSpaceReSTIRGIPass/RSMGeneration.cs.slang";
    const std::string& kRSMPhotonTracingFilePath = "RenderPasses/WorldSpaceReSTIRGIPass/RSMPhotonTracing.rt.slang";

    const std::string& kInputVBuffer = "vbuffer";
    const std::string& kInputDepthBuffer = "vDepth";
    const std::string& kInputNormBuffer = "vNormW";

    const std::string& kOutputColor = "outputColor";

    ChannelList InputChannel
    {
        {kInputVBuffer,"vbuffer","",false,ResourceFormat::Unknown},
        {kInputDepthBuffer,"depth","",false,ResourceFormat::Unknown},
        {kInputNormBuffer,"vNormW","",false,ResourceFormat::Unknown}
    };

    ChannelList OutputChannel
    {
        {kOutputColor,"outputColor","",false,ResourceFormat::RGBA32Float}
    };

    const uint32_t kMaxPayloadSizeBytes = 256u;
}

// Don't remove this. it's required for hot-reload to function properly
extern "C" __declspec(dllexport) const char* getProjDir()
{
    return PROJECT_DIR;
}

extern "C" __declspec(dllexport) void getPasses(Falcor::RenderPassLibrary& lib)
{
    lib.registerClass("WorldSpaceReSTIRGIPass", kDesc, WorldSpaceReSTIRGIPass::create);
}

WorldSpaceReSTIRGIPass::SharedPtr WorldSpaceReSTIRGIPass::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    SharedPtr pPass = SharedPtr(new WorldSpaceReSTIRGIPass);
    return pPass;
}

WorldSpaceReSTIRGIPass::WorldSpaceReSTIRGIPass()
{
    mOptions = WorldSpaceReSTIRGI::Options::create();
    mpPixelDebug = PixelDebug::create();
    mpPhotonPrefixSum = PrefixSum::create();
}

std::string WorldSpaceReSTIRGIPass::getDesc() { return kDesc; }

Dictionary WorldSpaceReSTIRGIPass::getScriptingDictionary()
{
    return Dictionary();
}

RenderPassReflection WorldSpaceReSTIRGIPass::reflect(const CompileData& compileData)
{
    // Define the required resources here
    RenderPassReflection reflector;
    addRenderPassInputs(reflector, InputChannel);
    addRenderPassOutputs(reflector, OutputChannel);
    return reflector;
}

void WorldSpaceReSTIRGIPass::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    params.frameDim = compileData.defaultTexDims;
}

void WorldSpaceReSTIRGIPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    //mpPixelDebug->beginFrame(pRenderContext, params.frameDim);
    //mpPixelDebug->endFrame(pRenderContext);

    auto& dict = renderData.getDictionary();

    const auto& pOutputColor = renderData[kOutputColor]->asTexture();
    assert(pOutputColor);

    if (!mpScene)
    {
        auto clear = [&](const ChannelDesc& channel)
        {
            auto pTex = renderData[channel.name]->asTexture();
            if (pTex) pRenderContext->clearUAV(pTex->getUAV().get(), float4(0.f));
        };
        for (const auto& channel : OutputChannel) clear(channel);
        return;
    }

    if (mOptionChanged)
    {
        auto flags = dict.getValue(kRenderPassRefreshFlags, RenderPassRefreshFlags::None);
        dict[Falcor::kRenderPassRefreshFlags] = flags | Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
        mOptionChanged = false;
    }

    params.frameDim = uint2(pOutputColor->getWidth(), pOutputColor->getHeight());

    mpPixelDebug->beginFrame(pRenderContext, params.frameDim);

    // Trace caustic photons if enabled
    if (mPtOptions.useCausticPhotonMapping && mpScene->useEmissiveLights())
    {
        UpdatePhotonResources();
        
        // Clear PPM statistics on first frame
        if (params.frameCount == 0)
        {
            if (mpPPMStatisticsBuffer) pRenderContext->clearUAV(mpPPMStatisticsBuffer->getUAV().get(), float4(0.f));
            if (mpPPMFluxBuffer) pRenderContext->clearUAV(mpPPMFluxBuffer->getUAV().get(), float4(0.f));
        }
        
        if (mPtOptions.useRSMCaustics)
        {
            // RSM-based photon tracing
            UpdateRSMResources();
            GenerateRSM(pRenderContext);
            TraceRSMPhotons(pRenderContext);
        }
        else
        {
            // Traditional random photon tracing
            TraceCausticPhotons(pRenderContext);
        }
        BuildPhotonHashGrid(pRenderContext);
    }

    for (uint32_t i = 0; i < reSTIRInstances.size(); i++)
    {
        params.currentGIInstance = i;
        UpdateResource();
        UpdateProgram();
        //std::cout << "heer";
        reSTIRInstances[i]->BeginFrame(pRenderContext, params.frameDim);
        PrepareGIData(pRenderContext, renderData);
        //reSTIRInstances[i]->params._pad = float3(pad, 0, 0);
        reSTIRInstances[i]->UpdateReSTIRGI(pRenderContext, mpInitialSample,renderData[kInputNormBuffer]->asTexture(), renderData[kInputDepthBuffer]->asTexture(), mpReconnectionData,renderData[kInputVBuffer]->asTexture());
        FinalShading(pRenderContext, renderData, i);
        reSTIRInstances[i]->EndFrame(pRenderContext);
    }

    params.frameCount++;

    mpPixelDebug->endFrame(pRenderContext);
}

void WorldSpaceReSTIRGIPass::renderUI(Gui::Widgets& widget)
{
    bool staticDirty = false;
    bool runtimeDirty = false;

    if (widget.group("path tracing options"))
    {
        staticDirty |= widget.checkbox("useReSTIRDI", mPtOptions.usedReSTIRDI);
        staticDirty |= widget.checkbox("useNEE", mPtOptions.usedNEE);
        if (mPtOptions.usedNEE)
            staticDirty |= widget.checkbox("useMIS", mPtOptions.usedMIS);
        staticDirty |= widget.var("gibounce", mPtOptions.maxBounces, 1u, 10u);
    }
    
    // Caustic Photon Mapping UI (PPM)
    if (widget.group("Caustic Photon Mapping (PPM)", true))
    {
        runtimeDirty |= widget.checkbox("Enable Caustics", mPtOptions.useCausticPhotonMapping);
        if (mPtOptions.useCausticPhotonMapping)
        {
            runtimeDirty |= widget.checkbox("Use RSM (Image-Space)", mPtOptions.useRSMCaustics);
            widget.tooltip("Use Reflective Shadow Maps for efficient photon generation.\nReference: Real-Time Caustics Using Cascaded Image-Space Photon Tracing");
            
            if (mPtOptions.useRSMCaustics)
            {
                runtimeDirty |= widget.var("RSM Resolution", mPtOptions.rsmResolution, 64u, 512u);
                widget.tooltip("RSM texture resolution. Total photons = resolution^2");
            }
            else
            {
                runtimeDirty |= widget.var("Photons per frame", mPtOptions.photonsPerFrame, 10000u, 2000000u);
            }
            
            runtimeDirty |= widget.var("Max photon bounces", mPtOptions.maxPhotonBounces, 1u, 16u);
            runtimeDirty |= widget.var("Initial radius", mPtOptions.photonInitialRadius, 0.01f, 1.0f);
            runtimeDirty |= widget.var("Max gather photons", mPtOptions.maxGatherPhotons, 50u, 5000u);
            runtimeDirty |= widget.var("PPM Alpha", mPtOptions.ppmAlpha, 0.5f, 0.95f);
        }
    }

    staticDirty |= widget.var("giInstance", numReSTIRInstances, 1u, 6u);

    if (!reSTIRInstances.empty() && reSTIRInstances[0])
    {
        mOptionChanged = reSTIRInstances[0]->renderUI(widget);
        staticDirty |= mOptionChanged;
        for (size_t i = 1; i < reSTIRInstances.size(); i++)
        {
            reSTIRInstances[i]->CopyRecompileState(reSTIRInstances[0]);
        }
    }

    //runtimeDirty |= renderDebugUI(widget);
    //if (auto group = widget.group("Debugging", true))
    //{
    //    //dirty |= group.checkbox("Use fixed seed", mParams.useFixedSeed);
    //    //group.tooltip("Forces a fixed random seed for each frame.\n\n"
    //    //    "This should produce exactly the same image each frame, which can be useful for debugging.");
    //    //if (mParams.useFixedSeed)
    //    //{
    //    //    dirty |= group.var("Seed", mParams.fixedSeed);
    //    //}

    //    mpPixelDebug->renderUI(group);
    //}
    auto group = widget.group("Debugging", true);
    mpPixelDebug->renderUI(group);

    //runtimeDirty |= widget.var("11", pad, 0u, 2u);

    if (staticDirty) mRecompile = true;
    bool dirty = staticDirty || runtimeDirty;
    if (dirty) mOptionChanged = true;
}

void WorldSpaceReSTIRGIPass::setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene)
{
    mpScene = pScene;
    params.frameCount = 0u;

    mPathTracingPass.mpProgram = nullptr;
    mPathTracingPass.mpBindTable = nullptr;
    mPathTracingPass.mpVars = nullptr;

    mpSampleGenerator = SampleGenerator::create(SAMPLE_GENERATOR_UNIFORM);

    if (mpScene->getRenderSettings().useEmissiveLights)
    {
        mpScene->getLightCollection(pRenderContext);
    }

    if (mpScene->useEnvLight())
    {
        mpEnvMapSampler = EnvMapSampler::create(pRenderContext, mpScene->getEnvMap());
    }

    if (mpScene->useEmissiveLights())
    {
        mpEmissiveSampler = EmissiveUniformSampler::create(pRenderContext, mpScene);
    }

    RtProgram::Desc desc;
    desc.addShaderLibrary(kTracePassFilePath);
    desc.setShaderModel(kShaderMode);

    desc.setMaxAttributeSize(mpScene->getRaytracingMaxAttributeSize());
    desc.setMaxPayloadSize(kMaxPayloadSizeBytes);
    desc.setMaxTraceRecursionDepth(1);

    mPathTracingPass.mpBindTable = RtBindingTable::create(1, 1, mpScene->getGeometryCount());
    mPathTracingPass.mpBindTable->setRayGen(desc.addRayGen("RayGen"));
    mPathTracingPass.mpBindTable->setMiss(0, desc.addMiss("ScatterMiss"));

    if (mpScene->hasGeometryType(Scene::GeometryType::TriangleMesh))
    {
        mPathTracingPass.mpBindTable->setHitGroupByType(0, mpScene, Scene::GeometryType::TriangleMesh, desc.addHitGroup("ScatterTriangleClosestHit", "ScatterTriangleAnyHit"));
    }

    if (mpScene->hasGeometryType(Scene::GeometryType::DisplacedTriangleMesh))
    {
        mPathTracingPass.mpBindTable->setHitGroupByType(0, mpScene, Scene::GeometryType::DisplacedTriangleMesh, desc.addHitGroup("ScatterDisplacedTriangleMeshClosestHit", "", "DisplacedTriangleMeshIntersection"));
    }

    auto defines = GetDefines();

    desc.addDefines(defines);
    mPathTracingPass.mpProgram = RtProgram::create(desc);
    mPathTracingPass.mpVars = RtProgramVars::create(mPathTracingPass.mpProgram, mPathTracingPass.mpBindTable);

    mpReflectTypePass = ComputePass::create(Program::Desc(kReflectTypeFilePath).setShaderModel(kShaderMode).csEntry("main"), defines);
    mpFinalShadingPass = ComputePass::create(Program::Desc(kFinalShadingFilePath).setShaderModel(kShaderMode).csEntry("main"), defines);
    mpBuildPhotonHashGridPass = ComputePass::create(Program::Desc(kBuildPhotonHashGridFilePath).setShaderModel(kShaderMode).csEntry("main"), defines);

    // Setup Photon Tracing Pass
    if (mpScene->useEmissiveLights())
    {
        RtProgram::Desc photonDesc;
        photonDesc.addShaderLibrary(kLightTracingPassFilePath);
        photonDesc.setShaderModel(kShaderMode);
        photonDesc.setMaxAttributeSize(mpScene->getRaytracingMaxAttributeSize());
        photonDesc.setMaxPayloadSize(kMaxPayloadSizeBytes);
        photonDesc.setMaxTraceRecursionDepth(1);
        
        mPhotonTracingPass.mpBindTable = RtBindingTable::create(1, 1, mpScene->getGeometryCount());
        mPhotonTracingPass.mpBindTable->setRayGen(photonDesc.addRayGen("PhotonRayGen"));
        mPhotonTracingPass.mpBindTable->setMiss(0, photonDesc.addMiss("PhotonMiss"));
        
        if (mpScene->hasGeometryType(Scene::GeometryType::TriangleMesh))
        {
            mPhotonTracingPass.mpBindTable->setHitGroupByType(0, mpScene, Scene::GeometryType::TriangleMesh, photonDesc.addHitGroup("PhotonClosestHit", "PhotonAnyHit"));
        }
        
        photonDesc.addDefines(defines);
        mPhotonTracingPass.mpProgram = RtProgram::create(photonDesc);
        mPhotonTracingPass.mpVars = RtProgramVars::create(mPhotonTracingPass.mpProgram, mPhotonTracingPass.mpBindTable);
    }

    {
        reSTIRInstances.clear();

        reSTIRInstances.resize(numReSTIRInstances);
        params.numGIInstance = numReSTIRInstances;
        for (uint32_t i = 0; i < numReSTIRInstances; i++)
        {
            reSTIRInstances[i] = WorldSpaceReSTIRGI::create(mpScene, mOptions, i, numReSTIRInstances);
        }
    }
}

bool WorldSpaceReSTIRGIPass::onMouseEvent(const MouseEvent& mouseEvent)
{
    return mpPixelDebug->onMouseEvent(mouseEvent);
}

void WorldSpaceReSTIRGIPass::UpdateProgram()
{
    if (!mRecompile) return;

    auto defines = GetDefines();

    RtProgram::Desc desc = mPathTracingPass.mpProgram->getRtDesc();
    desc.addDefines(defines);
   
    mPathTracingPass.mpProgram = RtProgram::create(desc);
    
    mPathTracingPass.mpVars = RtProgramVars::create(mPathTracingPass.mpProgram, mPathTracingPass.mpBindTable);

    mRecompile = false;

    if (reSTIRInstances.size() != numReSTIRInstances)
    {
        reSTIRInstances.clear();

        reSTIRInstances.resize(numReSTIRInstances);
        params.numGIInstance = numReSTIRInstances;
        for (uint32_t i = 0; i < numReSTIRInstances; i++)
        {
            reSTIRInstances[i] = WorldSpaceReSTIRGI::create(mpScene, mOptions, i, numReSTIRInstances);
        }
    }
}

void WorldSpaceReSTIRGIPass::UpdateResource()
{
    uint32_t elementCount = params.frameDim.x * params.frameDim.y;
    if (!mpInitialSample || mpInitialSample->getElementCount() != elementCount)
    {
        mpInitialSample = Buffer::createStructured(mpReflectTypePass["initialSamples"], elementCount, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess, Buffer::CpuAccess::None, nullptr, false);
    }
    if (!mpReconnectionData || mpReconnectionData->getElementCount() != elementCount)
    {
        mpReconnectionData = Buffer::createStructured(mpReflectTypePass["reconnectionDataBuffer"], elementCount, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess, Buffer::CpuAccess::None, nullptr, false);
    }
}

Program::DefineList WorldSpaceReSTIRGIPass::GetDefines()
{
    Program::DefineList defines = {};

    if (mpSampleGenerator) defines.add(mpSampleGenerator->getDefines());
    if (mpEmissiveSampler) defines.add(mpEmissiveSampler->getDefines());

    if (mpScene)
    {
        defines.add(mpScene->getSceneDefines());
        defines.add("USE_EMISSIVE_LIGHTS", mpScene->useEmissiveLights() ? "1" : "0");
        defines.add("USE_ANALYTIC_LIGHTS", mpScene->useAnalyticLights() ? "1" : "0");
        defines.add("USE_ENV_LIGHT", mpScene->useEnvLight() ? "1" : "0");
    }

    defines.add("USE_RESTIRDI", mPtOptions.usedReSTIRDI ? "1" : "0");
    defines.add("USE_NEE", mPtOptions.usedNEE ? "1" : "0");
    defines.add("USE_MIS", mPtOptions.usedMIS ? "1" : "0");
    defines.add("MAX_GI_BOUNCE", std::to_string(mPtOptions.maxBounces));
    defines.add("GI_ROUGHNESS_THRESHOLD", std::to_string(mOptions->roughnessThreshold));

    return defines;
}

bool WorldSpaceReSTIRGIPass::renderDebugUI(Gui::Widgets& widget)
{
    bool dirty = false;

    if (auto group = widget.group("Debugging", true))
    {
        //dirty |= group.checkbox("Use fixed seed", mParams.useFixedSeed);
        //group.tooltip("Forces a fixed random seed for each frame.\n\n"
        //    "This should produce exactly the same image each frame, which can be useful for debugging.");
        //if (mParams.useFixedSeed)
        //{
        //    dirty |= group.var("Seed", mParams.fixedSeed);
        //}

        mpPixelDebug->renderUI(group);
    }

    return dirty;
}

void WorldSpaceReSTIRGIPass::PrepareGIData(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto vars = mPathTracingPass.mpVars->getRootVar();

    vars["sampleInitializer"]["vbuffer"] = renderData[kInputVBuffer]->asTexture();
    vars["sampleInitializer"]["initialSamples"] = mpInitialSample;
    vars["sampleInitializer"]["outputColor"] = renderData[kOutputColor]->asTexture();
    vars["sampleInitializer"]["reconnectionDataBuffer"] = mpReconnectionData;
    vars["sampleInitializer"]["roughnessThreshold"] = mOptions->roughnessThreshold;

    vars["pathtracer"]["params"].setBlob(params);
    vars["gScene"] = mpScene->getParameterBlock();

    mpPixelDebug->prepareProgram(mPathTracingPass.mpProgram, vars);

    if (mpEnvMapSampler) mpEnvMapSampler->setShaderData(vars["pathtracer"]["envMapSampler"]);
    if (mpEmissiveSampler) mpEmissiveSampler->setShaderData(vars["pathtracer"]["emissiveSampler"]);

    mpScene->raytrace(pRenderContext, mPathTracingPass.mpProgram.get(), mPathTracingPass.mpVars, uint3(params.frameDim.x, params.frameDim.y, 1u));
}

void WorldSpaceReSTIRGIPass::FinalShading(RenderContext* pRenderContext, const RenderData& renderData, uint currentInstance)
{
    auto vars = mpFinalShadingPass->getRootVar();

    vars["finalShading"]["vbuffer"] = renderData[kInputVBuffer]->asTexture();
    vars["finalShading"]["reconnectionDataBuffer"] = mpReconnectionData;

    vars["finalShading"]["outputColor"] = renderData[kOutputColor]->asTexture();

    vars["finalShading"]["params"].setBlob(params);
    vars["finalShading"]["finalSample"] = reSTIRInstances[currentInstance]->mpFinalSample;

    vars["gScene"] = mpScene->getParameterBlock();
    
    // Set PPM caustic photon mapping parameters
    float3 sceneBBMin = mpScene->getSceneBounds().minPoint - float3(0.1f, 0.1f, 0.1f);
    float3 boundingSize = abs(mpScene->getSceneBounds().maxPoint - mpScene->getSceneBounds().minPoint);
    float cellSize = std::max(boundingSize.x, std::max(boundingSize.y, boundingSize.z)) / 80.0f;
    
    CausticPhotonCBData cbData;
    cbData.sceneBBMin = sceneBBMin;
    cbData.cellSize = cellSize;
    cbData.initialRadius = mPtOptions.photonInitialRadius;
    cbData.maxGatherPhotons = mPtOptions.maxGatherPhotons;
    cbData.hashTableSize = 100000u;
    
    // Use RSM resolution if RSM mode is enabled
    uint32_t totalPhotons = mPtOptions.useRSMCaustics ? 
        (mPtOptions.rsmResolution * mPtOptions.rsmResolution) : mPtOptions.photonsPerFrame;
    
    cbData.totalPhotons = totalPhotons;
    cbData.useCausticPhotonMapping = (mPtOptions.useCausticPhotonMapping && mpScene->useEmissiveLights()) ? 1u : 0u;
    cbData.frameIndex = params.frameCount;
    cbData.ppmAlpha = mPtOptions.ppmAlpha;
    cbData.photonsPerFrame = totalPhotons;
    
    vars["CausticPhotonCB"].setBlob(cbData);
    
    // Set photon buffers
    if (mpPhotonBuffer) vars["gPhotonBuffer"] = mpPhotonBuffer;
    if (mpPhotonCellStorage) vars["gPhotonCellStorage"] = mpPhotonCellStorage;
    if (mpPhotonIndexBuffer) vars["gPhotonIndexBuffer"] = mpPhotonIndexBuffer;
    if (mpPhotonCheckSumBuffer) vars["gPhotonCheckSumBuffer"] = mpPhotonCheckSumBuffer;
    
    // Set PPM statistics buffers
    if (mpPPMStatisticsBuffer) vars["gPPMStats"] = mpPPMStatisticsBuffer;
    if (mpPPMFluxBuffer) vars["gPPMFlux"] = mpPPMFluxBuffer;

    mpFinalShadingPass->execute(pRenderContext, uint3(params.frameDim.x, params.frameDim.y, 1u));
}

void WorldSpaceReSTIRGIPass::UpdatePhotonResources()
{
    uint32_t photonCount = mPtOptions.photonsPerFrame;
    
    // Create photon buffer
    if (!mpPhotonBuffer || mpPhotonBuffer->getElementCount() != photonCount)
    {
        mpPhotonBuffer = Buffer::createStructured(sizeof(float) * 8, photonCount, // PackedCausticPhoton size
            Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess,
            Buffer::CpuAccess::None, nullptr, false);
        mpPhotonBuffer->setName("PhotonBuffer");
    }
    
    // Create photon append buffer
    if (!mpPhotonAppendBuffer || mpPhotonAppendBuffer->getElementCount() != photonCount)
    {
        mpPhotonAppendBuffer = Buffer::createStructured(sizeof(uint32_t) * 4, photonCount, // PhotonAppendData size
            Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess,
            Buffer::CpuAccess::None, nullptr, false);
        mpPhotonAppendBuffer->setName("PhotonAppendBuffer");
    }
    
    // Create photon cell storage
    if (!mpPhotonCellStorage || mpPhotonCellStorage->getElementCount() != photonCount)
    {
        mpPhotonCellStorage = Buffer::createStructured(sizeof(uint32_t), photonCount,
            Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess,
            Buffer::CpuAccess::None, nullptr, false);
        mpPhotonCellStorage->setName("PhotonCellStorage");
    }
    
    // Create hash grid buffers
    uint32_t hashBufferSize = 3200000 * sizeof(uint32_t);
    if (!mpPhotonIndexBuffer)
    {
        mpPhotonIndexBuffer = Buffer::create(hashBufferSize,
            Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess,
            Buffer::CpuAccess::None);
        mpPhotonIndexBuffer->setName("PhotonIndexBuffer");
    }
    
    if (!mpPhotonCheckSumBuffer)
    {
        mpPhotonCheckSumBuffer = Buffer::create(hashBufferSize,
            Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess,
            Buffer::CpuAccess::None);
        mpPhotonCheckSumBuffer->setName("PhotonCheckSumBuffer");
    }
    
    if (!mpPhotonCellCounters)
    {
        mpPhotonCellCounters = Buffer::create(hashBufferSize,
            Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess,
            Buffer::CpuAccess::None);
        mpPhotonCellCounters->setName("PhotonCellCounters");
    }
    
    // Create PPM per-pixel statistics buffers
    uint2 frameDim = params.frameDim;
    if (!mpPPMStatisticsBuffer || mpPPMStatisticsBuffer->getWidth() != frameDim.x || mpPPMStatisticsBuffer->getHeight() != frameDim.y)
    {
        mpPPMStatisticsBuffer = Texture::create2D(frameDim.x, frameDim.y, ResourceFormat::RGBA32Float, 1, 1,
            nullptr, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess);
        mpPPMStatisticsBuffer->setName("PPMStatisticsBuffer");
    }
    
    if (!mpPPMFluxBuffer || mpPPMFluxBuffer->getWidth() != frameDim.x || mpPPMFluxBuffer->getHeight() != frameDim.y)
    {
        mpPPMFluxBuffer = Texture::create2D(frameDim.x, frameDim.y, ResourceFormat::RGBA32Float, 1, 1,
            nullptr, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess);
        mpPPMFluxBuffer->setName("PPMFluxBuffer");
    }
}

void WorldSpaceReSTIRGIPass::TraceCausticPhotons(RenderContext* pRenderContext)
{
    PROFILE("TraceCausticPhotons");
    
    if (!mPhotonTracingPass.mpProgram || !mPhotonTracingPass.mpVars) return;
    if (!mpPhotonBuffer) return;
    
    // Clear buffers
    pRenderContext->clearUAV(mpPhotonBuffer->getUAV().get(), uint4(0));
    pRenderContext->clearUAV(mpPhotonCheckSumBuffer->getUAV().get(), uint4(0));
    pRenderContext->clearUAV(mpPhotonCellCounters->getUAV().get(), uint4(0));
    pRenderContext->clearUAV(mpPhotonIndexBuffer->getUAV().get(), uint4(0));
    
    auto vars = mPhotonTracingPass.mpVars->getRootVar();
    
    // Set photon tracing parameters
    float3 sceneBBMin = mpScene->getSceneBounds().minPoint - float3(0.1f, 0.1f, 0.1f);
    float3 boundingSize = abs(mpScene->getSceneBounds().maxPoint - mpScene->getSceneBounds().minPoint);
    float cellSize = std::max(boundingSize.x, std::max(boundingSize.y, boundingSize.z)) / 80.0f;
    
    vars["PhotonTracingCB"]["gPhotonsPerLight"] = mPtOptions.photonsPerFrame;
    vars["PhotonTracingCB"]["gMaxPhotonBounces"] = mPtOptions.maxPhotonBounces;
    vars["PhotonTracingCB"]["gFrameCount"] = params.frameCount;
    vars["PhotonTracingCB"]["gTotalPhotons"] = mPtOptions.photonsPerFrame;
    vars["PhotonTracingCB"]["gSceneBBMin"] = sceneBBMin;
    vars["PhotonTracingCB"]["gCellSize"] = cellSize;
    vars["PhotonTracingCB"]["gHashTableSize"] = 100000u;
    
    // Set buffers
    vars["gPhotonBuffer"] = mpPhotonBuffer;
    vars["gPhotonAppendBuffer"] = mpPhotonAppendBuffer;
    vars["gPhotonCheckSum"] = mpPhotonCheckSumBuffer;
    vars["gPhotonCellCounters"] = mpPhotonCellCounters;
    
    // Set scene and samplers
    vars["gScene"] = mpScene->getParameterBlock();
    if (mpEmissiveSampler) mpEmissiveSampler->setShaderData(vars["gEmissiveSampler"]);
    
    // Dispatch photon tracing
    mpScene->raytrace(pRenderContext, mPhotonTracingPass.mpProgram.get(), mPhotonTracingPass.mpVars, 
        uint3(mPtOptions.photonsPerFrame, 1u, 1u));
    
    // UAV barrier to ensure writes are visible
    pRenderContext->uavBarrier(mpPhotonBuffer.get());
}

void WorldSpaceReSTIRGIPass::BuildPhotonHashGrid(RenderContext* pRenderContext)
{
    PROFILE("BuildPhotonHashGrid");
    
    // Run prefix sum on cell counters to get index buffer
    // Hash table has 100000 * 32 = 3,200,000 cells
    uint32_t numCells = 100000u * 32u;
    pRenderContext->copyBufferRegion(mpPhotonIndexBuffer.get(), 0, mpPhotonCellCounters.get(), 0, numCells * sizeof(uint32_t));
    mpPhotonPrefixSum->execute(pRenderContext, mpPhotonIndexBuffer, numCells);
    
    // Build hash grid
    auto vars = mpBuildPhotonHashGridPass->getRootVar();
    
    // Use RSM resolution if RSM mode is enabled
    uint32_t totalPhotons = mPtOptions.useRSMCaustics ? 
        (mPtOptions.rsmResolution * mPtOptions.rsmResolution) : mPtOptions.photonsPerFrame;
    
    vars["PhotonGridCB"]["gTotalPhotons"] = totalPhotons;
    vars["gPhotonAppendBuffer"] = mpPhotonAppendBuffer;
    vars["gPhotonIndexBuffer"] = mpPhotonIndexBuffer;
    vars["gPhotonCellStorage"] = mpPhotonCellStorage;
    
    uint32_t numGroups = (totalPhotons + 255) / 256;
    mpBuildPhotonHashGridPass->execute(pRenderContext, uint3(numGroups * 256, 1u, 1u));
}

void WorldSpaceReSTIRGIPass::UpdateRSMResources()
{
    uint32_t rsmRes = mPtOptions.rsmResolution;
    uint32_t photonCount = rsmRes * rsmRes;
    
    // Create RSM textures
    if (!mpRSMPosition || mpRSMPosition->getWidth() != rsmRes)
    {
        mpRSMPosition = Texture::create2D(rsmRes, rsmRes, ResourceFormat::RGBA32Float, 1, 1,
            nullptr, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess);
        mpRSMPosition->setName("RSMPosition");
    }
    
    if (!mpRSMNormal || mpRSMNormal->getWidth() != rsmRes)
    {
        mpRSMNormal = Texture::create2D(rsmRes, rsmRes, ResourceFormat::RGBA32Float, 1, 1,
            nullptr, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess);
        mpRSMNormal->setName("RSMNormal");
    }
    
    if (!mpRSMFlux || mpRSMFlux->getWidth() != rsmRes)
    {
        mpRSMFlux = Texture::create2D(rsmRes, rsmRes, ResourceFormat::RGBA32Float, 1, 1,
            nullptr, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess);
        mpRSMFlux->setName("RSMFlux");
    }
    
    // Resize photon buffers for RSM photon count
    if (!mpPhotonBuffer || mpPhotonBuffer->getElementCount() != photonCount)
    {
        mpPhotonBuffer = Buffer::createStructured(sizeof(float) * 8, photonCount,
            Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess,
            Buffer::CpuAccess::None, nullptr, false);
        mpPhotonBuffer->setName("PhotonBuffer");
    }
    
    if (!mpPhotonAppendBuffer || mpPhotonAppendBuffer->getElementCount() != photonCount)
    {
        mpPhotonAppendBuffer = Buffer::createStructured(sizeof(uint32_t) * 4, photonCount,
            Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess,
            Buffer::CpuAccess::None, nullptr, false);
        mpPhotonAppendBuffer->setName("PhotonAppendBuffer");
    }
    
    if (!mpPhotonCellStorage || mpPhotonCellStorage->getElementCount() != photonCount)
    {
        mpPhotonCellStorage = Buffer::createStructured(sizeof(uint32_t), photonCount,
            Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess,
            Buffer::CpuAccess::None, nullptr, false);
        mpPhotonCellStorage->setName("PhotonCellStorage");
    }
}

void WorldSpaceReSTIRGIPass::GenerateRSM(RenderContext* pRenderContext)
{
    PROFILE("GenerateRSM");
    
    if (!mpRSMGenerationPass)
    {
        Program::Desc desc;
        desc.addShaderLibrary(kRSMGenerationFilePath).setShaderModel(kShaderMode).csEntry("main");
        desc.addTypeConformances(mpScene->getTypeConformances());
        
        Program::DefineList defines = mpScene->getSceneDefines();
        mpRSMGenerationPass = ComputePass::create(desc, defines);
    }
    
    if (!mpRSMGenerationPass) return;
    
    auto vars = mpRSMGenerationPass->getRootVar();
    
    // Set RSM generation parameters
    // For Cornell Box area light at (0, 0.549, 0) pointing down
    float4x4 lightView = glm::lookAt(
        float3(0.0f, 0.549f, 0.0f),  // Light position
        float3(0.0f, 0.0f, 0.0f),     // Look at center
        float3(0.0f, 0.0f, 1.0f)      // Up vector
    );
    float4x4 lightProj = glm::perspective(glm::radians(90.0f), 1.0f, 0.01f, 10.0f);
    float4x4 lightViewProj = lightProj * lightView;
    
    vars["RSMGenerationCB"]["gLightViewProj"] = lightViewProj;
    vars["RSMGenerationCB"]["gLightView"] = lightView;
    vars["RSMGenerationCB"]["gLightPosition"] = float3(0.0f, 0.548f, 0.0f);
    vars["RSMGenerationCB"]["gLightIntensity"] = 5.0f;
    vars["RSMGenerationCB"]["gLightColor"] = float3(17.0f, 12.0f, 4.0f);
    vars["RSMGenerationCB"]["gRSMResolution"] = mPtOptions.rsmResolution;
    vars["RSMGenerationCB"]["gLightDirection"] = float3(0.0f, -1.0f, 0.0f);
    vars["RSMGenerationCB"]["gLightArea"] = 0.13f * 0.13f;
    vars["RSMGenerationCB"]["gFrameCount"] = params.frameCount;
    
    // Set output textures
    vars["gRSMPosition"] = mpRSMPosition;
    vars["gRSMNormal"] = mpRSMNormal;
    vars["gRSMFlux"] = mpRSMFlux;
    
    // Set scene
    vars["gScene"] = mpScene->getParameterBlock();
    
    // Dispatch
    uint32_t numGroups = (mPtOptions.rsmResolution + 15) / 16;
    mpRSMGenerationPass->execute(pRenderContext, uint3(numGroups * 16, numGroups * 16, 1u));
    
    // UAV barrier
    pRenderContext->uavBarrier(mpRSMPosition.get());
    pRenderContext->uavBarrier(mpRSMNormal.get());
    pRenderContext->uavBarrier(mpRSMFlux.get());
}

void WorldSpaceReSTIRGIPass::TraceRSMPhotons(RenderContext* pRenderContext)
{
    PROFILE("TraceRSMPhotons");
    
    // Create RSM photon tracing pass if needed
    if (!mRSMPhotonTracingPass.mpProgram)
    {
        Program::DefineList defines = mpScene->getSceneDefines();
        defines.add(mpSampleGenerator->getDefines());
        
        RtProgram::Desc rsmPhotonDesc;
        rsmPhotonDesc.addShaderLibrary(kRSMPhotonTracingFilePath);
        rsmPhotonDesc.setShaderModel(kShaderMode);
        rsmPhotonDesc.setMaxAttributeSize(mpScene->getRaytracingMaxAttributeSize());
        rsmPhotonDesc.setMaxPayloadSize(kMaxPayloadSizeBytes);
        rsmPhotonDesc.setMaxTraceRecursionDepth(1);
        
        mRSMPhotonTracingPass.mpBindTable = RtBindingTable::create(1, 1, mpScene->getGeometryCount());
        mRSMPhotonTracingPass.mpBindTable->setRayGen(rsmPhotonDesc.addRayGen("RSMPhotonRayGen"));
        mRSMPhotonTracingPass.mpBindTable->setMiss(0, rsmPhotonDesc.addMiss("RSMPhotonMiss"));
        
        if (mpScene->hasGeometryType(Scene::GeometryType::TriangleMesh))
        {
            mRSMPhotonTracingPass.mpBindTable->setHitGroupByType(0, mpScene, Scene::GeometryType::TriangleMesh, 
                rsmPhotonDesc.addHitGroup("RSMPhotonClosestHit", "RSMPhotonAnyHit"));
        }
        
        rsmPhotonDesc.addDefines(defines);
        mRSMPhotonTracingPass.mpProgram = RtProgram::create(rsmPhotonDesc);
        mRSMPhotonTracingPass.mpVars = RtProgramVars::create(mRSMPhotonTracingPass.mpProgram, mRSMPhotonTracingPass.mpBindTable);
    }
    
    if (!mRSMPhotonTracingPass.mpProgram || !mRSMPhotonTracingPass.mpVars) return;
    if (!mpPhotonBuffer) return;
    
    uint32_t totalPhotons = mPtOptions.rsmResolution * mPtOptions.rsmResolution;
    
    // Clear buffers
    pRenderContext->clearUAV(mpPhotonBuffer->getUAV().get(), uint4(0));
    pRenderContext->clearUAV(mpPhotonCheckSumBuffer->getUAV().get(), uint4(0));
    pRenderContext->clearUAV(mpPhotonCellCounters->getUAV().get(), uint4(0));
    pRenderContext->clearUAV(mpPhotonIndexBuffer->getUAV().get(), uint4(0));
    
    auto vars = mRSMPhotonTracingPass.mpVars->getRootVar();
    
    // Set RSM photon tracing parameters
    float3 sceneBBMin = mpScene->getSceneBounds().minPoint - float3(0.1f, 0.1f, 0.1f);
    float3 boundingSize = abs(mpScene->getSceneBounds().maxPoint - mpScene->getSceneBounds().minPoint);
    float cellSize = std::max(boundingSize.x, std::max(boundingSize.y, boundingSize.z)) / 80.0f;
    
    vars["RSMPhotonTracingCB"]["gRSMResolution"] = mPtOptions.rsmResolution;
    vars["RSMPhotonTracingCB"]["gMaxPhotonBounces"] = mPtOptions.maxPhotonBounces;
    vars["RSMPhotonTracingCB"]["gFrameCount"] = params.frameCount;
    vars["RSMPhotonTracingCB"]["gTotalPhotons"] = totalPhotons;
    vars["RSMPhotonTracingCB"]["gSceneBBMin"] = sceneBBMin;
    vars["RSMPhotonTracingCB"]["gCellSize"] = cellSize;
    vars["RSMPhotonTracingCB"]["gHashTableSize"] = 100000u;
    vars["RSMPhotonTracingCB"]["gPhotonFluxScale"] = 1.0f;
    
    // Set RSM textures
    vars["gRSMPosition"] = mpRSMPosition;
    vars["gRSMNormal"] = mpRSMNormal;
    vars["gRSMFlux"] = mpRSMFlux;
    
    // Set output buffers
    vars["gPhotonBuffer"] = mpPhotonBuffer;
    vars["gPhotonAppendBuffer"] = mpPhotonAppendBuffer;
    vars["gPhotonCheckSum"] = mpPhotonCheckSumBuffer;
    vars["gPhotonCellCounters"] = mpPhotonCellCounters;
    
    // Set scene
    vars["gScene"] = mpScene->getParameterBlock();
    
    // Dispatch photon tracing (one thread per RSM pixel)
    mpScene->raytrace(pRenderContext, mRSMPhotonTracingPass.mpProgram.get(), mRSMPhotonTracingPass.mpVars, 
        uint3(totalPhotons, 1u, 1u));
    
    // UAV barrier
    pRenderContext->uavBarrier(mpPhotonBuffer.get());
}





