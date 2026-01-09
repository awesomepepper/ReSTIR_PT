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
#pragma once
#include "Falcor.h"
#include "Experimental/WorldSpaceReSTIRGI/WorldSpaceReSTIRGI.h"
#include "Utils/Debug/PixelDebug.h"
#include "Utils/Sampling/SampleGenerator.h"
#include "Utils/Algorithm/PrefixSum.h"
#include "Rendering/Lights/EmissiveUniformSampler.h"
#include "Rendering/Lights/EnvMapSampler.h"
#include "Params.slang"


using namespace Falcor;

class WorldSpaceReSTIRGIPass : public RenderPass
{
public:
    using SharedPtr = std::shared_ptr<WorldSpaceReSTIRGIPass>;

    /** Create a new render pass object.
        \param[in] pRenderContext The render context.
        \param[in] dict Dictionary of serialized parameters.
        \return A new object, or an exception is thrown if creation failed.
    */
    static SharedPtr create(RenderContext* pRenderContext = nullptr, const Dictionary& dict = {});

    virtual std::string getDesc() override;
    virtual Dictionary getScriptingDictionary() override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void compile(RenderContext* pRenderContext, const CompileData& compileData) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene) override;
    //virtual bool onMouseEvent(const MouseEvent& mouseEvent) override { return false; }
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override;
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }

private:
    WorldSpaceReSTIRGIPass();

    void UpdateProgram();
    void UpdateResource();
    Program::DefineList GetDefines();
    bool renderDebugUI(Gui::Widgets& widget);

    void PrepareGIData(RenderContext* pRenderContext, const RenderData& renderData);
    void FinalShading(RenderContext* pRenderContext, const RenderData& renderData, uint currentInstance);
    
    // Caustic Photon Mapping
    void TraceCausticPhotons(RenderContext* pRenderContext);
    void BuildPhotonHashGrid(RenderContext* pRenderContext);
    void UpdatePhotonResources();

    ComputePass::SharedPtr mpFinalShadingPass;
    ComputePass::SharedPtr mpReflectTypePass;

    struct RtPass
    {
        RtProgram::SharedPtr mpProgram;
        RtBindingTable::SharedPtr mpBindTable;
        RtProgramVars::SharedPtr mpVars;
    } mPathTracingPass;
    
    // Caustic Photon Tracing Pass
    RtPass mPhotonTracingPass;
    ComputePass::SharedPtr mpBuildPhotonHashGridPass;

    /// <summary>
    /// changed required recompile
    /// </summary>
    struct PathTracerOptions
    {
        bool usedReSTIRDI = false;
        bool usedNEE = true;
        bool usedMIS = true;
        uint maxBounces = 15u;
        // Caustic Photon Mapping options
        bool useCausticPhotonMapping = false;
        uint photonsPerFrame = 200000u;  // Increased for better coverage
        uint maxPhotonBounces = 10u;     // Bounces for glass
        float photonGatherRadius = 0.08f; // Larger radius for more photon hits
        uint maxGatherPhotons = 200u;    // More photons per gather
    } mPtOptions;

    bool mOptionChanged = false;
    bool mRecompile = false;
    bool mNeedRecreateReSTIRGIInstance = false;

    uint numReSTIRInstances = 1u;

    WorldSpaceReSTIRGI::Options::SharedPtr mOptions;
    std::vector<WorldSpaceReSTIRGI::SharedPtr> reSTIRInstances;

    Buffer::SharedPtr mpInitialSample;
    Buffer::SharedPtr mpReconnectionData;
    
    // Caustic Photon Mapping buffers
    Buffer::SharedPtr mpPhotonBuffer;           // Stores caustic photons
    Buffer::SharedPtr mpPhotonAppendBuffer;     // For building hash grid
    Buffer::SharedPtr mpPhotonCellStorage;      // Hash grid cell storage
    Buffer::SharedPtr mpPhotonIndexBuffer;      // Hash grid index buffer
    Buffer::SharedPtr mpPhotonCheckSumBuffer;   // Hash grid checksum
    Buffer::SharedPtr mpPhotonCellCounters;     // Hash grid cell counters
    PrefixSum::SharedPtr mpPhotonPrefixSum;     // For building hash grid

    Scene::SharedPtr mpScene;
    SampleGenerator::SharedPtr mpSampleGenerator;
    EnvMapSampler::SharedPtr mpEnvMapSampler;
    EmissiveLightSampler::SharedPtr mpEmissiveSampler;
    PixelDebug::SharedPtr           mpPixelDebug;               ///< Utility class for pixel debugging (print in shaders).

    PTRuntimeParams params;

    // Caustic Photon CB - must match shader layout exactly (36 bytes)
    struct CausticPhotonCBData
    {
        float3 sceneBBMin;          // 0-11
        float cellSize;             // 12-15
        float gatherRadius;         // 16-19
        uint32_t maxGatherPhotons;  // 20-23
        uint32_t hashTableSize;     // 24-27
        uint32_t totalPhotons;      // 28-31
        uint32_t useCausticPhotonMapping; // 32-35
    };

    uint pad = 0;
};
