from falcor import *
import os

def render_graph_PPM_CausticsTest():
    g = RenderGraph("PPM_CausticsTest")
    loadRenderPassLibrary("AccumulatePass.dll")
    loadRenderPassLibrary("GBuffer.dll")
    loadRenderPassLibrary("ToneMapper.dll")
    loadRenderPassLibrary("WorldSpaceReSTIRGIPass.dll")

    # G-Buffer pass for geometry information
    GBufferRT = createPass("GBufferRT", {
        'samplePattern': SamplePattern.Center, 
        'sampleCount': 1, 
        'texLOD': TexLODMode.Mip0, 
        'useAlphaTest': True
    })
    g.addPass(GBufferRT, "GBufferRT")
    
    # WorldSpace ReSTIR GI Pass with PPM caustics enabled
    WorldSpaceReSTIRGIPass = createPass("WorldSpaceReSTIRGIPass")    
    g.addPass(WorldSpaceReSTIRGIPass, "WorldSpaceReSTIRGIPass")
    
    # Accumulation pass for temporal stability
    AccumulatePass = createPass("AccumulatePass", {
        'enableAccumulation': False, 
        'precisionMode': AccumulatePrecision.Double
    })
    g.addPass(AccumulatePass, "AccumulatePass")
    
    # Tone mapping for display
    ToneMapper = createPass("ToneMapper", {
        'autoExposure': False, 
        'exposureCompensation': 0.0, 
        'operator': ToneMapOp.Linear
    })
    g.addPass(ToneMapper, "ToneMapper")
    
    # Connect the passes
    g.addEdge("GBufferRT.vbuffer", "WorldSpaceReSTIRGIPass.vbuffer")    
    g.addEdge("GBufferRT.depth", "WorldSpaceReSTIRGIPass.vDepth")    
    g.addEdge("GBufferRT.faceNormalW", "WorldSpaceReSTIRGIPass.vNormW")    
    
    g.addEdge("WorldSpaceReSTIRGIPass.outputColor", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")
    
    g.markOutput("ToneMapper.dst")

    return g

# Create the render graph
graph_PPM_CausticsTest = render_graph_PPM_CausticsTest()
m.addGraph(graph_PPM_CausticsTest)

# Load the Water Caustics test scene
m.loadScene('E:/ReSTIR/ReSTIR_PT/Media/TestScenes/WaterCausticsTest.pyscene')

# NOTE: PPM settings need to be configured manually in the UI:
# 1. Enable "Caustic Photon Mapping (PPM)" checkbox
# 2. Adjust parameters as needed:
#    - Photons per frame: 100000-300000
#    - Max photon bounces: 10-20
#    - Initial radius: 0.05-0.2
#    - Max gather photons: 300-1000
#    - PPM Alpha: 0.6-0.9