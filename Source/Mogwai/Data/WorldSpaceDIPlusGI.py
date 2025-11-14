from falcor import *
import os


def render_graph_WorldSpaceDIPlusGI():
    g = RenderGraph("WorldSpaceDIPlusGI")
    loadRenderPassLibrary("AccumulatePass.dll")
    loadRenderPassLibrary("GBuffer.dll")
    # loadRenderPassLibrary("ReSTIRPTPass.dll")
    loadRenderPassLibrary("ToneMapper.dll")
    loadRenderPassLibrary("ScreenSpaceReSTIRPass.dll")
    loadRenderPassLibrary("WorldSpaceReSTIRGIPass.dll")
    loadRenderPassLibrary("ErrorMeasurePass.dll")
    loadRenderPassLibrary("ImageLoader.dll")

    ReSTIRGIPlusPass = createPass("ReSTIRPTPass", {'samplesPerPixel': 1})
    g.addPass(ReSTIRGIPlusPass, "ReSTIRPTPass")
    # VBufferRT = createPass("VBufferRT", {'samplePattern': SamplePattern.Center, 'sampleCount': 1, 'texLOD': TexLODMode.Mip0, 'useAlphaTest': True})
    # g.addPass(VBufferRT, "VBufferRT")
    # GBufferRaster = createPass("GBufferRaster")
    # g.addPass(GBufferRaster, "GBufferRaster")
    GBufferRT = createPass("GBufferRT", {'samplePattern': SamplePattern.Center, 'sampleCount': 1, 'texLOD': TexLODMode.Mip0, 'useAlphaTest': True})
    g.addPass(GBufferRT, "GBufferRT")
    AccumulatePass = createPass("AccumulatePass", {'enableAccumulation': False, 'precisionMode': AccumulatePrecision.Double})
    g.addPass(AccumulatePass, "AccumulatePass")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0, 'operator': ToneMapOp.Linear})
    g.addPass(ToneMapper, "ToneMapper")
    ScreenSpaceReSTIRPass = createPass("ScreenSpaceReSTIRPass")    
    g.addPass(ScreenSpaceReSTIRPass, "ScreenSpaceReSTIRPass")
    WorldSpaceReSTIRGIPass = createPass("WorldSpaceReSTIRGIPass")    
    g.addPass(WorldSpaceReSTIRGIPass, "WorldSpaceReSTIRGIPass")
    
    # g.addEdge("VBufferRT.vbuffer", "ReSTIRPTPass.vbuffer")   
    # g.addEdge("VBufferRT.mvec", "ReSTIRPTPass.motionVectors")    
    
    g.addEdge("GBufferRT.vbuffer", "WorldSpaceReSTIRGIPass.vbuffer")    
    # g.addEdge("VBufferRT.mvec", "WorldSpaceReSTIRGIPass.motionVectors")    
    g.addEdge("GBufferRT.depth", "WorldSpaceReSTIRGIPass.vDepth")    
    g.addEdge("GBufferRT.faceNormalW", "WorldSpaceReSTIRGIPass.vNormW")    

    g.addEdge("GBufferRT.vbuffer", "ScreenSpaceReSTIRPass.vbuffer")   
    g.addEdge("GBufferRT.mvec", "ScreenSpaceReSTIRPass.motionVectors")    
    g.addEdge("WorldSpaceReSTIRGIPass.outputColor", "ScreenSpaceReSTIRPass.GIColor")
    # g.addEdge("ScreenSpaceReSTIRPass.color", "ReSTIRPTPass.directLighting")    
    
    # g.addEdge("ReSTIRPTPass.color", "AccumulatePass.input")
    # g.addEdge("WorldSpaceReSTIRGIPass.outputColor", "AccumulatePass.input")
    g.addEdge("ScreenSpaceReSTIRPass.color", "AccumulatePass.input")    
    g.addEdge("AccumulatePass.output", "ToneMapper.src")
    
    g.markOutput("ToneMapper.dst")
    g.markOutput("AccumulatePass.output")  

    return g

graph_WorldSpaceDIPlusGI = render_graph_WorldSpaceDIPlusGI()

m.addGraph(graph_WorldSpaceDIPlusGI)
