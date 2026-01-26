from falcor import *
import os

# ============== Path Tracing GT Graph ==============
def render_graph_PathTracingGT():
    """Path Tracing Ground Truth for comparison"""
    g = RenderGraph("PathTracingGT")
    loadRenderPassLibrary("AccumulatePass.dll")
    loadRenderPassLibrary("GBuffer.dll")
    loadRenderPassLibrary("ToneMapper.dll")
    loadRenderPassLibrary("ReSTIRPTPass.dll")
    loadRenderPassLibrary("ScreenSpaceReSTIRPass.dll")

    # VBuffer for geometry
    VBufferRT = createPass("VBufferRT", {
        'samplePattern': SamplePattern.Center, 
        'sampleCount': 1, 
        'texLOD': TexLODMode.Mip0, 
        'useAlphaTest': True
    })
    g.addPass(VBufferRT, "VBufferRT")
    
    # ReSTIR PT Pass - used as standard path tracer
    ReSTIRPTPass = createPass("ReSTIRPTPass", {'samplesPerPixel': 1})
    g.addPass(ReSTIRPTPass, "ReSTIRPTPass")
    
    # Screen Space ReSTIR for direct lighting
    ScreenSpaceReSTIRPass = createPass("ScreenSpaceReSTIRPass")    
    g.addPass(ScreenSpaceReSTIRPass, "ScreenSpaceReSTIRPass")
    
    # Accumulation pass for convergence
    AccumulatePass = createPass("AccumulatePass", {
        'enableAccumulation': True, 
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
    g.addEdge("VBufferRT.vbuffer", "ReSTIRPTPass.vbuffer")   
    g.addEdge("VBufferRT.mvec", "ReSTIRPTPass.motionVectors")    
    
    g.addEdge("VBufferRT.vbuffer", "ScreenSpaceReSTIRPass.vbuffer")   
    g.addEdge("VBufferRT.mvec", "ScreenSpaceReSTIRPass.motionVectors")    
    g.addEdge("ScreenSpaceReSTIRPass.color", "ReSTIRPTPass.directLighting")    
    
    g.addEdge("ReSTIRPTPass.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")
    
    g.markOutput("ToneMapper.dst")
    g.markOutput("AccumulatePass.output")

    return g

# ============== PPM Caustics Test Graph ==============
def render_graph_PPM_CausticsTest():
    """WorldSpace ReSTIR GI with PPM Caustics (supports both random and RSM-based)"""
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

# ============== Create and add graphs ==============

# Add Path Tracing GT graph (for ground truth comparison)
graph_PathTracingGT = render_graph_PathTracingGT()
m.addGraph(graph_PathTracingGT)

# Add PPM Caustics Test graph
graph_PPM_CausticsTest = render_graph_PPM_CausticsTest()
m.addGraph(graph_PPM_CausticsTest)

# Load the Water Caustics test scene
m.loadScene('E:/ReSTIR/ReSTIR_PT/Media/TestScenes/WaterCausticsTest.pyscene')

# ============== Usage Instructions ==============
# 
# Two render graphs are available:
#
# 1. "PathTracingGT" - Standard path tracing for ground truth reference
#    - Uses ReSTIR PT with accumulation enabled
#    - Let it run for many frames to converge
#
# 2. "PPM_CausticsTest" - WorldSpace ReSTIR GI with PPM caustics
#    - Enable "Caustic Photon Mapping (PPM)" in UI
#    - Two photon generation methods available:
#
#      A) Random Sampling (default):
#         - Photons per frame: 100000-300000
#         - Traditional random light sampling
#
#      B) RSM-based (Image-Space Photon Tracing):
#         - Check "Use RSM (Image-Space)" checkbox
#         - RSM Resolution: 64-512 (total photons = resolution^2)
#         - Reference: "Real-Time Caustics Using Cascaded Image-Space Photon Tracing"
#         - More efficient photon distribution on visible surfaces
#
#    Common parameters:
#    - Max photon bounces: 10-20
#    - Initial radius: 0.05-0.2
#    - Max gather photons: 300-1000
#    - PPM Alpha: 0.6-0.9
#
# Switch between graphs in the UI to compare results

# ============== RSM Implementation Details ==============
#
# The RSM (Reflective Shadow Map) based method implements:
# "Real-Time Caustics Using Cascaded Image-Space Photon Tracing"
#
# Pipeline:
# 1. RSM Generation Pass:
#    - Traces rays from light source into scene
#    - Stores position, normal, and flux at each hit point
#    - Resolution determines photon count (res^2 photons)
#
# 2. RSM Photon Tracing Pass:
#    - Uses RSM pixels as virtual photon sources
#    - Each RSM pixel emits one photon
#    - Traces photons through specular surfaces (glass)
#    - Stores caustic photons on diffuse surfaces
#
# 3. Photon Gathering (in FinalShading):
#    - Same as traditional PPM
#    - Hash grid for efficient neighbor search
#    - Kernel density estimation for smooth results
#
# Advantages of RSM approach:
# - Photons are pre-filtered to light-visible surfaces
# - Better distribution than random sampling
# - Deterministic photon positions (good for temporal stability)
# - Can be extended to cascaded approach for multi-scale caustics
