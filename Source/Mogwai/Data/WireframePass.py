from falcor import *
import os

def render_graph_WireframePass():
    g = RenderGraph('WireframePass')
    loadRenderPassLibrary('WireframePass.dll')
    Wireframe = createPass('WireframePass')
    g.addPass(Wireframe, 'WireframePass')
    g.markOutput('WireframePass.output')
    return g

WireframePass = render_graph_WireframePass()
try: m.addGraph(WireframePass)
except NameError: None