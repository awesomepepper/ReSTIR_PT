############################################################################
# Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
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
###########################################################################

import glob
import os
from falcor import *

# Creates an array of default non-normal mapped materials.
def createDefaultMaterials():
    materials = []

    stdMtl = StandardMaterial('stdMtl')
    stdMtl.baseColor = float4(0.50, 0.70, 0.80, 1.0)
    stdMtl.roughness = 0.5
    materials.append(stdMtl)

    clothMtl = ClothMaterial('clothMtl')
    clothMtl.baseColor = float4(0.31, 0.04, 0.10, 1.0)
    clothMtl.roughness = 0.9
    materials.append(clothMtl)

    hairMtl = HairMaterial('hairMtl')
    hairMtl.baseColor = float4(1.0, 0.85, 0.45, 1.0)
    hairMtl.specularParams = float4(0.75, 0.1, 0.5, 1.0)
    materials.append(hairMtl)

    # Lambertian BRDF with kd=(0.5) stored in the MERL file format
    merlMtl = MERLMaterial('merlMtl', 'data/gray-lambert.binary')
    materials.append(merlMtl)

    return materials

# Creates an array of normal mapped materials.
def createNormalMappedMaterials(transmissive=True):
    materials = []

    stdMtl = StandardMaterial('stdMtl')
    stdMtl.loadTexture(MaterialTextureSlot.Normal, 'textures/checker_tile_normal.png', False)
    stdMtl.baseColor = float4(0.50, 0.70, 0.80, 1.0)
    stdMtl.roughness = 0.5
    materials.append(stdMtl)

    clothMtl = ClothMaterial('clothMtl')
    clothMtl.loadTexture(MaterialTextureSlot.Normal, 'textures/checker_tile_normal.png', False)
    clothMtl.baseColor = float4(0.31, 0.04, 0.10, 1.0)
    clothMtl.roughness = 0.9
    materials.append(clothMtl)

    searchpath = os.path.join(os.path.dirname(__file__), 'data/*.binary')
    brdfs = glob.glob(searchpath)
    merlMixMtl = MERLMixMaterial('merlMixMtl', brdfs)
    merlMixMtl.loadTexture(MaterialTextureSlot.Normal, 'textures/checker_tile_normal.png', False)
    merlMixMtl.loadTexture(MaterialTextureSlot.Index, 'textures/indices.png', False)
    materials.append(merlMixMtl)

    pbrtCoatedConductorMtl = PBRTCoatedConductorMaterial('pbrtCoatedConductorMtl')
    pbrtCoatedConductorMtl.loadTexture(MaterialTextureSlot.Normal, 'textures/checker_tile_normal.png', False)
    materials.append(pbrtCoatedConductorMtl)

    pbrtCoatedDiffuseMtl = PBRTCoatedDiffuseMaterial('pbrtCoatedDiffuseMtl')
    pbrtCoatedDiffuseMtl.loadTexture(MaterialTextureSlot.Normal, 'textures/checker_tile_normal.png', False)
    materials.append(pbrtCoatedDiffuseMtl)

    pbrtConductorMtl = PBRTConductorMaterial('pbrtConductorMtl')
    pbrtConductorMtl.loadTexture(MaterialTextureSlot.Normal, 'textures/checker_tile_normal.png', False)
    materials.append(pbrtConductorMtl)

    pbrtDiffuseMtl = PBRTDiffuseMaterial('pbrtDiffuseMtl')
    pbrtDiffuseMtl.loadTexture(MaterialTextureSlot.Normal, 'textures/checker_tile_normal.png', False)
    materials.append(pbrtDiffuseMtl)

    if transmissive:
        pbrtDielectricMtl = PBRTDielectricMaterial('pbrtDielectricMtl')
        pbrtDielectricMtl.loadTexture(MaterialTextureSlot.Normal, 'textures/checker_tile_normal.png', False)
        materials.append(pbrtDielectricMtl)

        pbrtDiffuseTransmissionMtl = PBRTDiffuseTransmissionMaterial('pbrtDiffuseTransmissionMtl')
        pbrtDiffuseTransmissionMtl.loadTexture(MaterialTextureSlot.Normal, 'textures/checker_tile_normal.png', False)
        materials.append(pbrtDiffuseTransmissionMtl)

    return materials
