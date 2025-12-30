/***************************************************************************
 # Build Photon Hash Grid
 # Organizes photons into spatial hash grid for efficient queries
 **************************************************************************/

import CausticPhoton;

cbuffer PhotonGridCB
{
    uint gTotalPhotons;
};

StructuredBuffer<PhotonAppendData> gPhotonAppendBuffer;
RWByteAddressBuffer gPhotonIndexBuffer;
RWStructuredBuffer<uint> gPhotonCellStorage;

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint photonIndex = dispatchThreadId.x;
    if (photonIndex >= gTotalPhotons) return;
    
    PhotonAppendData appendData = gPhotonAppendBuffer[photonIndex];
    
    if (appendData.isValid == 0) return;
    
    // Get the start index for this cell from prefix sum
    uint cellIdx = appendData.cellIdx;
    uint startIdx = cellIdx > 0 ? gPhotonIndexBuffer.Load((cellIdx - 1) * 4) : 0;
    uint inCellIdx = appendData.inCellIdx;
    
    // Store photon index in cell storage
    uint storageIdx = startIdx + inCellIdx;
    gPhotonCellStorage[storageIdx] = appendData.photonIndex;
}
