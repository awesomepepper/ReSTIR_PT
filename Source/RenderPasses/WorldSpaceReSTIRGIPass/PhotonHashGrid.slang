/***************************************************************************
 # Photon Hash Grid for Caustic Photon Mapping
 # Spatial data structure for efficient photon queries
 **************************************************************************/

import Utils.Math.HashUtils;
import CausticPhoton;

/// Photon hash grid parameters
struct PhotonGridParams
{
    float3 sceneBBMin;      // Scene bounding box minimum
    float cellSize;         // Grid cell size
    uint hashTableSize;     // Size of hash table
    uint maxPhotonsPerCell; // Maximum photons per cell
    float gatherRadius;     // Radius for photon gathering
    uint maxGatherPhotons;  // Maximum photons to gather
};

/// PCG random number generator
uint photonPcg32(uint input)
{
    uint state = input * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

/// Compute cell index from position
int3 positionToCell(float3 pos, float3 sceneBBMin, float cellSize)
{
    return int3(floor((pos - sceneBBMin) / cellSize));
}

/// Compute hash from cell coordinates
uint cellToHash(int3 cell, uint hashTableSize)
{
    uint h = photonPcg32(uint(cell.x) + photonPcg32(uint(cell.y) + photonPcg32(uint(cell.z))));
    return h % hashTableSize;
}

/// Compute checksum for collision detection
uint cellToChecksum(int3 cell)
{
    return max(jenkinsHash(uint(cell.x) + jenkinsHash(uint(cell.y) + jenkinsHash(uint(cell.z)))), 1u);
}

/// Find or insert a cell in the hash grid (for building)
int findOrInsertPhotonCell(float3 pos, PhotonGridParams gridParams, RWByteAddressBuffer checkSumBuffer, RWByteAddressBuffer cellCounters)
{
    int3 cell = positionToCell(pos, gridParams.sceneBBMin, gridParams.cellSize);
    uint cellIndex = cellToHash(cell, gridParams.hashTableSize);
    uint checkSum = cellToChecksum(cell);
    
    // Linear probing with 32 slots
    for (uint i = 0; i < 32; i++)
    {
        uint idx = (cellIndex * 32 + i) * 4; // byte offset
        uint checkSumPre;
        
        checkSumBuffer.InterlockedCompareExchange(idx, 0, checkSum, checkSumPre);
        if (checkSumPre == 0 || checkSumPre == checkSum)
        {
            return int(cellIndex * 32 + i);
        }
    }
    
    return -1;
}

/// Find a cell in the hash grid (for querying)
int findPhotonCell(float3 pos, PhotonGridParams gridParams, ByteAddressBuffer checkSumBuffer)
{
    int3 cell = positionToCell(pos, gridParams.sceneBBMin, gridParams.cellSize);
    uint cellIndex = cellToHash(cell, gridParams.hashTableSize);
    uint checkSum = cellToChecksum(cell);
    
    for (uint i = 0; i < 32; i++)
    {
        uint idx = (cellIndex * 32 + i) * 4;
        if (checkSumBuffer.Load(idx) == checkSum)
        {
            return int(cellIndex * 32 + i);
        }
    }
    
    return -1;
}

/// Gather photons within radius using hash grid
struct PhotonGatherer
{
    PhotonGridParams gridParams;
    ByteAddressBuffer checkSumBuffer;
    ByteAddressBuffer indexBuffer;
    StructuredBuffer<PackedCausticPhoton> photonBuffer;
    StructuredBuffer<uint> cellStorage;
    
    /// Gather caustic contribution at a point
    float3 gatherCaustics(float3 pos, float3 normal, float3 viewDir)
    {
        float3 causticRadiance = float3(0);
        float radiusSq = gridParams.gatherRadius * gridParams.gatherRadius;
        uint photonCount = 0;
        
        // Search neighboring cells
        int3 baseCell = positionToCell(pos, gridParams.sceneBBMin, gridParams.cellSize);
        int searchRadius = int(ceil(gridParams.gatherRadius / gridParams.cellSize));
        
        for (int dx = -searchRadius; dx <= searchRadius; dx++)
        {
            for (int dy = -searchRadius; dy <= searchRadius; dy++)
            {
                for (int dz = -searchRadius; dz <= searchRadius; dz++)
                {
                    int3 neighborCell = baseCell + int3(dx, dy, dz);
                    float3 cellPos = float3(neighborCell) * gridParams.cellSize + gridParams.sceneBBMin;
                    
                    int cellIdx = findPhotonCellByCoord(neighborCell);
                    if (cellIdx < 0) continue;
                    
                    // Get photon range for this cell
                    uint startIdx = cellIdx > 0 ? indexBuffer.Load((cellIdx - 1) * 4) : 0;
                    uint endIdx = indexBuffer.Load(cellIdx * 4);
                    
                    for (uint i = startIdx; i < endIdx && photonCount < gridParams.maxGatherPhotons; i++)
                    {
                        uint photonIdx = cellStorage[i];
                        PackedCausticPhoton packed = photonBuffer[photonIdx];
                        
                        if (!packed.isValid()) continue;
                        
                        CausticPhoton photon = packed.unpack();
                        
                        // Distance check
                        float3 diff = photon.position - pos;
                        float distSq = dot(diff, diff);
                        if (distSq > radiusSq) continue;
                        
                        // Normal check - photon should be on same surface
                        if (dot(photon.normal, normal) < 0.8) continue;
                        
                        // Cone filter kernel (better than box filter)
                        float dist = sqrt(distSq);
                        float k = 1.0 - dist / gridParams.gatherRadius;
                        float weight = k * k; // Cone filter
                        
                        // BRDF evaluation (assuming diffuse surface for caustics)
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
    
    int findPhotonCellByCoord(int3 cell)
    {
        uint cellIndex = cellToHash(cell, gridParams.hashTableSize);
        uint checkSum = cellToChecksum(cell);
        
        for (uint i = 0; i < 32; i++)
        {
            uint idx = (cellIndex * 32 + i) * 4;
            if (checkSumBuffer.Load(idx) == checkSum)
            {
                return int(cellIndex * 32 + i);
            }
        }
        
        return -1;
    }
};
