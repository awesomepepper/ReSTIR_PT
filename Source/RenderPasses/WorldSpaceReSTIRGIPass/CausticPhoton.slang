/***************************************************************************
 # Caustic Photon Data Structure
 # Used for Photon Mapping to capture caustic effects
 **************************************************************************/

__exported import Utils.Math.MathHelpers;
__exported import Utils.Math.PackedFormats;

/// Caustic photon structure - stores photon information at diffuse surfaces
struct CausticPhoton
{
    float3 position;    // World position where photon hit diffuse surface
    float3 normal;      // Surface normal at hit point
    float3 flux;        // Photon flux (power)
    float3 incidentDir; // Incident direction (from light)
    
    // Pack/unpack for storage efficiency
    static CausticPhoton create(float3 pos, float3 norm, float3 flx, float3 dir)
    {
        CausticPhoton p;
        p.position = pos;
        p.normal = norm;
        p.flux = flx;
        p.incidentDir = dir;
        return p;
    }
    
    static CausticPhoton createInvalid()
    {
        CausticPhoton p;
        p.position = float3(0);
        p.normal = float3(0);
        p.flux = float3(0);
        p.incidentDir = float3(0);
        return p;
    }
    
    bool isValid()
    {
        return any(flux > 0);
    }
};

/// Packed caustic photon for compact storage (48 bytes -> 32 bytes)
struct PackedCausticPhoton
{
    float3 position;        // 12 bytes
    uint packedNormal;      // 4 bytes (octahedral encoding)
    float3 flux;            // 12 bytes
    uint packedIncidentDir; // 4 bytes (octahedral encoding)
    
    static PackedCausticPhoton pack(CausticPhoton p)
    {
        PackedCausticPhoton packed;
        packed.position = p.position;
        packed.packedNormal = encodeNormal2x16(p.normal);
        packed.flux = p.flux;
        packed.packedIncidentDir = encodeNormal2x16(p.incidentDir);
        return packed;
    }
    
    CausticPhoton unpack()
    {
        CausticPhoton p;
        p.position = position;
        p.normal = decodeNormal2x16(packedNormal);
        p.flux = flux;
        p.incidentDir = decodeNormal2x16(packedIncidentDir);
        return p;
    }
    
    bool isValid()
    {
        return any(flux > 0);
    }
};

/// Hash grid cell for photon storage
struct PhotonCell
{
    uint photonIndex;   // Index into photon buffer
    uint photonCount;   // Number of photons in this cell
};

/// Append data for building photon hash grid
struct PhotonAppendData
{
    uint isValid;
    uint photonIndex;
    uint cellIdx;
    uint inCellIdx;
};
