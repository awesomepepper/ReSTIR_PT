/***************************************************************************
 # RSM (Reflective Shadow Map) Data Structures
 # For "Real-Time Caustics Using Cascaded Image-Space Photon Tracing"
 **************************************************************************/

import Utils.Math.MathHelpers;

// RSM pixel data structure
struct RSMPixel
{
    float3 position;    // World position
    float3 normal;      // World normal
    float3 flux;        // Light flux at this point
    float depth;        // Depth from light
    
    static RSMPixel create(float3 pos, float3 norm, float3 f, float d)
    {
        RSMPixel p;
        p.position = pos;
        p.normal = norm;
        p.flux = f;
        p.depth = d;
        return p;
    }
    
    static RSMPixel createInvalid()
    {
        RSMPixel p;
        p.position = float3(0);
        p.normal = float3(0);
        p.flux = float3(0);
        p.depth = -1.0f;
        return p;
    }
    
    bool isValid()
    {
        return depth >= 0.0f && any(flux > 0.0f);
    }
};

// Packed RSM pixel for storage (32 bytes)
struct PackedRSMPixel
{
    float4 positionDepth;   // xyz = position, w = depth
    float4 normalFluxR;     // xyz = normal, w = flux.r
    float4 fluxGB_pad;      // xy = flux.gb, zw = padding
    
    static PackedRSMPixel pack(RSMPixel p)
    {
        PackedRSMPixel packed;
        packed.positionDepth = float4(p.position, p.depth);
        packed.normalFluxR = float4(p.normal, p.flux.r);
        packed.fluxGB_pad = float4(p.flux.g, p.flux.b, 0, 0);
        return packed;
    }
    
    static RSMPixel unpack(PackedRSMPixel packed)
    {
        RSMPixel p;
        p.position = packed.positionDepth.xyz;
        p.depth = packed.positionDepth.w;
        p.normal = packed.normalFluxR.xyz;
        p.flux = float3(packed.normalFluxR.w, packed.fluxGB_pad.xy);
        return p;
    }
};

// RSM configuration
struct RSMConfig
{
    float4x4 lightViewProj;     // Light view-projection matrix
    float4x4 lightViewProjInv;  // Inverse for unprojection
    float3 lightPosition;       // Light position in world space
    float lightRadius;          // Light source radius (for area lights)
    float3 lightDirection;      // Light direction (for directional/spot)
    float lightIntensity;       // Light intensity multiplier
    float3 lightColor;          // Light color
    uint rsmResolution;         // RSM texture resolution
    float nearPlane;            // Near plane for light frustum
    float farPlane;             // Far plane for light frustum
    float2 _pad;
};

// Cascade configuration for cascaded RSM
struct CascadeConfig
{
    float4x4 viewProj;
    float nearZ;
    float farZ;
    float2 _pad;
};

static const uint kMaxCascades = 4;
