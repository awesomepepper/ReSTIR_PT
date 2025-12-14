import Scene.HitInfo;

struct ReconnectionData
{
    PackedHitInfo preRcVertexHitInfo;
    float3 pathPreThp; /// path pre thp indicates path throughput before the preRcVertex
    float3 pathPreRadiance;
    float3 preRcVertexWo; // the outgoing direction at the preRcVertex, also the direction to the Vertex before the preRcVertex
    uint pathLength;
};
