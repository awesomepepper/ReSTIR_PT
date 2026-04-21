"""
Convert Tungsten 'water-caustic' scene to Falcor pyscene format.

This script:
1. Parses .wo3 binary mesh files and converts them to .obj format
2. Generates .obj files for quad/cube primitives with baked transforms
3. Generates the WaterCaustic.pyscene file for Falcor
4. Copies the reference render files (EXR/PNG)
"""

import struct
import os
import shutil
import math

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'water-caustic')
DST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'water-caustic-pyscene')

# ============================================================
# Vector math utilities
# ============================================================

def cross(a, b):
    return (
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    )

def normalize(v):
    length = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    if length > 1e-12:
        return (v[0]/length, v[1]/length, v[2]/length)
    return (0, 0, 0)

def sub(a, b):
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])

def euler_to_rotation_matrix(rx_deg, ry_deg, rz_deg):
    """ZYX Euler rotation matrix: R = Rz * Ry * Rx (Tungsten convention)."""
    rx = math.radians(rx_deg)
    ry = math.radians(ry_deg)
    rz = math.radians(rz_deg)

    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    return [
        [cy*cz, cz*sx*sy - cx*sz, cx*cz*sy + sx*sz],
        [cy*sz, cx*cz + sx*sy*sz, cx*sy*sz - cz*sx],
        [-sy,   cy*sx,            cx*cy            ]
    ]

def mat_vec(M, v):
    return (
        M[0][0]*v[0] + M[0][1]*v[1] + M[0][2]*v[2],
        M[1][0]*v[0] + M[1][1]*v[1] + M[1][2]*v[2],
        M[2][0]*v[0] + M[2][1]*v[1] + M[2][2]*v[2],
    )

def transform_point(p, scale, rotation_deg, position):
    """Apply Tungsten transform order: Scale -> Rotate -> Translate."""
    R = euler_to_rotation_matrix(*rotation_deg)
    sp = (p[0]*scale[0], p[1]*scale[1], p[2]*scale[2])
    rp = mat_vec(R, sp)
    return (rp[0]+position[0], rp[1]+position[1], rp[2]+position[2])

def transform_normal(n, scale, rotation_deg):
    """Transform normal via inverse transpose: (M^-1)^T = R * S^-1."""
    R = euler_to_rotation_matrix(*rotation_deg)
    si = tuple(1.0/s if abs(s) > 1e-12 else 0.0 for s in scale)
    sn = (n[0]*si[0], n[1]*si[1], n[2]*si[2])
    rn = mat_vec(R, sn)
    return normalize(rn)

# ============================================================
# WO3 binary mesh parser
# ============================================================

def parse_wo3(filepath):
    """
    Parse Tungsten .wo3 binary mesh format.
    Layout:
      - uint32 vertex_count
      - uint32 unused (always 0)
      - Per vertex (32 bytes): float x, y, z, nx, ny, nz, u, v
      - uint32 triangle_count
      - Per triangle (16 bytes): uint32 material_id, v0, v1, v2
      - uint32 trailing (ignored)
    """
    with open(filepath, 'rb') as f:
        data = f.read()

    vertex_count = struct.unpack('<I', data[0:4])[0]

    vertices = []
    for i in range(vertex_count):
        offset = 8 + i * 32
        x, y, z, nx, ny, nz, u, v = struct.unpack('<8f', data[offset:offset+32])
        vertices.append({'pos': (x, y, z), 'normal': (nx, ny, nz), 'uv': (u, v)})

    tri_offset = 8 + vertex_count * 32
    tri_count = struct.unpack('<I', data[tri_offset:tri_offset+4])[0]

    triangles = []
    for i in range(tri_count):
        offset = tri_offset + 4 + i * 16
        mat, v0, v1, v2 = struct.unpack('<4I', data[offset:offset+16])
        triangles.append((v0, v1, v2))

    return vertices, triangles

# ============================================================
# OBJ writer
# ============================================================

def write_obj(filepath, vertices, triangles):
    with open(filepath, 'w') as f:
        for v in vertices:
            p = v['pos']
            f.write(f'v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n')
        for v in vertices:
            n = v['normal']
            f.write(f'vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n')
        for v in vertices:
            uv = v['uv']
            f.write(f'vt {uv[0]:.6f} {uv[1]:.6f}\n')
        for tri in triangles:
            i0, i1, i2 = tri[0]+1, tri[1]+1, tri[2]+1
            f.write(f'f {i0}/{i0}/{i0} {i1}/{i1}/{i1} {i2}/{i2}/{i2}\n')

# ============================================================
# Normal recomputation
# ============================================================

def recompute_smooth_normals(vertices, triangles):
    """Area-weighted smooth normals for smooth shading."""
    normal_accum = [[0.0, 0.0, 0.0] for _ in range(len(vertices))]

    for i0, i1, i2 in triangles:
        v0 = vertices[i0]['pos']
        v1 = vertices[i1]['pos']
        v2 = vertices[i2]['pos']
        e1 = sub(v1, v0)
        e2 = sub(v2, v0)
        n = cross(e1, e2)  # area-weighted (not normalized)
        for vi in [i0, i1, i2]:
            normal_accum[vi][0] += n[0]
            normal_accum[vi][1] += n[1]
            normal_accum[vi][2] += n[2]

    new_vertices = []
    for i, v in enumerate(vertices):
        n = normalize(tuple(normal_accum[i]))
        new_vertices.append({'pos': v['pos'], 'normal': n, 'uv': v['uv']})

    return new_vertices, triangles

def recompute_flat_normals(vertices, triangles):
    """Per-face normals for flat shading. Duplicates vertices at face boundaries."""
    new_vertices = []
    new_triangles = []

    for i0, i1, i2 in triangles:
        v0 = vertices[i0]['pos']
        v1 = vertices[i1]['pos']
        v2 = vertices[i2]['pos']
        e1 = sub(v1, v0)
        e2 = sub(v2, v0)
        n = normalize(cross(e1, e2))

        base_idx = len(new_vertices)
        for vi in [i0, i1, i2]:
            new_vertices.append({
                'pos': vertices[vi]['pos'],
                'normal': n,
                'uv': vertices[vi]['uv']
            })
        new_triangles.append((base_idx, base_idx+1, base_idx+2))

    return new_vertices, new_triangles

# ============================================================
# Geometry generators
# ============================================================

def apply_transform_to_vertices(vertices, scale, rotation_deg, position):
    new_vertices = []
    for v in vertices:
        pos = transform_point(v['pos'], scale, rotation_deg, position)
        normal = transform_normal(v['normal'], scale, rotation_deg)
        new_vertices.append({'pos': pos, 'normal': normal, 'uv': v['uv']})
    return new_vertices

def generate_quad(scale, rotation_deg, position):
    """Tungsten-style unit quad on XZ plane: (-0.5,0,-0.5) to (0.5,0,0.5)."""
    base_verts = [(-0.5, 0, -0.5), (0.5, 0, -0.5),
                  (0.5, 0, 0.5),   (-0.5, 0, 0.5)]
    normal = (0, 1, 0)
    uvs = [(0,0), (1,0), (1,1), (0,1)]

    vertices = [{'pos': v, 'normal': normal, 'uv': uvs[i]}
                for i, v in enumerate(base_verts)]
    vertices = apply_transform_to_vertices(vertices, scale, rotation_deg, position)
    triangles = [(0, 1, 2), (0, 2, 3)]
    return vertices, triangles

def generate_cube(scale, rotation_deg, position):
    """Unit cube from (-0.5,-0.5,-0.5) to (0.5,0.5,0.5), 6 faces, 24 verts."""
    faces = [
        {'verts': [(-0.5,-0.5, 0.5),( 0.5,-0.5, 0.5),( 0.5, 0.5, 0.5),(-0.5, 0.5, 0.5)], 'normal': ( 0, 0, 1)},
        {'verts': [( 0.5,-0.5,-0.5),(-0.5,-0.5,-0.5),(-0.5, 0.5,-0.5),( 0.5, 0.5,-0.5)], 'normal': ( 0, 0,-1)},
        {'verts': [( 0.5,-0.5, 0.5),( 0.5,-0.5,-0.5),( 0.5, 0.5,-0.5),( 0.5, 0.5, 0.5)], 'normal': ( 1, 0, 0)},
        {'verts': [(-0.5,-0.5,-0.5),(-0.5,-0.5, 0.5),(-0.5, 0.5, 0.5),(-0.5, 0.5,-0.5)], 'normal': (-1, 0, 0)},
        {'verts': [(-0.5, 0.5, 0.5),( 0.5, 0.5, 0.5),( 0.5, 0.5,-0.5),(-0.5, 0.5,-0.5)], 'normal': ( 0, 1, 0)},
        {'verts': [(-0.5,-0.5,-0.5),( 0.5,-0.5,-0.5),( 0.5,-0.5, 0.5),(-0.5,-0.5, 0.5)], 'normal': ( 0,-1, 0)},
    ]

    vertices = []
    triangles = []
    uvs = [(0,0),(1,0),(1,1),(0,1)]

    for face in faces:
        base_idx = len(vertices)
        for i, v in enumerate(face['verts']):
            vertices.append({'pos': v, 'normal': face['normal'], 'uv': uvs[i]})
        triangles.append((base_idx, base_idx+1, base_idx+2))
        triangles.append((base_idx, base_idx+2, base_idx+3))

    vertices = apply_transform_to_vertices(vertices, scale, rotation_deg, position)
    return vertices, triangles

# ============================================================
# Main conversion logic
# ============================================================

def main():
    os.makedirs(os.path.join(DST_DIR, 'models'), exist_ok=True)

    # --------------------------------------------------------
    # 1. Convert .wo3 meshes (with scene transforms baked in)
    # --------------------------------------------------------
    print('=== Converting .wo3 meshes ===')

    # Mesh001.wo3: Water surface (smooth=true, recompute_normals=true, scale=(1,1.5,1))
    verts, tris = parse_wo3(os.path.join(SRC_DIR, 'models', 'Mesh001.wo3'))
    print(f'Mesh001.wo3: {len(verts)} vertices, {len(tris)} triangles (raw)')
    verts = apply_transform_to_vertices(verts, (1, 1.5, 1), (0, 0, 0), (0, 0, 0))
    verts, tris = recompute_smooth_normals(verts, tris)
    write_obj(os.path.join(DST_DIR, 'models', 'WaterSurface.obj'), verts, tris)
    print(f'  -> WaterSurface.obj: {len(verts)} vertices, {len(tris)} triangles')

    # Mesh000.wo3: Water bottom/volume (smooth=false, recompute_normals=true, scale=(1,1.5,1))
    verts, tris = parse_wo3(os.path.join(SRC_DIR, 'models', 'Mesh000.wo3'))
    print(f'Mesh000.wo3: {len(verts)} vertices, {len(tris)} triangles (raw)')
    verts = apply_transform_to_vertices(verts, (1, 1.5, 1), (0, 0, 0), (0, 0, 0))
    verts, tris = recompute_flat_normals(verts, tris)
    write_obj(os.path.join(DST_DIR, 'models', 'WaterBottom.obj'), verts, tris)
    print(f'  -> WaterBottom.obj: {len(verts)} vertices, {len(tris)} triangles')

    # --------------------------------------------------------
    # 2. Generate quad/cube primitives with baked transforms
    # --------------------------------------------------------
    print('\n=== Generating primitive geometry ===')

    primitives = {
        'Floor':     {'type': 'quad', 'scale': (2,4,2),                           'rotation': (0,90,0),        'position': (0,0,0)},
        'Ceiling':   {'type': 'quad', 'scale': (2,4,2),                           'rotation': (0,0,-180),      'position': (0,2,0)},
        'BackWall':  {'type': 'quad', 'scale': (2,4,2),                           'rotation': (0,90,90),       'position': (0,1,-1)},
        'RightWall': {'type': 'quad', 'scale': (2,4,2),                           'rotation': (0,180,90),      'position': (1,1,0)},
        'LeftWall':  {'type': 'quad', 'scale': (2,4,2),                           'rotation': (0,0,90),        'position': (-1,1,0)},
        'ShortBox':  {'type': 'cube', 'scale': (0.319977,0.325132,0.322768),      'rotation': (90,90,-66.034), 'position': (0.5132,0.15215,0.44471)},
        'TallBox':   {'type': 'cube', 'scale': (0.34664,0.341188,0.684957),       'rotation': (90,-180,157.627), 'position': (-0.528405,0.335942,-0.291415)},
        'Light':     {'type': 'quad', 'scale': (0.005,1,0.004),                   'rotation': (0,0,-180),      'position': (-0.005,1.98,-0.03)},
    }

    for name, prim in primitives.items():
        filepath = os.path.join(DST_DIR, 'models', f'{name}.obj')
        if prim['type'] == 'quad':
            verts, tris = generate_quad(prim['scale'], prim['rotation'], prim['position'])
        else:
            verts, tris = generate_cube(prim['scale'], prim['rotation'], prim['position'])
        write_obj(filepath, verts, tris)
        print(f'  {name}.obj: {len(verts)} verts, {len(tris)} tris')

    # --------------------------------------------------------
    # 3. Copy reference renders (EXR + PNG)
    # --------------------------------------------------------
    print('\n=== Copying reference files ===')
    for fname in ['TungstenRender.exr', 'TungstenRender.png']:
        src = os.path.join(SRC_DIR, fname)
        dst = os.path.join(DST_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f'  Copied {fname}')

    # --------------------------------------------------------
    # 4. Compute light emissive radiance from Tungsten power
    # --------------------------------------------------------
    # Tungsten "power" = total radiant flux (watts)
    # Falcor "emissiveColor" = radiance (W/sr/m^2)
    # Conversion: radiance = power / (area * pi)
    power = (34.0, 24.0, 8.0)
    quad_area = 0.005 * 0.004  # base quad (1x1) scaled by sx*sz
    radiance = tuple(p / (quad_area * math.pi) for p in power)
    print(f'\nLight conversion:')
    print(f'  Power:    ({power[0]}, {power[1]}, {power[2]}) W')
    print(f'  Area:     {quad_area} m^2')
    print(f'  Radiance: ({radiance[0]:.1f}, {radiance[1]:.1f}, {radiance[2]:.1f}) W/sr/m^2')

    # --------------------------------------------------------
    # 5. Compute camera focal length from Tungsten FOV
    # --------------------------------------------------------
    tungsten_fov = 19.5  # degrees (horizontal = vertical for 1024x1024)
    sensor_half_height = 12.0  # mm (Falcor default: 24mm sensor height)
    focal_length = sensor_half_height / math.tan(math.radians(tungsten_fov / 2.0))
    print(f'\nCamera: FOV={tungsten_fov}° -> focalLength={focal_length:.4f}mm')

    # --------------------------------------------------------
    # 6. Generate WaterCaustic.pyscene
    # --------------------------------------------------------
    print('\n=== Generating WaterCaustic.pyscene ===')

    pyscene_content = f'''# Generated WaterCaustic pyscene
# Converted from Tungsten water-caustic scene (scene.json)
# Original: Cornell box with water surface producing caustics
# Reference render: TungstenRender.exr / TungstenRender.png

# ============================================================
# Camera
# ============================================================
camera = Camera()
camera.position = float3(0.0, 0.990944, 6.83879)
camera.target = float3(0.0, 0.990944, 0.0)
camera.up = float3(0, 1, 0)
camera.focalLength = {focal_length:.4f}
sceneBuilder.addCamera(camera)

# ============================================================
# Materials
# ============================================================

# Left wall - red diffuse
LeftWall = Material('LeftWall')
LeftWall.baseColor = float4(0.63, 0.065, 0.05, 1)
LeftWall.roughness = 1.0

# Right wall - green diffuse
RightWall = Material('RightWall')
RightWall.baseColor = float4(0.14, 0.45, 0.091, 1)
RightWall.roughness = 1.0

# Floor - grey diffuse
Floor = Material('Floor')
Floor.baseColor = float4(0.725, 0.71, 0.68, 1)
Floor.roughness = 1.0

# Ceiling - grey diffuse
Ceiling = Material('Ceiling')
Ceiling.baseColor = float4(0.725, 0.71, 0.68, 1)
Ceiling.roughness = 1.0

# Back wall - grey diffuse
BackWall = Material('BackWall')
BackWall.baseColor = float4(0.725, 0.71, 0.68, 1)
BackWall.roughness = 1.0

# Short box - grey diffuse
ShortBox = Material('ShortBox')
ShortBox.baseColor = float4(0.725, 0.71, 0.68, 1)
ShortBox.roughness = 1.0

# Tall box - grey diffuse
TallBox = Material('TallBox')
TallBox.baseColor = float4(0.725, 0.71, 0.68, 1)
TallBox.roughness = 1.0

# Area light - emissive (radiance computed from Tungsten power / area / pi)
Light = Material('Light')
Light.baseColor = float4(0, 0, 0, 1)
Light.emissiveColor = float3({radiance[0]:.1f}, {radiance[1]:.1f}, {radiance[2]:.1f})

# Water - dielectric with refraction (glass-like, ior=1.8)
Water = Material('Water')
Water.specularTransmission = 1.0
Water.indexOfRefraction = 1.8
Water.roughness = 0.0

# ============================================================
# Geometry - all transforms baked into OBJ files
# ============================================================

# --- Cornell box walls ---

obj = TriangleMesh.createFromFile('models/Floor.obj')
sceneBuilder.addMeshInstance(
    sceneBuilder.addTriangleMesh(obj, Floor),
    sceneBuilder.addNode('Floor', Transform())
)

obj = TriangleMesh.createFromFile('models/Ceiling.obj')
sceneBuilder.addMeshInstance(
    sceneBuilder.addTriangleMesh(obj, Ceiling),
    sceneBuilder.addNode('Ceiling', Transform())
)

obj = TriangleMesh.createFromFile('models/BackWall.obj')
sceneBuilder.addMeshInstance(
    sceneBuilder.addTriangleMesh(obj, BackWall),
    sceneBuilder.addNode('BackWall', Transform())
)

obj = TriangleMesh.createFromFile('models/RightWall.obj')
sceneBuilder.addMeshInstance(
    sceneBuilder.addTriangleMesh(obj, RightWall),
    sceneBuilder.addNode('RightWall', Transform())
)

obj = TriangleMesh.createFromFile('models/LeftWall.obj')
sceneBuilder.addMeshInstance(
    sceneBuilder.addTriangleMesh(obj, LeftWall),
    sceneBuilder.addNode('LeftWall', Transform())
)

# --- Boxes ---

obj = TriangleMesh.createFromFile('models/ShortBox.obj')
sceneBuilder.addMeshInstance(
    sceneBuilder.addTriangleMesh(obj, ShortBox),
    sceneBuilder.addNode('ShortBox', Transform())
)

obj = TriangleMesh.createFromFile('models/TallBox.obj')
sceneBuilder.addMeshInstance(
    sceneBuilder.addTriangleMesh(obj, TallBox),
    sceneBuilder.addNode('TallBox', Transform())
)

# --- Light ---

obj = TriangleMesh.createFromFile('models/Light.obj')
sceneBuilder.addMeshInstance(
    sceneBuilder.addTriangleMesh(obj, Light),
    sceneBuilder.addNode('Light', Transform())
)

# --- Water meshes (dielectric surface producing caustics) ---

obj = TriangleMesh.createFromFile('models/WaterSurface.obj')
sceneBuilder.addMeshInstance(
    sceneBuilder.addTriangleMesh(obj, Water),
    sceneBuilder.addNode('WaterSurface', Transform())
)

obj = TriangleMesh.createFromFile('models/WaterBottom.obj')
sceneBuilder.addMeshInstance(
    sceneBuilder.addTriangleMesh(obj, Water),
    sceneBuilder.addNode('WaterBottom', Transform())
)

# ============================================================
# Reference: TungstenRender.exr is the reference render from
# the Tungsten renderer for visual comparison.
# To load as environment map (if needed):
#   sceneBuilder.envMap = EnvMap('TungstenRender.exr')
# ============================================================
'''

    pyscene_path = os.path.join(DST_DIR, 'WaterCaustic.pyscene')
    with open(pyscene_path, 'w') as f:
        f.write(pyscene_content)
    print(f'Written: {pyscene_path}')

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    print('\n' + '='*60)
    print('Conversion complete!')
    print(f'Output: {DST_DIR}')
    print()
    models_dir = os.path.join(DST_DIR, 'models')
    for fname in sorted(os.listdir(models_dir)):
        fpath = os.path.join(models_dir, fname)
        size = os.path.getsize(fpath)
        print(f'  models/{fname}  ({size:,} bytes)')
    print()
    for fname in sorted(os.listdir(DST_DIR)):
        fpath = os.path.join(DST_DIR, fname)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            print(f'  {fname}  ({size:,} bytes)')
    print('='*60)

if __name__ == '__main__':
    main()
