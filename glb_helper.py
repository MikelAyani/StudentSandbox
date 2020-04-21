

from gltflib import GLTF
import struct
import numpy as np
import warnings
warnings.filterwarnings("ignore")

_dtypes = {5120: "<i1",
           5121: "<u1",
           5122: "<i2",
           5123: "<u2",
           5125: "<u4",
           5126: "<f4"}

# GLTF data formats: numpy shapes
_shapes = {
    "SCALAR": 1,
    "VEC2": (2),
    "VEC3": (3),
    "VEC4": (4),
    "MAT2": (2, 2),
    "MAT3": (3, 3),
    "MAT4": (4, 4)}


def load_accessor(glb, id):
    accessor = glb.model.accessors[id]
    # number of items
    count = accessor.count
    # what is the datatype
    dtype = _dtypes[accessor.componentType]
    # basically how many columns
    per_item = _shapes[accessor.type]
    # use reported count to generate shape
    shape = np.append(count, per_item)
    # number of items when flattened
    # i.e. a (4, 4) MAT4 has 16
    per_count = np.abs(np.product(per_item))
    # data was stored in a buffer view so get raw bytes
    bufferview = glb.model.bufferViews[accessor.bufferView]
    buffer = glb.resources[bufferview.buffer]
    # is the accessor offset in a buffer
    start = bufferview.byteOffset
    # length is the number of bytes per item times total
    length = np.dtype(dtype).itemsize * count * per_count
    # load the bytes data into correct dtype and shape
    return np.frombuffer(buffer.data[start:start + length], dtype=dtype).reshape(shape)


def get_mesh_data(glb, mesh_id, vertex_only=False):
    mesh_data = {}
    mesh = glb.model.meshes[mesh_id]
    if len(mesh.primitives):
        mesh_data['primitives'] = []
        for primitive in mesh.primitives:
            # Valid attribute semantic property names include POSITION, NORMAL, TANGENT, TEXCOORD_0, TEXCOORD_1, COLOR_0,...
            prim_data = {}
            # VERTEX DATA 
            if primitive.attributes.POSITION is not None:
                prim_data['POSITION'] = load_accessor(glb, primitive.attributes.POSITION)
            else:
                prim_data['POSITION'] = None

            if not vertex_only:
                # NORMAL
                if primitive.attributes.NORMAL is not None:
                    prim_data['NORMAL'] = load_accessor(glb, primitive.attributes.NORMAL)
                else:
                    prim_data['NORMAL'] = None
                # TANGENT
                if primitive.attributes.TANGENT is not None:
                    prim_data['TANGENT'] = load_accessor(glb, primitive.attributes.TANGENT)
                else:
                    prim_data['TANGENT'] = None
                # TEXCOORD_0
                if primitive.attributes.TEXCOORD_0 is not None:
                    prim_data['TEXCOORD_0'] = load_accessor(glb, primitive.attributes.TEXCOORD_0)
                else:
                    prim_data['TEXCOORD_0'] = None 
                # COLOR_0
                if 'COLOR_0' in primitive.attributes.__dict__:
                    if primitive.attributes.COLOR_0 is not None:
                        prim_data['COLOR_0'] = load_accessor(glb, primitive.attributes.COLOR_0)
                # indices
                if primitive.indices is not None:
                    prim_data['indices'] = load_accessor(glb, primitive.indices)
                else:
                    prim_data['indices'] = None
                # Material
                if primitive.material is not None:
                    prim_data['material'] = glb.model.materials[primitive.material]
                else:
                    prim_data['material'] = None
                # Mode
                prim_data['mode'] = primitive.mode
            # Save primitive
            mesh_data['primitives'].append(prim_data)

    # Return node data
    return mesh_data


def get_node_TRS(glb, node_id):
    '''
    TRS properties are converted to matrices and postmultiplied in the T * R * S order to compose the transformation matrix; 
    first the scale is applied to the vertices, then the rotation, and then the translation.
    '''
    node = glb.model.nodes[node_id]
    # Grab matrix
    if node.matrix is not None:
        '''
        Asumimng:
          0   1   2   3
        [r00 r01 r02 tx]
          4   5   6   7
        [r10 r11 r12 ty]
          8   9  10  11 
        [r20 r21 r22 tz]
         12  13  14  15
        [sx  sy  sz  1]
        '''
        translation = [node.matrix[12], node.matrix[13], node.matrix[14]]
        w = np.sqrt(1.0 + node.matrix[0] + node.matrix[5] + node.matrix[10]) / 2.0
        w4 = (4.0 * w)
        x = (node.matrix[6] - node.matrix[9]) / w4
        y = (node.matrix[8] - node.matrix[2]) / w4
        z = (node.matrix[1] - node.matrix[4]) / w4
        rotation = [x, y, z, w]
        if round(node.matrix[0],3) == round(node.matrix[5],3) == round(node.matrix[10],3):
            scale = [node.matrix[0], node.matrix[5], node.matrix[10]]
        else:
            factor = np.sqrt(node.matrix[0]*node.matrix[0]+node.matrix[5]*node.matrix[5]+node.matrix[10]*node.matrix[10])
            scale = [factor, factor, factor]
        return translation, rotation, scale
    
    if node.translation is not None:
        translation = node.translation #(X, Y, Z)
    else:
        translation = [0, 0, 0]

    if node.rotation is not None:
        rotation = node.rotation # (X, Y, Z, W)
    else:
        rotation = [0, 0, 0, 1]

    if node.scale is not None:
        scale = node.scale # (X, Y, Z)
    else:
        scale = [1, 1, 1]

    return translation, rotation, scale

def get_node_TRS_2(glb, node_id):
    '''
    TRS properties are converted to matrices and postmultiplied in the T * R * S order to compose the transformation matrix; 
    first the scale is applied to the vertices, then the rotation, and then the translation.
    '''
    node = glb.model.nodes[node_id]
    # Grab matrix
    if node.matrix is not None:
        NM = np.array(node.matrix).reshape((4,4)).transpose()
        '''
        NM looks like
        [r00*sx r01*sy r02*sz tx]
        [r10*sx r11*sy r12*sz ty]
        [r20*sx r21*sy r22*sz tz]
        [   0      0      0    1]
        '''
        #print(NM)
        translation = [NM[0][3], NM[1][3], NM[2][3]]
        
        sx = np.sqrt(NM[0][0]**2 + NM[1][0]**2 + NM[2][0]**2)
        sy = np.sqrt(NM[0][1]**2 + NM[1][1]**2 + NM[2][1]**2)
        sz = np.sqrt(NM[0][2]**2 + NM[1][2]**2 + NM[2][2]**2)
        scale = [sx, sy, sz]
        
        #Recordar que las transformaciones para obtener q son desde la matriz de rotacion
        w = np.sqrt(1.0 + NM[0][0]/sx + NM[1][1]/sy + NM[2][2]/sz) / 2.0
        w4 = (4.0 * w)
        x = (NM[2][1]/sy - NM[1][2]/sz) / w4
        y = (NM[0][2]/sz - NM[2][0]/sx) / w4
        z = (NM[1][0]/sx - NM[0][1]/sy) / w4
        rotation = [x, y, z, w]
        
        return translation, rotation, scale
    
    if node.translation is not None:
        translation = node.translation #(X, Y, Z)
    else:
        translation = [0, 0, 0]

    if node.rotation is not None:
        rotation = node.rotation # (X, Y, Z, W)
    else:
        rotation = [0, 0, 0, 1]

    if node.scale is not None:
        scale = node.scale # (X, Y, Z)
    else:
        scale = [1, 1, 1]

    return translation, rotation, scale


def get_texture(glb, texture_id):
    texture = glb.model.textures[texture_id]
    if texture.source is not None:
        image = glb.model.images[texture.source]
        extension = image.mimeType.split('/')[-1]
        name = image.name if image.name is not None else f'Texture_{id}'
        bufferview = glb.model.bufferViews[image.bufferView]
        buffer = glb.resources[bufferview.buffer]
        return (f'{name}.{extension}', buffer.data[bufferview.byteOffset: bufferview.byteOffset +  bufferview.byteLength])
    else:
        return ('', None)


def save_texture_to_file(glb, texture_id):
    if glb.model.textures is not None:
        (texture_name, data) = get_texture(glb, i)
        if texture_name:
            print(f"Texture found: {texture_name}")
            with open(join(CACHE_FOLDER, texture_name), 'wb') as f:
                f.write(data)


if __name__ == '__main__':
    from oep_globals import CACHE_FOLDER
    from os import listdir
    from os.path import isfile, join


    for filename in ['ba90b73d-744a-11ea-9c8b-06fe7d6a1d26.glb','8da10e70-56ed-11ea-b8aa-063ea404e626.glb']:#listdir(CACHE_FOLDER):
        if isfile(join(CACHE_FOLDER, filename)) and '.glb' in filename:
            try:
                ''' NOTE! THIS EXAMPLE CODE ASUMES THAT THE GLB FILES HAVE JUST A NODE DISTRIBUTION WITH TWO LEVELS, THE SCENE AND ITS CHILDREN.'''
                # Load file
                print(filename)
                glb = GLTF.load_glb(join(CACHE_FOLDER, filename))

                # Scene
                if glb.model.scene is not None:
                    scene = glb.model.scenes[glb.model.scene]
                    for main_node in scene.nodes:
                        # First level
                        translation, rotation, scale = get_node_TRS(glb, main_node)
                        print(translation, rotation, scale)
                        node_mesh = glb.model.nodes[main_node].mesh
                        # If node has a mesh
                        if node_mesh is not None:
                            mesh_data = get_mesh_data(glb, node_mesh, vertex_only=True)
                            # A mesh may have several primitives
                            for primitive in mesh_data['primitives']:
                                vertices = primitive['POSITION']
                                print(f'Mesh {node_mesh} found with {len(vertices)} vertices.')

                        # Second level
                        if glb.model.nodes[main_node].children:
                            # A node may have several child nodes
                            for child_node in glb.model.nodes[main_node].children:
                                translation, rotation, scale = get_node_TRS(glb, child_node)
                                child_node_mesh = glb.model.nodes[child_node].mesh
                                # If child node has a mesh
                                if child_node_mesh is not None:
                                    mesh_data = get_mesh_data(glb, child_node_mesh, vertex_only=True)
                                    # A mesh may have several primitives
                                    for primitive in mesh_data['primitives']:
                                        vertices = primitive['POSITION']
                                        print(f'Mesh {child_node_mesh} found with {len(vertices)} vertices.')
                            
            except Exception as e:
                print('Exception:', filename, e)
            #break
            