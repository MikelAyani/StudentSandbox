from OpenGL.GL import *
from OpenGL.GLU import *
import glfw
import math
import numpy as np
import threading
import time
import png
from glb_helper import *
from gltflib import GLTF
import io

"""
This is a sandbox to test the synthetic camera functionality in Simumatik Open Emulation Platform.
Usage:
The camera class is a thread wich will render the scene data sent through the pipe.
"""

def Transform2Euler(transform):
    """ Converts transform in quaternion to transform in RPY angles in radians.
    This is a modified version of this:
    https://github.com/mrdoob/three.js/blob/22ed6755399fa180ede84bf18ff6cea0ad66f6c0/src/math/Matrix4.js#L24
    """
    [x, y, z, qx, qy, qz, qw] = transform

    t = 2 * (qx * qz + qw * qy)
    P = math.asin(max(min(1, t), -1))
    if abs(t) < 0.9999999:
        R = math.atan2(2.0 * (qw * qx - qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
        Y = math.atan2(2.0 * (qw * qz - qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    else:
        R = math.atan2(2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx * qx + qz * qz))
        Y = 0

    return [x, y, z, R, P, Y]


class synthetic_camera(threading.Thread):

    def __init__(self, name:str='camera', pipe=None, width:int=800, height:int=600, vertical_fov:float=45.0, near:float=0.1, far:float=100.0, image_format:str='RGB', output_path:str='', send_response:bool=False):
        """ Constructor. """
        # Inherit
        threading.Thread.__init__(self, name=name, daemon=True)
        # Setup
        self.pipe = pipe
        self.running = True
        self.width = width
        self.height = height
        self.vertical_fov = vertical_fov
        self.near = near
        self.far = far
        self.format = image_format
        self.frame = [0, 0, 0, 0, 0, 0]
        self.output_path = output_path
        self.response = send_response


    def run(self):
        # Initialize environment
        if glfw.init():

            # Create window
            glfw.window_hint(glfw.VISIBLE, False)
            window = glfw.create_window(800, 600, "hidden window", None, None)
            glfw.make_context_current(window)

            # Initialize
            self.initialize()

            # Loop
            while self.running and self.pipe:
                # Check pipe
                if self.pipe.poll():
                    start = time.perf_counter()
                    self.render(self.pipe.recv())
                    if self.response:
                        dt = time.perf_counter() - start
                        self.pipe.send(f'Image {self.format} rendered in {int(dt*1e3)}ms')
                # Sleep
                time.sleep(1e-3)

            # Destroy camera
            glfw.destroy_window(window)

        # Terminate environment
        glfw.terminate() 


    def stop(self):
        """ Stop Thread."""
        self.running = False


    def initialize(self):
        """ Initializes OpenGL environment."""
        # Make the window's context current
        glDepthFunc(GL_LESS)    #Set the mode of the depth buffer
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_DEPTH_TEST)
        glPolygonMode(GL_FRONT, GL_FILL)    
        glPolygonMode(GL_BACK, GL_FILL)     
        glShadeModel(GL_SMOOTH)                
        glMatrixMode(GL_PROJECTION)                 
        gluPerspective(self.vertical_fov, self.width/self.height, self.near, self.far)
        glMatrixMode(GL_MODELVIEW)


    def set_frame(self, frame:list=[0, 0, 0, 0, 0, 0, 1]):
        """ Sets the camera frame from [x, y, z, X, Y, Z, W]"""
        self.frame = Transform2Euler(frame)


    def render_Box(self, vectorLHW):
        """ Render a box given vector[Length, Height, Width]"""
        L=vectorLHW[0]/2
        H=vectorLHW[1]/2
        W=vectorLHW[2]/2
        glBegin(GL_QUADS)
        glVertex3f(-L, -H, -W);  glVertex3f(L, -H, -W); glVertex3f(L, -H, W); glVertex3f(-L, -H, W) #1 2 3 4
        glVertex3f(-L, H, -W); glVertex3f(L, H, -W); glVertex3f(L, H, W); glVertex3f(-L, H, W) #5 6 7 8
        glVertex3f(-L, -H, -W); glVertex3f(L, -H, -W); glVertex3f(L, H, -W); glVertex3f(-L, H, -W) #1 2 6 5
        glVertex3f(-L, -H, W); glVertex3f(L, -H, W); glVertex3f(L, H, W); glVertex3f(-L, H, W) #4 3 7 8
        glVertex3f(L, -H, -W); glVertex3f(L, -H, W); glVertex3f(L, H, W); glVertex3f(L, H, -W) #2 3 7 6
        glVertex3f(-L, -H, -W); glVertex3f(-L, -H, W); glVertex3f(-L, H, W); glVertex3f(-L, H, -W) #1 4 8 5
        glEnd()

        glBegin(GL_LINES)
        glColor3f(0, 0, 0)
        glVertex3f(-L, -H, -W); glVertex3f(L, -H, -W) #1 2
        glVertex3f(L, -H, -W); glVertex3f(L, -H, W) #2 3
        glVertex3f(L, -H, W); glVertex3f(-L, -H, W) #3 4
        glVertex3f(-L, -H, W); glVertex3f(-L, -H, -W) #4 1
        glVertex3f(-L, H, -W); glVertex3f(L, H, -W) #5 6
        glVertex3f(L, H, -W); glVertex3f(L, H, W) #6 7
        glVertex3f(-L, H, W); glVertex3f(-L, H, -W) #8 5
        glVertex3f(L, H, W); glVertex3f(-L, H, W) #7 8
        glVertex3f(-L, -H, -W); glVertex3f(-L, H, -W) #1 5
        glVertex3f(L, -H, -W); glVertex3f(L, H, -W) #2 6
        glVertex3f(L, -H, W); glVertex3f(L, H, W) #3 7
        glVertex3f(-L, -H, W); glVertex3f(-L, H, W) #4 8
        glColor3f(1, 1, 1)
        glEnd()


    def render_Capsule(self, radius, length):
        """ Render capsule given radius and length"""
        capsuleQ = gluNewQuadric()
        gluCylinder(capsuleQ, radius, radius, length, 100, 100)
        gluSphere(capsuleQ, radius, 36, 18)
        glTranslatef(0, 0, length)
        gluSphere(capsuleQ, radius, 36, 18)
        glTranslatef(0, 0, -length) #To get the camera as it was, just in case we don't use glPushMatrix before calling capsule


    def render_Cylinder(self, radius, length):
        """ Render cylinder given radius and length"""
        cylinderQ = gluNewQuadric()
        gluCylinder(cylinderQ, radius, radius, length, 100, 100)
        gluSphere(cylinderQ, radius, 36, 2)  
        glTranslate(0, 0, length)
        gluSphere(cylinderQ, radius, 36, 2)  
        glTranslate(0, 0, -length)  


    def render_Plane(self, normal):
        """ Render 2D plane given its normal vector"""
        xAngle=np.arctan2(normal[2],normal[1]) * (180/np.pi)
        yAngle=np.arctan2(normal[0],normal[2]) * (180/np.pi)
        zAngle=np.arctan2(normal[0],normal[1]) * (180/np.pi)
        glRotatef(xAngle, 1, 0, 0)
        glRotatef(yAngle, 0, 1, 0)
        glRotatef(zAngle, 0, 0, 1)
        glBegin(GL_QUADS)
        glColor3f(1, 0, 0)
        glVertex3f(-self.far, 0, self.far)        
        glVertex3f(self.far, 0, self.far)        
        glVertex3f(self.far, 0, -self.far)        
        glVertex3f(-self.far, 0, -self.far)
        glEnd()

    def draw_glb_with_textures(self, glb, primitive):
        vertices = primitive['POSITION']
        faces = np.reshape(primitive['indices'], (-1, 3))
        UV = primitive['TEXCOORD_0']
        text_ID = primitive['material'].pbrMetallicRoughness.baseColorTexture.index
        _, texture_data = get_texture(glb, text_ID)

        r = png.Reader(bytes=texture_data).read_flat()
        l = np.array(r[2], np.int8)
        tex_data = np.reshape(l, (r[0]*r[1], 3))
        
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, r[0], r[1], 0, GL_RGB, GL_UNSIGNED_BYTE, tex_data)
        glGenerateMipmap(GL_TEXTURE_2D)
        glBegin(GL_TRIANGLES)
        for a in range(len(faces)):
            glTexCoord2dv(UV[faces[a,0]])
            glVertex3fv(100*vertices[faces[a,0]])
            glTexCoord2dv(UV[faces[a,1]])
            glVertex3fv(100*vertices[faces[a,1]])
            glTexCoord2dv(UV[faces[a,2]])
            glVertex3fv(100*vertices[faces[a,2]])
        glEnd()
        glBindTexture(GL_TEXTURE_2D, 0)


    def draw_glb_without_textures(self, primitive):
        vertices = primitive['POSITION']
        faces = np.reshape(primitive['indices'], (-1, 3))
        glBegin(GL_TRIANGLES)
        for a in range(len(faces)):
            glVertex3fv(100*vertices[faces[a,0]])
            glVertex3fv(100*vertices[faces[a,1]])
            glVertex3fv(100*vertices[faces[a,2]])
        glEnd()

    def draw_glb(self, path_to_file):
        _dtypes = {5120: "<i1",5121: "<u1",5122: "<i2",5123: "<u2",5125: "<u4",5126: "<f4"}
        _shapes = {"SCALAR": 1,"VEC2": (2),"VEC3": (3),"VEC4": (4),"MAT2": (2, 2),"MAT3": (3, 3),"MAT4": (4, 4)}
        try:
            glb = GLTF.load_glb(path_to_file)
            # First level
            main_node = glb.model.scene # Scene is a pointer to the main node
            translation, rotation, scale = get_node_TRS(glb, main_node)
            node_mesh = glb.model.nodes[main_node].mesh
            
            # If node has a mesh
            if node_mesh is not None:
                mesh_data = get_mesh_data(glb, node_mesh, vertex_only=False)
                # A mesh may have several primitives
                for primitive in mesh_data['primitives']:
                    if primitive['TEXCOORD_0'] is not None:
                        self.draw_glb_with_textures(glb, primitive)
                    else:
                        self.draw_glb_without_textures(primitive)

            # Second level
            if glb.model.nodes[main_node].children:
                # A node may have several child nodes
                for child_node in glb.model.nodes[main_node].children:
                    translation, rotation, scale = get_node_TRS(glb, child_node)
                    child_node_mesh = glb.model.nodes[child_node].mesh
                    # If child node has a mesh
                    if child_node_mesh is not None:
                        mesh_data = get_mesh_data(glb, child_node_mesh, vertex_only=False)
                        # A mesh may have several primitives
                        for primitive in mesh_data['primitives']:
                            if primitive['TEXCOORD_0'] is not None:
                                self.draw_glb_with_textures(glb, primitive)
                            else:
                                self.draw_glb_without_textures(primitive)
                    
        except Exception as e:
            print('Exception loading', path_to_file, e)  


    def render(self, data:dict):
        ''' Main camera script.
        data: A dictionary containing objects to be rendered.
            'object_name':
                'frame': (float[7]) object global frame including position and transformation (quaternion) [x, y, z, qx, qy, qz, qw],
                'shapes': a dictionary of shapes included in the object
                    'shape_name':
                        'origin': (optional) (float[7]) shape local frame relative to the object [x, y, z, qx, qy, qz, qw],
                        'type': (str) shape type: 'plane', 'box', 'cylinder', 'sphere', 'capsule', 'mesh'
                        'attributes': shape specific attributes
                            # plane
                            'normal': (float[3]) x, y, z normal vector of the plane
                            # box
                            'size': (float[3]) x, y, z sizes of the box
                            # cylinder
                            'radius': (float) radius of the cylinder
                            'length': (float) length of the cylinder
                            # capsule
                            'radius': (float) radius of the capsule
                            'length': (float) length of the capsule
                            # sphere
                            'radius': (float) radius of the sphere
                            # mesh
                            'model': (str) path to the mesh model (GLB file)
                            'scale': (float[3]) x, y, z axis scale values of the mesh
        '''
        array_textures = [0, 0]
        textures = glGenTextures(2, array_textures)
        glBindTexture(GL_TEXTURE_2D, textures[0])
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.width, self.height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
        glBindTexture(GL_TEXTURE_2D, 0)

        glBindTexture(GL_TEXTURE_2D, textures[1])
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, self.width, self.height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)
        glBindTexture(GL_TEXTURE_2D, 0)

        array_fbo = [0]
        fbo = glGenFramebuffers(1, array_fbo)
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textures[0], 0)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, textures[1], 0)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT) 
 
        # Example response
        # camera_transform = Transform()
        # camera_transform.setOrigin(Vector3(self.frame[0], self.frame[1], self.frame[2]))
        # camera_transform.setRotation(Quaternion.fromScalars(self.frame[3], self.frame[4], self.frame[5], self.frame[6]))
        # inv_camera_transform = camera_transform.inverse()
        # pos = inv_camera_transform.getOrigin()
        # rot = inv_camera_transform.getRotation()
        # print(f'Camera inv. position: {pos.x} {pos.y} {pos.z} (in meters)')
        # print(f'Camera inv. rotation: {rot.GetX()} {rot.GetY()} {rot.GetZ()} {rot.GetW()} (in quaternion)')

        # To load objects data
        for _, object_data in data.items():
            obj_frame = Transform2Euler(object_data.get('frame', [0, 0, 0, 0, 0, 0, 1]))

            for shape_name in object_data['shapes']:
                shape_orig = Transform2Euler(object_data['shapes'][shape_name].get('origin', [0, 0, 0, 0, 0, 0, 1]))
                shape_type = object_data['shapes'][shape_name].get('type')

                glLoadIdentity()
                glTranslatef(self.frame[0], self.frame[1], self.frame[2])
                glRotatef(self.frame[3], 1, 0, 0);  glRotatef(self.frame[4], 0, 1, 0);  glRotatef(self.frame[5], 0, 0, 1)
                glRotatef(-90, 1, 0, 0)

                glTranslatef(obj_frame[0]+shape_orig[0], obj_frame[1]+shape_orig[1], obj_frame[2]+shape_orig[2])
                glRotatef(obj_frame[3]+shape_orig[3], 1, 0, 0); glRotatef(obj_frame[4]+shape_orig[4], 0, 1, 0); glRotatef(obj_frame[5]+shape_orig[5], 0, 0, 1)

                if shape_type == 'plane':
                    self.render_Plane(object_data['shapes'][shape_name]['attributes'].get('normal'))
                elif shape_type == 'box':
                    self.render_Box(object_data['shapes'][shape_name]['attributes'].get('sizes'))
                elif shape_type == 'cylinder':
                    self.render_Cylinder(object_data['shapes'][shape_name]['attributes'].get('radius'), object_data['shapes'][shape_name]['attributes'].get('length'))
                elif shape_type == 'capsule':
                    self.render_Capsule(object_data['shapes'][shape_name]['attributes'].get('radius'), object_data['shapes'][shape_name]['attributes'].get('length'))
                elif shape_type == 'sphere':
                    sphereQ = gluNewQuadric()
                    gluSphere(sphereQ, object_data['shapes'][shape_name]['attributes'].get('radius'), 36, 18)
                elif shape_type == 'mesh':
                    # if 'cache' not in object_data['shapes'][shape_name]:
                    #     # Load cache
                    #     object_data['shapes'][shape_name]['cache'] = self.get_glb_cache(object_data['shapes'][shape_name])
                    # self.draw_glb(object_data['shapes'][shape_name]['cache'])
                    self.draw_glb(object_data['shapes'][shape_name]['attributes'].get('model'))
                    
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        
        # Generate RGB image
        if self.format == 'RGB':
            glBindTexture(GL_TEXTURE_2D, textures[0])
            color_str = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE)
            glBindTexture(GL_TEXTURE_2D, 0)
            color_data = np.frombuffer(color_str, dtype=np.uint8)
            matColor = np.frombuffer(color_data, dtype=np.uint8).reshape(self.height, self.width*3)
            png.from_array(np.flip(matColor,0), mode="RGB").save(self.output_path)

        # Generate Grayscale image
        elif self.format == 'L':
            glBindTexture(GL_TEXTURE_2D, textures[0])
            color_str = glGetTexImage(GL_TEXTURE_2D, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE)
            glBindTexture(GL_TEXTURE_2D, 0)
            color_data = np.frombuffer(color_str, dtype=np.uint8)
            matColor = np.frombuffer(color_data, dtype=np.uint8).reshape(self.height, self.width)
            png.from_array(np.flip(matColor, 0), mode="L").save(self.output_path)

        # Generate Depth image
        elif self.format == 'D':
            glBindTexture(GL_TEXTURE_2D, textures[1])
            depth_str = glGetTexImage(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, GL_FLOAT)
            glBindTexture(GL_TEXTURE_2D, 0)
            depth_data = np.frombuffer(depth_str, dtype=np.float32)
            # Linearize depth values
            z = depth_data*2.0 - 1.0
            linearDepth = (2.0 * self.near * self.far) / (self.far + self.near - z * (self.far - self.near))
            linearDepth = linearDepth/self.far
            # Resize 1D matrix to 2D matrix
            matD = np.reshape((255-255*linearDepth).astype(np.uint8), (self.height, self.width))
            png.from_array(np.flip(matD, 0), mode="L").save(self.output_path)
            
        elif self.format == 'RGBD':
            #Generate D data
                #Obtain the depth data in a numpy array
            glBindTexture(GL_TEXTURE_2D, textures[1])
            depth_str = glGetTexImage(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, GL_FLOAT)
            glBindTexture(GL_TEXTURE_2D, 0)
            depth_data = np.frombuffer(depth_str, dtype=np.float32)
                #Linearize depth values
            z = depth_data*2.0 - 1.0
            linearDepth = (2.0 * self.near * self.far) / (self.far + self.near - z * (self.far - self.near))
            linearDepth = linearDepth/self.far

            #Generate RGB data
            glBindTexture(GL_TEXTURE_2D, textures[0])
            color_str = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE)
            glBindTexture(GL_TEXTURE_2D, 0)
            color_data = np.frombuffer(color_str, dtype=np.uint8)
            rgbd_data = np.append(color_data, (0-0*linearDepth).astype('int'), axis=None).reshape(self.height, self.width*4)
            #print(type(rgbd_data))
            print(rgbd_data.shape)
            #matRGBD = np.frombuffer(rgbd_data).reshape(self.height, self.width*4)
            #print(matRGBD.shape)
            #matColor = np.frombuffer(color_data, dtype=np.uint8)
            #matRGBD = np.concatenate((matColor, matD255), axis=1)
            
            #Take RGBD matrix
            #matRGBD = np.empty((self.height, self.width*4))
            #matRGBD[:, :self.width*3] = matColor
            #matRGBD[:, self.width*3:self.width*4] = matD
            #print(matRGBD.shape)
            png.from_array(rgbd_data, mode="RGBA;16").save(self.output_path)
        

if __name__ == '__main__':
    import time
    from multiprocessing import Pipe
    # Create some dummy data
    data = {
        'floor': {
            'frame': [0, 0, 0, 0, 0, 0, 1],
            'shapes': {
                'plane':{
                    'type': 'plane',
                    'attributes': {
                        'normal': [0.0, 0.0, 1.0]
                    }
                }
            }
        },
        'test_box2': {
            'frame': [0, 0, 0, 0, 0, 0, 1],
            'shapes': {
                'box':{
                    'type': 'box',
                    'attributes': {
                        'sizes': [0.1, 0.1, 0.1]
                    }

                }
            }
        },
        'test_box': {
            'frame': [0, 0, 1, 0, 0, 0, 1],
            'shapes': {
                'box':{
                    'type': 'box',
                    'attributes': {
                        'sizes': [1.0, 1.0, 1.0]
                    }

                }
            }
        },
        'test_multibody': {
            'frame': [1, 2, 1, 0, 0, 0, 1],
            'shapes': {
                'shape_1':{
                    'type': 'box',
                    'attributes': {
                        'sizes': [0.5, 0.5, 0.5]
                    }
                },
                'shape_2':{
                    'origin': [-0.5, 0, 0, 0, 0, 0, 1],
                    'type': 'sphere',
                    'attributes': {
                        'radius': 0.3
                    }
                }
            }
        },
        'glb_archive': {
            'frame': [4, 0, 1, 0, 0, 0, 1],
            'shapes': {
                'glb_1':{
                    'type': 'mesh',
                    'attributes': {
                        'model': 'data/duck.glb',
                        'scale': [1, 1, 1]
                    }
                }
            }
        }
    }

    # Create camera
    pipe, camera_pipe = Pipe()
    camera = synthetic_camera(
        pipe=camera_pipe, 
        image_format='RGB', 
        output_path='test.png',
        send_response=True)
    camera.set_frame([-1, -1, -10, 0, 0, 0, 1])
    camera.start()
    print("Camera started.")
    # Loop
    counter = 0
    start = time.perf_counter()
    for i in range(1):
        pipe.send(data)
        print(pipe.recv())  
    # Stop
    camera.stop()
    camera.join()
    print("Camera destroyed.")