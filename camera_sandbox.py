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

"""
This is a sandbox to test the synthetic camera functionality in Simumatik Open Emulation Platform.
Usage:
The camera class is a thread wich will render the scene data sent through the pipe.
"""

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
        glDepthFunc(GL_LESS)    
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_DEPTH_TEST)
        glPolygonMode(GL_FRONT, GL_FILL)    
        glPolygonMode(GL_BACK, GL_FILL)     
        glShadeModel(GL_SMOOTH)                

    def set_frame(self, frame:list=[0, 0, 0, 0, 0, 0, 1]):
        """ Sets the camera frame from [x, y, z, X, Y, Z, W]"""
        self.frame = self.Transform2Euler(frame)

    def Transform2Euler(self, transform):
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

        return [x, z, y, R*(180/np.pi), Y*(180/np.pi), P*(180/np.pi)] #Modified because OpenGL has the axis changed respected to Simumatik

    def render_Box(self, vectorLWH):
        """ Render a box given vector[Length, Width, Height]"""
        L=vectorLWH[0]/2
        W=vectorLWH[1]/2
        H=vectorLWH[2]/2
        glBegin(GL_QUADS)
        glColor3f(1,1,1)
        glVertex3f(-L, -H, -W);  glVertex3f(L, -H, -W); glVertex3f(L, -H, W); glVertex3f(-L, -H, W) #1 2 3 4
        glVertex3f(-L, H, -W); glVertex3f(L, H, -W); glVertex3f(L, H, W); glVertex3f(-L, H, W) #5 6 7 8
        glVertex3f(-L, -H, -W); glVertex3f(L, -H, -W); glVertex3f(L, H, -W); glVertex3f(-L, H, -W) #1 2 6 5
        glVertex3f(-L, -H, W); glVertex3f(L, -H, W); glVertex3f(L, H, W); glVertex3f(-L, H, W) #4 3 7 8
        glVertex3f(L, -H, -W); glVertex3f(L, -H, W); glVertex3f(L, H, W); glVertex3f(L, H, -W) #2 3 7 6
        glVertex3f(-L, -H, -W); glVertex3f(-L, -H, W); glVertex3f(-L, H, W); glVertex3f(-L, H, -W) #1 4 8 5
        glEnd()

    def render_Capsule(self, radius, length):
        """ Render capsule given radius and length"""
        glColor3f(1, 1, 1)
        capsuleQ = gluNewQuadric()
        gluCylinder(capsuleQ, radius, radius, length, 100, 100)
        gluSphere(capsuleQ, radius, 36, 18)
        glTranslatef(0, 0, length)
        gluSphere(capsuleQ, radius, 36, 18)

    def render_Cylinder(self, radius, length):
        """ Render cylinder given radius and length"""
        glColor3f(1, 1, 1)
        cylinderQ = gluNewQuadric()
        glTranslatef(0, 0, -length/2)
        gluCylinder(cylinderQ, radius, radius, length, 100, 100)
        gluDisk(cylinderQ, 0.0, radius, 64, 1) 
        glTranslatef(0, 0, length)
        gluDisk(cylinderQ, 0.0, radius, 64, 1)  

    def render_Plane(self, normal):
        """ Render 2D plane given its normal vector[x, y, z]"""
        xAngle=np.arctan2(normal[1],normal[2]) * (180/np.pi)
        yAngle=np.arctan2(normal[2],normal[1]) * (180/np.pi)
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

    def render_glb_with_textures(self, glb, primitive, scale):
        """ Render glb with textures"""
        vertices = primitive['POSITION']
        faces = np.reshape(primitive['indices'], (-1, 3))
        UV = primitive['TEXCOORD_0']
        text_ID = primitive['material'].pbrMetallicRoughness.baseColorTexture.index
        _, texture_bytes = get_texture(glb, text_ID)

        texture_data = png.Reader(bytes=texture_bytes).read_flat()
        tex_array = np.array(texture_data[2], np.int8).reshape(texture_data[0]*texture_data[1], 3)
        
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texture_data[0], texture_data[1], 0, GL_RGB, GL_UNSIGNED_BYTE, tex_array)
        glGenerateMipmap(GL_TEXTURE_2D)
        glBegin(GL_TRIANGLES)
        for a in range(len(faces)):
            glTexCoord2dv(UV[faces[a,0]])
            glVertex3fv(scale*vertices[faces[a,0]])
            glTexCoord2dv(UV[faces[a,1]])
            glVertex3fv(scale*vertices[faces[a,1]])
            glTexCoord2dv(UV[faces[a,2]])
            glVertex3fv(scale*vertices[faces[a,2]])
        glEnd()
        glBindTexture(GL_TEXTURE_2D, 0)

    def render_glb_without_textures(self, primitive, scale):
        """ Render glb shape without colors or textures"""
        vertices = primitive['POSITION']
        faces = np.reshape(primitive['indices'], (-1, 3))
        glBegin(GL_TRIANGLES)
        for a in range(len(faces)):
            glVertex3fv(scale*vertices[faces[a,0]])
            glVertex3fv(scale*vertices[faces[a,1]])
            glVertex3fv(scale*vertices[faces[a,2]])
        glEnd()

    def render_glb(self, path_to_file, scale):
        """ Render glb archive limited to the main node and his children"""
        #_dtypes = {5120: "<i1",5121: "<u1",5122: "<i2",5123: "<u2",5125: "<u4",5126: "<f4"}
        #_shapes = {"SCALAR": 1,"VEC2": (2),"VEC3": (3),"VEC4": (4),"MAT2": (2, 2),"MAT3": (3, 3),"MAT4": (4, 4)}
        try:
            scale = [scale[0], scale[2], scale[1]] #Modified because OpenGL has the axis changed respected to Simumatik
            glb = GLTF.load_glb(path_to_file)
            # First level
            main_node = glb.model.scene # Scene is a pointer to the main node
            #translation_node, rotation_node, scale_node = get_node_TRS(glb, main_node)
            #frame_node = self.Transform2Euler(np.append(translation_node,rotation_node))
            node_mesh = glb.model.nodes[main_node].mesh
            
            # If node has a mesh
            if node_mesh is not None:
                mesh_data = get_mesh_data(glb, node_mesh, vertex_only=False)
                # A mesh may have several primitives
                for primitive in mesh_data['primitives']:
                    if primitive['TEXCOORD_0'] is not None and self.format != 'D':
                        self.render_glb_with_textures(glb, primitive, scale)
                    else:
                        self.render_glb_without_textures(primitive, scale)

            # Second level
            if glb.model.nodes[main_node].children:
                # A node may have several child nodes
                for child_node in glb.model.nodes[main_node].children:
                    #translation_node, rotation_node, scale_node = get_node_TRS(glb, child_node)
                    child_node_mesh = glb.model.nodes[child_node].mesh
                    # If child node has a mesh
                    if child_node_mesh is not None:
                        mesh_data = get_mesh_data(glb, child_node_mesh, vertex_only=False)
                        # A mesh may have several primitives
                        for primitive in mesh_data['primitives']:
                            if primitive['TEXCOORD_0'] is not None and self.format != 'D':
                                self.render_glb_with_textures(glb, primitive, scale)
                            else:
                                self.render_glb_without_textures(primitive, scale)
                    
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

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()            
        gluPerspective(self.vertical_fov, self.width/self.height, self.near, self.far)
        glRotatef(self.frame[3], 1, 0, 0);  glRotatef(self.frame[4]+90, 0, 1, 0);  glRotatef(self.frame[5], 0, 0, 1)
        glTranslatef(-self.frame[0], -self.frame[1], -self.frame[2])
        glMatrixMode(GL_MODELVIEW)

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

        # To load objects data
        for _, object_data in data.items():
            obj_frame = self.Transform2Euler(object_data.get('frame', [0, 0, 0, 0, 0, 0, 1]))

            for shape_name in object_data['shapes']:
                shape_orig = self.Transform2Euler(object_data['shapes'][shape_name].get('origin', [0, 0, 0, 0, 0, 0, 1]))
                shape_type = object_data['shapes'][shape_name].get('type')

                glLoadIdentity()  
                glTranslatef(obj_frame[0], obj_frame[1], obj_frame[2])
                glRotatef(obj_frame[3], 1, 0, 0); glRotatef(obj_frame[4], 0, 1, 0); glRotatef(obj_frame[5], 0, 0, 1)
                glTranslatef(shape_orig[0], shape_orig[1], shape_orig[2])
                glRotatef(shape_orig[3], 1, 0, 0); glRotatef(shape_orig[4], 0, 1, 0); glRotatef(shape_orig[5], 0, 0, 1)

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
                    self.render_glb(object_data['shapes'][shape_name]['attributes'].get('model'), object_data['shapes'][shape_name]['attributes'].get('scale'))
        
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # Generate RGB image
        if self.format == 'RGB':
            glBindTexture(GL_TEXTURE_2D, textures[0])
            color_str = glGetTexImage(GL_TEXTURE_2D, 0, GL_BGR, GL_UNSIGNED_BYTE)
            glBindTexture(GL_TEXTURE_2D, 0)
            matColor = np.frombuffer(color_str, dtype=np.uint8).reshape(self.height, self.width*3)
            png.from_array(np.flip(matColor,(0,1)).copy(), mode="RGB").save(self.output_path)

        # Generate Grayscale image
        elif self.format == 'L':
            glBindTexture(GL_TEXTURE_2D, textures[0])
            color_str = glGetTexImage(GL_TEXTURE_2D, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE)
            glBindTexture(GL_TEXTURE_2D, 0)
            matColor = np.frombuffer(color_str, dtype=np.uint8).reshape(self.height, self.width)
            png.from_array(np.flip(matColor,(0,1)).copy(), mode="L").save(self.output_path)

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
            png.from_array(np.flip(matD,(0,1)).copy(), mode="L").save(self.output_path)
            
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
            matD = np.reshape((255-255*linearDepth).astype(np.uint8), (self.height*self.width, 1))
            matD = np.flip(matD, (0,1))

            #Generate RGB data
            glBindTexture(GL_TEXTURE_2D, textures[0])
            color_str = glGetTexImage(GL_TEXTURE_2D, 0, GL_BGR, GL_UNSIGNED_BYTE)
            glBindTexture(GL_TEXTURE_2D, 0)
            matColor = np.frombuffer(color_str, dtype=np.uint8).reshape(self.height*self.width, 3)
            matColor = np.flip(matColor, (0,1))
            
            #Combine them for RGBD
            matRGBD = np.append(matColor, matD, axis=1).reshape(self.height, self.width*4)
            png.from_array(matRGBD, mode="RGBA").save(self.output_path)

"""
if __name__ == '__main__':
    import time
    from multiprocessing import Pipe
    #Data for conveyors and door
    data = {'floor': {'frame': [0, 0, 0, 0, 0, 0, 1], 'shapes': {'plane': {'type': 'plane', 'attributes': {'normal': [0.0, 0.0, 1.0]}}}}, '530': {'frame': [2.32, 0.0, 0.391, 0.0, 0.0, 1.0, -0.009], 'shapes': {'545': {'type': 'box', 'attributes': {'sizes': [2.12, 0.68, 0.16]}, 'origin': [0.0, 0.0, 0.29, 0.0, 0.0, 0.0, 1.0]}, '550': {'type': 'cylinder', 'attributes': {'length': 0.68, 'radius': 0.08}, 'origin': [1.06, 0.0, 0.29, 0.0, 0.0, 0.0, 1.0]}, '556': {'type': 'cylinder', 'attributes': {'length': 0.68, 'radius': 0.08}, 'origin': [-1.06, 0.0, 0.29, 0.0, 0.0, 0.0, 1.0]}}}, '1088': {'frame': [-0.131, -0.043, 0.387, 0.0, 0.0, 1.0, -0.006], 'shapes': {'1103': {'type': 'box', 'attributes': {'sizes': [2.12, 0.68, 0.16]}, 'origin': [0.0, 0.0, 0.29, 0.0, 0.0, 0.0, 1.0]}, '1108': {'type': 'cylinder', 'attributes': {'length': 0.68, 'radius': 0.08}, 'origin': [1.06, 0.0, 0.29, 0.0, 0.0, 0.0, 1.0]}, '1114': {'type': 'cylinder', 'attributes': {'length': 0.68, 'radius': 0.08}, 'origin': [-1.06, 0.0, 0.29, 0.0, 0.0, 0.0, 1.0]}}}, '1178': {'frame': [1.095, 0.0, 1.06, 0.0, 0.0, 0.0, 1.0], 'shapes': {'1194': {'type': 'box', 'attributes': {'sizes': [0.1, 1.3, 2.0]}}}}, '1582': {'frame': [0.954, -0.649, 0.138, 0.0, 0.0, 0.707, 0.707], 'shapes': {'1598': {'type': 'box', 'attributes': {'sizes': [0.01, 0.1, 0.1]}}}}}

    # Create camera
    pipe, camera_pipe = Pipe()
    camera = synthetic_camera(
        pipe=camera_pipe, 
        image_format='RGBD', 
        output_path='test.png',
        send_response=True)
    camera.set_frame([-0.492, 4.04, 1.334, 0, 0, -0.707, 0.707]) #For conveyors and door
    camera.start()
    print("Camera started.")
    # Loop
    counter = 0
    start = time.perf_counter()
    for i in range(4):
        pipe.send(data)
        print(pipe.recv())  
    # Stop
    camera.stop()
    camera.join()
    print("Camera destroyed.")
"""