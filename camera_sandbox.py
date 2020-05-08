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
from oep_quaternion import Transform2EulerOpenGL

"""
This is a sandbox to test the synthetic camera functionality in Simumatik Open Emulation Platform.
Usage:
The camera class is a thread wich will render the scene data sent through the pipe.
"""

class synthetic_camera(threading.Thread):

    def __init__(self, name:str='camera', pipe=None, width:int=800, height:int=600, vertical_fov:float=45.0, near:float=0.1, far:float=5.0, image_format:str='RGB', output_path:str='', send_response:bool=False):
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
            window = glfw.create_window(self.width, self.height, "hidden window", None, None)
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
        self.frame = Transform2EulerOpenGL(frame)

    def get_Texture(self, data, bytes=False):
        """ Creates texture to apply to any figure given the image data. By default the image data is given by a path,
        but it can be given also as bytes if the variable 'bytes' is True"""
        
        if bytes == False:
            texture_data = png.Reader(filename=data).asRGB()
            tex_array = np.vstack(map(np.uint8, texture_data[2])).reshape(texture_data[0]*texture_data[1], 3)
        else:
            texture_data = png.Reader(bytes=data).read_flat()
            tex_array = np.array(texture_data[2], np.uint8).reshape(texture_data[0]*texture_data[1], 3)
        
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texture_data[0], texture_data[1], 0, GL_RGB, GL_FLOAT, tex_array/255)
        glGenerateMipmap(GL_TEXTURE_2D)

    def render_Box(self, vectorLWH, material):
        """ Render a box given vector[Length, Width, Height]"""
        L=vectorLWH[0]/2
        W=vectorLWH[1]/2
        H=vectorLWH[2]/2

        if isinstance(material, list): 
            glColor4f(material[0], material[1], material[2], material[3])
            glBegin(GL_QUADS)
            glVertex3f(-L, -H, -W);  glVertex3f(L, -H, -W); glVertex3f(L, -H, W); glVertex3f(-L, -H, W) #1 2 3 4
            glVertex3f(-L, H, -W); glVertex3f(L, H, -W); glVertex3f(L, H, W); glVertex3f(-L, H, W) #5 6 7 8
            glVertex3f(-L, -H, -W); glVertex3f(L, -H, -W); glVertex3f(L, H, -W); glVertex3f(-L, H, -W) #1 2 6 5
            glVertex3f(-L, -H, W); glVertex3f(L, -H, W); glVertex3f(L, H, W); glVertex3f(-L, H, W) #4 3 7 8
            glVertex3f(L, -H, -W); glVertex3f(L, -H, W); glVertex3f(L, H, W); glVertex3f(L, H, -W) #2 3 7 6
            glVertex3f(-L, -H, -W); glVertex3f(-L, -H, W); glVertex3f(-L, H, W); glVertex3f(-L, H, -W) #1 4 8 5
            glEnd()
        else:
            self.get_Texture(material)
            glBegin(GL_QUADS)
            glTexCoord2f(0.0, 0.0);glVertex3f(-L, -H, -W);glTexCoord2f(1.0, 0.0);glVertex3f(L, -H, -W);glTexCoord2f(1.0, 1.0);glVertex3f(L, -H, W);glTexCoord2f(0.0, 1.0);glVertex3f(-L, -H, W) #1 2 3 4
            glTexCoord2f(0.0, 0.0);glVertex3f(-L, H, -W);glTexCoord2f(1.0, 0.0);glVertex3f(L, H, -W);glTexCoord2f(1.0, 1.0);glVertex3f(L, H, W);glTexCoord2f(0.0, 1.0);glVertex3f(-L, H, W) #5 6 7 8
            glTexCoord2f(0.0, 0.0);glVertex3f(-L, -H, -W);glTexCoord2f(1.0, 0.0);glVertex3f(L, -H, -W);glTexCoord2f(1.0, 1.0);glVertex3f(L, H, -W);glTexCoord2f(0.0, 1.0);glVertex3f(-L, H, -W) #1 2 6 5
            glTexCoord2f(0.0, 0.0);glVertex3f(-L, -H, W);glTexCoord2f(1.0, 0.0);glVertex3f(L, -H, W);glTexCoord2f(1.0, 1.0);glVertex3f(L, H, W);glTexCoord2f(0.0, 1.0);glVertex3f(-L, H, W) #4 3 7 8
            glTexCoord2f(0.0, 0.0);glVertex3f(L, -H, -W);glTexCoord2f(1.0, 0.0);glVertex3f(L, -H, W);glTexCoord2f(1.0, 1.0);glVertex3f(L, H, W);glTexCoord2f(0.0, 1.0);glVertex3f(L, H, -W) #2 3 7 6
            glTexCoord2f(0.0, 0.0);glVertex3f(-L, -H, -W);glTexCoord2f(1.0, 0.0);glVertex3f(-L, -H, W);glTexCoord2f(1.0, 1.0);glVertex3f(-L, H, W);glTexCoord2f(0.0, 1.0);glVertex3f(-L, H, -W) #1 4 8 5
            glEnd()
            glBindTexture(GL_TEXTURE_2D, 0)

    def render_Capsule(self, radius, length, material):
        """ Render capsule given radius and length"""
        if isinstance(material, list): 
            glColor4f(material[0], material[1], material[2], material[3])
            capsuleQ = gluNewQuadric()
            glTranslatef(0, 0, -length/2)
            gluCylinder(capsuleQ, radius, radius, length, 100, 100)
            gluSphere(capsuleQ, radius, 36, 18)
            glTranslatef(0, 0, length)
            gluSphere(capsuleQ, radius, 36, 18)
        else:
            self.get_Texture(material)
            capsuleQ = gluNewQuadric()
            gluQuadricNormals(capsuleQ, GLU_SMOOTH)
            gluQuadricTexture(capsuleQ, GL_TRUE)
            glTranslatef(0, 0, -length/2)
            gluCylinder(capsuleQ, radius, radius, length, 100, 100)
            gluSphere(capsuleQ, radius, 36, 18)
            glTranslatef(0, 0, length)
            gluSphere(capsuleQ, radius, 36, 18)
            glBindTexture(GL_TEXTURE_2D, 0)

    def render_Cylinder(self, radius, length, material):
        """ Render cylinder given radius and length"""
        if isinstance(material, list): 
            glColor4f(material[0], material[1], material[2], material[3])
            cylinderQ = gluNewQuadric()
            glTranslatef(0, 0, -length/2)
            gluCylinder(cylinderQ, radius, radius, length, 100, 100)
            gluDisk(cylinderQ, 0.0, radius, 64, 1) 
            glTranslatef(0, 0, length)
            gluDisk(cylinderQ, 0.0, radius, 64, 1)
        else:
            self.get_Texture(material)
            cylinderQ = gluNewQuadric()
            gluQuadricNormals(cylinderQ, GLU_SMOOTH)
            gluQuadricTexture(cylinderQ, GL_TRUE)
            glTranslatef(0, 0, -length/2)
            gluCylinder(cylinderQ, radius, radius, length, 100, 100)
            gluDisk(cylinderQ, 0.0, radius, 64, 1) 
            glTranslatef(0, 0, length)
            gluDisk(cylinderQ, 0.0, radius, 64, 1)
            glBindTexture(GL_TEXTURE_2D, 0)

    def render_Sphere(self, radius, material):
        sphereQ = gluNewQuadric()
        if isinstance(material, list):
            glColor4f(material[0], material[1], material[2], material[3])
            gluSphere(sphereQ, radius, 36, 18)
        else: 
            self.get_Texture(material)
            gluQuadricNormals(sphereQ, GLU_SMOOTH)
            gluQuadricTexture(sphereQ, GL_TRUE)
            gluSphere(sphereQ, radius, 36, 18)
            glBindTexture(GL_TEXTURE_2D, 0)        

    def render_Plane(self, normal):
        """ Render 2D plane given its normal vector[x, y, z]"""
        xAngle=np.arctan2(normal[1],normal[2]) * (180/np.pi)
        yAngle=np.arctan2(normal[2],normal[1]) * (180/np.pi)
        zAngle=np.arctan2(normal[0],normal[1]) * (180/np.pi)
        glRotatef(xAngle, 1, 0, 0)
        glRotatef(yAngle, 0, 1, 0)
        glRotatef(zAngle, 0, 0, 1)
        glBegin(GL_QUADS)
        glColor4f(0.953, 0.953, 0.953, 1)
        glVertex3f(-100, 0, 100);glVertex3f(100, 0, 100);glVertex3f(100, 0, -100);glVertex3f(-100, 0, -100)
        glEnd()
        
    def render_glb_with_textures(self, glb, primitive, scale):
        """ Render glb primitive with textures"""
        vertices = primitive['POSITION']
        new_vertices = np.zeros([len(vertices), 3])
        new_vertices[:, 0] = vertices[:, 0]
        new_vertices[:, 1] = vertices[:, 2]
        new_vertices[:, 2] = vertices[:, 1]
        vertices = new_vertices
        faces = np.reshape(primitive['indices'], (-1, 3))
        UV = primitive['TEXCOORD_0']
        text_ID = primitive['material'].pbrMetallicRoughness.baseColorTexture.index
        _, texture_bytes = get_texture(glb, text_ID)

        self.get_Texture(texture_bytes, bytes=True)
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
        """ Render glb primitive without colors or textures"""
        vertices = primitive['POSITION']
        new_vertices = np.zeros([len(vertices), 3])
        new_vertices[:, 0] = vertices[:, 0]
        new_vertices[:, 1] = vertices[:, 2]
        new_vertices[:, 2] = vertices[:, 1]
        vertices = new_vertices
        faces = np.reshape(primitive['indices'], (-1, 3))
        
        try:
            color = primitive['COLOR_0'][0]/255
        except:
            if primitive['material'] is not None and primitive['material'].pbrMetallicRoughness.baseColorFactor is not None:
                color = primitive['material'].pbrMetallicRoughness.baseColorFactor
            else:
                color = [0.765, 0.765, 0.765, 1]
        
        glBegin(GL_TRIANGLES)       
        glColor4f(color[0], color[1], color[2], color[3])
        for a in range(len(faces)):
            glVertex3fv(scale*vertices[faces[a,0]])
            glVertex3fv(scale*vertices[faces[a,1]])
            glVertex3fv(scale*vertices[faces[a,2]])
        glEnd()

    def render_glb(self, path_to_file, scale):
        """ Render glb archive limited main nodes and their children"""

        try:
            glb = GLTF.load_glb(path_to_file, load_file_resources=True)
            
            if glb.model.scene is not None:
                scene = glb.model.scenes[glb.model.scene]
                for main_node in scene.nodes:
                    #First level   
                    glPushMatrix()                 
                    translation_node, rotation_node, scale_node = get_node_TRS(glb, main_node)
                    frame_main_node = Transform2EulerOpenGL(np.append(translation_node,rotation_node))
                    scale_main_node = [scale[0]*scale_node[0], scale[2]*scale_node[2], scale[1]*scale_node[1]] #Modified because OpenGL has the axis changed respected to Simumatik
                    node_mesh = glb.model.nodes[main_node].mesh
                    glTranslatef(frame_main_node[0], frame_main_node[1], frame_main_node[2])
                    glRotatef(-frame_main_node[5], 1, 0, 0); glRotatef(-frame_main_node[3], 0, 0, 1); glRotatef(-frame_main_node[4], 0, 1, 0) 

                    # If node has a mesh
                    if node_mesh is not None:
                        mesh_data = get_mesh_data(glb, node_mesh, vertex_only=False)
                        # A mesh may have several primitives
                        for primitive in mesh_data['primitives']:
                            try:
                                if primitive['TEXCOORD_0'] is not None and self.format != 'D' and primitive['material'].pbrMetallicRoughness.baseColorTexture is not None:
                                    self.render_glb_with_textures(glb, primitive, scale_main_node)
                                else:
                                    self.render_glb_without_textures(primitive, scale_main_node)
                            except:
                                pass

                    # Second level
                    if glb.model.nodes[main_node].children is not None:
                        # A node may have several child nodes
                        for child_node in glb.model.nodes[main_node].children:
                            translation_node, rotation_node, scale_node = get_node_TRS(glb, child_node)
                            frame_child_node = Transform2EulerOpenGL(np.append(translation_node,rotation_node))
                            scale_child_node = [scale[0]*scale_node[0], scale[2]*scale_node[2], scale[1]*scale_node[1]] #Modified because OpenGL has the axis changed respected to Simumatik
                            child_node_mesh = glb.model.nodes[child_node].mesh
                            glPushMatrix()
                            glTranslatef(frame_child_node[0], frame_child_node[1], frame_child_node[2])
                            glRotatef(-frame_child_node[5], 1, 0, 0); glRotatef(-frame_child_node[3], 0, 0, 1); glRotatef(-frame_child_node[4], 0, 1, 0) 
                            # If child node has a mesh
                            if child_node_mesh is not None:
                                mesh_data = get_mesh_data(glb, child_node_mesh, vertex_only=False)
                                # A mesh may have several primitives
                                for primitive in mesh_data['primitives']:
                                    try:
                                        if primitive['TEXCOORD_0'] is not None and self.format != 'D' and primitive['material'].pbrMetallicRoughness.baseColorTexture is not None:
                                            self.render_glb_with_textures(glb, primitive, scale_child_node)
                                        else:
                                            self.render_glb_without_textures(primitive, scale_child_node)
                                    except:
                                        pass
                            glPopMatrix()    

                    glPopMatrix()

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
                        'material': (optional) shape material. May include one of the following options.
                            'color': (float[4]) rgba values
                            'texture': (str) path to the texture (PNG, JPG,... file)
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
        print("Rendering from sandbox...")
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()            
        gluPerspective(self.vertical_fov, self.width/self.height, self.near, self.far)
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

        # Setup Camera
        glLoadIdentity()  
        glRotatef(90, 0, 1, 0) #This is required to ensure the camera is pointing on X axis
        glRotatef(self.frame[4], 0, 1, 0); glRotatef(self.frame[3], 0, 0, 1); glRotatef(self.frame[5], 1, 0, 0)
        glTranslatef(-self.frame[0], -self.frame[1], -self.frame[2])
        
        # To load objects data
        for _, object_data in data.items():
            obj_frame = Transform2EulerOpenGL(object_data.get('frame', [0, 0, 0, 0, 0, 0, 1]))

            glPushMatrix()
            glTranslatef(obj_frame[0], obj_frame[1], obj_frame[2])
            glRotatef(-obj_frame[5], 1, 0, 0); glRotatef(-obj_frame[3], 0, 0, 1); glRotatef(-obj_frame[4], 0, 1, 0)

            for shape_name in object_data['shapes']:
                shape_orig = Transform2EulerOpenGL(object_data['shapes'][shape_name].get('origin', [0, 0, 0, 0, 0, 0, 1]))

                glPushMatrix()
                glTranslatef(shape_orig[0], shape_orig[1], shape_orig[2])
                glRotatef(-shape_orig[5], 1, 0, 0); glRotatef(-shape_orig[3], 0, 0, 1); glRotatef(-shape_orig[4], 0, 1, 0)

                shape_type = object_data['shapes'][shape_name].get('type')
                
                if object_data['shapes'][shape_name].get('material') is not None:
                    if object_data['shapes'][shape_name]['material'].get('color') is not None:
                        shape_material = object_data['shapes'][shape_name]['material'].get('color')
                    elif object_data['shapes'][shape_name]['material'].get('texture') is not None and self.format != 'D': 
                        shape_material = object_data['shapes'][shape_name]['material'].get('texture')
                        print("path to PNG: " + shape_material)
                    else:
                        shape_material = [0.765, 0.765, 0.765, 1]
                else:
                    shape_material = [0.765, 0.765, 0.765, 1]


                if shape_type == 'plane':
                    self.render_Plane(object_data['shapes'][shape_name]['attributes'].get('normal'))
                elif shape_type == 'box':
                    self.render_Box(object_data['shapes'][shape_name]['attributes'].get('sizes'), shape_material)
                elif shape_type == 'cylinder':
                    self.render_Cylinder(object_data['shapes'][shape_name]['attributes'].get('radius'), object_data['shapes'][shape_name]['attributes'].get('length'), shape_material)
                elif shape_type == 'capsule':
                    self.render_Capsule(object_data['shapes'][shape_name]['attributes'].get('radius'), object_data['shapes'][shape_name]['attributes'].get('length'), shape_material)
                elif shape_type == 'sphere':
                    self.render_Sphere(object_data['shapes'][shape_name]['attributes'].get('radius'), shape_material)
                elif shape_type == 'mesh':
                    self.render_glb(object_data['shapes'][shape_name]['attributes'].get('model'), object_data['shapes'][shape_name]['attributes'].get('scale'))
                
                glPopMatrix()

            glPopMatrix()

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