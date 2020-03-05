"""
This is a sandbox to test the synthetic camera functionality in Simumatik Open Emulation Platform.

Usage:
The run method inside this script will be called each time the camera is required to render a scene.

transform: refers to a vector including position and transformation (quaternion) data [x, y, z, qx, qy, qz, qw].

settings: A dictionary containing the camera settings
    'frame': (float[7]) camera global frame [x, y, z, qx, qy, qz, qw],
    'width': (int) camera width pixels, 
    'height': (int) camera height pixels,
    'vertical_fov':  (float) camera vertical fov in degrees,
    'near':  (float) camera near plane,
    'far':  (float) camera far plane,
    'format':  (str) camera image format: 'L', 'RGB', 'D' or 'RGBD',
    'output_path': (str) camera image output path (filename),

data: A dictionary containing objects to be rendered.
    'object_name':
        'frame': (float[7]) object global frame [x, y, z, qx, qy, qz, qw],
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

"""

'''
HELPER METHODS
'''

def Box(vectorLHW):
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

def Capsule(radius, length):
    capsuleQ = gluNewQuadric()
    gluCylinder(capsuleQ, radius, radius, length, 100, 100)
    gluSphere(capsuleQ, radius, 36, 18)
    glTranslatef(0, 0, length)
    gluSphere(capsuleQ, radius, 36, 18)
    glTranslatef(0, 0, -length) #To get the camera as it was, just in case we don't use glPushMatrix before calling capsule

def Cylinder(radius, length):
    cylinderQ = gluNewQuadric()
    gluCylinder(cylinderQ, radius, radius, length, 100, 100)
    gluSphere(cylinderQ, radius, 36, 2)  
    glTranslate(0, 0, length)
    gluSphere(cylinderQ, radius, 36, 2)  
    glTranslate(0, 0, -length)  

def Plane(normal):
    global camera_far
    xAngle=np.arctan2(normal[2],normal[1]) * (180/np.pi)
    yAngle=np.arctan2(normal[0],normal[2]) * (180/np.pi)
    zAngle=np.arctan2(normal[0],normal[1]) * (180/np.pi)
    glRotatef(xAngle, 1, 0, 0)
    glRotatef(yAngle, 0, 1, 0)
    glRotatef(zAngle, 0, 0, 1)
    glBegin(GL_QUADS)
    glColor3f(1, 0, 0)
    glVertex3f(-camera_far, 0, camera_far)        
    glVertex3f(camera_far, 0, camera_far)        
    glVertex3f(camera_far, 0, -camera_far)        
    glVertex3f(-camera_far, 0, -camera_far)
    glEnd()


'''
MAIN SCRIPT
'''

# To load camera settings
'''
camera_frame = Transform2Euler(settings.get('frame', [0, 0, 0, 0, 0, 0, 1]))
camera_windowWidth = settings.get('width', 800)
camera_windowHeight = settings.get('height', 600)
camera_vertical_fov = settings.get('vertical_fov', 45.0)
camera_near = settings.get('near', 0.1)
camera_far = settings.get('far', 100.0)
camera_format = settings.get('format', 'RGB')
camera_output_path = settings.get('output_path', 'None')
'''

array_textures = [0, 0]
textures = glGenTextures(2, array_textures)
glBindTexture(GL_TEXTURE_2D, textures[0])
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, camera_windowWidth, camera_windowHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
glBindTexture(GL_TEXTURE_2D, 0)

glBindTexture(GL_TEXTURE_2D, textures[1])
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, camera_windowWidth, camera_windowHeight, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)
glBindTexture(GL_TEXTURE_2D, 0)

array_fbo = [0]
fbo = glGenFramebuffers(1, array_fbo)
glBindFramebuffer(GL_FRAMEBUFFER, fbo)
glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textures[0], 0)
glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, textures[1], 0)

glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT) 

# To load objects data
for object_name, object_data in data.items():
    obj_frame = Transform2Euler(object_data.get('frame', [0, 0, 0, 0, 0, 0, 1]))

    for shape_name in object_data['shapes']:
        shape_orig = Transform2Euler(object_data['shapes'][shape_name].get('origin', [0, 0, 0, 0, 0, 0, 1]))
        shape_type = object_data['shapes'][shape_name].get('type')

        glLoadIdentity()
        glTranslatef(camera_frame[0], camera_frame[1], camera_frame[2])
        glRotatef(camera_frame[3], 1, 0, 0);  glRotatef(camera_frame[4], 0, 1, 0);  glRotatef(camera_frame[5], 0, 0, 1)
        glRotatef(-90, 1, 0, 0)

        glPushMatrix()
        glTranslatef(obj_frame[0]+shape_orig[0], obj_frame[1]+shape_orig[1], obj_frame[2]+shape_orig[2])
        glRotatef(obj_frame[3]+shape_orig[3], 1, 0, 0); glRotatef(obj_frame[4]+shape_orig[4], 0, 1, 0); glRotatef(obj_frame[5]+shape_orig[5], 0, 0, 1)

        if shape_type == 'plane':
            Plane(object_data['shapes'][shape_name]['attributes'].get('normal'))
        elif shape_type == 'box':
            Box(object_data['shapes'][shape_name]['attributes'].get('sizes'))
        elif shape_type == 'cylinder':
            Cylinder(object_data['shapes'][shape_name]['attributes'].get('radius'), object_data['shapes'][shape_name]['attributes'].get('length'))
        elif shape_type == 'capsule':
            Capsule(object_data['shapes'][shape_name]['attributes'].get('radius'), object_data['shapes'][shape_name]['attributes'].get('length'))
        elif shape_type == 'sphere':
            sphereQ = gluNewQuadric()
            gluSphere(sphereQ, object_data['shapes'][shape_name]['attributes'].get('radius'), 36, 18)
        glPopMatrix()

glBindFramebuffer(GL_FRAMEBUFFER, 0)

# Generate RGB image
if camera_format == 'RGB':
    glBindTexture(GL_TEXTURE_2D, textures[0])
    color_str = glGetTexImage(GL_TEXTURE_2D, 0, GL_BGR, GL_UNSIGNED_BYTE)
    glBindTexture(GL_TEXTURE_2D, 0)
    color_data = np.frombuffer(color_str, dtype=np.uint8)
    matColor = np.frombuffer(color_data, dtype=np.uint8).reshape(camera_windowHeight, camera_windowWidth, 3)
    #matColor = cv2.cvtColor(matColor, cv2.COLORBGR2GRAY)
    cv2.imwrite(settings.get('output_path'), cv2.flip(matColor, 0))

# Generate Grayscale image
elif camera_format == 'L':
    glBindTexture(GL_TEXTURE_2D, textures[0])
    color_str = glGetTexImage(GL_TEXTURE_2D, 0, GL_BGR, GL_UNSIGNED_BYTE)
    glBindTexture(GL_TEXTURE_2D, 0)
    color_data = np.frombuffer(color_str, dtype=np.uint8)
    matColor = np.frombuffer(color_data, dtype=np.uint8).reshape(camera_windowHeight, camera_windowWidth, 3)
    matL = cv2.cvtColor(matColor, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(settings.get('output_path'), cv2.flip(matL, 0))

# Generate Depth image
elif camera_format == 'D':
    glBindTexture(GL_TEXTURE_2D, textures[1])
    depth_str = glGetTexImage(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, GL_FLOAT)
    glBindTexture(GL_TEXTURE_2D, 0)
    depth_data = np.frombuffer(depth_str, dtype=np.float32)
    # Linearize depth values
    z = depth_data*2.0 - 1.0
    linearDepth = (2.0 * camera_near * camera_far) / (camera_far + camera_near - z * (camera_far - camera_near))
    linearDepth = linearDepth/camera_far
    # Resize 1D matrix to 2D matrix
    matD = np.reshape((255-255*linearDepth).astype(np.uint8), (camera_windowHeight, camera_windowWidth))
    cv2.imwrite(settings.get('output_path'), cv2.flip(matD, 0))