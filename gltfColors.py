import cv2
import math
from PIL import Image
from PIL import ImageOps
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
import glfw

settings = {
    'frame': [0, 0, -10, 0, 0, 0, 1],
    'width': int(800), # TODO: Remove this temporary solution when DM fixed
    'height': int(600),
    'vertical_fov': 45.0,
    'near':  0.1,
    'far':  100.0,
    'format':  'D',
    'output_path': "C:/Users/Simumatik/Simumatik/foto.png"
    }
data = {
    'floor': {
        'frame': [0, -1, 0, 0, 0, 0, 1],
        'shapes': {
            'plane':{
                'type': 'plane',
                'attributes': {
                    'normal': [0.0, 1.0, 0.0]
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
    }
}

INSTANTIATE_WINDOW = False # Setting this to True allows rendering 10x times faster
camera_windowWidth, camera_windowHeight = settings.get('width', 800), settings.get('height', 600)
fbo, depth_texture = None, None

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

def init_gltf():
    global camera_windowWidth, camera_windowHeight
    # Initialize the library
    if not glfw.init():
        return
    # Set window hint NOT visible
    glfw.window_hint(glfw.VISIBLE, False)
    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(camera_windowWidth, camera_windowHeight, "hidden window", None, None)
    if not window:
        glfw.terminate()
        return

    # Make the window's context current
    glfw.make_context_current(window)
    return window

def clean_gltf(window):
    glfw.destroy_window(window)
    glfw.terminate()

def run(settings, data):
    global camera_near, camera_far, camera_vertical_fov, camera_windowWidth, camera_windowHeight, i

    # To load camera settings
    camera_frame = Transform2Euler(settings.get('frame', [0, 0, 0, 0, 0, 0, 1]))
    camera_windowWidth = settings.get('width', 800)
    camera_windowHeight = settings.get('height', 600)
    camera_vertical_fov = settings.get('vertical_fov', 45.0)
    camera_near = settings.get('near', 0.1)
    camera_far = settings.get('far', 100.0)
    camera_format = settings.get('format', 'RGB')
    camera_output_path = settings.get('output_path', 'None')

    if not INSTANTIATE_WINDOW:
            window = init_gltf()

    glClearColor(0.3, 0.3, 0.3, 0.0)   #RGB Alpha
    glClearDepth(1.0) #Values from 0 to 1 (Stablish the "background" depth)           
    glDepthFunc(GL_LESS)    #Set the mode of the depth buffer
    glEnable(GL_TEXTURE_2D)
    glEnable(GL_DEPTH_TEST)
    glPolygonMode(GL_FRONT, GL_FILL)    
    glPolygonMode(GL_BACK, GL_FILL)     
    glShadeModel(GL_SMOOTH)                
    glMatrixMode(GL_PROJECTION)                 
    gluPerspective(camera_vertical_fov, float(camera_windowWidth)/float(camera_windowHeight), camera_near, camera_far)
    glMatrixMode(GL_MODELVIEW)

    array_textures = [0, 0]
    textures = glGenTextures(2, array_textures)
    glBindTexture(GL_TEXTURE_2D, textures[0])
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, camera_windowWidth, camera_windowHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
    glBindTexture(GL_TEXTURE_2D, 0)

    depth_texture = glGenTextures(1)
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
        
        glLoadIdentity()
        glTranslatef(camera_frame[0], camera_frame[1], camera_frame[2])
        glRotatef(camera_frame[3], 1, 0, 0);  glRotatef(camera_frame[4], 0, 1, 0);  glRotatef(camera_frame[5], 0, 0, 1)

        for shape_name in object_data['shapes']:
            shape_orig = Transform2Euler(object_data['shapes'][shape_name].get('origin', [0, 0, 0, 0, 0, 0, 1]))
            shape_type = object_data['shapes'][shape_name].get('type')

            # Move to shape
            glLoadIdentity()
            glTranslatef(camera_frame[0], camera_frame[1], camera_frame[2])
            glRotatef(camera_frame[3], 1, 0, 0);  glRotatef(camera_frame[4], 0, 1, 0);  glRotatef(camera_frame[5], 0, 0, 1)

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
    
    #Generate RGB image
        #Obtain the color data in a numpy array
    glBindTexture(GL_TEXTURE_2D, textures[0])
    color_str = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE)
    glBindTexture(GL_TEXTURE_2D, 0)
    color_data = np.frombuffer(color_str, dtype=np.uint8)
    matColor = np.frombuffer(color_data, dtype=np.uint8).reshape(camera_windowHeight, camera_windowWidth, 3)
    #matColor = cv2.cvtColor(matColor, cv2.COLORBGR2GRAY)
    cv2.imwrite("C:/Users/Simumatik/Simumatik/imageRGB.png", cv2.flip(matColor, 0))

    #Generate D image
        #Obtain the depth data in a numpy array
    glBindTexture(GL_TEXTURE_2D, textures[1])
    depth_str = glGetTexImage(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, GL_FLOAT)
    glBindTexture(GL_TEXTURE_2D, 0)
    depth_data = np.frombuffer(depth_str, dtype=np.float32)
        #Linearize depth values
    z = depth_data*2.0 - 1.0
    linearDepth = (2.0 * camera_near * camera_far) / (camera_far + camera_near - z * (camera_far - camera_near))
    linearDepth = linearDepth/camera_far
        #Resize 1D matrix to 2D matrix
    matD = np.reshape((255-255*linearDepth).astype(np.uint8), (camera_windowHeight, camera_windowWidth))
    cv2.imwrite("C:/Users/Simumatik/Simumatik/imageD.png", cv2.flip(matD, 0))

    #glDeleteFramebuffers(1, fbo) 
    #glDeleteTextures(2, textures) 

    if not INSTANTIATE_WINDOW:
        clean_gltf(window)


if __name__ == "__main__":
    import time
    now = time.perf_counter()

    # Create Window
    if INSTANTIATE_WINDOW:
        window = init_gltf()

    # Loop
    count = 0
    while time.perf_counter()-now <= 1:
        run(settings, data)
        count += 1
    print(count)

    # Close
    if INSTANTIATE_WINDOW:
        clean_gltf(window)