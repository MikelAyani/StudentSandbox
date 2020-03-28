import cv2
from PIL import Image
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
import glfw
from PIL import ImageOps
from gltflib import GLTF
import struct

_dtypes = {5120: "<i1",5121: "<u1",5122: "<i2",5123: "<u2",5125: "<u4",5126: "<f4"}

# GLTF data formats: numpy shapes
_shapes = {"SCALAR": 1,"VEC2": (2),"VEC3": (3),"VEC4": (4),"MAT2": (2, 2),"MAT3": (3, 3),"MAT4": (4, 4)}


DISPLAY_WIDTH = 900
DISPLAY_HEIGHT = 900
INSTANTIATE_WINDOW = True # Setting this to True allows rendering 10x times faster
USE_PIL = False # OpenCV seems to be faster than PIL
image_on = False

def init_gltf():
    # Initialize the library
    if not glfw.init():
        return
    # Set window hint NOT visible
    glfw.window_hint(glfw.VISIBLE, False)
    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(DISPLAY_WIDTH, DISPLAY_HEIGHT, "hidden window", None, None)
    if not window:
        glfw.terminate()
        return

    # Make the window's context current
    glfw.make_context_current(window)
    return window

def clean_gltf(window):
    glfw.destroy_window(window)
    glfw.terminate()

def draw_glb():
    global image_on
    filename = "GLB/duck_vhacd.glb"
    gltf = GLTF.load_glb(filename)    

    for accessor in gltf.model.accessors:
        
        # number of items
        count = accessor.count
        # what is the datatype
        dtype = _dtypes[accessor.componentType]
        # basically how many columns 1,2,3
        per_item = _shapes[accessor.type]
        # use reported count to generate shape
        shape = np.append(count, per_item)
        # number of items when flattened
        # i.e. a (4, 4) MAT4 has 16
        per_count = np.abs(np.product(per_item))
        # data was stored in a buffer view so get raw bytes
        bufferview = gltf.model.bufferViews[accessor.bufferView]
        buffer = gltf.resources[bufferview.buffer]
        # is the accessor offset in a buffer
        start = bufferview.byteOffset
        # length is the number of bytes per item times total
        length = np.dtype(dtype).itemsize * count * per_count
        # load the bytes data into correct dtype and shape
 
        if per_item == 3:
            Vertex = np.frombuffer(buffer.data[start:start + length], dtype=dtype).reshape(shape)
        if per_item == 2:
            UV = np.frombuffer(buffer.data[start:start + length], dtype=dtype).reshape(shape) 
            image_on = True
        if per_item == 1 and image_on == True:
            Faces = np.frombuffer(buffer.data[start:start + length], dtype=dtype).reshape(-1,3)
            texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)	
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            img = Image.open('GLB/reee.png')
            img_data = np.array(list(img.getdata()), np.int8)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 512, 512, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data);
            glGenerateMipmap(GL_TEXTURE_2D);
            glBegin(GL_TRIANGLES)
            for a in range(len(Faces)):
                glTexCoord2dv(UV[Faces[a,0]])
                glVertex3fv(100*Vertex[Faces[a,0]])
                glTexCoord2dv(UV[Faces[a,1]])
                glVertex3fv(100*Vertex[Faces[a,1]])
                glTexCoord2dv(UV[Faces[a,2]])
                glVertex3fv(100*Vertex[Faces[a,2]])
            glEnd()
            glBindTexture(GL_TEXTURE_2D, 0)
        if per_item == 1 and image_on == False:
            Faces = np.frombuffer(buffer.data[start:start + length], dtype=dtype).reshape(-1,3)
            glBegin(GL_TRIANGLES)
            for a in range(len(Faces)):
                glVertex3fv(100*Vertex[Faces[a,0]])
                glVertex3fv(100*Vertex[Faces[a,1]])
                glVertex3fv(100*Vertex[Faces[a,2]])
            glEnd()


def main():

    if not INSTANTIATE_WINDOW:
        window = init_gltf()

    gluPerspective(45, (float(DISPLAY_WIDTH) / float(DISPLAY_HEIGHT)), 0.01, 12)

    glEnable(GL_TEXTURE_2D)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glTranslatef(0, 0, -5) # Move to position

    # Draw duck
    draw_glb()

    image_buffer = glReadPixels(0, 0, DISPLAY_WIDTH, DISPLAY_HEIGHT, OpenGL.GL.GL_RGB, OpenGL.GL.GL_UNSIGNED_BYTE)
    image = np.frombuffer(image_buffer, dtype=np.uint8).reshape(DISPLAY_WIDTH, DISPLAY_HEIGHT, 3)

    if USE_PIL:
        im = Image.fromarray(image)
        im = ImageOps.flip(im)
        im.save('image.png', "PNG")        
    else:
        image = cv2.flip(image,0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(r"image.png", image)

    if not INSTANTIATE_WINDOW:
        clean_gltf(window)


if __name__ == "__main__":
    import time
    now = time.perf_counter()

    # Create Window
    if INSTANTIATE_WINDOW:
        window = init_gltf()

    main()

    print()

    # Close
    if INSTANTIATE_WINDOW:
        clean_gltf(window)