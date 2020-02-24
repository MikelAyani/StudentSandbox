import cv2
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
import glfw


DISPLAY_WIDTH = 900
DISPLAY_HEIGHT = 900

INSTANTIATE_WINDOW = True # Setting this to True allows rendering 10x times faster


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


def main():

    if not INSTANTIATE_WINDOW:
        window = init_gltf()

    gluPerspective(90, (DISPLAY_WIDTH / DISPLAY_HEIGHT), 0.01, 12)

    glEnable(GL_TEXTURE_2D)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)

    glRotatef(-90, 1, 0, 0) # Straight rotation
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glRotatef(285, 0, 0, 1) # Rotate yaw
    glTranslatef(-5, -3, -2) # Move to position

    # Draw rectangle
    glBegin(GL_QUADS)
    glColor3f(1, 0, 0)
    glVertex3f(2, 2, 0)
    glVertex3f(2, 2, 2)
    glVertex3f(2, 6, 2)
    glVertex3f(2, 6, 0)
    glEnd()

    image_buffer = glReadPixels(0, 0, DISPLAY_WIDTH, DISPLAY_HEIGHT, OpenGL.GL.GL_RGB, OpenGL.GL.GL_UNSIGNED_BYTE)
    image = np.frombuffer(image_buffer, dtype=np.uint8).reshape(DISPLAY_WIDTH, DISPLAY_HEIGHT, 3)

    cv2.imwrite(r"image.png", image)

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
    while time.perf_counter()-1 <= 1.0:
        main()
        count += 1
    print(count)

    # Close
    if INSTANTIATE_WINDOW:
        clean_gltf(window)