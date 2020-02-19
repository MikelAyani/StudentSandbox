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

# Imports
# TODO: In order to use the python 'trimesh' library to deal with mesh files, note that Rtree library needs to be installed manually first in Windows.
# Use the corresponding wheel here: https://www.lfd.uci.edu/~gohlke/pythonlibs/#rtree
# TODO: Add required libraries here

#import numpy as np
#from PIL import Image
#from PIL import ImageOps
#from OpenGL.GL import *
#from OpenGL.GLU import *
#from OpenGL.GLUT import *
#import trimesh

#My first time using GitHub


# Script
def run(
    settings: dict,
    data: dict
    ):

    # To load camera settings
    width = settings.get('width', 800)
    height = settings.get('height', 600)
    print(f"camera size is {width} x {height}")

    for object_name, object_data in data.items():
        num_shapes = len(object_data.get('shapes'))
        print(f"rendering {object_name} with {num_shapes} shapes...")
        print(f"Frame of {object_name} is {object_data.get('frame')}")
        for shape_name in object_data['shapes']:
            origin = object_data['shapes'][shape_name].get('origin', [0, 0, 0, 0, 0, 0, 1])
            print(origin)
            #print(object_data['shapes'][shape_name]['origin'])
        
    filename = settings.get('output_path', 'None')
    print(f"saving results to {filename}...")


