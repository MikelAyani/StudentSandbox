"""
This is a sandbox to test the synthetic camera functionality in Simumatik Open Emulation Platform.

Usage:
The run method inside this script will be called each time the camera is required to render a scene.

transform: refers to a vector including position and transformation (quaternion) data [x, y, z, qx, qy, qz, qw].

settings:
    - Width (int): camera width pixels 
    - ...

data:
    
"""

# Imports
# TODO: In order to use the python 'trimesh' library to deal with mesh files, note that Rtree library needs to be installed manually first in Windows.
# Use the corresponding wheel here: https://www.lfd.uci.edu/~gohlke/pythonlibs/#rtree
# TODO: Add required libraries here

#My first time using GitHub


# Script
def run(
    settings: dict,
    data: dict
    ):

    print(settings)
    print(data)