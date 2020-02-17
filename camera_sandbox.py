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
# TODO: Add required libraries here


# Script
def run(
    settings: dict,
    data: dict
    ):

    print(settings)
    print(data)