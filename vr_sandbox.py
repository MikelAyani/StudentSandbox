"""
This is a sandbox to test the modeling of human body using VR data in Simumatik Open Emulation Platform.

Usage:
The run method inside this script will be called each time the human body is required to be recalculated.

transform: refers to a vector including position and transformation (quaternion) data [x, y, z, qx, qy, qz, qw].

vr_data:
    - head (transform): Head transformation in global coordinates. 
    - right_hand (transform): Right_hand transformation in global coordinates. 
    - left_hand (transform): Left_hand transformation in global coordinates. 

return: The script should return a dictionary with the links data, including
    - radius: link visual shape capsule radius
    - length: link visual shape capsule length
    - transform: link visual shape capsule transform
"""

# Imports
# TODO: Add required libraries here


# Script
def run(
    vr_data: dict
    )->dict:

    print(vr_data)

    res = {'link_1': {'radius':0.02, 'length':0.3, 'transform':[0 0 0 0 0 0 1]}}

    return res