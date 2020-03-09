"""
This is a sandbox to test the modeling of human body using VR data in Simumatik Open Emulation Platform.

Usage:
The run method inside this script will be called each time the human body is required to be recalculated.

transform: refers to a bullet transform including position and transformation (quaternion) data.

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
from smtk_bullet import (Vector3, Quaternion, Transform)
import math

RAD2DEG = 180.0/math.pi

# Script
def run(
    vr_data: dict
    ):
    # Example reading Head data
    head_transform = vr_data.get('head')
    pos = head_transform.getOrigin()
    rot = head_transform.getRotation()
    print(f'Head position: {pos.x} {pos.y} {pos.z} (in meters)')
    rot_euler = rot.getEulerAngles()
    print(f'Head rotation euler angles: {rot_euler.x} {rot_euler.y} {rot_euler.z} (in radians)')

    # Example response
    link_1_transform = Transform()
    link_1_transform.setOrigin(Vector3(0, 0, 0))
    link_1_transform.setRotation(Quaternion.fromScalars(0, 0, 0, 1))
    res = {'link_1': {'radius':0.02, 'length':0.3, 'transform':link_1_transform}}

    return res