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
from smtk_bullet import (Vector3, Quaternion, Transform)
from math import pi
import math

# import copy

RAD2DEG = 180.0 / math.pi


# Script
def Calibration(right_hand_trigger, left_hand_trigger):
    if right_hand_trigger == 1 and left_hand_trigger == 1:
        posrighthand = r_hand_transform.getOrigin()  # Gets the position of the right hand
        poslefthand = l_hand_transform.getOrigin()  # Gets the position of the left hand
        vector = posrighthand - poslefthand  # Vector3 that goes between the controllers
        distance = math.sqrt(vector.x ** 2 + vector.y ** 2 + vector.z ** 2)
    return a, c    # Return the length of the arms, and the height of the person

def CompleteArm(neck_transform, hand_transform, right_left):
    a = 0.25  # Lower Arm Length
    c = 0.33  # Upper Arm Length

    if right_left == 0:     # Right Arm
        r_l = 1
    else:                   # Left Arm
        r_l = -1

    # Shoulder
    shoulder_offset = Transform()  # Same as before (Check neck calculations)
    shoulder_offset.setOrigin(Vector3(r_l*0.2, 0, 0))           # Need to test the x,y,z
    shoulder_transform = neck_transform * shoulder_offset

    # Wrist
    wrist_offset = Transform()  # Same as before (Check neck calculations)
    wrist_offset.setOrigin(Vector3(r_l*0.02, 0.025, 0.15))
    wrist_transform = hand_transform * wrist_offset

    # Calculations for elbow
    poswrist = wrist_transform.getOrigin()  # Gets the position of the wrist
    posshoulder = shoulder_transform.getOrigin()  # Gets the position of the shoulder
    armvector = poswrist - posshoulder  # Vector3 that goes from the shoulder to the wrist
    poselbow = (posshoulder + poswrist) * 0.5  # Gets middle position between shoulder and wrist

    b = math.sqrt(armvector.x ** 2 + armvector.y ** 2 + armvector.z ** 2)  # Actual distance between shoulder and wrist
    if b == 0:
        alpha = 0
    elif b >= (a+c):
        b = a+c-0.01
        alpha = math.acos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))  # Angle between forearm and armvector
    else:
        alpha = math.acos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))  # Angle between forearm and armvector

    z = b * 0.5 * math.sin(alpha)  # Variable z
    x = b * 0.5 * math.cos(alpha)  # Variable x
    distA = a - x  # Distance that elbow moves in one direction
    distB = z  # Distance that elbow moves in other direction
    elbowinit = hand_transform  # Create a transform with rotation of the controller (Need to copy instead)!!!!!!!!!!
    elbowinit.setOrigin(poselbow)  # Move the transform into elbowinit position
    elbow_offset = Transform()  # Same as before (Check neck calculations)
    elbow_offset.setOrigin(Vector3(r_l*distB, 0, distA))  # Check "Controller Axis" at the end (need to test)
    elbow_transform_notyet = elbowinit * elbow_offset  # Moves the elbow to the real position(Still need quaternion)

    #Quaternion
    rot_offset = Transform()
    rot_offset.setOrigin(poselbow)
    rot = Quaternion.fromAxisAngle(armvector, r_l * pi/6)
    rot_offset.setRotation(rot)
    elbow_transform = elbow_transform_notyet * rot_offset

    return shoulder_transform, elbow_transform, wrist_transform

def run(
        vr_data: dict
):

    # Head and Hands
    head_transform = vr_data.get('head')
    r_hand_transform = vr_data.get('right_hand')
    l_hand_transform = vr_data.get('left_hand')
    # right_hand_trigger = vr_data.get('right_hand_trigger')
    # left_hand_trigger = vr_data.get('left_hand_trigger')

    # Neck
    neck_offset = Transform()                         # Create a transform
    neck_offset.setOrigin(Vector3(0, -0.14, 0.13))   # We move the transform where the neck is suppose to be
    neck_transform = head_transform * neck_offset       # Create the neck transform and apply the offset
    """posneck = neck_transform.getOrigin()            # Position of the neck, to show in console
    rotneck = neck_transform.getRotation()          # Rotation of the neck, to show in console
    print("Neck Transform: \n")
    print('posNeck \n', posneck.x, posneck.y, posneck.z)
    print('rotNeck \n', rotneck.getX(), rotneck.getY(), rotneck.getZ(), rotneck.getW())"""

    (r_shoulder_transform, r_elbow_transform, r_wrist_transform) = CompleteArm(neck_transform, r_hand_transform, 0)
    (l_shoulder_transform, l_elbow_transform, l_wrist_transform) = CompleteArm(neck_transform, l_hand_transform, 1)

    res = {'link_1': {'radius': 0.02, 'length': 0.15, 'transform': neck_transform},
           'link_2': {'radius': 0.02, 'length': 0.1, 'transform': r_shoulder_transform},
           'link_3': {'radius': 0.02, 'length': 0, 'transform': r_elbow_transform},
           'link_4': {'radius': 0.02, 'length': 0, 'transform': r_wrist_transform},
           'link_5': {'radius': 0.02, 'length': 0.1, 'transform': l_shoulder_transform},
           'link_6': {'radius': 0.02, 'length': 0, 'transform': l_elbow_transform},
           'link_7': {'radius': 0.02, 'length': 0, 'transform': l_wrist_transform}
           }

    return res


# Controller Axis (X,Y,Z)
# +X moves to the right of the controller
# +Y moves up of the controller
# +Z moves where the controller is pointing