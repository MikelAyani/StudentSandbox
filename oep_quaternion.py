import math
import numpy as np

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

def Transform2EulerOpenGL(transform):
    ret = Transform2Euler(transform)
    #Modified because OpenGL has the axis changed respected to Simumatik
    return [ret[0], ret[2], ret[1], np.rad2deg(ret[3]), np.rad2deg(ret[5]), np.rad2deg(ret[4])] 

def Transform2Quat(transform):
    """ Converts RPY angles in degrees to quaternion.
    This is a modified version of this but executed in ored X-Y-Z:
    https://github.com/mrdoob/three.js/blob/master/src/math/Quaternion.js
    """
    [x, y, z, roll, pitch, yaw] = transform

    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    qx = sr * cp * cy + cr * sp * sy
    qy = cr * sp * cy - sr * cp * sy
    qz = cr * cp * sy + sr * sp * cy
    qw = cr * cp * cy - sr * sp * sy
   
    return [x, y, z, qx,qy,qz,qw]


if __name__ == '__main__':
    transform = [0, 0, 0, 1.57, 0, 0]
    print(transform)
    transform = Transform2Quat(transform)
    print(transform)
    transform = Transform2Euler(transform)
    print(transform)