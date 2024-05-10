import numpy as np

from droid.misc.server_interface import ServerInterface


nuc_ip = '172.16.0.3'

_robot = ServerInterface(ip_address=nuc_ip)

_gripper_angle = 1.544
reset_joints = np.array(
    [0.0113871, 0.38554454, 0.00297691, -2.07930946, -0.0146654, 2.44864178, _gripper_angle])
def reset():
    _robot.update_gripper(0, velocity=False, blocking=True)
    _robot.update_joints(reset_joints, velocity=False, blocking=True)

reset()