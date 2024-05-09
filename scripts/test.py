from droid.misc.transformations import change_pose_frame, add_poses, euler_to_quat, pose_diff, quat_to_euler, angle_diff, add_angles
from droid.robot_ik.robot_ik_solver import RobotIKSolver

import numpy as np

_ik_solver = RobotIKSolver()
action = [0.1, 0.3, -0.5, -0.09, 0.1]
DoF = 3

ee_space = np.array([
    [0.38, -0.25, 0.07, 0.00],
    [0.70, 0.28, 0.35, 0.085],
])

_curr_pos = [0.6, 0., 0.2]
_curr_angle = [-3., 0.02, 0.01]



def _format_action(action, default_angle=np.array([np.pi, 0., 0.])):
    '''Returns [x,y,z], [yaw, pitch, roll], close_gripper'''
    default_delta_angle = angle_diff(default_angle, _curr_angle)
    if DoF == 3:
        delta_pos, delta_angle, gripper = action[:-1], default_delta_angle, action[-1:]
    elif DoF == 4:
        delta_pos, delta_angle, gripper = action[:3], [default_delta_angle[0], default_delta_angle[1],
                                                       action[3]], action[-1:]
    elif DoF == 6:
        delta_pos, delta_angle, gripper = action[:3], action[3:6], action[-1:]
    return np.array(delta_pos), np.array(delta_angle), gripper


def _get_valid_pos_and_gripper(pos, gripper):
    '''To avoid situations where robot can break the object / burn out joints,
    allowing us to specify (x, y, z, gripper) where the robot cannot enter. Gripper is included
    because (x, y, z) is different when gripper is open/closed.

    There are two ways to do this: (a) reject action and maintain current pose or (b) project back
    to valid space. Rejecting action works, but it might get stuck inside the box if no action can
    take it outside. Projection is a hard problem, as it is a non-convex set :(, but we can follow
    some rough heuristics.'''

    # clip commanded position to satisfy box constraints
    x_low, y_low, z_low = ee_space[0][:3]
    x_high, y_high, z_high = ee_space[1][:3]
    pos[0] = pos[0].clip(x_low, x_high)  # new x
    pos[1] = pos[1].clip(y_low, y_high)  # new y
    pos[2] = pos[2].clip(z_low, z_high)  # new z

    # todo: safety guarantee

    return pos, gripper


pos_vel, rot_vel, gripper_vel = _format_action(action)
cartesian_delta = _ik_solver.cartesian_velocity_to_delta(np.concatenate([pos_vel, rot_vel]))
lin_delta, rot_delta = cartesian_delta[:3], cartesian_delta[3:]
print(_curr_pos + lin_delta)
desired_pos, gripper_vel = _get_valid_pos_and_gripper(_curr_pos + lin_delta, gripper_vel)
print(desired_pos)
desired_angle = add_angles(rot_delta, _curr_angle)
if DoF == 4:
    desired_angle[2] = desired_angle[2].clip(ee_space[0][3], ee_space[1][3])
pos_delta = desired_pos - _curr_pos
rot_delta = angle_diff(desired_angle, _curr_angle)
cartesian_vel = _ik_solver.cartesian_delta_to_velocity(np.concatenate([pos_delta, rot_delta]))
action = np.concatenate([cartesian_vel, gripper_vel])

print(action)