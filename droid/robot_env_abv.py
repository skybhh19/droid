from copy import deepcopy

import gym
import numpy as np
import time
from gym.spaces import Box, Dict


from transformations import add_angles, angle_diff

from droid.calibration.calibration_utils import load_calibration_info
from droid.camera_utils.info import camera_type_dict
from droid.camera_utils.wrappers.multi_camera_wrapper import MultiCameraWrapper
from droid.misc.parameters import hand_camera_id, nuc_ip
from droid.misc.server_interface import ServerInterface
from droid.misc.time import time_ms
from droid.misc.transformations import change_pose_frame


class RobotEnv(gym.Env):
    def __init__(self, action_space="cartesian_velocity", camera_kwargs={}, do_reset=True,
                 DoF=3):
        # Initialize Gym Environment
        super().__init__()

        # physics
        self.max_lin_vel = 0.2 # 0.1
        self.max_rot_vel = 2.0 # 0.5
        self.DoF = DoF
        self.control_hz = 10 # 15


        assert action_space in ["cartesian_position", "joint_position", "cartesian_velocity", "joint_velocity"]
        self.eef_bounds = None
        # self.eef_bounds = np.array([[0.37, -0.3, 0.1], [0.73, 0.26, 0.52]])
        print("Current robot space is", self.eef_bounds)
        self.action_space_type = action_space
        self.check_action_range = "velocity" in action_space

        # Robot Configuration
        self.reset_joints = np.array([0, -1 / 5 * np.pi, 0, -4 / 5 * np.pi, 0, 3 / 5 * np.pi, 0.0])
        self.randomize_low = np.array([-0.1, -0.2, -0.1, -0.3, -0.3, -0.3])
        self.randomize_high = np.array([0.1, 0.2, 0.1, 0.3, 0.3, 0.3])
        if self.DoF == 3:
            self.randomize_low[3:6] = np.zeros(3)
            self.randomize_high[3:6] = np.zeros(3)
        elif self.controller_type == 4:
            self.randomize_low[4:6] = np.zeros(3)
            self.randomize_high[4:6] = np.zeros(3)
        # self.DoF = 7 if ("cartesian" in action_space) else 8


        # EE position (x, y, z) + gripper width
        if self.DoF == 3:
            self.ee_space = Box(
                np.array([0.38, -0.25, 0.07, 0.00]),
                np.array([0.70, 0.28, 0.35, 0.085]),
            )
        elif self.DoF == 4:
            # EE position (x, y, z) + gripper width
            self.ee_space = Box(
                np.array([0.55, -0.06, 0.07, -1.57, 0.00]),
                np.array([0.73, 0.28, 0.35, 0.0, 0.085]),
            )


        if nuc_ip is None:
            from franka.robot import FrankaRobot

            self._robot = FrankaRobot()
        else:
            self._robot = ServerInterface(ip_address=nuc_ip)

        # Create Cameras
        self.camera_reader = MultiCameraWrapper(camera_kwargs)
        self.calibration_dict = load_calibration_info()
        self.camera_type_dict = camera_type_dict

        # Reset Robot
        if do_reset:
            self.reset()

    def _format_action(self, action):
        '''Returns [x,y,z], [yaw, pitch, roll], close_gripper'''
        default_delta_angle = angle_diff(self._default_angle, self._curr_angle)
        if self.DoF == 3:
            delta_pos, delta_angle, gripper = action[:-1], default_delta_angle, action[-1]
        elif self.DoF == 4:
            delta_pos, delta_angle, gripper = action[:3], [default_delta_angle[0], default_delta_angle[1], action[3]], action[-1]
        elif self.DoF == 6:
            delta_pos, delta_angle, gripper = action[:3], action[3:6], action[-1]
        return np.array(delta_pos), np.array(delta_angle), gripper

    def _limit_velocity(self, lin_vel, rot_vel):
        """Scales down the linear and angular magnitudes of the action"""
        lin_vel_norm = np.linalg.norm(lin_vel)
        rot_vel_norm = np.linalg.norm(rot_vel)
        if lin_vel_norm > 1:
            lin_vel = lin_vel / lin_vel_norm
        if rot_vel_norm > 1:
            rot_vel = rot_vel / rot_vel_norm
        lin_vel, rot_vel = lin_vel * self.max_lin_vel / self.hz, rot_vel * self.max_rot_vel / self.hz
        return lin_vel, rot_vel

    def _get_valid_pos_and_gripper(self, pos, gripper):
        '''To avoid situations where robot can break the object / burn out joints,
        allowing us to specify (x, y, z, gripper) where the robot cannot enter. Gripper is included
        because (x, y, z) is different when gripper is open/closed.

        There are two ways to do this: (a) reject action and maintain current pose or (b) project back
        to valid space. Rejecting action works, but it might get stuck inside the box if no action can
        take it outside. Projection is a hard problem, as it is a non-convex set :(, but we can follow
        some rough heuristics.'''

        # clip commanded position to satisfy box constraints
        x_low, y_low, z_low = self.ee_space.low[:3]
        x_high, y_high, z_high = self.ee_space.high[:3]
        pos[0] = pos[0].clip(x_low, x_high) # new x
        pos[1] = pos[1].clip(y_low, y_high) # new y
        pos[2] = pos[2].clip(z_low, z_high) # new z

        # todo: collision safety

        return pos, gripper

    @property
    def _curr_pos(self):
        return self._robot.get_ee_pos()

    @property
    def _curr_angle(self):
        return self._robot.get_ee_angle()


    def step(self, action):
        start_time = time.time()
        # Check Action
        assert len(action) == (self.DoF + 1)
        if self.check_action_range:
            assert (action.max() <= 1) and (action.min() >= -1), f'action: {action}'

        pos_action, angle_action, gripper = self._format_action(action)
        lin_vel, rot_vel = self._limit_velocity(pos_action, angle_action)
        desired_pos, gripper = self._get_valid_pos_and_gripper(self._curr_pos + lin_vel, gripper)
        desired_angle = add_angles(rot_vel, self._curr_angle)
        if self.DoF == 4:
            desired_angle[2] = desired_angle[2].clip(self.ee_space.low[3], self.ee_space.high[3])
        self._update_robot(desired_pos, desired_angle, gripper)

        if self.eef_bounds is not None:
            # Constrain the eef positions within the bounds
            eef_pos = self.get_observation()['robot_state']['cartesian_position']
            for pos_i in range(3):
                if eef_pos[pos_i] < self.eef_bounds[0][pos_i] and action[pos_i] < 0:
                    action[pos_i] = 0
                elif eef_pos[pos_i] > self.eef_bounds[1][pos_i] and action[pos_i] > 0:
                    action[pos_i] = 0

        # Update Robot
        action_info = self.update_robot(
            action,
            action_space=self.action_space_type,
        )

        # Return Action Info
        return action_info

    def reset(self, randomize=True):
        self._robot.update_gripper(0, velocity=False, blocking=True)

        if randomize:
            noise = np.random.uniform(low=self.randomize_low, high=self.randomize_high)
        else:
            noise = None

        self._robot.update_joints(self.reset_joints, velocity=False, blocking=True, cartesian_noise=noise)
        return self.get_observation()

    def _update_robot(self, action, action_space="cartesian_velocity", blocking=False):
        action_info = self._robot.update_command(
            action,
            action_space=action_space,
            blocking=blocking
        )
        return action_info

    def create_action_dict(self, action):
        return self._robot.create_action_dict(action)

    def read_cameras(self):
        return self.camera_reader.read_cameras()

    def get_state(self):
        read_start = time_ms()
        state_dict, timestamp_dict = self._robot.get_robot_state()
        timestamp_dict["read_start"] = read_start
        timestamp_dict["read_end"] = time_ms()
        return state_dict, timestamp_dict

    def get_camera_extrinsics(self, state_dict):
        # Adjust gripper camere by current pose
        extrinsics = deepcopy(self.calibration_dict)
        for cam_id in self.calibration_dict:
            if hand_camera_id not in cam_id:
                continue
            gripper_pose = state_dict["cartesian_position"]
            extrinsics[cam_id + "_gripper_offset"] = extrinsics[cam_id]
            extrinsics[cam_id] = change_pose_frame(extrinsics[cam_id], gripper_pose)
        return extrinsics

    def get_observation(self):
        obs_dict = {"timestamp": {}}

        # Robot State #
        state_dict, timestamp_dict = self.get_state()
        obs_dict["robot_state"] = state_dict
        obs_dict["timestamp"]["robot_state"] = timestamp_dict

        # Camera Readings #
        camera_obs, camera_timestamp = self.read_cameras()
        camera_dict = dict(image=dict(
            wrist_image=camera_obs["image"]["19824535_left"][:, :, 0:3],
            side_image=camera_obs["image"]["23404442_left"][:, :, 0:3],
        ))


        obs_dict.update(camera_obs)
        # obs_dict.update(camera_dict)
        obs_dict["timestamp"]["cameras"] = camera_timestamp

        # Camera Info #
        obs_dict["camera_type"] = deepcopy(self.camera_type_dict)
        extrinsics = self.get_camera_extrinsics(state_dict)
        obs_dict["camera_extrinsics"] = extrinsics

        intrinsics = {}
        for cam in self.camera_reader.camera_dict.values():
            cam_intr_info = cam.get_intrinsics()
            for (full_cam_id, info) in cam_intr_info.items():
                intrinsics[full_cam_id] = info["cameraMatrix"]
        obs_dict["camera_intrinsics"] = intrinsics

        return obs_dict
