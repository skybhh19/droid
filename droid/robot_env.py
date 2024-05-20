from copy import deepcopy

import gym
import numpy as np
import time
from gym.spaces import Box, Dict


from droid.calibration.calibration_utils import load_calibration_info
from droid.camera_utils.info import camera_type_dict
from droid.camera_utils.wrappers.multi_camera_wrapper import MultiCameraWrapper
from droid.misc.parameters import hand_camera_id, nuc_ip
from droid.misc.server_interface import ServerInterface
from droid.misc.time import time_ms
from droid.misc.transformations import change_pose_frame, add_poses, euler_to_quat, pose_diff, quat_to_euler, angle_diff, add_angles
from droid.robot_ik.robot_ik_solver import RobotIKSolver



class RobotEnv(gym.Env):
    def __init__(self, action_space_type="cartesian_velocity", camera_kwargs={}, do_reset=True,
                 DoF=3):
        # Initialize Gym Environment
        super().__init__()

        # physics
        # self.DoF = 7 if ("cartesian" in action_space_type) else 8
        self.DoF = DoF
        assert self.DoF in [3, 4, 6]
        self.control_hz = 15

        # Robot Configuration
        # self.reset_joints = np.array([0, -1 / 5 * np.pi, 0, -4 / 5 * np.pi, 0, 3 / 5 * np.pi, 0.0])
        self._gripper_angle = 1.544
        if self._gripper_angle == 1.544:
            self._default_angle = np.array([3.11231174, 0.00372336, -1.49605473])
        elif self._gripper_angle == 0.:
            self._default_angle = np.array([np.pi, 0., 0.])
        self.reset_joints = np.array([0.0113871, 0.38554454, 0.00297691, -2.07930946, -0.0146654, 2.44864178, self._gripper_angle])
        # self.reset_joints = np.array([0, 0.423, 0, -1.944, 0., 2.219, self._gripper_angle])

        assert action_space_type in ["cartesian_velocity"]
        self.eef_bounds = None
        # self.eef_bounds = np.array([[0.37, -0.3, 0.1], [0.73, 0.26, 0.52]])
        # print("Current robot space is", self.eef_bounds)
        self.action_space_type = action_space_type
        self.check_action_range = "velocity" in action_space_type

        # EE position (x, y, z) + gripper width
        if self.DoF == 3:
            # self.ee_space = Box(
            #     np.array([0.38, -0.25, 0.07, 0.00]),
            #     np.array([0.70, 0.28, 0.35, 0.085]),
            # )
            self.ee_space = Box(
                np.array([0.45, -0.24, 0.12]),
                np.array([0.7, 0.17, 0.3]),
            )
        elif self.DoF == 4:
            # EE position (x, y, z) + gripper width
            self.ee_space = Box(
                np.array([0.55, -0.06, 0.12, -1.57]),
                np.array([0.73, 0.25, 0.35, 0.0]),
            )

        if self.DoF < 6:
            self._ik_solver = RobotIKSolver()


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
            print(self.get_ee_angle())

    def get_ee_pos(self):
        '''Returns [x,y,z]'''
        pose = self._robot.get_ee_pose()
        return pose[:3]

    def get_ee_angle(self):
        '''Returns [yaw, pitch, roll]'''
        pose = self._robot.get_ee_pose()
        return pose[3:]

    def _format_action(self, action):
        '''Returns [x,y,z], [yaw, pitch, roll], close_gripper'''
        default_delta_angle = angle_diff(self._default_angle, self._curr_angle)
        if self.DoF == 3:
            delta_pos, delta_angle, gripper = action[:-1], default_delta_angle, action[-1:]
        elif self.DoF == 4:
            delta_pos, delta_angle, gripper = action[:3], [default_delta_angle[0], default_delta_angle[1], action[3]], action[-1:]
        elif self.DoF == 6:
            delta_pos, delta_angle, gripper = action[:3], action[3:6], action[-1:]
        return np.array(delta_pos), np.array(delta_angle), gripper

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

        # todo: safety guarantee

        return pos, gripper

    @property
    def _curr_pos(self):
        return self.get_ee_pos()

    @property
    def _curr_angle(self):
        return self.get_ee_angle()

    def step(self, action):
        input_action = action.copy()
        assert len(action) == (self.DoF + 1)
        if self.DoF < 6:
            pos_vel, rot_vel, gripper_vel = self._format_action(action)
            cartesian_delta = self._ik_solver.cartesian_velocity_to_delta(np.concatenate([pos_vel, rot_vel]))
            lin_delta, rot_delta = cartesian_delta[:3], cartesian_delta[3:]
            desired_pos, gripper_vel = self._get_valid_pos_and_gripper(self._curr_pos + lin_delta, gripper_vel)
            desired_angle = add_angles(rot_delta, self._curr_angle)
            if self.DoF == 4:
                desired_angle[2] = desired_angle[2].clip(self.ee_space.low[3], self.ee_space.high[3])
            pos_delta = desired_pos - self._curr_pos
            rot_delta = angle_diff(desired_angle, self._curr_angle)
            cartesian_vel = self._ik_solver.cartesian_delta_to_velocity(np.concatenate([pos_delta, rot_delta]))
            action = np.concatenate([cartesian_vel, gripper_vel])

        action = action.clip(-1, 1)

        # Check Action
        if self.check_action_range:
            assert (action.max() <= 1) and (action.min() >= -1), f'action: {action}'

        # Update Robot
        action_info = self.update_robot(
            action,
            action_space_type=self.action_space_type,
        )
        assert len(input_action) == (self.DoF + 1)
        # print("cur pos", self.get_ee_pos())
        # Return Action Info
        action_info['input_action'] = input_action
        return action_info

    def reset(self, randomize=False):
        self._robot.update_gripper(0, velocity=False, blocking=True)

        if randomize:
            noise = np.random.uniform(low=self.randomize_low, high=self.randomize_high)
        else:
            noise = None

        self._robot.update_joints(self.reset_joints, velocity=False, blocking=True, cartesian_noise=noise)
        return self.get_observation()

    def update_robot(self, action, action_space_type="cartesian_velocity", blocking=False):
        action_info = self._robot.update_command(
            action,
            action_space_type=action_space_type,
            blocking=blocking,
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

    def _normalize_ee_cartesian(self, cartesian_position):
        assert len(cartesian_position) == 6
        if self.DoF == 3:
            obs = cartesian_position[:3]
        elif self.DoF == 4:
            obs = np.concatenate([cartesian_position[:3], cartesian_position[-1:]])
        else:
            raise NotImplementedError
        # print("ee_obs", obs)
        """Normalizes low-dim obs between [-1,1]."""
        # x_new = 2 * (x - min(x)) / (max(x) - min(x)) - 1
        # x = (x_new + 1) * (max (x) - min(x)) / 2 + min(x)
        # Source: https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
        normalized_obs = 2 * (obs - self.ee_space.low) / (self.ee_space.high - self.ee_space.low) - 1
        return normalized_obs

    def get_observation(self):
        obs_dict = {"timestamp": {}}

        # Robot State #
        state_dict, timestamp_dict = self.get_state()
        obs_dict["robot_state"] = state_dict
        obs_dict["norm_ee_obs"] = np.concatenate([self._normalize_ee_cartesian(state_dict["cartesian_position"]), np.array([state_dict["gripper_position"]])])
        # print("norm_ee_obs", obs_dict["norm_ee_obs"])
        obs_dict["timestamp"]["robot_state"] = timestamp_dict

        # Camera Readings #
        camera_obs, camera_timestamp = self.read_cameras()

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
