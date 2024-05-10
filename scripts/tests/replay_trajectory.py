from droid.robot_env import RobotEnv
from droid.trajectory_utils.misc import replay_trajectory

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    args = parser.parse_args()
    return args

args = get_args()
# trajectory_folderpath = "/home/sasha/droid/data/cube_stacking/success/2024-05-08/Wed_May__8_19:40:08_2024/"
action_space_type = "cartesian_velocity"

# Make the robot env
env = RobotEnv(action_space_type=action_space_type)

# Replay Trajectory #
h5_filepath = args.folder + "/trajectory.h5"
replay_trajectory(env, filepath=h5_filepath)
