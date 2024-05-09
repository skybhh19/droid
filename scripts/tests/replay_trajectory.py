from droid.robot_env import RobotEnv
from droid.trajectory_utils.misc import replay_trajectory

trajectory_folderpath = "/home/sasha/droid/data/cube_stacking/success/2024-05-08/Wed_May__8_19:40:08_2024/"
action_space_type = "cartesian_velocity"

# Make the robot env
env = RobotEnv(action_space_type=action_space_type)

# Replay Trajectory #
h5_filepath = trajectory_folderpath + "/trajectory.h5"
replay_trajectory(env, filepath=h5_filepath)
