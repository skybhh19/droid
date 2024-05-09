from droid.controllers.oculus_controller import VRPolicy
from droid.robot_env import RobotEnv
from droid.user_interface.data_collector import DataCollecter
from droid.user_interface.gui import RobotGUI

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--DoF", default=3, type=int)
    parser.add_argument("--task", type=str)
    args = parser.parse_args()
    return args

# Make the robot env
args = get_args()
env = RobotEnv(DoF=args.DoF)
controller = VRPolicy()

# Make the data collector
data_collector = DataCollecter(env=env, controller=controller, task=args.task)

# Make the GUI
user_interface = RobotGUI(robot=data_collector)
