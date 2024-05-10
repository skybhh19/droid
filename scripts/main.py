from droid.controllers.oculus_controller import VRPolicy
from droid.robot_env import RobotEnv
from droid.user_interface.data_collector import DataCollecter
from droid.user_interface.gui import RobotGUI

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--DoF", type=int, required=True)
    parser.add_argument("--task", type=str, required=True)
    args = parser.parse_args()
    return args

# Make the robot env
args = get_args()
env = RobotEnv(DoF=args.DoF)
controller = VRPolicy(DoF=args.DoF)

# Make the data collector
data_collector = DataCollecter(env=env, controller=controller, task=args.task, DoF=args.DoF)

# Make the GUI
user_interface = RobotGUI(robot=data_collector)
