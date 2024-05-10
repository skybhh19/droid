import h5py
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    help="path to hdf5 dataset",
)
args = parser.parse_args()

f = h5py.File(args.dataset, "r")
group_keys = ["observation", "action"]
# print(f['action/robot_state/cartesian_position'][()])
# print(f['action/cartesian_velocity'][()][:30])
print(f['observation/robot_state']['cartesian_position'])
print(f['observation/timestamp/robot_state'].keys())

for g_key in group_keys:
    print("="*10, g_key, "="*10)
    for k in f[g_key]:
        print(k)



