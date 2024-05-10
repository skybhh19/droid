"""
Add image information to existing droid hdf5 file
"""
import h5py
import os
import numpy as np
import glob
from tqdm import tqdm
import argparse
import shutil
import torch
import random
import traceback
import json
import cv2

"""
Follow instructions here to setup zed:
https://www.stereolabs.com/docs/installation/linux/
"""
import pyzed.sl as sl

import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils

from droid.camera_utils.wrappers.recorded_multi_camera_wrapper import RecordedMultiCameraWrapper
from droid.trajectory_utils.trajectory_reader import TrajectoryReader
from droid.camera_utils.info import camera_type_to_string_dict

from droid.camera_utils.camera_readers.zed_camera import ZedCamera, standard_params


def get_cam_instrinsics(svo_path):
    """
    utility function to get camera intrinsics
    """
    intrinsics = {}

    return intrinsics

def convert_dataset(path, args, ep_data_grp):
    recording_folderpath = os.path.join(os.path.dirname(path), "recordings", "SVO")
    
    num_svo_files = len([f for f in os.listdir(recording_folderpath) if os.path.isfile(os.path.join(recording_folderpath, f))])
    assert(num_svo_files == 2), "Didnt find 2 svos!"
    camera_kwargs = dict(
        hand_camera=dict(image=True, concatenate_images=False, resolution=(args.imsize, args.imsize), resize_func="cv2"),
        varied_camera=dict(image=True, concatenate_images=False, resolution=(args.imsize, args.imsize), resize_func="cv2"),
    )
    camera_reader = RecordedMultiCameraWrapper(recording_folderpath, camera_kwargs)

    # shutil.copyfile(path, output_path)
    f = h5py.File(path, "r")

    demo_len = f["action"]["cartesian_velocity"].shape[0]

    # if "camera" not in f["observation"]:
    #     f["observation"].create_group("camera").create_group("image")
    # image_grp = f["observation/camera/image"]

    """
    Extract camera type and keys. Examples of what they should look like:
    camera_type_dict = {
        '17225336': 'hand_camera',
        '24013089': 'varied_camera',
        '25047636': 'varied_camera'
    }
    CAM_NAME_TO_KEY_MAPPING = {
        "hand_camera_left_image": "17225336_left",
        "hand_camera_right_image": "17225336_right",
        "varied_camera_1_left_image": "24013089_left",
        "varied_camera_1_right_image": "24013089_right",
        "varied_camera_2_left_image": "25047636_left",
        "varied_camera_2_right_image": "25047636_right",
    }
    """

    CAM_ID_TO_TYPE = {}
    hand_cam_ids = []
    varied_cam_ids = []
    for k in f["observation"]["camera_type"]:
        cam_type = camera_type_to_string_dict[f["observation"]["camera_type"][k][0]]
        CAM_ID_TO_TYPE[k] = cam_type
        if cam_type == "hand_camera":
            hand_cam_ids.append(k)
        elif cam_type == "varied_camera":
            varied_cam_ids.append(k)
        else:
            raise ValueError
        

    # sort the camera ids: important to maintain consistency of cams between train and eval!
    hand_cam_ids = sorted(hand_cam_ids)
    varied_cam_ids = sorted(varied_cam_ids)

    IMAGE_NAME_TO_CAM_KEY_MAPPING = {}
    # IMAGE_NAME_TO_CAM_KEY_MAPPING["hand_camera_left_image"] = "{}_left".format(hand_cam_ids[0])
    IMAGE_NAME_TO_CAM_KEY_MAPPING["hand_camera_right_image"] = "{}_right".format(hand_cam_ids[0])

    # set up mapping for varied cameras
    for i in range(len(varied_cam_ids)):
        # for side in ["left", "right"]:
        for side in ["right"]:
            cam_name = "varied_camera_{}_{}_image".format(i+1, side)
            cam_key = "{}_{}".format(varied_cam_ids[i], side)
            IMAGE_NAME_TO_CAM_KEY_MAPPING[cam_name] = cam_key

    cam_data = {cam_name: [] for cam_name in IMAGE_NAME_TO_CAM_KEY_MAPPING.keys()}
    traj_reader = TrajectoryReader(path, read_images=False)

    for index in range(demo_len):
        
        timestep = traj_reader.read_timestep(index=index)
        timestamp_dict = timestep["observation"]["timestamp"]["cameras"]
        
        timestamp_dict = {}
        camera_obs = camera_reader.read_cameras(
            index=index, camera_type_dict=CAM_ID_TO_TYPE, timestamp_dict=timestamp_dict
        )
        for cam_name in IMAGE_NAME_TO_CAM_KEY_MAPPING.keys():
            if camera_obs is None:
                im = np.zeros((args.imsize, args.imsize, 3))
            else:
                im_key = IMAGE_NAME_TO_CAM_KEY_MAPPING[cam_name]
                im = camera_obs["image"][im_key]
                im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)

            # perform bgr_to_rgb operation
            im = im[:,:,::-1]
            
            cam_data[cam_name].append(im)

    for cam_name in cam_data.keys():
        cam_data[cam_name] = np.array(cam_data[cam_name]).astype(np.uint8)
        ep_data_grp.create_dataset('obs/{}'.format(cam_name), data=cam_data[cam_name], compression="gzip")
        # if cam_name in image_grp:
        #     del image_grp[cam_name]
        # image_grp.create_dataset(cam_name, data=cam_data[cam_name], compression="gzip")

    ep_data_grp.create_dataset('obs/norm_ee_obs', data=np.array(f["observation"]["norm_ee_obs"][:]))
    # extract camera extrinsics data
    # if "extrinsics" not in f["observation/camera"]:
    #     f["observation/camera"].create_group("extrinsics")
    # extrinsics_grp = f["observation/camera/extrinsics"]
    # for (im_name, cam_key) in IMAGE_NAME_TO_CAM_KEY_MAPPING.items():
    #     raw_data = f["observation/camera_extrinsics"][cam_key][:]
    #     raw_data = torch.from_numpy(raw_data)
    #     pos = raw_data[:,0:3]
    #     rot_mat = TorchUtils.euler_angles_to_matrix(raw_data[:,3:6], convention="XYZ")
    #     extrinsics = np.zeros((len(pos), 4, 4))
    #     extrinsics[:,:3,:3] = TensorUtils.to_numpy(rot_mat)
    #     extrinsics[:,:3,3] = TensorUtils.to_numpy(pos)
    #     extrinsics[:,3,3] = 1.0
    #     # invert the matrix to represent standard definition of extrinsics: from world to cam
    #     extrinsics = np.linalg.inv(extrinsics)
    #     extr_name = "_".join(im_name.split("_")[:-1])
    #     extrinsics_grp.create_dataset(extr_name, data=extrinsics)
    
    # svo_path = os.path.join(os.path.dirname(path), "recordings", "SVO")
    # cam_reader_svo = camera_reader #RecordedMultiCameraWrapper(svo_path, camera_kwargs)
    # if "intrinsics" not in f["observation/camera"]:
    #     f["observation/camera"].create_group("intrinsics")
    # intrinsics_grp = f["observation/camera/intrinsics"]
    # for cam_id, svo_reader in cam_reader_svo.camera_dict.items():
    #     cam = svo_reader._cam
    #     calib_params = cam.get_camera_information().camera_configuration.calibration_parameters
    #     # for (posftix, params)in zip(
    #     #     ["_left", "_right"],
    #     #     [calib_params.left_cam, calib_params.right_cam]
    #     # ):
    #     for (posftix, params) in zip(
    #             ["_right"],
    #             [calib_params.right_cam]
    #     ):
    #         # get name to store intrinsics under
    #         cam_key = cam_id + posftix
    #         # reverse search for image name
    #         im_name = None
    #         for (k, v) in IMAGE_NAME_TO_CAM_KEY_MAPPING.items():
    #             if v == cam_key:
    #                 im_name = k
    #                 break
    #         if im_name is None: # sometimes the raw_key doesn't correspond to any camera we have images for
    #             continue
    #         intr_name = "_".join(im_name.split("_")[:-1])
    #
    #         # if intr_name not in intrinsics_grp:
    #         #     intrinsics_grp.create_group(intr_name)
    #         # cam_intr_grp = intrinsics_grp[intr_name]
    #
    #         # these lines are copied from _process_intrinsics function in svo_reader.py
    #         cam_intrinsics = np.array([[params.fx, 0, params.cx], [0, params.fy, params.cy], [0, 0, 1]])
    #         data = np.repeat(cam_intrinsics[None], demo_len, axis=0)
    #         intrinsics_grp.create_dataset(intr_name, data=data)
    #         # {
    #         #     "camera_matrix": np.array([[params.fx, 0, params.cx], [0, params.fy, params.cy], [0, 0, 1]]),
    #         #     "dist_coeffs": np.array(list(params.disto)),
    #         # }
    #         # # batchify across trajectory
    #         # for k in cam_intrinsics:
    #         #     data = np.repeat(cam_intrinsics[k][None], demo_len, axis=0)
    #         #     cam_intr_grp.create_dataset(k, data=data)

    # extract action key data
    # action_dict_group = f["action"]
    # for in_ac_key in ["cartesian_position", "cartesian_velocity"]:
    #     in_action = action_dict_group[in_ac_key][:]
    #     in_pos = in_action[:,:3].astype(np.float64)
    #     in_rot = in_action[:,3:6].astype(np.float64) # in euler format
    #     rot_ = torch.from_numpy(in_rot)
    #     rot_6d = TorchUtils.euler_angles_to_rot_6d(
    #         rot_, convention="XYZ",
    #     )
    #     rot_6d = rot_6d.numpy().astype(np.float64)
    #
    #     if in_ac_key == "cartesian_position":
    #         prefix = "abs_"
    #     elif in_ac_key == "cartesian_velocity":
    #         prefix = "rel_"
    #     else:
    #         raise ValueError
    #
    #     this_action_dict = {
    #         prefix + 'pos': in_pos,
    #         prefix + 'rot_euler': in_rot,
    #         prefix + 'rot_6d': rot_6d,
    #     }
    #     for key, data in this_action_dict.items():
    #         if key in action_dict_group:
    #             del action_dict_group[key]
    #         action_dict_group.create_dataset(key, data=data)

    actions = np.array(f["action"]["input_action"][:])
    ep_data_grp.create_dataset('actions', data=actions)
    ep_data_grp.attrs["num_samples"] = len(actions)
    # ensure all action keys are batched (ie., are not 0-dimensional)
    # for k in action_dict_group:
    #     if isinstance(action_dict_group[k], h5py.Dataset) and len(action_dict_group[k].shape) == 1:
    #         reshaped_values = np.reshape(action_dict_group[k][:], (-1, 1))
    #         del action_dict_group[k]
    #         action_dict_group.create_dataset(k, data=reshaped_values)

    # post-processing: remove timesteps where robot movement is disabled
    movement_enabled = f["observation/controller_info/movement_enabled"][:]
    assert np.all(movement_enabled)
    # timesteps_to_remove = np.where(movement_enabled == False)[0]
    #
    # if not args.keep_idle_timesteps:
    #     remove_timesteps(f, timesteps_to_remove)

    f.close()
    camera_reader.disable_cameras()
    del camera_reader
    return len(actions)

def remove_timesteps(f, timesteps_to_remove):
    total_timesteps = f["action/cartesian_position"].shape[0]
    
    def remove_timesteps_for_group(g):
        for k in g:
            if isinstance(g[k], h5py._hl.dataset.Dataset):
                if g[k].shape[0] != total_timesteps:
                    print("skipping {}".format(k))
                    continue
                new_dataset = np.delete(g[k], timesteps_to_remove, axis=0)
                del g[k]
                g.create_dataset(k, data=new_dataset)
            elif isinstance(g[k], h5py._hl.group.Group):
                remove_timesteps_for_group(g[k])
            else:
                raise NotImplementedError

    for k in f:
        remove_timesteps_for_group(f[k])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--folder",
        type=str,
        help="folder containing hdf5's to add camera images to",
        default="~/datasets/droid/success"
    )

    parser.add_argument(
        "--imsize",
        type=int,
        default=128,
        help="image size (w and h)",
    )

    parser.add_argument(
        "--keep_idle_timesteps",
        action="store_true",
        help="override the default behavior of truncating idle timesteps",
    )

    parser.add_argument(
        "--num_demo",
        type=int,
    )
    
    args = parser.parse_args()
    output_path = os.path.join(args.folder, "trajectory_im{}.hdf5".format(args.imsize))
    print("output file", output_path)
    assert not os.path.exists(output_path)
    out_f = h5py.File(output_path, "w")
    out_f_grp = out_f.create_group('data')

    env_meta = dict()
    out_f["data"].attrs["env_args"] = json.dumps(env_meta, indent=4)

    datasets = []
    j = os.walk(os.path.expanduser(args.folder))
    # import pdb; pdb.set_trace()
    for root, dirs, files in j:
        for f in files:
            if f == "trajectory.h5":
                # if "success" in root:
                datasets.append(os.path.join(root, f))
                print(len(datasets))

    print("converting datasets...")
    random.shuffle(datasets)
    failed = 0
    total_samples = 0
    for idx in tqdm(range(len(datasets))):
        if idx >= args.num_demo:
            break
        d = datasets[idx]
        ep_data_grp = out_f_grp.create_group("demo_{}".format(idx))
        d = os.path.expanduser(d)
        # try:
        num_samples = convert_dataset(d, args, ep_data_grp)
        total_samples += num_samples
        # except Exception as e:
        #     failed += 1
        #     print(f"{failed} Failed")
    out_f_grp.attrs["total"] = total_samples
    out_f.close()