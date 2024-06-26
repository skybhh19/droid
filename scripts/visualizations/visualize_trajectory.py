from droid.trajectory_utils.misc import visualize_trajectory

trajectory_folderpath = "/home/sasha/droid/droid/data_loading/../../data/cube_stacking/success/2024-05-07/Tue_May__7_19:21:57_2024"

camera_kwargs = dict(
    hand_camera=dict(image=True, resolution=(0, 0)),
    varied_camera=dict(image=True, resolution=(0, 0)),
)

h5_filepath = trajectory_folderpath + "/trajectory.h5"
recording_folderpath = trajectory_folderpath + "/recordings/SVO"
visualize_trajectory(filepath=h5_filepath, recording_folderpath=recording_folderpath, camera_kwargs=camera_kwargs)
