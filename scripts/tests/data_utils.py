import enum
import torch
import numpy as np
class StrEnum(str, enum.Enum):
    pass

class StateEncoding(StrEnum):
    """Defines state keys for datasets."""

    EE_POS = "EE_POS"  # 3 Dim EE Pos
    EE_EULER = "EE_EULER"  # 3 Dim EE Euler
    EE_QUAT = "EE_QUAT"  # 4 Dim EE Quat
    EE_ROT6D = "EE_ROT6D"  # 6 Dim Rot6D
    JOINT_POS = "JOINT_POS"  # 7 x joint
    JOINT_VEL = "JOINT_VEL"  # 7 x joint
    GRIPPER = "GRIPPER"  # 1 x gripper open / close.
    EE_VEL_LIN = "EE_VEL_LIN"
    EE_VEL_ANG = "EE_VEL_ANG"
    MISC = "MISC"  # Other miscellaneous objects



def convert_array_to_state(actions):
    assert len(actions) == 7
    state_actions = {
        StateEncoding.EE_POS: actions[:3],
        StateEncoding.EE_EULER: actions[3:6],
        StateEncoding.GRIPPER: [actions[6]],
    }
    return state_actions

def convert_state_to_array(state_actions, joint_vel=None):
    if joint_vel is not None:
        actions = np.array(joint_vel)
    else:
        actions = np.concatenate([state_actions[StateEncoding.EE_POS], state_actions[StateEncoding.EE_EULER], state_actions[StateEncoding.GRIPPER]])
    return actions

class RobotType(StrEnum):
    """Defines Enum For different robot types"""

    PANDA = "PANDA"
    WIDOWX = "WIDOWX"
    FR3 = "FR3"
    META = "META"
    KUKA = "KUKA"
    JACO = "JACO"
    SAWYER = "SAWYER"
    UR5 = "UR5"
    XARM = "XARM"
    STRETCH = "STRETCH"
    UNKNOWN = "UNKNOWN"


class NormalizationType(StrEnum):
    NONE = "NONE"
    BOUNDS = "BOUNDS"
    GAUSSIAN = "GAUSSIAN"
    BOUNDS_5STDV = "BOUNDS_5STDV"


class DataType(StrEnum):
    BOOL = "BOOL"
    DISCRETE = "DISCRETE"
    CONTINUOUS = "CONTINUOUS"
    CONSTANT = "CONSTANT"


def rmat_to_rot6d(rmat: torch.Tensor) -> torch.Tensor:
    r6 = rmat[..., :2, :]
    r6_0, r6_1 = r6[..., 0, :], r6[..., 1, :]
    return torch.concat([r6_0, r6_1], axis=-1)


def binarize_gripper_actions(gripper_actions: torch.Tensor) -> torch.Tensor:
    """
    Binarizes the gripper to 0 (open) or 1 (closed).
    Taken from https://github.com/octo-models/octo/blob/main/octo/data/utils/data_utils.py#L292
    """
    open_mask = gripper_actions < 0.05
    closed_mask = gripper_actions > 0.95
    in_between_mask = torch.logical_not(torch.logical_or(open_mask, closed_mask))
    is_closed_float = torch.cast(closed_mask, torch.float32)

    def scan_fn(carry, i):
        return torch.cond(
            in_between_mask[i],
            lambda: torch.cast(carry, torch.float32),  # If we are in between, return the future gripper state.
            lambda: is_closed_float[i],  # If we are not in between, return 1 if closed.
        )

    return torch.scan(scan_fn, torch.range(torch.shape(gripper_actions)[0]), is_closed_float[-1], reverse=True)


def rel2abs_gripper_actions(
    actions: torch.Tensor,
    threshold: float = 0.1,
):
    """
    Attribution:
    largely borrowed from https://github.com/octo-models/octo/blob/main/octo/data/utils/data_utils.py

    Converts relative gripper actions (+1 for closing, -1 for opening) to absolute gripper actions:
    0 for open, 1 for closed.
    Assumes that the first relative gripper is not redundant (i.e. close when already closed).
    """
    opening_mask = actions < -threshold
    closing_mask = actions > threshold

    # 1 for closing, -1 for opening, 0 for no change
    thresholded_actions = torch.where(opening_mask, -1, torch.where(closing_mask, 1, 0))

    def scan_fn(carry, i):
        # set the gripper action to be the previous one if zero, otherwize set it to be the thresholded action
        return torch.cond(
            thresholded_actions[i] == 0,
            lambda: carry,
            lambda: thresholded_actions[i],
        )

    # Get the action at the first position of change
    first_action = thresholded_actions[torch.argmax(thresholded_actions != 0, axis=0)]
    # If my frist action is 0 (no change) or 1 (closing), then start = open (-1).
    # If my first action is -1 (open), then start = closing (1)
    start = torch.cond(first_action == -1, lambda: 1, lambda: -1)

    # Resulting actions are -1 to 1.
    new_actions = torch.scan(scan_fn, torch.range(torch.shape(actions)[0]), start)
    return torch.cast(new_actions, torch.float32) / 2 + 0.5


def gripper_state_from_width(gripper_state: torch.Tensor, max_width: float = 0.079):
    gripper_state = torch.clip_by_value(gripper_state, 0, max_width) / max_width
    return 1 - gripper_state
import cv2

def create_video_from_camera_dict(image_dict, image_keys, output_video_path, fps=30):
    """
    Creates a video from a dictionary containing multiple cameras' images captured over time.

    Args:
    - image_dict (dict): Dictionary where keys are camera names and values are lists of NumPy images (frames over time).
    - output_video_path (str): Path to save the output video.
    - fps (int): Frames per second.
    """

    # Ensure all cameras have the same number of frames
    num_frames = len(image_dict)  # Get frame count from first camera
    camera_names = image_keys  # Sort keys to maintain order

    # Get image dimensions (assuming all frames have the same size)
    sample_frame = image_dict[0][camera_names[0]] # Get first frame from first camera
    height, width, channels = sample_frame.shape

    # Define video writer properties
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
    video_width = width * len(camera_names)  # Combined width of all cameras
    video_height = height  # Height remains the same
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (video_width, video_height))

    # Process each timestep
    for i in range(num_frames):
        # Collect images for the current timestep from all cameras
        frame_list = [image_dict[i][camera] for camera in camera_names]

        # Concatenate images side by side
        concatenated_frame = cv2.hconcat(frame_list)

        # Write the frame to the video
        video_writer.write(concatenated_frame)

    video_writer.release()
    print(f"Video saved at {output_video_path}")

