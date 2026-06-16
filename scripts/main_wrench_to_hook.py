# ruff: noqa

import contextlib
import dataclasses
import datetime
import faulthandler
import os
import signal
import time
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
import tqdm
import tyro
from droid.robot_env import RobotEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from PIL import Image

faulthandler.enable()

# DROID data collection frequency -- we slow down execution to match this frequency
DROID_CONTROL_FREQUENCY = 15


@dataclasses.dataclass
class Args:
    # Hardware parameters
    external_camera_id: str = "23404442"  # e.g., "24259877"
    wrist_camera_id: str = "17471093"  # e.g., "13062452"

    # Rollout parameters
    max_timesteps: int = 600
    # How many actions to execute from a predicted action chunk before querying policy server again
    # 8 is usually a good default (equals 0.5 seconds of action execution).
    open_loop_horizon: int = 8
    # Whether to render an mp4 rollout video.
    render_video: bool = True
    # Directory where rollout videos are saved.
    video_out_dir: str = "."
    # Rendered rollout video FPS.
    video_fps: int = 10

    # Remote server parameters
    remote_host: str = "0.0.0.0"  # point this to the IP address of the policy server, e.g., "192.168.1.100"
    remote_port: int = 8123  # default wrench-on-hook policy server port


# We are using Ctrl+C to optionally terminate rollouts early -- however, if we press Ctrl+C while the policy server is
# waiting for a new action chunk, it will raise an exception and the server connection dies.
# This context manager temporarily prevents Ctrl+C and delays it after the server call is complete.
@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Temporarily prevent keyboard interrupts by delaying them until after the protected code."""
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


def main(args: Args):
    # Initialize the Panda environment. Using joint velocity action space and gripper position action space is very important.
    env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")
    print("Created the droid env!")

    # Connect to the policy server
    policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)

    df = pd.DataFrame(columns=["success", "duration", "video_filename", "num_rollouts", "success_rate_so_far"])

    instruction = "Hang the wrench on the hook"
    while True:
        # instruction = input("Enter instruction: ")

        # Rollout parameters
        actions_from_chunk_completed = 0
        pred_action_chunk = None

        # Prepare to save video of rollout
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
        video = []
        bar = tqdm.tqdm(range(args.max_timesteps))
        print("Running rollout... press Ctrl+C to stop early.")
        for t_step in bar:
            start_time = time.time()
            try:
                # Get the current observation
                curr_obs = _extract_observation(
                    args,
                    env.get_observation(),
                    # Save the first observation to disk
                    save_to_disk=t_step == 0,
                )

                video.append(
                    _make_video_frame(
                        curr_obs["external_image"],
                        curr_obs["wrist_image"],
                    )
                )

                # Send websocket request to policy server if it's time to predict a new chunk
                if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:
                    actions_from_chunk_completed = 0

                    # We resize images on the robot laptop to minimize the amount of data sent to the policy server
                    # and improve latency.
                    request_data = {
                        "observation/exterior_image_1_left": image_tools.resize_with_pad(
                            curr_obs["external_image"], 224, 224
                        ),
                        "observation/wrist_image_left": image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
                        "observation/joint_position": curr_obs["joint_position"],
                        "observation/gripper_position": curr_obs["gripper_position"],
                        "prompt": instruction,
                    }

                    # Wrap the server call in a context manager to prevent Ctrl+C from interrupting it
                    # Ctrl+C will be handled after the server call is complete
                    with prevent_keyboard_interrupt():
                        # this returns action chunk [10, 8] of 10 joint velocity actions (7) + gripper position (1)
                        pred_action_chunk = policy_client.infer(request_data)["actions"]
                    # assert pred_action_chunk.shape == (10, 8)

                # Select current action to execute from chunk
                action = pred_action_chunk[actions_from_chunk_completed]
                actions_from_chunk_completed += 1

                # Binarize gripper action
                if action[-1].item() > 0.5:
                    # action[-1] = 1.0
                    action = np.concatenate([action[:-1], np.ones((1,))])
                else:
                    # action[-1] = 0.0
                    action = np.concatenate([action[:-1], np.zeros((1,))])

                # clip all dimensions of action to [-1, 1]
                action = np.clip(action, -1, 1)

                env.step(action)

                # Sleep to match DROID data collection frequency
                elapsed_time = time.time() - start_time
                if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                    time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)
            except KeyboardInterrupt:
                break

        success: Optional[Union[str, float]] = None
        while not isinstance(success, float):
            success_input = input(
                "Did the rollout succeed? (enter y for 100%, n for 0%), or a numeric value 0-100 based on the evaluation spec"
            ).strip().lower()
            if success_input == "y":
                success = 1.0
            elif success_input == "n":
                success = 0.0
            else:
                success = float(success_input) / 100

            if not (0 <= success <= 1):
                print(f"Success must be a number in [0, 100] but got: {success * 100}")
                success = None

        save_filename = "wrench_on_hook_video_" + timestamp
        video_path = ""
        if args.render_video:
            outcome_dir = "success" if success > 0 else "fail"
            video_dir = os.path.join(args.video_out_dir, outcome_dir)
            os.makedirs(video_dir, exist_ok=True)
            video_path = os.path.join(video_dir, save_filename + ".mp4")
            _write_video(video_path, video, fps=args.video_fps)

        num_rollouts = len(df) + 1
        success_rate_so_far = (df["success"].sum() + success) / num_rollouts
        print(f"Success rate so far: {success_rate_so_far:.3f} ({df['success'].sum() + success:g}/{num_rollouts})")

        df = df.append(
            {
                "success": success,
                "duration": t_step,
                "video_filename": video_path,
                "num_rollouts": num_rollouts,
                "success_rate_so_far": success_rate_so_far,
            },
            ignore_index=True,
        )

        if input("Do one more eval? (enter y or n) ").lower() != "y":
            break
        env.reset()

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%I:%M%p_%B_%d_%Y")
    csv_filename = os.path.join("results", f"eval_wrench_on_hook_{timestamp}.csv")
    df.to_csv(csv_filename)
    print(f"Results saved to {csv_filename}")


def _extract_observation(args: Args, obs_dict, *, save_to_disk=False):
    image_observations = obs_dict["image"]
    external_image, wrist_image = None, None
    for key in image_observations:
        # Note the "left" below refers to the left camera in the stereo pair.
        # The model is only trained on left stereo cams, so we only feed those.
        if args.external_camera_id in key and "left" in key:
            external_image = image_observations[key]
        elif args.wrist_camera_id in key and "left" in key:
            wrist_image = image_observations[key]

    if external_image is None:
        raise ValueError(f"Could not find external camera {args.external_camera_id!r} in observation keys.")
    if wrist_image is None:
        raise ValueError(f"Could not find wrist camera {args.wrist_camera_id!r} in observation keys.")

    # Drop the alpha dimension
    external_image = external_image[..., :3]
    wrist_image = wrist_image[..., :3]

    # Convert to RGB
    external_image = external_image[..., ::-1]
    wrist_image = wrist_image[..., ::-1]

    # In addition to image observations, also capture the proprioceptive state
    robot_state = obs_dict["robot_state"]
    cartesian_position = np.array(robot_state["cartesian_position"])
    joint_position = np.array(robot_state["joint_positions"])
    gripper_position = np.array([robot_state["gripper_position"]])

    # Save the images to disk so that they can be viewed live while the robot is running
    # Create one combined image to make live viewing easy
    if save_to_disk:
        combined_image = _make_video_frame(external_image, wrist_image)
        combined_image = Image.fromarray(combined_image)
        combined_image.save("robot_camera_views.png")

    return {
        "external_image": external_image,
        "wrist_image": wrist_image,
        "cartesian_position": cartesian_position,
        "joint_position": joint_position,
        "gripper_position": gripper_position,
    }


def _make_video_frame(external_image, wrist_image):
    external_image = np.asarray(external_image)[..., :3]
    wrist_image = np.asarray(wrist_image)[..., :3]

    if external_image.shape[0] != wrist_image.shape[0]:
        target_height = external_image.shape[0]
        target_width = round(wrist_image.shape[1] * target_height / wrist_image.shape[0])
        wrist_image = np.asarray(Image.fromarray(wrist_image).resize((target_width, target_height)))

    return np.concatenate([external_image, wrist_image], axis=1)


def _write_video(path, frames, fps):
    frames = [np.ascontiguousarray(np.asarray(frame)[..., :3].astype(np.uint8)) for frame in frames]
    if not frames:
        return

    try:
        try:
            import imageio.v2 as imageio
        except ImportError:
            import imageio

        imageio.mimsave(path, frames, fps=fps)
        return
    except ImportError:
        pass

    try:
        import cv2
    except ImportError as exc:
        raise ImportError("Saving rollout videos requires either imageio or opencv-python in this environment.") from exc

    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)
