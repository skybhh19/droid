# ruff: noqa

import contextlib
import dataclasses
import datetime
import faulthandler
import os
import signal
import time
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import tqdm
import tyro
from droid.robot_env import RobotEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from PIL import Image

faulthandler.enable()

DROID_CONTROL_FREQUENCY = 15
TASK_PROMPT = "Insert the tool into the holder"

# Franka Emika Panda joint limits, in radians.
FRANKA_JOINT_LIMIT_LOW = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
FRANKA_JOINT_LIMIT_HIGH = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])


@dataclasses.dataclass
class Args:
    # Hardware parameters
    external_camera_id: str = "23404442"  # e.g., "24259877"
    wrist_camera_id: str = "17471093"  # e.g., "13062452"

    # Policy parameters
    prompt: str = TASK_PROMPT

    # Rollout parameters
    max_timesteps: int = 600
    # How many actions to execute from a predicted action chunk before querying policy server again.
    open_loop_horizon: int = 8
    # Maximum absolute joint-target change sent at a single 15 Hz step. Set <= 0 to disable.
    max_joint_delta: float = 0.2
    # Clip absolute joint targets to Panda joint limits before sending them.
    enforce_joint_limits: bool = True
    # Keep gripper execution identical to the joint-velocity eval scripts.
    binarize_gripper: bool = True
    gripper_threshold: float = 0.5
    # Whether to render an mp4 rollout video.
    render_video: bool = True
    # Directory where rollout videos are saved.
    video_out_dir: str = "."
    # Rendered rollout video FPS.
    video_fps: int = 10

    # Remote server parameters
    remote_host: str = "0.0.0.0"  # point this to the IP address of the policy server, e.g., "192.168.1.100"
    remote_port: int = 8123  # default tool-in-holder policy server port


@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Delay Ctrl+C until after the protected block."""
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
    if args.open_loop_horizon < 1:
        raise ValueError(f"open_loop_horizon must be >= 1, got {args.open_loop_horizon}")
    if args.max_joint_delta < 0:
        raise ValueError(f"max_joint_delta must be >= 0, got {args.max_joint_delta}")

    # This script is for policies served with LeRobotDROIDJointPositionDataConfig, whose websocket outputs are
    # [absolute_joint_position[7], gripper_position[1]] after OpenPI's output transforms.
    env = RobotEnv(action_space="joint_position", gripper_action_space="position")
    print("Created the droid env with joint-position arm control and gripper-position control.")

    policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)
    print(f"Connected to policy server at {args.remote_host}:{args.remote_port}")
    metadata = policy_client.get_server_metadata()
    if metadata:
        print(f"Policy server metadata: {metadata}")

    records: List[Dict[str, Union[str, float, int]]] = []

    while True:
        actions_from_chunk_completed = 0
        actions_in_current_chunk = 0
        pred_action_chunk = None
        num_policy_queries = 0

        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
        video = []
        last_t_step = -1
        max_commanded_joint_delta = 0.0

        bar = tqdm.tqdm(range(args.max_timesteps))
        print("Running rollout... press Ctrl+C to stop early.")
        for t_step in bar:
            last_t_step = t_step
            start_time = time.time()
            try:
                curr_obs = _extract_observation(
                    args,
                    env.get_observation(),
                    save_to_disk=t_step == 0,
                )

                video.append(
                    _make_video_frame(
                        curr_obs["external_image"],
                        curr_obs["wrist_image"],
                    )
                )

                if pred_action_chunk is None or actions_from_chunk_completed >= actions_in_current_chunk:
                    request_data = {
                        "observation/exterior_image_1_left": image_tools.resize_with_pad(
                            curr_obs["external_image"], 224, 224
                        ),
                        "observation/wrist_image_left": image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
                        "observation/joint_position": curr_obs["joint_position"],
                        "observation/gripper_position": curr_obs["gripper_position"],
                        "prompt": args.prompt,
                    }

                    with prevent_keyboard_interrupt():
                        pred_action_chunk = np.asarray(policy_client.infer(request_data)["actions"], dtype=np.float32)
                    _validate_action_chunk(pred_action_chunk)
                    actions_from_chunk_completed = 0
                    actions_in_current_chunk = min(args.open_loop_horizon, len(pred_action_chunk))
                    num_policy_queries += 1

                raw_action = pred_action_chunk[actions_from_chunk_completed]
                actions_from_chunk_completed += 1

                action = _prepare_joint_position_action(raw_action, curr_obs["joint_position"], args)
                max_commanded_joint_delta = max(
                    max_commanded_joint_delta,
                    float(np.max(np.abs(action[:7] - curr_obs["joint_position"]))),
                )

                env.step(action)

                elapsed_time = time.time() - start_time
                if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                    time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)
            except KeyboardInterrupt:
                break

        success = _read_success()

        save_filename = "tool_in_holder_joint_position_video_" + timestamp
        video_path = ""
        if args.render_video:
            outcome_dir = "success" if success > 0 else "fail"
            video_dir = os.path.join(args.video_out_dir, outcome_dir)
            os.makedirs(video_dir, exist_ok=True)
            video_path = os.path.join(video_dir, save_filename + ".mp4")
            _write_video(video_path, video, fps=args.video_fps)

        num_rollouts = len(records) + 1
        total_success = sum(float(record["success"]) for record in records) + success
        success_rate_so_far = total_success / num_rollouts
        print(f"Success rate so far: {success_rate_so_far:.3f} ({total_success:g}/{num_rollouts})")

        records.append(
            {
                "success": success,
                "duration": last_t_step,
                "video_filename": video_path,
                "num_rollouts": num_rollouts,
                "success_rate_so_far": success_rate_so_far,
                "num_policy_queries": num_policy_queries,
                "max_commanded_joint_delta": max_commanded_joint_delta,
            }
        )

        if input("Do one more eval? (enter y or n) ").strip().lower() != "y":
            break
        env.reset()

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%I:%M%p_%B_%d_%Y")
    csv_filename = os.path.join("results", f"eval_tool_in_holder_joint_position_{timestamp}.csv")
    pd.DataFrame(records).to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")


def _validate_action_chunk(action_chunk):
    if action_chunk.ndim != 2 or action_chunk.shape[1] != 8:
        raise ValueError(f"Expected policy action chunk with shape [horizon, 8], got {action_chunk.shape}")
    if len(action_chunk) == 0:
        raise ValueError("Policy returned an empty action chunk.")
    if not np.all(np.isfinite(action_chunk)):
        raise ValueError("Policy returned NaN or Inf actions.")


def _prepare_joint_position_action(raw_action, current_joint_position, args: Args):
    raw_action = np.asarray(raw_action, dtype=np.float32)
    current_joint_position = np.asarray(current_joint_position, dtype=np.float32)

    if raw_action.shape != (8,):
        raise ValueError(f"Expected one action with shape (8,), got {raw_action.shape}")
    if current_joint_position.shape != (7,):
        raise ValueError(f"Expected current joint position with shape (7,), got {current_joint_position.shape}")

    joint_position = raw_action[:7].copy()
    if args.enforce_joint_limits:
        joint_position = np.clip(joint_position, FRANKA_JOINT_LIMIT_LOW, FRANKA_JOINT_LIMIT_HIGH)

    if args.max_joint_delta > 0:
        joint_position = np.clip(
            joint_position,
            current_joint_position - args.max_joint_delta,
            current_joint_position + args.max_joint_delta,
        )
        if args.enforce_joint_limits:
            joint_position = np.clip(joint_position, FRANKA_JOINT_LIMIT_LOW, FRANKA_JOINT_LIMIT_HIGH)

    gripper_position = float(np.clip(raw_action[-1], 0, 1))
    if args.binarize_gripper:
        gripper_position = 1.0 if gripper_position > args.gripper_threshold else 0.0

    return np.concatenate([joint_position, np.array([gripper_position], dtype=np.float32)]).astype(
        np.float32, copy=False
    )


def _read_success():
    while True:
        success_input = input(
            "Did the rollout succeed? (enter y for 100%, n for 0%), or a numeric value 0-100 based on the evaluation spec"
        ).strip().lower()
        if success_input == "y":
            return 1.0
        if success_input == "n":
            return 0.0

        try:
            success = float(success_input) / 100
        except ValueError:
            print(f"Expected y, n, or a numeric value in [0, 100], got: {success_input!r}")
            continue

        if 0 <= success <= 1:
            return success
        print(f"Success must be a number in [0, 100] but got: {success * 100}")


def _extract_observation(args: Args, obs_dict, *, save_to_disk=False):
    image_observations = obs_dict["image"]
    external_image, wrist_image = None, None
    for key in image_observations:
        if args.external_camera_id in key and "left" in key:
            external_image = image_observations[key]
        elif args.wrist_camera_id in key and "left" in key:
            wrist_image = image_observations[key]

    if external_image is None:
        raise ValueError(f"Could not find external camera {args.external_camera_id!r} in observation keys.")
    if wrist_image is None:
        raise ValueError(f"Could not find wrist camera {args.wrist_camera_id!r} in observation keys.")

    external_image = external_image[..., :3]
    wrist_image = wrist_image[..., :3]

    external_image = external_image[..., ::-1]
    wrist_image = wrist_image[..., ::-1]

    robot_state = obs_dict["robot_state"]
    cartesian_position = np.array(robot_state["cartesian_position"])
    joint_position = np.array(robot_state["joint_positions"])
    gripper_position = np.array([robot_state["gripper_position"]])

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
