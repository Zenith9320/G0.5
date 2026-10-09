"""Shared helpers for the YAM smoke tests.

Not runnable on its own — imported by test_*.py so the five smoke tests don't
duplicate config loading, obs building, and action mapping.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `ws_transport` (one directory up) importable regardless of cwd.
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import yaml

from ws_transport import PolicyWebSocketClient  # noqa: E402

ARM_JOINT_DIM = 6  # YAM arm has 6 real joints; left_control is padded to 9


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def default_config_path() -> str:
    return str(_REPO / "config.yaml")


def add_config_arg(parser) -> None:
    parser.add_argument("--config", default=default_config_path())


def make_robot(cfg: dict):
    """Return (left, right) YAMRobot instances."""
    from gello.robots.yam import YAMRobot

    return (
        YAMRobot(channel=cfg["robot"]["left_channel"]),
        YAMRobot(channel=cfg["robot"]["right_channel"]),
    )


def make_cameras(cfg: dict) -> dict:
    from gello.zmq_core.camera_node import ZMQClientCamera

    host = cfg["cameras"].get("host", "127.0.0.1")
    return {
        key: ZMQClientCamera(port=port, host=host)
        for key, port in cfg["cameras"]["ports"].items()
    }


def make_uri(cfg: dict) -> str:
    return f"ws://{cfg['server']['host']}:{cfg['server']['port']}"


def read_state(left, right) -> dict:
    s_l = np.asarray(left.get_joint_state(), dtype=np.float32)
    s_r = np.asarray(right.get_joint_state(), dtype=np.float32)
    return {
        "left_arm": s_l[:ARM_JOINT_DIM],
        "left_gripper": s_l[ARM_JOINT_DIM : ARM_JOINT_DIM + 1],
        "right_arm": s_r[:ARM_JOINT_DIM],
        "right_gripper": s_r[ARM_JOINT_DIM : ARM_JOINT_DIM + 1],
    }


def build_raw_obs(cfg: dict, cameras: dict, state: dict) -> dict:
    images = {}
    for key, cam in cameras.items():
        img, _ = cam.read()  # [H, W, 3] uint8
        images[key] = np.ascontiguousarray(img.transpose(2, 0, 1))
    return {
        "task": cfg["policy"]["task"],
        "embodiment_type": cfg["policy"].get("embodiment_type", "galaxea_r1lite"),
        "frequency": cfg["policy"].get("frequency", 30),
        "images": images,
        "state": state,
    }


def action_to_cmd(action: dict):
    """Map the 27-dim action back to per-arm 7-dim [6 arm + 1 gripper]."""
    left = np.concatenate(
        [
            np.asarray(action["left_control"]).ravel()[:ARM_JOINT_DIM],
            np.asarray(action["left_gripper"]).ravel()[:1],
        ]
    )
    right = np.concatenate(
        [
            np.asarray(action["right_control"]).ravel()[:ARM_JOINT_DIM],
            np.asarray(action["right_gripper"]).ravel()[:1],
        ]
    )
    return left.astype(np.float32), right.astype(np.float32)
