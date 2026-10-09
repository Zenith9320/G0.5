"""G0.5 YAM real-robot policy client (YAMRobot-based).

Control loop that runs on the industrial PC (工控机):

    read YAM joint state (YAMRobot.get_joint_state) + cameras
      -> build raw_obs
      -> call serve_policy.py over WebSocket
      -> command YAM (YAMRobot.command_joint_state, 7-dim: 6 arm + 1 gripper)

This replaces the earlier ROS2-topic version. YAM's real control interface is
the `gello.robots.yam.YAMRobot` class (CAN -> i2rt driver), NOT ROS2 topics.

Run INSIDE gello_software's venv so `import gello.robots.yam` resolves:

    cd /home/xiangyushun/workspace/YAM/gello_software && source .venv/bin/activate
    uv pip install websockets msgpack numpy      # extras for ws_transport.py
    python ../g05_yam_deploy/policy_client.py --config ../g05_yam_deploy/config.yaml

Notes:
    * The action dict returned by the server uses parts_meta names
      (left_control / left_gripper / right_control / right_gripper / lower_body).
      `left_control`/`right_control` are padded to 9 dims; the first 6 are the
      real YAM arm joints.
    * YAMRobot wants a single 7-dim command [6 arm + 1 gripper], so we
      concatenate arm[:6] + gripper[:1] per side.
    * Camera ports / count / color order are TODO and must be confirmed on the
      real machine (see read_images()).
"""

from __future__ import annotations

import argparse
import asyncio

import numpy as np
import yaml

from gello.robots.yam import YAMRobot
from ws_transport import PolicyWebSocketClient

ARM_JOINT_DIM = 6  # YAM arm has 6 real joints; left_control is padded to 9


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


class YAMPolicyClient:
    def __init__(self, cfg: dict):
        self.cfg = cfg

        # One YAMRobot per arm, each on its own CAN channel.
        self.left = YAMRobot(channel=cfg["robot"]["left_channel"])
        self.right = YAMRobot(channel=cfg["robot"]["right_channel"])

        self.task = cfg["policy"]["task"]
        self.embodiment_type = cfg["policy"].get("embodiment_type", "galaxea_r1lite")
        self.frequency = cfg["policy"].get("frequency", 30)

        self.cameras = self._init_cameras(cfg.get("cameras", {}))

    def _init_cameras(self, cam_cfg: dict) -> dict:
        # YAM cameras are served over ZMQ (see gello_software's
        # launch_camera_clients.py / launch_camera_nodes.py). Fill the real
        # ports in config.yaml; keys must match the image names the server
        # expects: head_rgb, left_wrist_rgb, right_wrist_rgb.
        from gello.zmq_core.camera_node import ZMQClientCamera

        cameras = {}
        host = cam_cfg.get("host", "127.0.0.1")
        for key, port in cam_cfg.get("ports", {}).items():
            cameras[key] = ZMQClientCamera(port=port, host=host)
        return cameras

    # ── observation reading ──────────────────────────────────────────────
    def read_state(self) -> dict:
        # YAMRobot.get_joint_state() -> 7-dim [6 arm, 1 gripper] per side.
        s_l = np.asarray(self.left.get_joint_state(), dtype=np.float32)
        s_r = np.asarray(self.right.get_joint_state(), dtype=np.float32)
        return {
            "left_arm": s_l[:6],
            "left_gripper": s_l[6:7],
            "right_arm": s_r[:6],
            "right_gripper": s_r[6:7],
        }

    def read_images(self) -> dict:
        # Each image must be [C, H, W] uint8. The ZMQ camera returns a color
        # image as [H, W, 3] uint8.
        # TODO: confirm color order (RGB vs BGR). If red/blue look swapped,
        #       flip with img[:, :, ::-1] before transposing.
        images = {}
        for key, cam in self.cameras.items():
            img, _depth = cam.read()  # [H, W, 3] uint8
            images[key] = np.ascontiguousarray(img.transpose(2, 0, 1))
        return images

    def build_raw_obs(self) -> dict:
        if not self.cameras:
            raise RuntimeError("no cameras configured; fill config.yaml['cameras']")
        return {
            "task": self.task,
            "embodiment_type": self.embodiment_type,
            "frequency": self.frequency,
            "images": self.read_images(),
            "state": self.read_state(),
        }

    # ── action commanding ────────────────────────────────────────────────
    def command(self, action: dict) -> None:
        # action values are already denormalized + absolute (server-side
        # postprocess did the inverse transform). Map 27-dim back to 7-dim/arm.
        left_cmd = np.concatenate(
            [
                np.asarray(action["left_control"]).ravel()[:ARM_JOINT_DIM],
                np.asarray(action["left_gripper"]).ravel()[:1],
            ]
        )
        right_cmd = np.concatenate(
            [
                np.asarray(action["right_control"]).ravel()[:ARM_JOINT_DIM],
                np.asarray(action["right_gripper"]).ravel()[:1],
            ]
        )
        # TODO: verify gripper scaling — G0.5 outputs a normalized gripper
        # value, but YAMRobot's 7th joint may expect the i2rt gripper unit.
        self.left.command_joint_state(left_cmd.astype(np.float32))
        self.right.command_joint_state(right_cmd.astype(np.float32))

    # ── control loop ─────────────────────────────────────────────────────
    async def run(self) -> None:
        uri = f"ws://{self.cfg['server']['host']}:{self.cfg['server']['port']}"
        period = 1.0 / self.frequency

        async with PolicyWebSocketClient(uri) as client:
            print(f"connected to {uri} (embodiment={self.embodiment_type}, hz={self.frequency})")
            while True:
                try:
                    raw_obs = self.build_raw_obs()
                except RuntimeError as exc:
                    print(f"obs error: {exc}")
                    await asyncio.sleep(0.01)
                    continue

                resp = await client.infer(raw_obs)
                action = resp.get("action")
                if action is not None:
                    self.command(action)
                await asyncio.sleep(period)


def main():
    parser = argparse.ArgumentParser(description="YAM policy client (YAMRobot-based)")
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    client = YAMPolicyClient(cfg)
    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
