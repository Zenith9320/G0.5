"""Smoke test 3/5: WebSocket + wire-format round-trip (no robot, no motion).

Uses a zero dummy obs so it validates the format + server _validate_obs without
touching the robot or cameras.

    python tests/test_transport.py
"""

from __future__ import annotations

import argparse
import asyncio

import numpy as np

from _helpers import (
    PolicyWebSocketClient,
    add_config_arg,
    load_config,
    make_uri,
)


def main():
    parser = argparse.ArgumentParser()
    add_config_arg(parser)
    cfg = load_config(parser.parse_args().config)

    dummy = {
        "task": cfg["policy"]["task"],
        "embodiment_type": cfg["policy"].get("embodiment_type", "galaxea_r1lite"),
        "frequency": cfg["policy"].get("frequency", 30),
        "images": {
            "head_rgb": np.zeros((3, 480, 640), dtype=np.uint8),
            "left_wrist_rgb": np.zeros((3, 480, 640), dtype=np.uint8),
            "right_wrist_rgb": np.zeros((3, 480, 640), dtype=np.uint8),
        },
        "state": {
            "left_arm": np.zeros(6, dtype=np.float32),
            "left_gripper": np.zeros(1, dtype=np.float32),
            "right_arm": np.zeros(6, dtype=np.float32),
            "right_gripper": np.zeros(1, dtype=np.float32),
        },
    }
    uri = make_uri(cfg)

    async def run():
        async with PolicyWebSocketClient(uri) as client:
            print(f"  connected to {uri}")
            print("  server metadata:", client.metadata)
            resp = await client.infer(dummy)
            print("  resp keys:", sorted(resp.keys()))
            assert "action" in resp, f"no 'action': {sorted(resp.keys())}"
            print("  action keys:", sorted(resp["action"].keys()))
            print("  need_obs:", resp.get("need_obs"))

    asyncio.run(run())
    print("PASS: transport + format round-trip through the real server")


if __name__ == "__main__":
    main()
