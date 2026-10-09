"""Smoke test 5/5: actually command the robot (DANGER).

Run only after test_predict.py prints sane deltas.

    python tests/test_live.py
"""

from __future__ import annotations

import argparse
import asyncio
import time

import numpy as np

from _helpers import (
    PolicyWebSocketClient,
    action_to_cmd,
    add_config_arg,
    build_raw_obs,
    load_config,
    make_cameras,
    make_robot,
    make_uri,
    read_state,
)


def main():
    parser = argparse.ArgumentParser()
    add_config_arg(parser)
    cfg = load_config(parser.parse_args().config)

    left, right = make_robot(cfg)
    cameras = make_cameras(cfg)
    uri = make_uri(cfg)
    period = 1.0 / cfg["policy"].get("frequency", 30)

    async def run():
        async with PolicyWebSocketClient(uri) as client:
            print("DANGER: commanding the robot. Ctrl+C to stop.")
            for i in range(3, 0, -1):
                print(f"  starting in {i} ...")
                time.sleep(1)
            while True:
                state = read_state(left, right)
                raw_obs = build_raw_obs(cfg, cameras, state)
                action = (await client.infer(raw_obs))["action"]
                l_cmd, r_cmd = action_to_cmd(action)
                left.command_joint_state(l_cmd)
                right.command_joint_state(r_cmd)
                print(f"  commanded | L={np.round(l_cmd, 3)} R={np.round(r_cmd, 3)}")
                await asyncio.sleep(period)

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("stopped. Run: scripts/safety hold")


if __name__ == "__main__":
    main()
