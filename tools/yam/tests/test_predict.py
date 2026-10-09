"""Smoke test 4/5: real obs -> action, PRINT only (no motion).

Reviews predicted-vs-current deltas to catch sign/offset errors before `live`.

    python tests/test_predict.py
"""

from __future__ import annotations

import argparse
import asyncio

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
    state = read_state(left, right)
    raw_obs = build_raw_obs(cfg, cameras, state)
    uri = make_uri(cfg)

    async def run():
        async with PolicyWebSocketClient(uri) as client:
            action = (await client.infer(raw_obs))["action"]
            l_cmd, r_cmd = action_to_cmd(action)
            print("  current left_arm :", np.round(state["left_arm"], 4).tolist())
            print("  action  left_arm :", np.round(l_cmd[:6], 4).tolist())
            print("  current right_arm:", np.round(state["right_arm"], 4).tolist())
            print("  action  right_arm:", np.round(r_cmd[:6], 4).tolist())
            print("  current gripper  :", state["left_gripper"], state["right_gripper"])
            print("  action  gripper  :", l_cmd[6:7], r_cmd[6:7])
            d = np.concatenate(
                [l_cmd[:6] - state["left_arm"], r_cmd[:6] - state["right_arm"]]
            )
            print(f"  max |arm delta|  : {np.abs(d).max():.4f} rad")
            if np.abs(d).max() > 0.5:
                print("  WARNING: large delta — check joint signs/offsets before `live`")

    asyncio.run(run())
    print("PASS: prediction printed (no motion). Review deltas before `test_live`.")


if __name__ == "__main__":
    main()
