"""Smoke test 1/5: environment + CAN connection + read joint state (read-only).

    python tests/test_env.py
"""

from __future__ import annotations

import argparse

import numpy as np

from _helpers import add_config_arg, load_config, make_robot, read_state


def main():
    parser = argparse.ArgumentParser()
    add_config_arg(parser)
    cfg = load_config(parser.parse_args().config)

    left, right = make_robot(cfg)
    state = read_state(left, right)
    print("PASS: YAMRobot instantiated + state read")
    for k, v in state.items():
        assert np.isfinite(v).all(), f"{k} has NaN/inf: {v}"
        print(f"  {k:14s} {v.tolist()}")
    assert state["left_arm"].shape == (6,), f"left_arm shape {state['left_arm'].shape}"
    print("PASS: all state finite, shapes correct")


if __name__ == "__main__":
    main()
