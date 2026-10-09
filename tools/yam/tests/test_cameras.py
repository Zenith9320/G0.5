"""Smoke test 2/5: read one frame per camera, save PNG for color check (read-only).

    python tests/test_cameras.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _helpers import add_config_arg, load_config, make_cameras

OUT_DIR = Path("/tmp/yam_smoke")


def main():
    parser = argparse.ArgumentParser()
    add_config_arg(parser)
    cfg = load_config(parser.parse_args().config)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
    except ImportError:
        raise SystemExit("pillow not installed: `uv pip install pillow`")

    for key, cam in make_cameras(cfg).items():
        img, _ = cam.read()  # [H, W, 3] uint8
        assert img.ndim == 3 and img.shape[2] == 3, f"{key}: shape {img.shape}"
        assert img.dtype == np.uint8, f"{key}: dtype {img.dtype}"
        # Save as-is so you can eyeball RGB vs BGR (a red object should look red).
        Image.fromarray(img).save(OUT_DIR / f"{key}.png")
        print(f"  {key:16s} shape={img.shape} -> {OUT_DIR / f'{key}.png'}")
    print("PASS: cameras read; open the PNGs to confirm color order and content")


if __name__ == "__main__":
    main()
