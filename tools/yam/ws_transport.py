"""Standalone WebSocket transport for the G0.5 policy client.

Copied from the official GalaxeaVLA repo and de-coupled from `g05` so it can
run on the robot's industrial PC (工控机) WITHOUT installing the full model
code / torch / weights. Only dependencies: `msgpack`, `numpy`, `websockets`.

The wire format must match `serve_policy.py` exactly:
  - ndarray    -> {"__ndarray__": True, "data": <bytes>, "dtype": str, "shape": tuple}
  - np scalar  -> {"__npgeneric__": True, "data": <scalar>, "dtype": str}
"""

from __future__ import annotations

import asyncio
import functools
from typing import Any

import msgpack
import numpy as np

_UNSUPPORTED_KINDS = ("V", "O", "c")  # void, object, complex


def _pack(obj: Any) -> Any:
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in _UNSUPPORTED_KINDS:
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            "__ndarray__": True,
            "data": obj.tobytes(),
            "dtype": obj.dtype.str,
            "shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {"__npgeneric__": True, "data": obj.item(), "dtype": obj.dtype.str}
    return obj


def _bytes_to_str_keys(obj: Any) -> Any:
    """Recursively decode bytes-typed dict keys to str."""
    if isinstance(obj, dict):
        return {
            (k.decode() if isinstance(k, bytes) else k): _bytes_to_str_keys(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_bytes_to_str_keys(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_bytes_to_str_keys(v) for v in obj)
    return obj


def _unpack(obj: Any) -> Any:
    obj = _bytes_to_str_keys(obj)
    if "__ndarray__" in obj:
        return np.ndarray(
            buffer=obj["data"],
            dtype=np.dtype(obj["dtype"]),
            shape=tuple(obj["shape"]),
        )
    if "__npgeneric__" in obj:
        return np.dtype(obj["dtype"]).type(obj["data"])
    return obj


packb = functools.partial(msgpack.packb, default=_pack)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack)


class PolicyWebSocketClient:
    """Thin async WebSocket client for `serve_policy.py`.

    Mirrors `scripts/utils/policy_ws_client.py` from the official repo, minus
    the `g05` import (packb/unpackb are defined locally above).
    """

    def __init__(self, uri: str, timeout_s: float = 30.0):
        self.uri = uri
        self.timeout_s = timeout_s
        self._ws = None
        self.metadata: dict[str, Any] | None = None

    async def connect(self) -> dict[str, Any]:
        import websockets

        connect_kwargs = {
            "max_size": None,
            "ping_interval": 30,
            "ping_timeout": 120,
            "proxy": None,
        }
        try:
            self._ws = await websockets.connect(self.uri, **connect_kwargs)
        except TypeError as exc:
            if "proxy" not in str(exc):
                raise
            connect_kwargs.pop("proxy")
            self._ws = await websockets.connect(self.uri, **connect_kwargs)
        handshake = await asyncio.wait_for(self._ws.recv(), timeout=self.timeout_s)
        self.metadata = unpackb(handshake)
        return self.metadata

    async def infer(
        self, raw_obs: dict[str, Any], current_step: int | None = None
    ) -> dict[str, Any]:
        if self._ws is None:
            raise RuntimeError("WebSocket client is not connected")
        if current_step is not None:
            raw_obs = {**raw_obs, "current_step": current_step}
        await asyncio.wait_for(self._ws.send(packb(raw_obs)), timeout=self.timeout_s)
        response = await asyncio.wait_for(self._ws.recv(), timeout=self.timeout_s)
        return unpackb(response)

    async def reset(self) -> None:
        if self._ws is None:
            raise RuntimeError("WebSocket client is not connected")
        await asyncio.wait_for(self._ws.send(packb({"__reset__": True})), timeout=self.timeout_s)
        response = await asyncio.wait_for(self._ws.recv(), timeout=self.timeout_s)
        resp = unpackb(response)
        if not isinstance(resp, dict) or not resp.get("__reset__"):
            raise RuntimeError(f"Unexpected reset response: {resp}")

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    async def __aenter__(self) -> "PolicyWebSocketClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()
