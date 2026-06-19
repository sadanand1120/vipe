import json
import os
import time

from pathlib import Path
from typing import Any

import numpy as np
import torch


class SLAMDebugTrace:
    def __init__(self, path: str | os.PathLike[str] | None) -> None:
        self.path = Path(path) if path else None
        self._fp = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fp = self.path.open("w", encoding="utf-8")

    @classmethod
    def from_env(cls) -> "SLAMDebugTrace":
        return cls(os.environ.get("VIPE_SLAM_DEBUG_TRACE_PATH"))

    @property
    def enabled(self) -> bool:
        return self._fp is not None

    def close(self) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None

    def write(self, event: str, **payload: Any) -> None:
        if self._fp is None:
            return
        record = {"event": event, "time": time.time(), **payload}
        self._fp.write(json.dumps(record, default=self._json_default, sort_keys=True) + "\n")
        self._fp.flush()

    @staticmethod
    def _json_default(obj: Any):
        if isinstance(obj, torch.Tensor):
            obj = obj.detach().cpu()
            if obj.numel() <= 64:
                return obj.tolist()
            return {"shape": list(obj.shape), "dtype": str(obj.dtype)}
        if isinstance(obj, np.ndarray):
            if obj.size <= 64:
                return obj.tolist()
            return {"shape": list(obj.shape), "dtype": str(obj.dtype)}
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
