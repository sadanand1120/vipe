import json

from pathlib import Path
from typing import Any


class BATraceLogger:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8")
        self._stage_cycles: dict[str, int] = {}

    def close(self) -> None:
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def record_losses(self, context: dict[str, Any], losses: list[tuple[float, float]]) -> None:
        stage = str(context["stage"])
        cycle_base = int(context.get("cycle_base", 0))
        outer_iter = context.get("outer_iter")

        for ba_idx, (loss_pre, loss_post) in enumerate(losses, start=1):
            self._stage_cycles[stage] = self._stage_cycles.get(stage, 0) + 1
            record = dict(context)
            record["ba_iter"] = ba_idx
            record["cycle"] = cycle_base + ba_idx
            record["stage_cycle"] = self._stage_cycles[stage]
            record["nested"] = f"{outer_iter}.{ba_idx}" if outer_iter is not None else str(ba_idx)
            record["loss_pre"] = float(loss_pre)
            record["loss_post"] = float(loss_post)
            record["loss_delta"] = float(loss_pre - loss_post)
            record["loss"] = float(loss_post)
            self._file.write(json.dumps(record, sort_keys=True) + "\n")

        self._file.flush()
