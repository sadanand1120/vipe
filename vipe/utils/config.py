from pathlib import Path
from typing import Any

import yaml


class AttrDict(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def copy(self):
        return AttrDict({key: _to_attr_dict(value) for key, value in self.items()})


def _to_attr_dict(value: Any) -> Any:
    if isinstance(value, dict):
        return AttrDict({key: _to_attr_dict(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_attr_dict(item) for item in value]
    return value


def load_yaml_config(path: str | Path) -> AttrDict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML config: {path}")
    return _to_attr_dict(config)
