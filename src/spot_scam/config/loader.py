from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional

import yaml

from spot_scam.utils.paths import CONFIG_DIR, ensure_directories


class ConfigError(Exception):
    """Raised when configuration loading fails"""


def _deep_merge(
    base: MutableMapping[str, Any], updates: Mapping[str, Any]
) -> MutableMapping[str, Any]:
    """Recursively merge two dictionaries"""
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            base[key] = _deep_merge(dict(base[key]), value)  # type: ignore[arg-type]
        else:
            base[key] = copy.deepcopy(value)
    return base


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Configuration file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        try:
            return yaml.safe_load(handle) or {}
        except yaml.YAMLError as exc:  # pragma: no cover
            raise ConfigError(f"Failed to parse YAML at {path}: {exc}") from exc


def load_config(
    config_path: Optional[Path] = None,
    overrides: Optional[Mapping[str, Any]] = None,
    default_name: str = "defaults.yaml",
) -> Dict[str, Any]:
    ensure_directories()
    default_path = CONFIG_DIR / default_name
    config: Dict[str, Any] = _read_yaml(default_path)

    if config_path:
        candidate = Path(config_path)
        if candidate.is_absolute():
            resolved = candidate
        else:
            cwd_candidate = Path.cwd() / candidate
            resolved = cwd_candidate if cwd_candidate.exists() else CONFIG_DIR / candidate
        user_conf = _read_yaml(resolved)
        config = _deep_merge(config, user_conf)

    if overrides:
        config = _deep_merge(config, _expand_dot_keys(overrides))

    return config


def _expand_dot_keys(mapping: Mapping[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in mapping.items():
        segments = key.split(".")
        cursor = result
        for segment in segments[:-1]:
            cursor = cursor.setdefault(segment, {})
        cursor[segments[-1]] = value
    return result


def dump_config(config: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, sort_keys=False)


def config_hash(config: Mapping[str, Any]) -> str:
    serialized = json.dumps(config, sort_keys=True, separators=(",", ":"))

    hash_val = 0xCBF29CE484222325
    fnv_prime = 0x100000001B3
    for char in serialized:
        hash_val ^= ord(char)
        hash_val = (hash_val * fnv_prime) & 0xFFFFFFFFFFFFFFFF
    return f"{hash_val:016x}"


__all__ = [
    "ConfigError",
    "load_config",
    "dump_config",
    "config_hash",
]
