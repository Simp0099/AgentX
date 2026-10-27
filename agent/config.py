"""agent/config.py: typed accessor over configs/default.yaml.

Reads the YAML config once and provides it to the agent components.
No network access, no model modification. Pure configuration.
"""

from __future__ import annotations

import os
from typing import Any

import yaml

_DEFCONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "configs", "default.yaml"
)

_config_cache: dict[str, Any] | None = None


def get_config() -> dict[str, Any]:
    """Return the configuration dict, loading from default.yaml once."""
    global _config_cache
    if _config_cache is None:
        if os.path.isfile(_DEFCONFIG_PATH):
            with open(_DEFCONFIG_PATH) as f:
                _config_cache = yaml.safe_load(f)
        else:
            _config_cache = {
                "data": {"root": "data/CASIA2", "image_size": [224, 224]},
                "training": {"batch_size": 32, "epochs_frozen": 10},
                "output": {"checkpoint_dir": "checkpoints/", "threshold": 0.5},
                "forensics": {},
            }
    return _config_cache


def get(section: str, key: str, default: Any = None) -> Any:
    """Convenience accessor: get config[section][key] or default."""
    config = get_config()
    return config.get(section, {}).get(key, default)
