"""
Configuration loader.

Single entry point for reading config.yaml so every module gets the same
settings. Also resolves relative paths against the project root, so code
works no matter which directory you run it from.
"""

from pathlib import Path
from typing import Any, Dict

import yaml

# Project root = parent of the "core" folder this file lives in
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# Sections that must exist in config.yaml
REQUIRED_SECTIONS = ["company", "data_period", "paths", "logging"]


class ConfigError(Exception):
    """Raised when the configuration file is missing or invalid."""


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load and validate config.yaml. Returns the config as a dictionary."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Config file is not valid YAML: {e}") from e

    if not isinstance(config, dict):
        raise ConfigError("Config file is empty or not a mapping.")

    missing = [s for s in REQUIRED_SECTIONS if s not in config]
    if missing:
        raise ConfigError(f"Config is missing required sections: {missing}")

    for key in ("name", "ticker"):
        if not config["company"].get(key):
            raise ConfigError(f"company.{key} must be set in config.yaml")

    start = config["data_period"].get("start_year")
    end = config["data_period"].get("end_year")
    if start is None or end is None or start > end:
        raise ConfigError("data_period.start_year must be <= end_year")

    return config


def get_path(config: Dict[str, Any], key: str) -> Path:
    """Return an absolute Path for a key under config['paths']."""
    try:
        relative = config["paths"][key]
    except KeyError as e:
        raise ConfigError(f"Unknown path key in config: {key}") from e
    return PROJECT_ROOT / relative