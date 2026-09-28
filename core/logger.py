"""
Central logger. Every module calls get_logger(__name__) and gets a logger
that writes to both the console and logs/app.log.
"""

import logging
import sys
from pathlib import Path

from core.config_loader import PROJECT_ROOT, load_config

_configured = False


def _setup_logging() -> None:
    """Configure the root logger once (console + file)."""
    global _configured
    if _configured:
        return

    config = load_config()
    level_name = config["logging"].get("level", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    log_file = PROJECT_ROOT / config["logging"].get("file", "logs/app.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(console)
    root.addHandler(file_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger. Usage: log = get_logger(__name__)"""
    _setup_logging()
    return logging.getLogger(name)