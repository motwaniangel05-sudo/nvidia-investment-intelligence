"""Phase 1 tests: config loading, folder structure, and logging."""

from pathlib import Path

import pytest
import yaml

from core.config_loader import (
    PROJECT_ROOT,
    ConfigError,
    get_path,
    load_config,
)
from core.logger import get_logger


def test_config_loads():
    config = load_config()
    assert config["company"]["ticker"] == "NVDA"
    assert config["data_period"]["start_year"] <= config["data_period"]["end_year"]


def test_config_missing_file_raises():
    with pytest.raises(ConfigError):
        load_config(Path("does_not_exist.yaml"))


def test_config_missing_section_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.dump({"company": {"name": "X", "ticker": "X"}}))
    with pytest.raises(ConfigError):
        load_config(bad)


def test_config_bad_years_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        yaml.dump(
            {
                "company": {"name": "X", "ticker": "X"},
                "data_period": {"start_year": 2030, "end_year": 2020},
                "paths": {},
                "logging": {},
            }
        )
    )
    with pytest.raises(ConfigError):
        load_config(bad)


def test_required_folders_exist():
    config = load_config()
    for key in ("data_raw", "data_processed", "data_financial",
                "data_market", "data_news", "data_documents",
                "reports", "charts", "agent_results", "logs"):
        assert get_path(config, key).is_dir(), f"Missing folder for '{key}'"


def test_unknown_path_key_raises():
    with pytest.raises(ConfigError):
        get_path(load_config(), "not_a_real_key")


def test_logger_writes_to_file():
    config = load_config()
    log = get_logger("test_setup")
    log.info("phase1 logger test message")
    for handler in log.handlers + log.parent.handlers:
        handler.flush()
    log_file = PROJECT_ROOT / config["logging"]["file"]
    assert log_file.exists()
    assert "phase1 logger test message" in log_file.read_text()