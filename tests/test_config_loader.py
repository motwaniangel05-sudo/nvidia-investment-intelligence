"""Tests for load_config validation errors."""

import copy

import pytest
import yaml

from core.config_loader import ConfigError, load_config


def _write(tmp_path, content):
    path = tmp_path / "config.yaml"
    path.write_text(content)
    return path


def test_invalid_yaml_raises_config_error(tmp_path):
    path = _write(tmp_path, "company: [unclosed")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(path)


@pytest.mark.parametrize("content", ["", "- a\n- b\n", "just a string"])
def test_empty_or_non_mapping_config_raises_config_error(tmp_path, content):
    path = _write(tmp_path, content)
    with pytest.raises(ConfigError, match="empty or not a mapping"):
        load_config(path)


@pytest.mark.parametrize("key", ["name", "ticker"])
def test_blank_company_field_raises_config_error(tmp_path, key):
    cfg = copy.deepcopy(load_config())  # the real, valid config
    cfg["company"][key] = ""
    path = _write(tmp_path, yaml.safe_dump(cfg))

    with pytest.raises(ConfigError, match=f"company.{key}"):
        load_config(path)


def test_valid_config_round_trips_through_a_temp_file(tmp_path):
    cfg = load_config()
    path = _write(tmp_path, yaml.safe_dump(cfg))
    assert load_config(path) == cfg
