"""Tests for SecClient: header validation, retry logic, throttling."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from data_acquisition.sec_client import SecClient, SecClientError

GOOD_CONFIG = {
    "sec": {"user_agent": "Test User test@example.com"},
    "acquisition": {
        "sec_min_seconds_between_requests": 0,
        "http_timeout_seconds": 5,
        "http_max_retries": 3,
    },
}


def make_response(status=200, json_data=None):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = json_data or {}
    if status >= 400:
        r.raise_for_status.side_effect = requests.HTTPError(
            f"HTTP {status}", response=r
        )
    return r


def test_missing_user_agent_raises():
    with pytest.raises(SecClientError):
        SecClient({"sec": {"user_agent": ""}, "acquisition": {}})


def test_user_agent_without_email_raises():
    with pytest.raises(SecClientError):
        SecClient({"sec": {"user_agent": "no email here"}, "acquisition": {}})


def test_user_agent_header_is_set():
    client = SecClient(GOOD_CONFIG)
    assert client.session.headers["User-Agent"] == "Test User test@example.com"


@patch("data_acquisition.sec_client.time.sleep")
def test_retries_then_succeeds(mock_sleep):
    client = SecClient(GOOD_CONFIG)
    client.session.get = MagicMock(
        side_effect=[make_response(503), make_response(200, {"ok": True})]
    )
    assert client.get_json("http://x") == {"ok": True}
    assert client.session.get.call_count == 2


@patch("data_acquisition.sec_client.time.sleep")
def test_gives_up_after_max_retries(mock_sleep):
    client = SecClient(GOOD_CONFIG)
    client.session.get = MagicMock(return_value=make_response(503))
    with pytest.raises(SecClientError):
        client.get("http://x")
    assert client.session.get.call_count == 3


@patch("data_acquisition.sec_client.time.sleep")
def test_404_does_not_retry(mock_sleep):
    client = SecClient(GOOD_CONFIG)
    client.session.get = MagicMock(return_value=make_response(404))
    with pytest.raises(SecClientError):
        client.get("http://x")
    assert client.session.get.call_count == 1


def test_cik_is_zero_padded():
    client = SecClient(GOOD_CONFIG)
    client.get_json = MagicMock(return_value={})
    client.get_submissions("1045810")
    called_url = client.get_json.call_args[0][0]
    assert "CIK0001045810.json" in called_url
