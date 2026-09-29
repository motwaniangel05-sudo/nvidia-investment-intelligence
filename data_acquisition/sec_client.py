"""
Polite HTTP client for SEC EDGAR.

Responsibilities:
  - Always send the required User-Agent header
  - Throttle requests (SEC limit is 10/sec; we go slower)
  - Retry on temporary failures with increasing wait times
  - Raise clear errors instead of failing silently
"""

import time
from typing import Any, Dict

import requests

from core.config_loader import load_config
from core.logger import get_logger

log = get_logger(__name__)

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"


class SecClientError(Exception):
    """Raised when a SEC request fails after all retries."""


class SecClient:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or load_config()

        user_agent = self.config.get("sec", {}).get("user_agent", "")
        if not user_agent or "@" not in user_agent:
            raise SecClientError(
                "config.yaml sec.user_agent must contain your name and email "
                "(SEC requires it). Example: 'Angel Motwani you@example.com'"
            )

        acq = self.config.get("acquisition", {})
        self.min_interval = acq.get("sec_min_seconds_between_requests", 0.25)
        self.timeout = acq.get("http_timeout_seconds", 30)
        self.max_retries = acq.get("http_max_retries", 3)

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self._last_request_time = 0.0

    def _throttle(self) -> None:
        """Sleep if needed so we never exceed our request rate."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request_time = time.monotonic()

    def get(self, url: str) -> requests.Response:
        """GET a URL with throttling and retries. Returns the Response."""
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                response = self.session.get(url, timeout=self.timeout)
                # 429 = rate limited, 5xx = server trouble: worth retrying
                if response.status_code == 429 or response.status_code >= 500:
                    raise requests.HTTPError(
                        f"HTTP {response.status_code}", response=response
                    )
                response.raise_for_status()  # other 4xx (like 404) -> error
                return response
            except requests.RequestException as e:
                last_error = e
                status = getattr(getattr(e, "response", None), "status_code", None)
                # 4xx errors other than 429 will not get better on retry
                if status is not None and 400 <= status < 500 and status != 429:
                    break
                wait = 2 ** attempt
                log.warning(
                    "Request failed (attempt %d/%d): %s. Waiting %ds.",
                    attempt, self.max_retries, e, wait,
                )
                time.sleep(wait)

        raise SecClientError(f"Failed to fetch {url}: {last_error}")

    def get_json(self, url: str) -> Dict[str, Any]:
        """GET a URL and parse the JSON body."""
        response = self.get(url)
        try:
            return response.json()
        except ValueError as e:
            raise SecClientError(f"Response from {url} was not valid JSON") from e

    def get_submissions(self, cik: str) -> Dict[str, Any]:
        """Fetch the filing catalog for a company. CIK is zero-padded to 10."""
        cik_padded = str(cik).zfill(10)
        url = SUBMISSIONS_URL.format(cik=cik_padded)
        log.info("Fetching submissions catalog: %s", url)
        return self.get_json(url)
