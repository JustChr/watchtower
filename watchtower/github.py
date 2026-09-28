"""Read-only GitHub client: REST with ETags (a 304 costs no rate limit), GraphQL for discussions."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

API = "https://api.github.com"
MAX_PAGES = 10
_NEXT = re.compile(r'<([^>]+)>;\s*rel="next"')


class GitHubError(Exception):
    """A GitHub call failed. The message never contains the token."""


class RateLimited(GitHubError):
    def __init__(self, reset_at: float) -> None:
        super().__init__(
            f"rate limited until {time.strftime('%H:%M:%S', time.gmtime(reset_at))} UTC"
        )
        self.reset_at = reset_at


class GitHub:
    def __init__(self, token: str, *, timeout: float = 30) -> None:
        self._token = token
        self._timeout = timeout
        self._etags: dict[str, str] = {}

    def _request(
        self, url: str, *, body: dict | None = None, etag: str | None = None
    ) -> tuple[int, dict[str, str], Any]:
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "watchtower",
        }
        if etag:
            headers["If-None-Match"] = etag
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, headers=headers)
        where = urllib.parse.urlsplit(url).path
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                lowered = {k.lower(): v for k, v in response.headers.items()}
                return response.status, lowered, json.load(response)
        except urllib.error.HTTPError as err:
            lowered = {k.lower(): v for k, v in err.headers.items()}
            if err.code == 304:
                return 304, lowered, None
            if err.code == 429 or (err.code == 403 and lowered.get("x-ratelimit-remaining") == "0"):
                if "retry-after" in lowered:
                    raise RateLimited(time.time() + float(lowered["retry-after"])) from None
                reset = float(lowered.get("x-ratelimit-reset") or time.time() + 60)
                raise RateLimited(reset) from None
            raise GitHubError(f"HTTP {err.code} {err.reason} for {where}") from None
        except (urllib.error.URLError, OSError, ValueError) as err:
            raise GitHubError(f"{type(err).__name__} for {where}") from None

    def get_list(self, path: str, **params: Any) -> list[dict]:
        """Every item of a paginated list, or ``[]`` when nothing changed since the last call."""

        url = f"{API}{path}?{urllib.parse.urlencode(params)}"
        status, headers, data = self._request(url, etag=self._etags.get(url))
        if status == 304:
            return []
        if "etag" in headers:
            self._etags[url] = headers["etag"]
        items = list(data)
        for _ in range(MAX_PAGES - 1):
            match = _NEXT.search(headers.get("link", ""))
            if not match:
                break
            _, headers, data = self._request(match.group(1))
            items.extend(data)
        return items

    def graphql(self, query: str, variables: dict[str, Any]) -> dict:
        _, _, data = self._request(f"{API}/graphql", body={"query": query, "variables": variables})
        if data.get("errors"):
            messages = "; ".join(e.get("message", "?") for e in data["errors"])
            raise GitHubError(f"GraphQL: {messages}")
        return data["data"]
