"""Posting as a GitHub App, with short-lived tokens from the App's private key.

The key signs a JWT (RS256, valid ten minutes) that proves being the App; the
JWT buys an installation token for one repo, valid an hour and limited to
writing issues and discussions (comments, and labels on issues). Only the
poster holds the key.
"""

from __future__ import annotations

import base64
import calendar
import json
import time
from collections.abc import Callable
from typing import Any, Protocol

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from .github import GitHub, GitHubError
from .store import Draft

# Renew an installation token this long before it expires (seconds).
TOKEN_MARGIN = 300
PERMISSIONS = {"issues": "write", "discussions": "write"}

DISCUSSION_ID = """
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) { discussion(number: $number) { id } }
}
"""
ADD_DISCUSSION_COMMENT = """
mutation($discussion: ID!, $body: String!, $replyTo: ID) {
  addDiscussionComment(input: {discussionId: $discussion, body: $body, replyToId: $replyTo}) {
    comment { url }
  }
}
"""


class Client(Protocol):
    def get_json(self, path: str) -> Any: ...
    def post_json(self, path: str, body: dict) -> Any: ...
    def graphql(self, query: str, variables: dict[str, Any]) -> dict: ...


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _json(value: dict) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode()


class App:
    def __init__(
        self,
        app_id: str,
        key_pem: str,
        *,
        connect: Callable[[str], Client] = GitHub,
        clock: Callable[[], float] = time.time,
    ) -> None:
        key = serialization.load_pem_private_key(key_pem.encode(), password=None)
        if not isinstance(key, rsa.RSAPrivateKey):
            raise ValueError("the GitHub App key must be an RSA private key")
        self._key = key
        # Numeric App ID or Client ID ("Iv23..."): GitHub accepts either as issuer.
        self._issuer: int | str = int(app_id) if app_id.isdigit() else app_id
        self._connect = connect
        self._clock = clock
        self._tokens: dict[str, tuple[str, float]] = {}

    def jwt(self) -> str:
        now = int(self._clock())
        # Issued a minute back, against clock drift; GitHub allows at most ten minutes.
        payload = {"iat": now - 60, "exp": now + 540, "iss": self._issuer}
        signing_input = f"{_b64(_json({'alg': 'RS256', 'typ': 'JWT'}))}.{_b64(_json(payload))}"
        signature = self._key.sign(signing_input.encode(), padding.PKCS1v15(), hashes.SHA256())
        return f"{signing_input}.{_b64(signature)}"

    def slug(self) -> str:
        """The App's name on GitHub; its comments show as ``<slug>[bot]``."""

        return self._connect(self.jwt()).get_json("/app")["slug"]

    def token(self, repo: str) -> str:
        """An installation token for ``repo`` only, reused until shortly before it expires."""

        cached = self._tokens.get(repo)
        if cached and cached[1] - TOKEN_MARGIN > self._clock():
            return cached[0]
        app = self._connect(self.jwt())
        installation = app.get_json(f"/repos/{repo}/installation")["id"]
        data = app.post_json(
            f"/app/installations/{int(installation)}/access_tokens",
            {"repositories": [repo.split("/", 1)[1]], "permissions": PERMISSIONS},
        )
        expires = calendar.timegm(time.strptime(data["expires_at"], "%Y-%m-%dT%H:%M:%SZ"))
        self._tokens[repo] = (data["token"], expires)
        return data["token"]

    def post(self, draft: Draft, body: str) -> str:
        """Post ``body`` as a comment where ``draft`` belongs; returns the comment's URL.

        Anything failing before the write itself raises a ``definite`` ``GitHubError``:
        nothing was posted.
        """

        try:
            client = self._connect(self.token(draft.repo))
            discussion = None
            if draft.kind == "discussion":
                owner, name = draft.repo.split("/")
                variables = {"owner": owner, "name": name, "number": draft.number}
                data = client.graphql(DISCUSSION_ID, variables)
                discussion = data["repository"]["discussion"]["id"]
        except GitHubError as err:
            raise GitHubError(str(err), definite=True) from None
        except (KeyError, TypeError, ValueError) as err:
            raise GitHubError(f"unexpected answer: {type(err).__name__}", definite=True) from None

        if discussion is not None:
            variables = {"discussion": discussion, "body": body, "replyTo": draft.reply_to or None}
            data = client.graphql(ADD_DISCUSSION_COMMENT, variables)
            return data["addDiscussionComment"]["comment"]["url"]
        path = f"/repos/{draft.repo}/issues/{draft.number}/comments"
        return client.post_json(path, {"body": body})["html_url"]

    def label(self, draft: Draft, name: str) -> None:
        """Add the label ``name`` to the draft's issue (GitHub creates it if it's new)."""

        client = self._connect(self.token(draft.repo))
        client.post_json(f"/repos/{draft.repo}/issues/{draft.number}/labels", {"labels": [name]})
