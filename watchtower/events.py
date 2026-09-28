"""Turn GitHub's lists into new-activity events.

Rules, per repository:

- a **baseline** is stored on the first poll (now, minus ``backfill_hours``);
  nothing created before it is ever reported, so a first start doesn't flood;
- each list keeps a **cursor** (the newest ``updated_at`` seen). The URL only
  changes when the cursor moves, so an unchanged repo answers 304, free;
- cursors are returned, not stored: the watcher stores them only after the
  events are queued, so a crash in between re-reports instead of losing.

Dedup against already-reported events is the watcher's job (``Store.is_seen``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

from .config import Config
from .store import Store

DISCUSSIONS_QUERY = """
query($owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) {
    discussions(first: 15, orderBy: {field: UPDATED_AT, direction: DESC}) {
      nodes {
        number title url createdAt body author { login } authorAssociation
        comments(last: 20) {
          nodes {
            id url createdAt body author { login } authorAssociation
            replies(last: 10) {
              nodes { id url createdAt body author { login } authorAssociation }
            }
          }
        }
      }
    }
  }
}
"""


class Source(Protocol):
    def get_list(self, path: str, **params: object) -> list[dict]: ...
    def get_json(self, path: str) -> object: ...
    def graphql(self, query: str, variables: dict) -> dict: ...


@dataclass(frozen=True)
class Event:
    key: str
    topic: str
    kind: str
    repo: str
    number: int
    title: str
    author: str
    body: str
    url: str
    association: str = "NONE"  # GitHub's author_association: OWNER, MEMBER, ...
    reply_to: str = ""  # discussions: the top-level comment a reply goes under

    @property
    def is_bot(self) -> bool:
        return self.author.endswith("[bot]")

    @property
    def thread_kind(self) -> str | None:
        """Where a reply to this goes: ``issue`` or ``discussion`` (``None``: PRs)."""

        return {
            "issue": "issue",
            "issue_comment": "issue",
            "discussion": "discussion",
            "discussion_comment": "discussion",
        }.get(self.kind)

    def for_model(self) -> dict[str, str]:
        return {
            "type": self.kind,
            "repository": self.repo,
            "title": self.title,
            "author": self.author,
            "body": self.body,
        }


@dataclass
class Poll:
    events: list[Event] = field(default_factory=list)
    cursors: dict[str, str] = field(default_factory=dict)


def iso(timestamp: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp))


def baseline(store: Store, cfg: Config, repo: str, now: float) -> str:
    name = f"baseline:{repo}"
    value = store.get_cursor(name)
    if value is None:
        value = iso(now - cfg.backfill_hours * 3600)
        store.set_cursor(name, value)
    return value


def _login(node: dict | None) -> str:
    return (node or {}).get("login") or "ghost"


def poll_repo(
    source: Source,
    store: Store,
    cfg: Config,
    repo: str,
    titles: dict[tuple[str, int], tuple[str, bool]],
    *,
    now: float | None = None,
) -> Poll:
    """New events in one repository, oldest first within each kind.

    ``titles`` maps (repo, number) to (title, is_pr); the caller keeps it between polls.
    """

    start = baseline(store, cfg, repo, time.time() if now is None else now)
    poll = Poll()

    # Issues and pull requests (the issues endpoint lists both).
    name = f"issues:{repo}"
    since = store.get_cursor(name) or start
    newest = since
    for item in source.get_list(
        f"/repos/{repo}/issues",
        state="all",
        since=since,
        sort="updated",
        direction="asc",
        per_page=100,
    ):
        is_pr = "pull_request" in item
        titles[(repo, item["number"])] = (item["title"], is_pr)
        newest = max(newest, item["updated_at"])
        if item["created_at"] < start:
            continue
        poll.events.append(
            Event(
                key=f"{repo}#{item['number']}",
                topic="reviews" if is_pr else "triage",
                kind="pr" if is_pr else "issue",
                repo=repo,
                number=item["number"],
                title=item["title"],
                author=_login(item.get("user")),
                body=item.get("body") or "",
                url=item["html_url"],
                association=item.get("author_association") or "NONE",
            )
        )
    if newest != since:
        poll.cursors[name] = newest

    # Conversation comments on issues and pull requests.
    name = f"comments:{repo}"
    since = store.get_cursor(name) or start
    newest = since
    for comment in source.get_list(
        f"/repos/{repo}/issues/comments",
        since=since,
        sort="updated",
        direction="asc",
        per_page=100,
    ):
        newest = max(newest, comment["updated_at"])
        if comment["created_at"] < start:
            continue
        number = int(comment["issue_url"].rsplit("/", 1)[1])
        title, is_pr = titles.get((repo, number), (f"#{number}", "/pull/" in comment["html_url"]))
        poll.events.append(
            Event(
                key=f"{repo}#comment-{comment['id']}",
                topic="reviews" if is_pr else "triage",
                kind="pr_comment" if is_pr else "issue_comment",
                repo=repo,
                number=number,
                title=title,
                author=_login(comment.get("user")),
                body=comment.get("body") or "",
                url=comment["html_url"],
                association=comment.get("author_association") or "NONE",
            )
        )
    if newest != since:
        poll.cursors[name] = newest

    # Discussions, their comments and replies (GraphQL only; no cursor needed).
    owner, repo_name = repo.split("/")
    data = source.graphql(DISCUSSIONS_QUERY, {"owner": owner, "name": repo_name})
    discussions = ((data.get("repository") or {}).get("discussions") or {}).get("nodes") or []
    found: list[tuple[str, Event]] = []
    for discussion in discussions:
        number, title = discussion["number"], discussion["title"]
        if discussion["createdAt"] >= start:
            found.append(
                (
                    discussion["createdAt"],
                    Event(
                        key=f"{repo}#discussion-{number}",
                        topic="replies",
                        kind="discussion",
                        repo=repo,
                        number=number,
                        title=title,
                        author=_login(discussion.get("author")),
                        body=discussion.get("body") or "",
                        url=discussion["url"],
                        association=discussion.get("authorAssociation") or "NONE",
                    ),
                )
            )
        for comment in (discussion.get("comments") or {}).get("nodes") or []:
            replies = (comment.get("replies") or {}).get("nodes") or []
            for node in (comment, *replies):
                if node["createdAt"] < start:
                    continue
                found.append(
                    (
                        node["createdAt"],
                        Event(
                            key=f"{repo}#discussion-comment-{node['id']}",
                            topic="replies",
                            kind="discussion_comment",
                            repo=repo,
                            number=number,
                            title=title,
                            author=_login(node.get("author")),
                            body=node.get("body") or "",
                            url=node["url"],
                            association=node.get("authorAssociation") or "NONE",
                            # A reply goes under the same top-level comment.
                            reply_to=comment["id"],
                        ),
                    )
                )
    poll.events.extend(event for _, event in sorted(found, key=lambda pair: pair[0]))

    poll.events = [e for e in poll.events if e.author.lower() not in cfg.ignore_authors]
    return poll
