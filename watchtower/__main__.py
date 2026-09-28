"""Entry point: ``python -m watchtower watcher|gateway|health <name> <max-age-seconds>|history ...``."""

from __future__ import annotations

import logging
import os
import re
import sys
import time

from . import config


def health(name: str, max_age: float) -> int:
    """Exit 0 if ``name`` wrote a heartbeat within ``max_age`` seconds (Docker healthcheck)."""

    from .store import Store

    store = Store(config.DATA_DIR / "watchtower.db")
    beat = store.heartbeats().get(name)
    return 0 if beat and time.time() - beat[0] < max_age else 1


_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def history(argv: list[str]) -> int:
    """Check the history from a shell. Its text is strangers', so control characters
    (terminal escape sequences) are removed before printing."""

    from .history import History

    db = History(config.DATA_DIR / "history.db")
    match argv:
        case ["search", repo, *words] if words:
            lines = [
                f"#{h.number} [{h.kind}, {h.state}] {h.title}"
                for h in db.search(repo, " ".join(words))
            ]
        case ["show", repo, number] if number.isdigit():
            thread = db.thread(repo, int(number))
            if thread is None:
                lines = ["not found"]
            else:
                lines = [
                    f"#{thread['number']} [{thread['kind']}, {thread['state']}] {thread['title']}",
                    f"{thread['author']} ({thread['association']}): {thread['body']}",
                    *(
                        f"\n{c['author']} ({c['association']}): {c['body']}"
                        for c in thread["comments"]
                    ),
                ]
        case [repo]:
            lines = [str(db.counts(repo))]
        case _:
            print(
                "python -m watchtower history <repo> | search <repo> <words> | show <repo> <number>",
                file=sys.stderr,
            )
            return 2
    print("\n".join(_CONTROL.sub("", line) for line in lines))
    return 0


def main(argv: list[str]) -> int:
    logging.basicConfig(
        level=os.environ.get("WATCHTOWER_LOG", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    match argv:
        case ["watcher"]:
            from . import watcher

            watcher.run(config.load())
        case ["gateway"]:
            from . import gateway

            gateway.run(config.load())
        case ["health", name, max_age]:
            return health(name, float(max_age))
        case ["history", *rest]:
            return history(rest)
        case _:
            print(__doc__, file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
