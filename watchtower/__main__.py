"""Entry point: ``python -m watchtower watcher|gateway|worker|poster``, or
``health <name> <max-age-seconds>``, or ``history ...``, or ``eval ...``."""

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


EVAL_USAGE = "python -m watchtower eval <owner/name> [--limit N] [number[@comment] ...]"
_CASE = re.compile(r"(\d+)(?:@(\d+))?")


def evaluate(argv: list[str]) -> int:
    """Replay closed issues (``evaluate``): run in the watcher container, it fetches
    what the cases need and queues the replay for the worker (the model).
    ``160@5`` replays #160 cut at its 5th comment."""

    from . import evaluate as replay
    from .github import GitHub
    from .history import History
    from .store import Store

    limit = None
    if "--limit" in argv:
        at = argv.index("--limit")
        if at + 1 >= len(argv) or not argv[at + 1].isdigit():
            print(EVAL_USAGE, file=sys.stderr)
            return 2
        limit = int(argv[at + 1])
        argv = argv[:at] + argv[at + 2 :]
    if not argv or argv[0].count("/") != 1 or not all(_CASE.fullmatch(a) for a in argv[1:]):
        print(EVAL_USAGE, file=sys.stderr)
        return 2
    cfg = config.load()
    if not cfg.agent_model:
        print("llm.agent_model is not set", file=sys.stderr)
        return 2
    repo = argv[0]
    stamp = time.strftime("%Y%m%d-%H%M", time.gmtime())
    out = config.DATA_DIR / "eval" / f"{repo.split('/')[1]}-{stamp}.md"
    numbers = [_case(a) for a in argv[1:]]
    history = History(config.DATA_DIR / "history.db")
    selected = replay.cases(history, repo, numbers)[:limit]
    print(f"{len(selected)} issue(s): fetching their files and code")
    replay.prepare(
        history,
        repo,
        selected,
        config.DATA_DIR / "repos",
        config.DATA_DIR / "attachments",
        GitHub(config.read_secret("github_read")),
    )
    payload = {"repo": repo, "numbers": numbers, "limit": limit, "out": str(out)}
    Store(config.DATA_DIR / "watchtower.db").add_job("eval", f"eval:{repo}:{stamp}", payload)
    print(f"Queued for the worker; Telegram says when it's done. Report: {out}")
    return 0


def _case(arg: str) -> int | tuple[int, int]:
    number, at = _CASE.fullmatch(arg).groups()
    return (int(number), int(at)) if at else int(number)


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
        case ["worker"]:
            from . import worker

            worker.run(config.load())
        case ["poster"]:
            from . import poster

            poster.run(config.load())
        case ["health", name, max_age]:
            return health(name, float(max_age))
        case ["history", *rest]:
            return history(rest)
        case ["eval", *rest]:
            return evaluate(rest)
        case _:
            print(__doc__, file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
