"""Entry point: ``python -m watchtower watcher|gateway|worker|poster|web|runner|toolchain``, or
``health <name> <max-age-seconds>``, or ``history ...``, or ``eval ...``, or
``review <owner/name> <number>``, or ``selftest``."""

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


def selftest() -> int:
    """Prove the sandbox on this box: run in the watcher container, it puts a job of
    harmless probes where the runner takes it and prints the runner's result. Every
    probe should pass except the failing one, which must be reported as failed."""

    import tempfile
    from pathlib import Path

    from . import sandbox
    from .watcher import SANDBOX_DIR

    if sandbox.busy(SANDBOX_DIR):
        print("a job is in the sandbox already; try again in a few minutes", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory() as tree:
        (Path(tree) / "README.md").write_text("selftest")
        sandbox.write_job(SANDBOX_DIR, sandbox.selftest_job(), Path(tree))
    print("job written; waiting for the runner (up to two minutes)...")
    result = None
    for _ in range(60):
        result = sandbox.read_result(SANDBOX_DIR, "selftest")
        if result is not None:
            break
        time.sleep(2)
    sandbox.remove_job(SANDBOX_DIR, "selftest")
    if result is None:
        print("no result: is the runner service up? (docker logs watchtower-runner-1)")
        return 1
    for s in result.steps:
        print(f"{s.status:8} {s.name}" + (f"  -> {s.output.strip()[:100]}" if s.output else ""))
    expected = {s.name: "failed" if "failing" in s.name else "passed" for s in result.steps}
    bad = [s.name for s in result.steps if s.status != expected[s.name]]
    print("ISOLATED" if not bad else f"NOT AS EXPECTED: {', '.join(bad)}")
    return 1 if bad else 0


def review(argv: list[str]) -> int:
    """Queue a review of an existing PR (run in the watcher container): the watcher
    fetches it, runs its checks if they are on, the worker reviews it, Telegram offers it."""

    from . import watcher
    from .github import GitHub
    from .store import Store

    if len(argv) != 2 or argv[0].count("/") != 1 or not argv[1].isdigit():
        print("python -m watchtower review <owner/name> <pull request number>", file=sys.stderr)
        return 2
    cfg = config.load()
    if argv[0] not in cfg.repos or not cfg.reviews:
        print(
            "that repo isn't watched, or llm.review_prs / the agent model is off", file=sys.stderr
        )
        return 2
    store = Store(config.DATA_DIR / "watchtower.db")
    print(
        watcher.queue_review(
            GitHub(config.read_secret("github_read")), store, argv[0], int(argv[1])
        )
    )
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
        case ["worker"]:
            from . import worker

            worker.run(config.load())
        case ["web"]:
            from . import web

            web.run(config.load())
        case ["poster"]:
            from . import poster

            poster.run(config.load())
        case ["runner"]:  # no config, no secrets: it runs a stranger's code
            from . import runner

            runner.run()  # one job, then exit: compose starts a fresh container
        case ["toolchain"]:  # a network, and nothing else: it installs the default branch's deps
            from . import toolchain

            toolchain.run()
        case ["review", *rest]:
            return review(rest)
        case ["selftest"]:
            return selftest()
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
