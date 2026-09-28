"""Entry point: ``python -m watchtower watcher|gateway|health <name> <max-age-seconds>``."""

from __future__ import annotations

import logging
import os
import sys
import time

from . import config


def health(name: str, max_age: float) -> int:
    """Exit 0 if ``name`` wrote a heartbeat within ``max_age`` seconds (Docker healthcheck)."""

    from .store import Store

    store = Store(config.DATA_DIR / "watchtower.db")
    beat = store.heartbeats().get(name)
    return 0 if beat and time.time() - beat[0] < max_age else 1


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
        case _:
            print(__doc__, file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
