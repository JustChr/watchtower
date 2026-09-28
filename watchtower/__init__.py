"""Watchtower: watch GitHub repositories, summarise new activity locally, report to Telegram.

Two processes share one SQLite database in ``/data``:

- ``watcher`` polls GitHub (read-only token), asks a local Ollama model for a
  one-line summary, and queues a message in the outbox.
- ``gateway`` owns the Telegram bot: it delivers the outbox and answers
  commands, but only from the one configured user in the one configured group.

Neither process can write to GitHub. Standard library only, on purpose.
"""
