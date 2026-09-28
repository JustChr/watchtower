# One image, four services (watcher, gateway, drafter, poster) -- they differ only in
# command, secrets and networks.
FROM python:3.14-slim

RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin watchtower

# The one dependency: RS256 for the GitHub App's JWT (poster only).
RUN pip install --no-cache-dir "cryptography==50.0.1"

WORKDIR /app
COPY watchtower/ /app/watchtower/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER 10001:10001
ENTRYPOINT ["python", "-m", "watchtower"]
