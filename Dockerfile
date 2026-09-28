# One image, two services (watcher, gateway) -- they differ only in command and secrets.
FROM python:3.14-slim

RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin watchtower

WORKDIR /app
COPY watchtower/ /app/watchtower/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER 10001:10001
ENTRYPOINT ["python", "-m", "watchtower"]
