# Reproducible environment for the multiverse-consensus pipeline.
#   docker build -t metabolomic-pipeline .
#   docker run --rm metabolomic-pipeline --help
#   docker run --rm -v "$PWD/data:/workspace/data" metabolomic-pipeline multiverse run
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.8.4 /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_LINK_MODE=copy PATH="/opt/venv/bin:$PATH"

# g++: line-profiler (via bioservices) has no wheel on some platforms.
# libexpat1: needed by kaleido (static PNG export of report figures).
RUN apt-get update && apt-get install -y --no-install-recommends g++ libexpat1 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Dependencies first (cached layer), resolved exactly from uv.lock.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --all-extras --no-install-project

COPY src/ src/
COPY conf/ conf/
COPY scripts/ scripts/
COPY tests/ tests/
COPY dvc.yaml params.yaml ./
RUN uv sync --frozen --all-extras

ENTRYPOINT ["biodisc"]
CMD ["--help"]
