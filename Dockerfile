# Runtime image for the ingest API and the device fleet. The training extra
# (PyTorch) is left out on purpose: devices and the API only need onnxruntime,
# which is the point of the compression work.
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

FROM python:3.13-slim-bookworm AS runtime
RUN groupadd --system pulse && useradd --system --gid pulse --create-home pulse
WORKDIR /app
COPY --from=builder --chown=pulse:pulse /app/.venv /app/.venv
RUN mkdir -p /app/data /app/artifacts && chown -R pulse:pulse /app
ENV PATH="/app/.venv/bin:$PATH" PULSE_DB_PATH=/app/data/pulse.db PULSE_REGISTRY_DIR=/app/data/registry
USER pulse
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --retries=3 CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"]
CMD ["pulse", "serve", "--host", "0.0.0.0", "--port", "8000"]
