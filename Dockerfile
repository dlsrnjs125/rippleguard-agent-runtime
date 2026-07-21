FROM python:3.12.8-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_ENV=container \
    CONTRACTS_ROOT=/app/contracts \
    MODEL_MANIFEST_PATH=/app/artifacts/manifests/phase2-loan-xgboost.v1.0.0.json \
    MODEL_ARTIFACT_ROOT=/app/artifacts/models \
    HOST=0.0.0.0 \
    PORT=8080

WORKDIR /app

RUN useradd --create-home --uid 10001 appuser

COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
COPY artifacts ./artifacts

RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir .

USER appuser

EXPOSE 8080
CMD ["python", "-m", "uvicorn", "rippleguard_agent_runtime.app.api:app", "--host", "0.0.0.0", "--port", "8080"]
