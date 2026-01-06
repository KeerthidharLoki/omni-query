# Multi-stage build — slim Python runtime for the FastAPI backend
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.11-slim AS runtime
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY src/ ./src/
COPY run_api.py .
ENV PATH=/root/.local/bin:$PATH
EXPOSE 9100
CMD ["python", "run_api.py"]
