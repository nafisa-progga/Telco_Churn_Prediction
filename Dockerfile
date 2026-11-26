# Lightweight Python runtime for serving the FastAPI churn model
FROM python:3.12-slim

WORKDIR /app

# Avoid writing .pyc files and buffer stdout/stderr for container logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install Python dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy source code, API app, and trained model artifacts
COPY src/ ./src/
COPY app/ ./app/
COPY artifacts/ ./artifacts/

EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]