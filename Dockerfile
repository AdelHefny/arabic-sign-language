# ============================================================
# Arabic Sign Language Translation System — Dockerfile
# ============================================================
# Multi-stage build:
#   stage 1 (builder) — install Python deps into a venv
#   stage 2 (runtime) — lean image with only what's needed
# ============================================================

# ── Stage 1: Builder ─────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for OpenCV / MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create isolated venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install only production requirements
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir \
        fastapi \
        uvicorn[standard] \
        websockets \
        python-multipart \
        httpx \
        onnxruntime \
        numpy \
        pyarabic \
        python-dotenv \
        pyyaml

# ── Stage 2: Runtime ─────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime system libs
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy project source
COPY backend/ ./backend/
COPY models/   ./models/
COPY labels.json .

# Non-root user for security
RUN useradd -m -u 1001 arsl
USER arsl

# Expose API port
EXPOSE 8000

# Health-check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

# Launch
CMD ["uvicorn", "backend.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
