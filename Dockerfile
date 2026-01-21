# ============================================================================
# LYRA CLEAN - PRODUCTION DOCKERFILE
# ============================================================================
# Multi-stage build for minimal image size
# Final image: ~200MB (Python slim + dependencies)
# ============================================================================

# ============================================================================
# STAGE 1: Builder (install dependencies)
# ============================================================================
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ============================================================================
# STAGE 2: Runtime (minimal image)
# ============================================================================
FROM python:3.11-slim

LABEL maintainer="lyra-clean"
LABEL description="Lyra Clean API - Physics-driven semantic LLM system"

# Create non-root user for security
RUN useradd -m -u 1000 lyra && \
    mkdir -p /app/data /app/logs && \
    chown -R lyra:lyra /app

WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /home/lyra/.local

# Copy application code
COPY --chown=lyra:lyra . .

# Set Python path
ENV PATH=/home/lyra/.local/bin:$PATH
ENV PYTHONPATH=/app:$PYTHONPATH

# Switch to non-root user
USER lyra

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health')" || exit 1

# Run application
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
