# =============================================================================
# DESPIECE-BOT - Dockerfile Simple (Solo AI Stack)
# =============================================================================

FROM python:3.11-slim

# Metadata
LABEL maintainer="coagente.com"
LABEL description="Despiece-Bot: Simple AI Stack with Google GenAI + CrewAI + DSPy"
LABEL version="1.0"

# Set working directory
WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY .env* ./

# Create data directory
RUN mkdir -p /app/data && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uvicorn", "src.despiece_bot.main:app", "--host", "0.0.0.0", "--port", "8000"]
