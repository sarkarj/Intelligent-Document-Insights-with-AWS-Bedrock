# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false \
    STREAMLIT_GLOBAL_DEVELOPMENT_MODE=false

# Set work directory
WORKDIR /app

# Install system dependencies in a single layer with cleanup
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        curl \
        tesseract-ocr \
        tesseract-ocr-eng && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

# Create a non-root user with proper home directory
RUN groupadd -r appuser && \
    useradd -r -g appuser -d /home/appuser -m appuser && \
    mkdir -p /home/appuser/.streamlit && \
    chown -R appuser:appuser /home/appuser

# Copy application code
COPY --chown=appuser:appuser . .

# Create logs directory and set permissions
RUN mkdir -p /app/logs && \
    chown -R appuser:appuser /app

# Create Streamlit config to avoid permission issues
RUN echo '[server]\nheadless = true\nenableCORS = false\nenableXsrfProtection = false\n[global]\ndevelopmentMode = false\n[browser]\ngatherUsageStats = false' > /home/appuser/.streamlit/config.toml && \
    chown appuser:appuser /home/appuser/.streamlit/config.toml

# Switch to non-root user
USER appuser

# Set HOME environment variable
ENV HOME=/home/appuser

# Expose ports for Streamlit apps
EXPOSE 8502 8503 8504

# Default command - run user_app
CMD ["streamlit", "run", "user_app/app.py", "--server.port=8502", "--server.address=0.0.0.0"]