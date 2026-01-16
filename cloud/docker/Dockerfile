# Dockerfile for Landuse Global Pipeline
# Multi-stage build for optimized image size

# Base stage with common dependencies
FROM python:3.10-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    libproj-dev \
    libgeos-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set GDAL environment variables
ENV GDAL_CONFIG=/usr/bin/gdal-config
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.10-slim as runtime

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    libproj-dev \
    libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

ENV GDAL_CONFIG=/usr/bin/gdal-config

WORKDIR /app

# Copy Python packages from base stage
COPY --from=base /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=base /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY pipelines/ ./pipelines/
COPY configs/ ./configs/

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs /app/output

# Set Python path
ENV PYTHONPATH=/app/src:${PYTHONPATH}

# Default command (can be overridden)
ENTRYPOINT ["python"]
CMD ["pipelines/global/stage0_ingest.py", "--config", "configs/cloud.yaml"]

# Labels
LABEL maintainer="Pengyu Zhong"
LABEL version="1.0.0"
LABEL description="Global Pipeline for PV Deployment Suitability Analysis"
