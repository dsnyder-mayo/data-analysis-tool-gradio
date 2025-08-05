# Simple, reliable Dockerfile for Cloud Run
FROM python:3.12-slim AS build

# Update package list and install essential build tools only
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

FROM ubuntu:latest

RUN apt-get update -qq \
    && apt-get upgrade -y \
    && apt-get install -y \
        ca-certificates \
        libsqlite3-0 \
        libpango-1.0-0 \
        libharfbuzz0b \
        libpangoft2-1.0-0 \
        libharfbuzz-subset0 \
        libffi-dev libjpeg-dev \
        libopenjp2-7-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /usr/local/bin/python3.12 /usr/bin/python3.12
COPY --from=build /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=build /usr/local/lib/libpython3.12.* /usr/local/lib/
COPY --from=build /usr/local/lib/python3.12 /usr/local/lib/
COPY --from=build /usr/local/bin /usr/local/bin

RUN \
  ln -s /usr/bin/python3.12 /usr/bin/python \
  && ln -s /usr/bin/python3.12 /usr/bin/python3 \
  && ldconfig

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Create a non-root user for security and set up proper directories
RUN groupadd -r appuser && useradd -r -g appuser appuser && \
    mkdir -p /home/appuser/.config/matplotlib && \
    mkdir -p /home/appuser/.dspy_cache && \
    mkdir -p /tmp/matplotlib && \
    chown -R appuser:appuser /app /home/appuser /tmp/matplotlib
USER appuser

# Expose the port that Gradio runs on
EXPOSE 5000

# Set environment variables for Cloud Run and fix cache issues
ENV PORT=5000
ENV PYTHONUNBUFFERED=1
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV DSPY_CACHEDIR=/home/appuser/.dspy_cache
ENV HOME=/home/appuser

# Command to run the application
CMD ["python", "app.py"]
