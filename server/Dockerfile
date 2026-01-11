# Dockerfile for the federated server
# Build with the repository root as build context:
# docker build -f Dockerfile.server -t myacr.azurecr.io/fed-skin-server:latest .

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies required by some Python packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       git \
       ffmpeg \
       libsm6 \
       libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy repository source
COPY . /app

# Expose default server port (matches FEDERATED_CONFIG server_address port)
EXPOSE 8080

# Default command: start the federated server
CMD ["python", "main_server.py"]
