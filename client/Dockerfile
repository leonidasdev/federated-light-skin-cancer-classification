# Dockerfile for the federated client
# Build with the repository root as build context:
# docker build -f Dockerfile.client -t myacr.azurecr.io/fed-skin-client:latest .

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
       bash \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy repository source
COPY . /app

# Add entrypoint
COPY entrypoint_client.sh /usr/local/bin/entrypoint_client.sh
RUN chmod +x /usr/local/bin/entrypoint_client.sh

# Default port placeholder (client typically initiates connections)
EXPOSE 0

# Entrypoint constructs args from environment variables and runs the client
ENTRYPOINT ["/usr/local/bin/entrypoint_client.sh"]
