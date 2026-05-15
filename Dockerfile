# Dockerfile for feature extraction container
# Build with: docker build -t prognosis-feature-extractor .
# Run with: docker run --rm -v /path/to/repo:/app -v /path/to/WSI:/app/WSI -v /path/to/simclr:/app/simclr prognosis-feature-extractor --dataset prognosis --backbone UNI2-h

FROM registry.ihelse.net/docker.io/library/python:3.11-slim@sha256:6d85378d88a19cd4d76079817532d62232be95757cb45945a99fec8e8084b9c2

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies required for OpenSlide, image handling, and PyTorch packages.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libgl1 \
        libopenjp2-7 \
        libopenslide0 \
        openslide-tools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first to leverage Docker cache.
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install -r /app/requirements.txt

# Copy repository sources.
COPY . /app

ENTRYPOINT ["python", "python_scripts/prepare_for_MIL/compute_feats.py"]
CMD ["--help"]
