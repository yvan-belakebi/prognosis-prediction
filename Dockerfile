FROM python:3.11-slim

# Required for the NVIDIA Container Toolkit to mount CUDA driver libs when
# the container is run with `--gpus`; torch's pip wheel already bundles the
# CUDA runtime, so no nvidia/cuda base image is needed.
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends micro

COPY requirements.txt .
COPY python_scripts/external_repositories/ python_scripts/external_repositories/
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y opencv-python \
    && pip install --no-cache-dir --force-reinstall --no-deps opencv-python-headless==4.13.0.92