FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends micro

COPY requirements.txt .
COPY python_scripts/external_repositories/ python_scripts/external_repositories/
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y opencv-python \
    && pip install --no-cache-dir --force-reinstall --no-deps opencv-python-headless==4.13.0.92