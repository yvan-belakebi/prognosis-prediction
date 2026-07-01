FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
COPY python_scripts/external_repositories/ python_scripts/external_repositories/
RUN pip install --no-cache-dir -r requirements.txt