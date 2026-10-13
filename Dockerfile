FROM python:3.11-slim

# Required for the NVIDIA Container Toolkit to mount CUDA driver libs when
# the container is run with `--gpus`; torch's pip wheel already bundles the
# CUDA runtime, so no nvidia/cuda base image is needed.
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends micro

COPY requirements.txt .
COPY python_scripts/external_repositories/TRIDENT-main/ python_scripts/external_repositories/TRIDENT-main/
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y opencv-python \
    && pip install --no-cache-dir --force-reinstall --no-deps opencv-python-headless==4.13.0.92
ENTRYPOINT ["python", "python_scripts/external_repositories/TRIDENT-main/run_batch_of_slides.py", \
            "--task", "feat", "--patch_encoder", "uni_v2", \
            "--patch_encoder_ckpt_path", "/nfs-share/models/uni2h/pytorch_model.bin"]
CMD ["--wsi_dir", "/nfs-share/test_docker/wsis", "--job_dir", "/nfs-share/test_docker/trident", "--wsi_ext", \
     ".svs", "--mag", "20", "--patch_size", "224", "--overlap", "0", "--batch_size", "256", "--skip_errors"]
