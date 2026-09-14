# CPU image: the plate models run on the CPU anyway, and no CUDA driver is required
FROM python:3.13-slim

# the CPU build of torch is a fraction of the size of the default one, which ships CUDA
ENV PIP_INDEX_URL=https://download.pytorch.org/whl/cpu \
    PIP_EXTRA_INDEX_URL=https://pypi.org/simple \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# onnxruntime links against the system OpenMP runtime
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
# both opencv packages provide cv2, and ultralytics pulls in the windowed one, which needs libGL;
# removing it deletes the shared cv2 files, so the headless build is reinstalled afterwards
RUN sed '/^opencv-python==/d' requirements.txt > requirements-image.txt \
 && pip install -r requirements-image.txt opencv-python-headless==5.0.0.93 \
 && pip uninstall -y opencv-python \
 && pip install --force-reinstall --no-deps opencv-python-headless==5.0.0.93

COPY vehicle_tracker ./vehicle_tracker
COPY scripts ./scripts

# the models are baked in so the container also starts on a device that is already offline
RUN python scripts/fetch_assets.py models

# python as pid 1 ignores SIGTERM, so docker stop would kill it before the video file is closed
STOPSIGNAL SIGINT

# the image has no display, so runs need --no-display:
#   docker run --rm -v "$PWD/data:/data" vehicle-tracker \
#       --source /data/clip.mp4 --no-display --save-video /data/annotated.mp4
ENTRYPOINT ["python", "-m", "vehicle_tracker"]
CMD ["--help"]
