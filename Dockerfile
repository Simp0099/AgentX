# Serves the Gradio demo. CPU-only: inference on a single image needs no GPU,
# and CPU images are a fraction of the size of the CUDA ones.
FROM python:3.11-slim

# libgl/libglib are needed by Pillow/matplotlib image backends on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements alone first so the pip layer is cached across code changes.
COPY requirements.txt .
# Slow mirrors stall reads longer than pip's 15 s default read timeout, which
# kills the ~700 MB TensorFlow wheel download on flaky networks.
ENV PIP_DEFAULT_TIMEOUT=180
RUN pip install --no-cache-dir -r requirements.txt

# agent/ holds the verdict logic and schemas the app imports at startup.
COPY src/ ./src/
COPY agent/ ./agent/
COPY app/ ./app/
COPY configs/ ./configs/
# Checkpoint is optional: the app boots into ELA-preview mode without it,
# because the .gitkeep ensures the directory exists even without a committed checkpoint.
COPY checkpoints/ ./checkpoints/

ENV PORT=7860 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2
EXPOSE 7860

# Runs as an unprivileged user: this listens on a network port, so a container
# escape should not land on root.
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

CMD ["python", "app/app.py"]
