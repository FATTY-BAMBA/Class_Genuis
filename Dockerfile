# syntax=docker/dockerfile:1
# ---- Stage 1: The Builder ----
FROM python:3.10-slim AS builder

# Set ARGs and ENV vars needed for the build
ARG PADDLE_VERSION_GPU=2.6.1
ARG PADDLE_VERSION_CPU=2.6.1
ARG PYCAIRO_VERSION=1.26.1

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 PIP_DEFAULT_TIMEOUT=600

# Install system dependencies, including a modern Node.js for visualdl build
RUN apt-get update && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends \
    nodejs build-essential cmake git wget curl \
    libcairo2-dev libjpeg-dev libgif-dev pkg-config \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip in the global environment of this stable image
RUN python -m pip install --upgrade pip setuptools wheel

# Copy requirements early to leverage Docker cache
COPY requirements.txt /tmp/requirements.txt
COPY constraints.txt /tmp/constraints.txt
ENV PIP_CONSTRAINT=/tmp/constraints.txt

# Install core dependencies first (these are stable and less likely to conflict)
RUN python -m pip install \
    "packaging>=20.0" \
    "Cython==3.0.10" \
    "pybind11==2.12.0" \
    "meson==1.2.3" \
    "meson-python==0.15.0" \
    "ninja==1.11.1"

# Install main application requirements (this will handle all version resolution)
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt

# Install visualdl directly from PyPI
RUN python -m pip install "visualdl==2.5.3"

# Install paddlepaddle for the specified variant (CPU/GPU)
ARG BUILD_VARIANT=gpu
RUN if [ "${BUILD_VARIANT}" = "gpu" ]; then \
      python -m pip install --prefer-binary -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html "paddlepaddle-gpu==${PADDLE_VERSION_GPU}"; \
    else \
      python -m pip install --prefer-binary -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html "paddlepaddle==${PADDLE_VERSION_CPU}"; \
    fi
RUN python -m pip install "paddleocr==2.6.1"

# ---- Stage 2: The Final Image ----
FROM python:3.10-slim AS final

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    TZ=Asia/Taipei \
    HF_HOME=/workspace/models \
    WHISPER_CACHE=/workspace/models \
    CTRANSLATE2_CACHE=/workspace/models \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1

WORKDIR /app

# --- SURGICALLY ADD NVIDIA CUDA RUNTIME LIBRARIES ---
ARG BUILD_VARIANT=gpu
RUN if [ "${BUILD_VARIANT}" = "gpu" ]; then \
    apt-get update && apt-get install -y --no-install-recommends gnupg curl ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub | gpg --dearmor -o /usr/share/keyrings/nvidia-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/nvidia-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" > /etc/apt/sources.list.d/nvidia-cuda.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    cuda-cudart-11-8 \
    libcudnn8=8.9.7.29-1+cuda11.8 \
    && rm -rf /var/lib/apt/lists/*; \
    fi

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg redis-tools libsndfile1 libgl1 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire Python environment from the builder stage
COPY --from=builder /usr/local/ /usr/local/

# Copy application code only (no need to install anything)
COPY . .

RUN chmod +x /app/start.sh
RUN printf '#!/bin/bash\nexec uvicorn chapter_llama.main:app --host 0.0.0.0 --port 8000 --log-level warning\n' > /app/start_chapter_llama.sh && chmod +x /app/start_chapter_llama.sh
RUN printf '#!/bin/bash\nset -e\n./start_chapter_llama.sh &\nsleep 5\ncurl -sf http://localhost:8000/health || exit 1\nexec ./start.sh\n' > /app/start_supervisor.sh && chmod +x /app/start_supervisor.sh

RUN useradd -ms /bin/bash appuser && \
    mkdir -p /app/uploads /app/segments /workspace && \
    chown -R appuser:appuser /app /workspace
USER appuser

EXPOSE 5000 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD curl -sf http://localhost:8000/health && curl -sf http://localhost:5000/healthz || exit 1

CMD ["./start_supervisor.sh"]