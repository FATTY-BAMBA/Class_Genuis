# syntax=docker/dockerfile:1
# ---- Stage 1: The Builder ----
# Start from a guaranteed-stable official Python image
FROM python:3.10-slim AS builder

# Set ARGs and ENV vars needed for the build
ARG PADDLE_VERSION_GPU=2.6.1
ARG PADDLE_VERSION_CPU=2.6.1
ARG PYCAIRO_VERSION=1.26.1

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 PIP_DEFAULT_TIMEOUT=600

# Install system dependencies, including Node.js for visualdl build
RUN apt-get update && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends \
    nodejs build-essential cmake git wget curl \
    libcairo2-dev libjpeg-dev libgif-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

# ### THIS IS THE FIX ###
# Upgrade pip to the latest version BEFORE installing other packages
RUN python -m pip install --upgrade pip setuptools wheel

# Install all Python dependencies
RUN python -m pip install \
    "numpy==1.26.4" \
    "packaging==23.2" \
    "Cython==3.0.10" \
    "pybind11==2.12.0" \
    "meson==1.2.3" \
    "meson-python==0.15.0" \
    "ninja==1.11.1" \
    "onnxruntime==1.15.1" \
    "opencv-python-headless==4.7.0.72" \
    "scikit-learn==1.0.2" \
    "scikit-image==0.21.0" \
    "PyMuPDF==1.22.5" \
    "python-docx==1.2.0" \
    "fonttools==4.51.0" \
    "pyclipper==1.3.0.post6" \
    "attrdict==2.0.1" \
    "beautifulsoup4==4.13.4" \
    "fire==0.7.1" \
    "lmdb==1.7.3" \
    "openpyxl==3.1.5" \
    "premailer==3.10.0" \
    "rapidfuzz==3.13.0" \
    "imgaug==0.4.0" \
    "pdf2docx==0.5.8" \
    "lanms-neo==1.0.2" \
    "Polygon3==3.0.9.1" \
    "pycairo==${PYCAIRO_VERSION}"

# Manual install for visualdl from a known-good branch
RUN git clone --depth 1 --branch release/2.5 https://github.com/PaddlePaddle/VisualDL.git /tmp/visualdl_src && \
    python -m pip install /tmp/visualdl_src && \
    rm -rf /tmp/visualdl_src

# Install paddlepaddle for the specified variant (CPU/GPU)
ARG BUILD_VARIANT=gpu
RUN if [ "${BUILD_VARIANT}" = "gpu" ]; then \
      python -m pip install --prefer-binary -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html "paddlepaddle-gpu==${PADDLE_VERSION_GPU}"; \
    else \
      python -m pip install --prefer-binary -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html "paddlepaddle==${PADDLE_VERSION_CPU}"; \
    fi
RUN python -m pip install "paddleocr==2.6.1"

# ---- Stage 2: The Final Image ----
# Start from the same stable Python base
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

# Install only the necessary runtime shared libraries from the OS
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg redis-tools libsndfile1 libgl1 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire Python environment from the builder stage
COPY --from=builder /usr/local/ /usr/local/

# Copy application-specific dependencies and code
COPY requirements.txt .
COPY constraints.txt /tmp/constraints.txt
ENV PIP_CONSTRAINT=/tmp/constraints.txt
RUN python -m pip install --no-cache-dir -r requirements.txt
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