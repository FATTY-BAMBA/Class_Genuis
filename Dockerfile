# syntax=docker/dockerfile:1

# ==================== BASE: CUDA 11.8 + cuDNN 8 on Ubuntu 20.04 ====================
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04 AS builder

# ==================== ENVIRONMENT VARIABLES ====================
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Taipei \
    CUDA_VISIBLE_DEVICES=0 \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

# ==================== SYSTEM DEPENDENCIES ====================
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip python3.10-distutils \
    build-essential cmake git curl ca-certificates \
    wget nodejs npm patchelf execstack \
    libcairo2-dev libjpeg-dev libgif-dev pkg-config \
    libopenblas-dev libssl-dev && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# ==================== PYTHON SETUP ====================
RUN python -m pip install --upgrade pip==24.0 setuptools wheel

# ==================== BUILD DEPENDENCIES ====================
WORKDIR /build

# Copy dependency lists
COPY requirements.txt constraints.txt /build/

# Remove conflicting packages from requirements
RUN grep -v "ctranslate2\|faster-whisper\|tokenizers\|transformers" requirements.txt > requirements_filtered.txt || cp requirements.txt requirements_filtered.txt

# Install core dependencies
RUN python -m pip install --no-cache-dir \
    packaging>=20.0 \
    Cython==3.0.10 \
    pybind11==2.12.0 \
    meson==1.2.3 \
    meson-python==0.15.0 \
    ninja==1.11.1

# Install PyTorch with CUDA 11.8 support
RUN python -m pip install \
    torch==2.2.2+cu118 \
    torchvision==0.17.2+cu118 \
    torchaudio==2.2.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Install other requirements
RUN python -m pip install --no-cache-dir -r requirements_filtered.txt

# Install Polygon3
RUN python -m pip install --no-cache-dir "Polygon3==3.0.9.1"

# Install compatible versions for whisper
RUN python -m pip install --no-cache-dir \
    faster-whisper==0.10.1 \
    ctranslate2==3.24.0 \
    transformers==4.36.2

# Install PaddlePaddle and PaddleOCR
RUN python -m pip install --no-cache-dir \
    paddlepaddle-gpu==2.6.1 \
    -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html && \
    python -m pip install --no-cache-dir paddleocr==2.6.1

# Install VisualDL
RUN python -m pip install --no-cache-dir visualdl==2.5.3

# ==================== RUNTIME STAGE ====================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04 AS final

# ==================== ENVIRONMENT VARIABLES ====================
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Taipei \
    HF_HOME=/workspace/models \
    WHISPER_CACHE=/workspace/models \
    CTRANSLATE2_CACHE=/workspace/models \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    GLOG_minloglevel=2 \
    GLOG_logtostderr=0 \
    FLAGS_fraction_of_gpu_memory_to_use=0.9 \
    CUDA_VISIBLE_DEVICES=0 \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/lib:${LD_LIBRARY_PATH}

# ==================== RUNTIME DEPENDENCIES ====================
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-distutils \
    ffmpeg redis-server redis-tools \
    libsndfile1 libgl1 libgomp1 libglib2.0-0 \
    libsm6 libxext6 libxrender1 libcairo2 \
    curl aria2 netcat-openbsd procps net-tools lsof \
    patchelf execstack && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# ==================== COPY PYTHON ENVIRONMENT ====================
COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=builder /usr/local/bin /usr/local/bin

# ==================== APPLICATION ====================
WORKDIR /app
COPY . .

# Fix numpy.int deprecation
RUN echo "import numpy as np; np.int = int if not hasattr(np, 'int') else np.int" > /usr/local/lib/python3.10/dist-packages/numpy_patch.py && \
    echo "try: import numpy_patch\nexcept: pass" >> /usr/local/lib/python3.10/dist-packages/sitecustomize.py

# ==================== CREATE NON-ROOT USER ====================
RUN useradd -ms /bin/bash appuser && \
    mkdir -p /app/uploads /app/segments /workspace/logs /workspace/models /workspace/uploads && \
    chown -R appuser:appuser /app /workspace && \
    chmod +x /app/start.sh

# ==================== SWITCH TO NON-ROOT ====================
USER appuser

EXPOSE 5000 8888

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:5000/healthz || exit 1

CMD ["./start.sh"]
