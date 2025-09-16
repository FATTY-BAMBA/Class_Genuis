# syntax=docker/dockerfile:1

# ------------------------------------------------------------------
# Stage 1 – Builder: compile everything once, keep the final image slim
# ------------------------------------------------------------------
FROM python:3.10-slim AS builder

# ---- Build-time arguments ----------------------------------------------------
ARG PADDLE_VERSION_GPU=2.6.1
ARG PADDLE_VERSION_CPU=2.6.1
ARG PYCAIRO_VERSION=1.26.1
ARG BUILD_VARIANT=gpu

# ---- Build-time environment --------------------------------------------------
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=1200 \
    PIP_CONSTRAINT=/tmp/constraints.txt

# ---- System dependencies (Node needed by VisualDL) ---------------------------
RUN apt-get update && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends \
        nodejs build-essential cmake git wget curl \
        libcairo2-dev libjpeg-dev libgif-dev pkg-config python3-dev \
        libopenblas-dev libssl-dev && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/*

# ---- Copy dependency lists early for maximum cache hit -----------------------
COPY requirements.txt constraints.txt /tmp/
ENV PIP_CONSTRAINT=/tmp/constraints.txt

# ---- Python packaging tooling (pinned for reproducibility) -----------------
RUN python -m pip install --upgrade pip==24.0 setuptools wheel

# ---- Core build helpers ------------------------------------------------------
RUN python -m pip install \
        packaging>=20.0 \
        Cython==3.0.10 \
        pybind11==2.12.0 \
        meson==1.2.3 \
        meson-python==0.15.0 \
        ninja==1.11.1

# ---- Two-pass pip install with BuildKit cache mount -------------------------
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-deps --no-cache-dir -r /tmp/requirements.txt && \
    python -m pip install --no-cache-dir -r /tmp/requirements.txt

# ---- Install Polygon3, after build-essential is available ----
RUN pip install --no-cache-dir "Polygon3==3.0.9.1"

# ---- Fix ctranslate2 executable stack issue (CORRECTED) ----
# Install execstack tool temporarily for fixing the library
RUN apt-get update && \
    apt-get install -y --no-install-recommends execstack && \
    # Clear executable stack flag on ALL ctranslate2 libraries
    find /usr/local/lib/python*/site-packages -name "*.so*" -path "*/ctranslate2*" -exec execstack -c {} \; 2>/dev/null || true && \
    # Also fix any other potentially problematic libraries
    find /usr/local/lib -name "libctranslate2*.so*" -exec execstack -c {} \; 2>/dev/null || true && \
    # Clean up execstack as we don't need it in runtime
    apt-get purge -y execstack && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# ---- VisualDL (not in requirements.txt) -------------------------------------
RUN python -m pip install --no-cache-dir visualdl==2.5.3

# ---- PaddlePaddle GPU or CPU -------------------------------------------------
RUN if [ "$BUILD_VARIANT" = "gpu" ]; then \
        python -m pip install --no-cache-dir -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html \
            paddlepaddle-gpu==${PADDLE_VERSION_GPU}; \
    else \
        python -m pip install --no-cache-dir -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html \
            paddlepaddle==${PADDLE_VERSION_CPU}; \
    fi

# ---- PaddleOCR --------------------------------------------------------------
RUN python -m pip install --no-cache-dir paddleocr==2.6.1

# ---- Final clean-up in builder ----------------------------------------------
RUN find /usr/local -type f -name "*.pyc" -delete && \
    find /usr/local -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    rm -rf /tmp/* /root/.cache /var/cache/apt/*

# ------------------------------------------------------------------
# Stage 2 – Runtime: minimal footprint, only runtime libs
# ------------------------------------------------------------------
FROM python:3.10-slim AS final

ARG BUILD_VARIANT=gpu

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Taipei \
    HF_HOME=/workspace/models \
    WHISPER_CACHE=/workspace/models \
    CTRANSLATE2_CACHE=/workspace/models \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    RUNPOD_DEBUG_ENABLED=${RUNPOD_DEBUG_ENABLED:-false} \
    GLOG_minloglevel=2 \
    GLOG_logtostderr=0 \
    FLAGS_fraction_of_gpu_memory_to_use=0.9 \
    LD_PRELOAD="" \
    EXECSTACK_DISABLE=1

WORKDIR /app

# ---- CUDA 11.8 runtime (GPU variant only) -----------------------------------
RUN if [ "$BUILD_VARIANT" = "gpu" ]; then \
        apt-get update && \
        apt-get install -y --no-install-recommends gnupg curl ca-certificates && \
        curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub \
            | gpg --dearmor -o /usr/share/keyrings/nvidia-archive-keyring.gpg && \
        echo "deb [signed-by=/usr/share/keyrings/nvidia-archive-keyring.gpg] \
            https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" \
            > /etc/apt/sources.list.d/nvidia-cuda.list && \
        apt-get update && \
        apt-get install -y --no-install-recommends \
            cuda-cudart-11-8 libcudnn8=8.9.7.29-1+cuda11.8 && \
        rm -rf /var/lib/apt/lists/* /var/cache/apt/*; \
    fi

# ---- Runtime dependencies & handy debug tools --------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg redis-server redis-tools libsndfile1 libgl1 libgomp1 \
        curl aria2 netcat-openbsd procps net-tools lsof \
        # Add patchelf for additional library fixing if needed
        patchelf && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/*

# ---- Copy entire Python environment from builder -----------------------------
COPY --from=builder /usr/local /usr/local

# ---- Additional runtime fix for ctranslate2 (belt and suspenders approach) ----
# This ensures the libraries are properly configured in the final image
RUN ldconfig && \
    # Use patchelf to remove EXECSTACK flag as a fallback
    find /usr/local/lib/python*/site-packages -name "*.so*" -path "*/ctranslate2*" \
        -exec patchelf --remove-needed EXECSTACK {} \; 2>/dev/null || true && \
    # Update library cache
    ldconfig

# ---- Clean Python cache in final image --------------------------------------
RUN find /usr/local -type f -name "*.pyc" -delete && \
    find /usr/local -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# ---- Application code --------------------------------------------------------
COPY . .
RUN chmod +x /app/start.sh

# ---- Create startup script to handle runtime library issues -----------------
RUN cat > /app/fix_libraries.sh << 'EOF'
#!/bin/bash
# Runtime library fix script
echo "Checking ctranslate2 libraries..."
# Test if ctranslate2 can be imported
python -c "import ctranslate2; print('ctranslate2 OK')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "ctranslate2 import failed, attempting runtime fix..."
    # Try to fix with patchelf if available
    if command -v patchelf >/dev/null 2>&1; then
        find /usr/local/lib -name "*ctranslate2*.so*" -exec patchelf --set-execstack false {} \; 2>/dev/null
    fi
    # Update library cache
    ldconfig 2>/dev/null || true
fi
EOF
RUN chmod +x /app/fix_libraries.sh

# ---- Non-root user & directories ---------------------------------------------
RUN useradd -ms /bin/bash appuser && \
    mkdir -p /app/uploads /app/segments /workspace/logs /workspace/models && \
    chown -R appuser:appuser /app /workspace
USER appuser

EXPOSE 5000 8888

# ---- Health check ------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:5000/healthz || exit 1

# ---- Modified entrypoint to run fix script first ----------------------------
ENTRYPOINT ["/bin/bash", "-c", "/app/fix_libraries.sh && exec /app/start.sh"]
