#!/usr/bin/env bash
set -Eeuo pipefail

### ---------- Env ----------
export PYTHONUNBUFFERED=1
export CUDA_MODULE_LOADING=${CUDA_MODULE_LOADING:-LAZY}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export WHISPER_DEVICE=${WHISPER_DEVICE:-cuda}
export WHISPER_IMPL=${WHISPER_IMPL:-fast}
export WHISPER_MODEL=${WHISPER_MODEL:-large-v2}
export PYTHONPATH="/app:${PYTHONPATH:-}"

# Detect RunPod → prefer local Redis inside container; else use external if provided
if [ "${RUNPOD_POD_ID:-}" != "" ]; then
  echo "Detected RunPod environment - using local Redis"
  export REDIS_URL="redis://localhost:6379/0"
else
  export REDIS_URL=${REDIS_URL:-rediss://red-d332fu3uibrs73a90n5g:R8Nfl8YuJhfRc9djsmWPEnr7tJOx4prK@virginia-keyvalue.render.com:6379/0}
fi

export REDIS_WAIT_SEC=${REDIS_WAIT_SEC:-30}
export CELERY_QUEUES=${CELERY_QUEUES:-video_processing,qa_generation,maintenance,monitoring,celery}
export CELERY_LOGLEVEL=${CELERY_LOGLEVEL:-INFO}
export CELERY_POOL=${CELERY_POOL:-solo}
export CELERY_CONCURRENCY=${CELERY_CONCURRENCY:-1}
export CELERY_PREFETCH=${CELERY_PREFETCH:-1}
export CELERY_MAX_TASKS_PER_CHILD=${CELERY_MAX_TASKS_PER_CHILD:-0}
export CELERY_LEAN=${CELERY_LEAN:-false}
export LOG_LEVEL=${LOG_LEVEL:-INFO}

# Suppress PaddlePaddle noise
export GLOG_minloglevel=2
export GLOG_logtostderr=0

# Logs
mkdir -p /workspace/logs
CELERY_LOG=/workspace/logs/celery.worker.log
GUNICORN_LOG=/workspace/logs/gunicorn.log

echo "========= CUDA / GPU ========="
if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi || true; else echo "nvidia-smi not found"; fi
python - <<'PY'
import torch, os
print(f"torch {torch.__version__} | cuda_available={torch.cuda.is_available()} | device_env={os.environ.get('WHISPER_DEVICE','auto')}")
if torch.cuda.is_available():
    print("cuda device count:", torch.cuda.device_count())
    print("cuda name:", torch.cuda.get_device_name(0))
PY
echo "==============================="

### ---------- Redis ----------
MASKED_REDIS_URL="${REDIS_URL//:[^@]*@/:*****@}"
echo "REDIS_URL detected: ${MASKED_REDIS_URL}"
REDIS_STARTED_LOCALLY=0

# Start local redis if URL is localhost
if [[ "${REDIS_URL}" =~ ^redis://(127\.0\.0\.1|localhost)(:|/|$) ]]; then
  echo "Starting local Redis..."
  redis-server --daemonize yes --save "" --appendonly no
  REDIS_STARTED_LOCALLY=1
  sleep 2
else
  echo "Using external Redis at: ${MASKED_REDIS_URL}"
fi

echo -n "Waiting for Redis "
for (( i=1; i<=REDIS_WAIT_SEC; i++ )); do
  if redis-cli -u "${REDIS_URL}" ping >/dev/null 2>&1; then echo " ✓"; break; fi
  echo -n "."
  sleep 1
done

if ! redis-cli -u "${REDIS_URL}" ping >/dev/null 2>&1; then
  echo
  echo "ERROR: Redis not reachable after ${REDIS_WAIT_SEC}s"
  if [ "${RUNPOD_POD_ID:-}" != "" ] && [ "${REDIS_STARTED_LOCALLY}" = "0" ]; then
    echo "Attempting local Redis fallback..."
    redis-server --daemonize yes --save "" --appendonly no
    export REDIS_URL="redis://localhost:6379/0"
    sleep 2
    if ! redis-cli -u "${REDIS_URL}" ping >/dev/null 2>&1; then
      echo "Failed to start local Redis"; exit 1
    fi
    REDIS_STARTED_LOCALLY=1
  else
    exit 1
  fi
fi

### ---------- Cleanup handlers ----------
cleanup() {
  echo "Shutting down..."
  if [[ "${TAIL_PID:-}" != "" ]]; then
    kill -TERM "${TAIL_PID}" 2>/dev/null || true
  fi
  if [[ -f /tmp/gunicorn.pid ]]; then
    PID=$(cat /tmp/gunicorn.pid || true)
    [[ -n "${PID}" ]] && kill -TERM "${PID}" 2>/dev/null || true
  fi
  if [[ -f /tmp/celery.pid ]]; then
    PID=$(cat /tmp/celery.pid || true)
    [[ -n "${PID}" ]] && kill -TERM "${PID}" 2>/dev/null || true
  fi
  if [[ "${REDIS_STARTED_LOCALLY}" = "1" ]]; then
    redis-cli -u "${REDIS_URL}" shutdown || true
  fi
}
trap cleanup EXIT INT TERM

### ---------- Celery options ----------
CELERY_COMMON_OPTS=(
  --loglevel="${CELERY_LOGLEVEL}"
  -P "${CELERY_POOL}"
  --concurrency="${CELERY_CONCURRENCY}"
  --prefetch-multiplier="${CELERY_PREFETCH}"
  --max-tasks-per-child="${CELERY_MAX_TASKS_PER_CHILD}"
  -Q "${CELERY_QUEUES}"
  --hostname="worker@%h"
)
if [[ "${CELERY_LEAN}" == "true" ]]; then
  CELERY_COMMON_OPTS+=( --without-gossip --without-mingle --without-heartbeat )
fi

### ---------- Start Celery (logs → file) ----------
echo "Starting Celery worker..."
# Clear or create log file (optional)
touch "${CELERY_LOG}"; chmod 666 "${CELERY_LOG}"

celery -A tasks.tasks:celery worker "${CELERY_COMMON_OPTS[@]}" \
  --logfile="${CELERY_LOG}" \
  --pidfile=/tmp/celery.pid &
CELERY_BG_PID=$!

sleep 2
CELERY_PID=$(cat /tmp/celery.pid 2>/dev/null || echo "")
echo "Celery PID: ${CELERY_PID:-unknown}"

### ---------- Start Gunicorn (capture output) ----------
echo "Starting Gunicorn server..."
echo "Binding to 0.0.0.0:5000"
touch "${GUNICORN_LOG}"; chmod 666 "${GUNICORN_LOG}"

gunicorn app.app:app \
  -b 0.0.0.0:5000 \
  --workers 1 \
  --threads 4 \
  --timeout 600 \
  --graceful-timeout 60 \
  --capture-output \
  --access-logfile "${GUNICORN_LOG}" \
  --error-logfile  "${GUNICORN_LOG}" \
  --pid /tmp/gunicorn.pid \
  --daemon

sleep 2
GUNICORN_PID=$(cat /tmp/gunicorn.pid 2>/dev/null || echo "")
echo "Gunicorn PID: ${GUNICORN_PID:-unknown}"

echo
echo "Current running processes:"
ps aux | grep -E "(gunicorn|celery|redis)" | grep -v grep || true
echo

### ---------- Verify up ----------
sleep 3

if ! kill -0 "${GUNICORN_PID}" 2>/dev/null; then
  echo "ERROR: Gunicorn process is not running"
  tail -n 50 "${GUNICORN_LOG}" || true
  exit 1
fi

if ! kill -0 "${CELERY_PID}" 2>/dev/null; then
  echo "ERROR: Celery process is not running"
  tail -n 50 "${CELERY_LOG}" || true
  exit 1
fi

# Verify port is listening
if command -v ss >/dev/null 2>&1; then
  ss -lntp | grep -q ":5000" || echo "WARNING: Port 5000 doesn't appear to be listening"
elif command -v netstat >/dev/null 2>&1; then
  netstat -tlnp 2>/dev/null | grep -q ":5000" || echo "WARNING: Port 5000 doesn't appear to be listening"
fi

echo "============================================"
echo "✅ All services started successfully"
echo "📍 Application: http://0.0.0.0:5000"
echo "📋 Logs saved to:"
echo "   - Celery  : ${CELERY_LOG}"
echo "   - Gunicorn: ${GUNICORN_LOG}"
echo "============================================"
echo
echo "📊 Streaming Celery logs (filtered)..."
echo "   Hiding: PaddlePaddle warnings"
echo "   Showing: Downloads, ASR, Processing, Errors"
echo "============================================"
echo

# Stream Celery logs to console with filtering
tail -f "${CELERY_LOG}" | grep -v "Fail to fscanf\|default_variables.cpp" &
TAIL_PID=$!

# Keep script alive
wait "${CELERY_BG_PID}"
