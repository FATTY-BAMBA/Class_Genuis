#!/usr/bin/env bash
set -Eeuo pipefail

# ---------- env ----------
export PYTHONUNBUFFERED=1
export CUDA_MODULE_LOADING=${CUDA_MODULE_LOADING:-LAZY}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export WHISPER_DEVICE=${WHISPER_DEVICE:-cuda}
export WHISPER_IMPL=${WHISPER_IMPL:-fast}
export WHISPER_MODEL=${WHISPER_MODEL:-large-v2}
export PYTHONPATH="/app:${PYTHONPATH:-}"

# Detect if we're running in RunPod and override Redis URL if needed
if [ "${RUNPOD_POD_ID:-}" != "" ]; then
    echo "Detected RunPod environment - using local Redis"
    export REDIS_URL="redis://localhost:6379/0"
else
    # Use the external Redis for non-RunPod environments
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

mkdir -p /workspace/logs

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

# ---------- Redis ----------
MASKED_REDIS_URL="${REDIS_URL//:[^@]*@/:*****@}"
echo "REDIS_URL detected: ${MASKED_REDIS_URL}"
REDIS_STARTED_LOCALLY=0

# Check if we need to start local Redis
if [[ "${REDIS_URL}" =~ ^redis://(127\.0\.0\.1|localhost)(:|/|$) ]]; then
  echo "Starting local Redis..."
  redis-server --daemonize yes --save "" --appendonly no
  REDIS_STARTED_LOCALLY=1
  sleep 2
else
  echo "Using external Redis at: ${MASKED_REDIS_URL}"
fi

# Wait for Redis connection
echo -n "Waiting for Redis "
for (( i=1; i<=REDIS_WAIT_SEC; i++ )); do
  if redis-cli -u "${REDIS_URL}" ping >/dev/null 2>&1; then 
    echo " ✓"
    break
  fi
  echo -n "."
  sleep 1
done

# Check if Redis is reachable
if ! redis-cli -u "${REDIS_URL}" ping >/dev/null 2>&1; then
  echo
  echo "ERROR: Redis not reachable after ${REDIS_WAIT_SEC}s"
  
  # If we're in RunPod and Redis failed, try to start it
  if [ "${RUNPOD_POD_ID:-}" != "" ] && [ "${REDIS_STARTED_LOCALLY}" == "0" ]; then
    echo "Attempting to start local Redis as fallback..."
    redis-server --daemonize yes --save "" --appendonly no
    export REDIS_URL="redis://localhost:6379/0"
    sleep 2
    if redis-cli -u "${REDIS_URL}" ping >/dev/null 2>&1; then
      echo "Local Redis started successfully as fallback"
      REDIS_STARTED_LOCALLY=1
    else
      echo "Failed to start local Redis"
      exit 1
    fi
  else
    exit 1
  fi
fi

# ---------- Celery options ----------
CELERY_COMMON_OPTS=(
  --loglevel="${CELERY_LOGLEVEL}"
  --logfile=-
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

# ---------- graceful shutdown ----------
cleanup() {
  echo "Shutting down..."
  if [[ -f /tmp/gunicorn.pid ]]; then
    PID=$(cat /tmp/gunicorn.pid || true)
    [[ -n "${PID}" ]] && kill -TERM "${PID}" 2>/dev/null || true
  fi
  if [[ -f /tmp/celery.pid ]]; then
    PID=$(cat /tmp/celery.pid || true)
    [[ -n "${PID}" ]] && kill -TERM "${PID}" 2>/dev/null || true
  fi
  if [[ "${REDIS_STARTED_LOCALLY}" == "1" ]]; then
    redis-cli -u "${REDIS_URL}" shutdown || true
  fi
}
trap cleanup EXIT INT TERM

# ---------- start Celery (stdout -> tee -> file) ----------
echo "Starting Celery worker..."
stdbuf -oL -eL celery -A tasks.tasks:celery worker "${CELERY_COMMON_OPTS[@]}" --pidfile=/tmp/celery.pid \
  2>&1 | tee -a /workspace/logs/celery.log &
sleep 2
CELERY_PID=$(cat /tmp/celery.pid 2>/dev/null || echo "")
echo "Celery PID: ${CELERY_PID:-unknown}"

# ---------- start Gunicorn (stdout -> tee -> file) ----------
echo "Starting Gunicorn server..."
echo "Binding to 0.0.0.0:5000"
stdbuf -oL -eL gunicorn app.app:app \
  -b 0.0.0.0:5000 \
  --workers 1 \
  --threads 4 \
  --timeout 600 \
  --graceful-timeout 60 \
  --access-logfile - \
  --error-logfile  - \
  --pid /tmp/gunicorn.pid \
  2>&1 | tee -a /workspace/logs/gunicorn.log &
sleep 2
GUNICORN_PID=$(cat /tmp/gunicorn.pid 2>/dev/null || echo "")
echo "Gunicorn PID: ${GUNICORN_PID:-unknown}"

# Debug: Show what's running
echo "Current running processes:"
ps aux | grep -E "(gunicorn|celery|redis)" | grep -v grep

# Wait longer before checking if processes are running
sleep 5

# Check if processes are still running
if ! kill -0 $GUNICORN_PID 2>/dev/null; then
    echo "ERROR: Gunicorn process is not running. Checking logs..."
    tail -20 /workspace/logs/gunicorn.log
    exit 1
fi

if ! kill -0 $CELERY_PID 2>/dev/null; then
    echo "ERROR: Celery process is not running. Checking logs..."
    tail -20 /workspace/logs/celery.log
    exit 1
fi

# Extra check for port binding
if ! netstat -tlnp 2>/dev/null | grep -q ":5000"; then
    echo "WARNING: Port 5000 doesn't appear to be listening"
    echo "Checking with lsof:"
    lsof -i :5000 || true
fi

echo "All services started successfully. Application running on http://0.0.0.0:5000"
echo "Monitoring processes..."

# ---------- supervise ----------
wait -n
STATUS=$?
echo "A service exited (status=${STATUS})."
exit "${STATUS}"
