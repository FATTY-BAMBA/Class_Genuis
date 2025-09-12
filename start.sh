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
export REDIS_URL=${REDIS_URL:-redis://localhost:6379/0}
export REDIS_WAIT_SEC=${REDIS_WAIT_SEC:-30}
export CELERY_QUEUES=${CELERY_QUEUES:-video_processing,qa_generation,maintenance,monitoring,celery}
export CELERY_LOGLEVEL=${CELERY_LOGLEVEL:-INFO}
export CELERY_POOL=${CELERY_POOL:-solo}
export CELERY_CONCURRENCY=${CELERY_CONCURRENCY:-1}
export CELERY_PREFETCH=${CELERY_PREFETCH:-1}
export CELERY_MAX_TASKS_PER_CHILD=${CELERY_MAX_TASKS_PER_CHILD:-0}
export CELERY_LEAN=${CELERY_LEAN:-false}
export LOG_LEVEL=${LOG_LEVEL:-INFO}   # NEW: app logging level

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

# ---------- Chapter-Llama ----------
echo "Starting Chapter-Llama service..."
./start_chapter_llama.sh &
sleep 5
curl -sf http://localhost:8000/health || { echo "Chapter-Llama not ready"; exit 1; }
echo "Chapter-Llama started on :8000"

# ---------- Redis ----------
MASKED_REDIS_URL="${REDIS_URL//:[^@]*@/:*****@}"
echo "REDIS_URL detected: ${MASKED_REDIS_URL}"
REDIS_STARTED_LOCALLY=0
if [[ "${REDIS_URL}" =~ ^redis://(127\.0\.0\.1|localhost)(:|/|$) ]]; then
  echo "Starting local Redis..."
  redis-server --daemonize yes --save "" --appendonly no
  REDIS_STARTED_LOCALLY=1
else
  echo "Using external Redis at: ${MASKED_REDIS_URL}"
fi
echo -n "Waiting for Redis "
for (( i=1; i<=REDIS_WAIT_SEC; i++ )); do
  if redis-cli -u "${REDIS_URL}" ping >/dev/null 2>&1; then echo " ✓"; break; fi
  echo -n "."; sleep 1
done
if ! redis-cli -u "${REDIS_URL}" ping >/dev/null 2>&1; then
  echo; echo "ERROR: Redis not reachable after ${REDIS_WAIT_SEC}s"; exit 1
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
# line-buffer so logs appear instantly
stdbuf -oL -eL celery -A tasks.tasks:celery worker "${CELERY_COMMON_OPTS[@]}" --pidfile=/tmp/celery.pid \
  2>&1 | tee -a /workspace/logs/celery.log &
sleep 1
CELERY_PID=$(cat /tmp/celery.pid || true)
echo "Celery PID: ${CELERY_PID:-unknown}"

# ---------- start Gunicorn (stdout -> tee -> file) ----------
echo "Starting Gunicorn server..."
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
sleep 1
GUNICORN_PID=$(cat /tmp/gunicorn.pid || true)
echo "Gunicorn PID: ${GUNICORN_PID:-unknown}"

# ---------- supervise ----------
wait -n
STATUS=$?
echo "A service exited (status=${STATUS})."
exit "${STATUS}"
