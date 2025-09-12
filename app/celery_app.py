import os
from celery import Celery

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
celery = Celery("tasks", broker=REDIS_URL, backend=REDIS_URL)

# (optional) point Celery to your tasks package
celery.autodiscover_tasks(["tasks"])
