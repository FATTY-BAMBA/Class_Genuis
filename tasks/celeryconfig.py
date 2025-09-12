from celery.schedules import crontab

beat_schedule = {
    "clean-old-uploads-every-hour": {
        "task": "tasks.clean_old_uploads",
        "schedule": crontab(minute=0),  # Every hour, adjust as needed
    },
}
