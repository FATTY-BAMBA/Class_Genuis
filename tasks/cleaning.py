import logging
from app.celery_app import celery
from .file_maintenance import clean_old_files

@celery.task(name="tasks.clean_old_uploads")
def clean_uploads_task():
    logging.info("🧹 Cleaning up old uploaded files...")
    clean_old_files("/app/uploads", max_age_hours=6)
    clean_old_files("/app/logs", max_age_hours=6)
    clean_old_files("/app/sent_payloads", max_age_hours=6)
