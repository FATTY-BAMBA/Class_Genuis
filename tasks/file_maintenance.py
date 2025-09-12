# tasks/file_maintenance.py

import logging
import os
import time

def clean_old_files(directory, max_age_hours=2):
    now = time.time()
    max_age_seconds = max_age_hours * 3600
    if not os.path.exists(directory):
        return

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        try:
            if os.path.isfile(filepath) and time.time() - os.path.getmtime(filepath) > max_age_seconds:
                os.remove(filepath)
                logging.info(f"🗑️ Deleted old file: {filepath}")
        except Exception as e:
            logging.warning(f"⚠️ Failed to delete {filepath}: {e}")
