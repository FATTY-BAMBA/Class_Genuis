#app/file_saving.py

import json
import logging

# ==================== Convert Q&A List + Notes to Final JSON Payload ====================

def qa_text_to_json(qa_content, id, team_id, section_no, created_at, course_note):
    """
    Wraps structured Q&A content and course notes into final JSON format.

    Parameters:
    - qa_content: List of question dicts (already structured)
    - course_note: Markdown string of lecture notes

    Returns:
    - Dictionary ready to be exported or saved
    """

    # Safety: Reassign sequential question IDs in order (e.g., Q001, Q002, ...)
    for idx, question in enumerate(qa_content, start=1):
        question["QuestionId"] = f"Q{str(idx).zfill(3)}"

    payload = {
        "Id": id,
        "TeamId": team_id,
        "SectionNo": section_no,
        "CreatedAt": created_at,
        "Questions": qa_content,
        "CourseNote": course_note.strip()
    }

    logging.info(f"📦 Packaged {len(qa_content)} questions into final JSON payload.")
    return payload
