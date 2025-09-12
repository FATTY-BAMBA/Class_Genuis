# app/openai_utils.py

import os
import json
import re
import logging
from openai import OpenAI
from dotenv import load_dotenv

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("❌ OPENAI_API_KEY is not set in the environment.")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


def call_openai(prompt, role, model="gpt-4-turbo", max_tokens=4000):
    """
    Sends a prompt to OpenAI's chat API and returns the response text.
    """
    logger.info(f"🟢 Calling OpenAI model: {model} | Max tokens: {max_tokens}")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": role},
                {"role": "user", "content": prompt}
            ],
            temperature=0.25,
            max_tokens=max_tokens,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )

        result = response.choices[0].message.content.strip()
        logger.info(f"✅ OpenAI response received (length={len(result)}): {result[:500]}...")
        return result

    except Exception as e:
        logger.error(f"❌ OpenAI API call failed: {e}", exc_info=True)
        if hasattr(e, "response") and hasattr(e.response, "text"):
            logger.error(f"🔍 API error details: {e.response.text}")
        return ""


def parse_qa_json(response_text):
    """
    Extracts and parses JSON content from an OpenAI response, even if wrapped in markdown or surrounded by extra text.
    Returns a list of parsed JSON items or an empty list if parsing fails.
    """
    if not response_text or not isinstance(response_text, str):
        logger.warning("⚠️ Empty or invalid GPT response passed to parse_qa_json")
        return []

    try:
        # Remove markdown-style ```json blocks
        clean_text = re.sub(r"```(?:json)?", "", response_text, flags=re.IGNORECASE).strip("` \n")

        # Try direct parsing
        parsed = json.loads(clean_text)
        if isinstance(parsed, list):
            return parsed

    except json.JSONDecodeError as e1:
        logger.warning(f"⚠️ JSON parsing failed on first attempt: {e1}")

        # Try to find the first valid list in the text
        try:
            match = re.search(r"(\[\s*{.*?}\s*\])", response_text, re.DOTALL)
            if match:
                extracted_json = match.group(1)
                parsed = json.loads(extracted_json)
                if isinstance(parsed, list):
                    return parsed
        except Exception as e2:
            logger.error(f"❌ Second JSON parsing attempt failed: {e2}")

    # Log the first part of the raw response for debugging
    logger.debug("📄 Raw GPT response:\n" + response_text[:1000])
    logger.error("❌ parse_qa_json: Failed to extract valid JSON array.")
    return []
