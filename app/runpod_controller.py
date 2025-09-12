import os
import time
import json
import logging
import requests
from dotenv import load_dotenv

# === Load Environment Variables ===
load_dotenv()

# === Logging Setup ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# === Config ===
RUNPOD_API_URL = "https://api.runpod.io/graphql"
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_POD_ID = os.getenv("RUNPOD_POD_ID")

if not RUNPOD_API_KEY or not RUNPOD_POD_ID:
    logging.error("❌ RUNPOD_API_KEY or RUNPOD_POD_ID not set.")
    raise EnvironmentError("Missing RunPod credentials.")

# === Headers ===
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {RUNPOD_API_KEY}"
}


def run_graphql_query(query: str, variables: dict = None):
    """Send a GraphQL query to the RunPod API."""
    payload = {
        "query": query,
        "variables": variables or {}
    }

    try:
        response = requests.post(RUNPOD_API_URL, json=payload, headers=HEADERS)
        response.raise_for_status()
        data = response.json()

        if data.get("errors"):
            logging.error(f"❌ GraphQL errors: {data['errors']}")
            raise RuntimeError(data["errors"])

        return data.get("data", {})

    except requests.exceptions.RequestException as e:
        logging.error(f"❌ GraphQL request failed: {e}", exc_info=True)
        raise


def get_pod_status():
    """Return status of the pod matching RUNPOD_POD_ID using the 'myself' query."""
    query = """
    {
      myself {
        pods {
          id
          name
          desiredStatus
          runtime {
            uptimeInSeconds
          }
        }
      }
    }
    """

    try:
        data = run_graphql_query(query)
        pods = data.get("myself", {}).get("pods", [])

        for pod in pods:
            if pod["id"] == RUNPOD_POD_ID:
                status = pod.get("desiredStatus")
                uptime = pod.get("runtime", {}).get("uptimeInSeconds", 0)

                if status == "RUNNING" and int(uptime or 0) > 0:
                    logging.info(f"✅ Pod {pod['name']} is RUNNING (uptime: {uptime}s)")
                    return "RUNNING"
                elif status == "RUNNING":
                    logging.info(f"⏳ Pod {pod['name']} is STARTING...")
                    return "STARTING"
                else:
                    logging.info(f"📦 Pod {pod['name']} is in state: {status}")
                    return status

        logging.warning(f"❌ Pod ID '{RUNPOD_POD_ID}' not found among your pods.")
        return None

    except Exception as e:
        logging.error(f"❌ Failed to get pod status: {e}")
        return None


def start_pod():
    """Start or resume the pod."""
    mutation = """
    mutation($podId: ID!) {
      podResume(input: { podId: $podId }) {
        id
        desiredStatus
      }
    }
    """

    try:
        data = run_graphql_query(mutation, {"podId": RUNPOD_POD_ID})
        result = data.get("podResume")
        if result:
            logging.info(f"🚀 Pod resume triggered: {result}")
        else:
            logging.warning("⚠️ Pod resume returned no result.")
        return result

    except Exception as e:
        logging.error(f"❌ Failed to start pod: {e}")
        return None


def wait_for_pod_ready(timeout: int = 300, poll_interval: int = 5):
    """Polls the pod until it becomes RUNNING."""
    start_time = time.time()
    logging.info(f"⏳ Waiting for pod {RUNPOD_POD_ID} to be RUNNING...")

    while time.time() - start_time < timeout:
        status = get_pod_status()
        if status == "RUNNING":
            logging.info(f"✅ Pod {RUNPOD_POD_ID} is now active.")
            return True
        elif status is None:
            logging.warning("⚠️ Unable to fetch pod status.")
        else:
            logging.info(f"🔁 Pod status: {status}")
        time.sleep(poll_interval)

    logging.error(f"❌ Timeout: Pod {RUNPOD_POD_ID} not ready after {timeout} seconds.")
    return False
