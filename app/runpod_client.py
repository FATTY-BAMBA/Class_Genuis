# runpod_client.py

import os
import requests
import logging

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_POD_ID = os.getenv("RUNPOD_POD_ID")
RUNPOD_API_URL = "https://api.runpod.io/graphql"

headers = {
    "Authorization": RUNPOD_API_KEY,
    "Content-Type": "application/json"
}

def start_pod():
    query = """
    mutation {
      podResume(input: { podId: "%s" }) {
        id
        environment
        desiredStatus
      }
    }
    """ % RUNPOD_POD_ID
    response = requests.post(RUNPOD_API_URL, headers=headers, json={"query": query})
    try:
        return response.json()["data"]["podResume"]
    except Exception as e:
        logging.error(f"❌ Failed to start pod: {e}")
        return None

def stop_pod():
    query = """
    mutation {
      podStop(input: { podId: "%s" }) {
        id
        desiredStatus
      }
    }
    """ % RUNPOD_POD_ID
    response = requests.post(RUNPOD_API_URL, headers=headers, json={"query": query})
    try:
        return response.json()["data"]["podStop"]
    except Exception as e:
        logging.error(f"❌ Failed to stop pod: {e}")
        return None

def get_pod_status():
    query = """
    query {
      pod(podId: "%s") {
        id
        name
        desiredStatus
        uptimeSeconds
        gpuUtilPercent
        status
      }
    }
    """ % RUNPOD_POD_ID
    response = requests.post(RUNPOD_API_URL, headers=headers, json={"query": query})
    try:
        return response.json()["data"]["pod"]
    except Exception as e:
        logging.error(f"❌ Failed to fetch pod status: {e}")
        return None
