import cv2
import json
import base64
import boto3
import numpy as np
import logging
from botocore.exceptions import BotoCoreError, ClientError
import os
from dotenv import load_dotenv

import sys, os

# Add <project_root>/src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.local_models.ppe_code.inference import model_fn, predict_fn

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(BASE_DIR, "..", "local_models", "ppe_code")
model_dir = os.path.abspath(model_dir)

model = model_fn(model_dir)
#------------------------------------------------------------------------------- PPE Detection ------------------------------------------------------------------------------

# Load environment variables from .env
load_dotenv()

logger = logging.getLogger("detection")
logger.setLevel(logging.INFO)




# -------------------------------------------------------------------------------
# PPE Detection
# -------------------------------------------------------------------------------

# Load environment variables
load_dotenv()



logger = logging.getLogger("detection")
logger.setLevel(logging.INFO)


def ppe_detection(frame):
    """Send a frame to SageMaker endpoint and return (result, error_message, annotated_frame) safely."""
    try:

        result=predict_fn(frame,model)

        # Extract fields
        frame_id = result.get("frame", -1)
        detections = result.get("detections", [])
        annotated_b64 = result.get("annotated_frame", "")
        alert=result.get("alerts","")

        # Decode annotated frame
        annotated_frame = None
        if annotated_b64:
            try:
                annotated_bytes = base64.b64decode(annotated_b64)
                annotated_array = np.frombuffer(annotated_bytes, np.uint8)
                annotated_frame = cv2.imdecode(annotated_array, cv2.IMREAD_COLOR)
                if annotated_frame is None:
                    logger.warning("Failed to decode annotated frame from response")
            except Exception as e:
                msg = f"Error decoding annotated frame: {str(e)}"
                logger.exception(msg)
                # Still return result JSON even if image fails
                return {"frame_id": frame_id, "detections": detections}, msg, None

        # Success
        return {"frame_id": frame_id, "detections": detections}, None, annotated_frame,alert

    except Exception as e:
        msg = f"Unexpected error in ppe_detection: {str(e)}"
        logger.exception(msg)
        return None, msg, None



