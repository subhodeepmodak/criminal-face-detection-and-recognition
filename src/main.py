import os
import sys
from pathlib import Path

# Ensure src directory is available in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import random
import warnings
import argparse
import logging
import numpy as np
import torch
import time
from typing import List, Tuple
from models import SCRFD, ArcFace
from utils.helpers import compute_similarity, draw_bbox_info, draw_bbox
from telegram_bot import send_message, send_video

# Argument Parser
def parse_args():
    parser = argparse.ArgumentParser(description="Face Re-Identification")
    parser.add_argument("--det-weight", type=str, required=True, help="Path to detection model weights")
    parser.add_argument("--rec-weight", type=str, required=True, help="Path to recognition model weights")
    parser.add_argument("--faces-dir", type=str, required=True, help="Directory containing face images")
    parser.add_argument("--source", type=str, required=True, help="Video or webcam source")
    parser.add_argument("--similarity-thresh", type=float, default=0.5, help="Similarity threshold for recognition")
    parser.add_argument("--confidence-thresh", type=float, default=0.5, help="Confidence threshold for detection")
    parser.add_argument("--max-num", type=int, default=100, help="Maximum number of detected faces per frame")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--save-video", type=str, default="output_video.mp4", help="Output path for the saved video")
    parser.add_argument("--disable-telegram", action="store_true", help="Disable Telegram alert notifications")
    return parser.parse_args()

# Set Device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU")

warnings.filterwarnings("ignore")

# Tracking detected persons and best frames
detected_persons = {}
best_detections = {}

# Logging Setup
def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), None),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

def build_targets(detector, recognizer, params) -> List[Tuple[np.ndarray, str]]:
    targets = []
    for file_name in os.listdir(params.faces_dir):
        file_path = os.path.join(params.faces_dir, file_name)
        img = cv2.imread(file_path)
        if img is None:
            logging.warning(f"Could not read image {file_path}. Skipping...")
            continue

        bboxes, kpss = detector.detect(img, 1)
        if len(bboxes) == 0:
            logging.warning(f"No face detected in {file_path}. Skipping...")
            continue

        kps = kpss[0]
        embedding = recognizer(img, kps)
        name = os.path.splitext(file_name)[0]
        targets.append((embedding, name))
    return targets

def frame_processor(frame, detector, recognizer, targets, colors, params):
    global detected_persons, best_detections

    bboxes, kpss = detector.detect(frame, params.max_num)
    current_time = time.time()

    for bbox, kps in zip(bboxes, kpss):
        *bbox, conf_score = bbox.astype(float)
        conf_score = round(conf_score, 2)

        embedding = recognizer(frame, kps)
        max_similarity, best_match_name = 0, "Unknown"

        for target, name in targets:
            similarity = compute_similarity(target, embedding)
            if similarity > max_similarity and similarity > params.similarity_thresh:
                max_similarity, best_match_name = similarity, name

        if best_match_name != "Unknown":
            color = colors[best_match_name]
            draw_bbox_info(frame, bbox, similarity=max_similarity, name=best_match_name, color=color)

            if best_match_name not in detected_persons:
                detected_persons[best_match_name] = current_time
                message = f"CRIMINAL DETECTED: {best_match_name}"
                print(message)
                if not getattr(params, "disable_telegram", False):
                    try:
                        send_message(message)
                    except Exception as e:
                        logging.warning(f"Could not send Telegram alert: {e}")

            if best_match_name not in best_detections or conf_score > best_detections[best_match_name][0]:
                best_detections[best_match_name] = (conf_score, frame.copy())
                print(f"Updated best frame for {best_match_name} (Confidence: {conf_score:.2f})")
        else:
            draw_bbox(frame, bbox, (255, 0, 0))
    return frame

def main(params):
    setup_logging(params.log_level)

    detector = SCRFD(params.det_weight, input_size=(640, 640), conf_thres=params.confidence_thresh)
    detector.session.set_providers(["CUDAExecutionProvider", "CPUExecutionProvider"])
    recognizer = ArcFace(params.rec_weight)
    recognizer.session.set_providers(["CUDAExecutionProvider", "CPUExecutionProvider"])

    targets = build_targets(detector, recognizer, params)
    colors = {name: (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256)) for _, name in targets}

    cap = cv2.VideoCapture(params.source)
    if not cap.isOpened():
        raise Exception("Could not open video or webcam")

    width, height, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 25  # fallback default
    out = cv2.VideoWriter(params.save_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = frame_processor(frame, detector, recognizer, targets, colors, params)
        out.write(frame)
        frame = cv2.resize(frame, (1080, 720))
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    for name, (conf_score, best_frame) in best_detections.items():
        frame_path = f"detected_{name}.jpg"
        cv2.imwrite(frame_path, best_frame)
        print(f"Saved best frame for {name} (Confidence: {conf_score:.2f}) and a {frame_path}")

    print(f"[Video] Output video saved to: {params.save_video}")
    
if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    args = parse_args()
    if isinstance(args.source, str) and args.source.isdigit():
        args.source = int(args.source)
    main(args)
    print("Processing completed.")
