"""
Subject Drift Metrics

Metrics:
- Subject Flow: optical flow magnitude in subject region
- Global Flow: optical flow magnitude in entire frame
- Subject Drift: standard deviation of subject center position

Uses YOLO + SAM for accurate subject segmentation.
"""

import cv2
import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VEHICLE_CLASSES = [2, 5, 7]  # COCO: car, bus, truck


def load_models(sam_checkpoint="/data/sharedpvc/ComfyUI/models/sams/sam_vit_b_01ec64.pth"):
    """Load YOLO and SAM models"""
    from ultralytics import YOLO
    from segment_anything import sam_model_registry, SamPredictor

    yolo = YOLO("yolov8n.pt")
    sam = sam_model_registry['vit_b'](checkpoint=sam_checkpoint)
    sam.to(DEVICE)
    predictor = SamPredictor(sam)

    return yolo, predictor


def calc_optical_flow(frame1, frame2):
    """Calculate optical flow magnitude"""
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    return np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)


def detect_bbox(frame, yolo, conf_thresh=0.25):
    """Detect subject bbox using YOLO"""
    results = yolo(frame, verbose=False)[0]
    boxes = results.boxes

    if len(boxes) == 0:
        return None

    best_box, best_score = None, 0
    for box, cls, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
        if conf > conf_thresh and int(cls) in VEHICLE_CLASSES:
            if conf > best_score:
                best_score = conf.item()
                best_box = box.cpu().numpy()

    if best_box is None:
        areas = (boxes.xyxy[:, 2] - boxes.xyxy[:, 0]) * \
                (boxes.xyxy[:, 3] - boxes.xyxy[:, 1])
        idx = areas.argmax().item()
        if boxes.conf[idx] > 0.2:
            best_box = boxes.xyxy[idx].cpu().numpy()

    return best_box


def segment_subject(frame, bbox, predictor):
    """Segment subject using SAM"""
    predictor.set_image(frame)
    masks, scores, _ = predictor.predict(box=bbox, multimask_output=True)
    return masks[scores.argmax()]


def extract_frames(video_path, num_frames=16):
    """Extract frames from video"""
    cap = cv2.VideoCapture(video_path)
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    total = len(all_frames)
    if total == 0:
        return []

    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    return [all_frames[i] for i in indices]


def calculate_subject_drift(frames, yolo, predictor):
    """
    Calculate subject drift metrics using YOLO + SAM

    Args:
        frames: List of numpy arrays (RGB)
        yolo: YOLO model
        predictor: SAM predictor

    Returns:
        dict with subject_flow, global_flow, drift
    """
    centers = []
    global_flows = []
    subject_flows = []
    prev_mask = None

    for i, frame in enumerate(frames):
        bbox = detect_bbox(frame, yolo)
        mask = None

        if bbox is not None:
            mask = segment_subject(frame, bbox, predictor)
            ys, xs = np.where(mask)
            if len(xs) > 0:
                centers.append([xs.mean(), ys.mean()])

        if i < len(frames) - 1:
            mag = calc_optical_flow(frames[i], frames[i + 1])
            global_flows.append(np.mean(mag))

            if mask is not None and mask.sum() > 0:
                subject_flows.append(np.mean(mag[mask]))

        prev_mask = mask

    drift = -1
    if len(centers) >= 2:
        centers = np.array(centers)
        drift = np.sqrt(np.std(centers[:, 0])**2 + np.std(centers[:, 1])**2)

    return {
        "subject_flow": float(np.mean(subject_flows)) if subject_flows else -1,
        "global_flow": float(np.mean(global_flows)) if global_flows else -1,
        "drift": float(drift),
    }
