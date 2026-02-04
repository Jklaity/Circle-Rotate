"""
Subject Drift Metrics

Metrics:
- Subject Flow: optical flow magnitude in subject region
- Global Flow: optical flow magnitude in entire frame
- Subject Drift: standard deviation of subject center position
"""

import cv2
import numpy as np

VEHICLE_CLASSES = [2, 5, 7]  # COCO: car, bus, truck


def calc_optical_flow(frame1, frame2):
    """Calculate optical flow magnitude"""
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    return mag


def detect_subject(frame, model, conf_thresh=0.3):
    """Detect subject bbox using YOLO"""
    results = model(frame, verbose=False)[0]
    boxes = results.boxes

    if len(boxes) == 0:
        return None

    # Prefer vehicles
    best_box, best_score = None, 0
    for box, cls, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
        if conf > conf_thresh and int(cls) in VEHICLE_CLASSES:
            if conf > best_score:
                best_score = conf.item()
                best_box = box.cpu().numpy()

    # Fallback: largest box
    if best_box is None:
        areas = (boxes.xyxy[:, 2] - boxes.xyxy[:, 0]) * (boxes.xyxy[:, 3] - boxes.xyxy[:, 1])
        idx = areas.argmax().item()
        if boxes.conf[idx] > 0.25:
            best_box = boxes.xyxy[idx].cpu().numpy()

    return best_box


def extract_frames(video_path, num_frames=16):
    """Extract frames from video using cv2"""
    cap = cv2.VideoCapture(video_path)
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        all_frames.append(frame)
    cap.release()

    total = len(all_frames)
    if total == 0:
        return []

    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    return [all_frames[i] for i in indices]


def calculate_subject_drift(frames, model):
    """
    Calculate subject drift metrics

    Args:
        frames: List of numpy arrays (RGB)
        model: YOLO model

    Returns:
        dict with keys:
        - subject_flow: average optical flow in subject region
        - global_flow: average optical flow in entire frame
        - drift: standard deviation of subject center position
    """
    h, w = frames[0].shape[:2]

    subject_flows = []
    global_flows = []
    centers = []

    for i in range(len(frames) - 1):
        mag = calc_optical_flow(frames[i], frames[i + 1])
        global_flows.append(mag.mean())

        bbox = detect_subject(frames[i], model)
        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            subject_mag = mag[y1:y2, x1:x2]
            if subject_mag.size > 0:
                subject_flows.append(subject_mag.mean())

            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            centers.append((cx, cy))

    subject_flow = np.mean(subject_flows) if subject_flows else 0.0
    global_flow = np.mean(global_flows) if global_flows else 0.0

    if len(centers) >= 2:
        centers = np.array(centers)
        drift = np.std(centers[:, 0]) + np.std(centers[:, 1])
    else:
        drift = 0.0

    return {
        "subject_flow": float(subject_flow),
        "global_flow": float(global_flow),
        "drift": float(drift),
    }
