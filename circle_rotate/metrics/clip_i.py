"""
CLIP-I (Identity Preservation) Metrics

Metrics:
- First-Last Consistency: similarity between first and last frame
- Global Avg Consistency: average similarity of all frames to first frame
- Frame-to-Frame Smoothness: average similarity between adjacent frames
"""

import torch
import cv2
import numpy as np
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_frames(video_path, num_frames=16):
    """Extract frames from video (uniform sampling)"""
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
    frames = [Image.fromarray(all_frames[i]) for i in indices]
    return frames


def calculate_clip_i(frames, model, processor):
    """
    Calculate CLIP-I metrics

    Args:
        frames: List of PIL Images
        model: CLIP model
        processor: CLIP processor

    Returns:
        dict with keys:
        - first_last: First-Last frame consistency
        - global_avg: Global average consistency
        - frame_acc: Frame-to-frame smoothness
    """
    inputs = processor(images=frames, return_tensors="pt", padding=True).to(DEVICE)

    with torch.no_grad():
        features = model.get_image_features(**inputs)

    # L2 normalize
    features = features / features.norm(p=2, dim=-1, keepdim=True)

    # First-Last consistency
    first_last = torch.dot(features[0], features[-1]).item()

    # Global average consistency
    global_sims = torch.mm(features, features[0].unsqueeze(1)).squeeze()
    global_avg = global_sims.mean().item()

    # Frame-to-frame smoothness
    inter_sims = torch.nn.functional.cosine_similarity(features[:-1], features[1:])
    frame_acc = inter_sims.mean().item()

    return {
        "first_last": first_last,
        "global_avg": global_avg,
        "frame_acc": frame_acc,
    }
