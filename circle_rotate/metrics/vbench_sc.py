"""
VBench Subject Consistency Metrics

Uses DINO ViT-B/16 for feature extraction.
Official VBench implementation approach.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
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


def calculate_vbench_sc(frames, model, transform):
    """
    Calculate VBench Subject Consistency

    Args:
        frames: List of PIL Images
        model: DINO model
        transform: Image transform

    Returns:
        float: Subject consistency score (0-1)
    """
    features = []

    for frame in frames:
        img = transform(frame).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feat = model(img)
            feat = F.normalize(feat, dim=-1, p=2)
            features.append(feat)

    # VBench official: average of (sim_prev + sim_first) / 2
    sims = []
    for i in range(1, len(features)):
        sim_prev = max(0.0, F.cosine_similarity(features[i-1], features[i]).item())
        sim_first = max(0.0, F.cosine_similarity(features[0], features[i]).item())
        sims.append((sim_prev + sim_first) / 2)

    return float(np.mean(sims)) if sims else 0.0
