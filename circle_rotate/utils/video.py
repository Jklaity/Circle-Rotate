"""
Video utility functions
"""

import cv2
import numpy as np


def load_video(video_path, num_frames=None):
    """Load video frames"""
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    if num_frames and len(frames) > 0:
        indices = np.linspace(0, len(frames)-1, num_frames, dtype=int)
        frames = [frames[i] for i in indices]

    return frames


def save_video(frames, output_path, fps=16):
    """Save frames as video"""
    if len(frames) == 0:
        return None

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    return output_path
