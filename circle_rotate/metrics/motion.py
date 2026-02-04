"""
Motion Metrics

Metrics:
- Motion Magnitude: Average optical flow magnitude across frames
- Dynamic Degree: Ratio of changed pixels between frames
"""

import cv2
import numpy as np


def get_all_frames(video_path):
    """Extract all frames from video"""
    cap = cv2.VideoCapture(video_path)

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    return frames


def calc_motion_magnitude(video_path):
    """
    Calculate average optical flow magnitude

    Args:
        video_path: Path to video file

    Returns:
        float: Average motion magnitude
    """
    frames = get_all_frames(video_path)
    if len(frames) < 2:
        return 0.0

    magnitudes = []
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

    for i in range(1, len(frames)):
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )

        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        magnitudes.append(mag.mean())

        prev_gray = curr_gray

    return float(np.mean(magnitudes))


def calc_dynamic_degree(video_path, threshold=10):
    """
    Calculate dynamic degree (ratio of changed pixels)

    Args:
        video_path: Path to video file
        threshold: Pixel difference threshold

    Returns:
        float: Dynamic degree (0-1)
    """
    frames = get_all_frames(video_path)
    if len(frames) < 2:
        return 0.0

    ratios = []
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

    for i in range(1, len(frames)):
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

        diff = np.abs(curr_gray.astype(float) - prev_gray.astype(float))
        changed = np.sum(diff > threshold)
        total = diff.size
        ratios.append(changed / total)

        prev_gray = curr_gray

    return float(np.mean(ratios))


def calculate_motion(video_path, threshold=10):
    """
    Calculate all motion metrics

    Args:
        video_path: Path to video file
        threshold: Pixel difference threshold for dynamic degree

    Returns:
        dict with keys: motion_magnitude, dynamic_degree
    """
    return {
        "motion_magnitude": calc_motion_magnitude(video_path),
        "dynamic_degree": calc_dynamic_degree(video_path, threshold),
    }
