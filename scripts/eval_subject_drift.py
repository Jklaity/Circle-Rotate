#!/usr/bin/env python
"""
Subject Drift Evaluation Script

Usage:
    python scripts/eval_subject_drift.py --video path/to/video.mp4
"""

import argparse
from ultralytics import YOLO

from circle_rotate.metrics.subject_drift import (
    extract_frames, calculate_subject_drift
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=16)
    args = parser.parse_args()

    print("Loading YOLOv8 model...")
    model = YOLO("yolov8n.pt")

    frames = extract_frames(args.video, args.num_frames)
    if len(frames) < 2:
        print("Error: Not enough frames")
        return

    metrics = calculate_subject_drift(frames, model)

    print(f"\nResults for {args.video}:")
    print(f"  Subject Flow: {metrics['subject_flow']:.4f}")
    print(f"  Global Flow:  {metrics['global_flow']:.4f}")
    print(f"  Drift:        {metrics['drift']:.4f}")


if __name__ == "__main__":
    main()
