#!/usr/bin/env python
"""
Motion Metrics Evaluation Script

Usage:
    python scripts/eval_motion.py --video path/to/video.mp4
"""

import argparse

from circle_rotate.metrics.motion import calculate_motion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--threshold", type=int, default=10)
    args = parser.parse_args()

    metrics = calculate_motion(args.video, args.threshold)

    print(f"\nResults for {args.video}:")
    print(f"  Motion Magnitude: {metrics['motion_magnitude']:.4f}")
    print(f"  Dynamic Degree:   {metrics['dynamic_degree']:.4f}")


if __name__ == "__main__":
    main()
