#!/usr/bin/env python
"""
Optical Flow Visualization Script

Usage:
    python scripts/vis_optical_flow.py --video path/to/video.mp4
"""

import argparse
from circle_rotate.visualization.optical_flow import (
    extract_frames, compute_optical_flow, visualize_flow
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--output", type=str, default="optical_flow.png")
    parser.add_argument("--vmax", type=float, default=50)
    args = parser.parse_args()

    frames = extract_frames(args.video)
    if frames[0] is None or frames[1] is None:
        print("Error: Could not extract frames")
        return

    flow_mag, _ = compute_optical_flow(frames[0], frames[1])
    print(f"Mean flow: {flow_mag.mean():.2f}")

    visualize_flow(flow_mag, args.output, vmax=args.vmax)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
