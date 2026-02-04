#!/usr/bin/env python
"""
CLIP-I Evaluation Script

Usage:
    python scripts/eval_clip_i.py --video path/to/video.mp4
"""

import argparse
from transformers import CLIPProcessor, CLIPModel

from circle_rotate.metrics.clip_i import get_frames, calculate_clip_i

MODEL_ID = "openai/clip-vit-base-patch32"
DEVICE = "cuda"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=16)
    args = parser.parse_args()

    print(f"Loading CLIP model: {MODEL_ID}")
    model = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_ID)

    frames = get_frames(args.video, args.num_frames)
    metrics = calculate_clip_i(frames, model, processor)

    print(f"\nResults for {args.video}:")
    print(f"  First-Last: {metrics['first_last']:.4f}")
    print(f"  Global Avg: {metrics['global_avg']:.4f}")
    print(f"  Frame Acc:  {metrics['frame_acc']:.4f}")


if __name__ == "__main__":
    main()
