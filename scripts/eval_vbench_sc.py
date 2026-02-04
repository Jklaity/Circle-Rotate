#!/usr/bin/env python
"""
VBench Subject Consistency Evaluation Script

Usage:
    python scripts/eval_vbench_sc.py --video path/to/video.mp4
"""

import argparse
import torch
from torchvision import transforms

from circle_rotate.metrics.vbench_sc import get_frames, calculate_vbench_sc

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=16)
    args = parser.parse_args()

    print("Loading DINO ViT-B/16 model...")
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
    model = model.to(DEVICE).eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    frames = get_frames(args.video, args.num_frames)
    if len(frames) < 2:
        print("Error: Not enough frames")
        return

    score = calculate_vbench_sc(frames, model, transform)

    print(f"\nResults for {args.video}:")
    print(f"  VBench-SC: {score:.4f}")


if __name__ == "__main__":
    main()
