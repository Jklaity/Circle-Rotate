#!/usr/bin/env python
"""
Fidelity Evaluation Script (PSNR/SSIM/LPIPS)

Usage:
    python scripts/eval_fidelity.py --video path/to/video.mp4
"""

import argparse

from circle_rotate.metrics.fidelity import calculate_fidelity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--lpips", action="store_true", help="Calculate LPIPS")
    args = parser.parse_args()

    lpips_model = None
    if args.lpips:
        import lpips
        print("Loading LPIPS model...")
        lpips_model = lpips.LPIPS(net='alex').cuda().eval()

    metrics = calculate_fidelity(args.video, lpips_model)

    print(f"\nResults for {args.video}:")
    print(f"  PSNR: {metrics['psnr']:.4f}")
    print(f"  SSIM: {metrics['ssim']:.4f}")
    if metrics['lpips'] is not None:
        print(f"  LPIPS: {metrics['lpips']:.4f}")


if __name__ == "__main__":
    main()
