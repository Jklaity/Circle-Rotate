#!/usr/bin/env python
"""
Unified Evaluation Script - Run all metrics at once

Usage:
    python scripts/eval_all.py --video path/to/video.mp4
    python scripts/eval_all.py --video path/to/video.mp4 --output results.json
"""

import argparse
import json
import torch
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

from circle_rotate.metrics.subject_drift import (
    load_models, extract_frames, calculate_subject_drift
)
from circle_rotate.metrics.clip_i import get_frames, calculate_clip_i
from circle_rotate.metrics.fidelity import calculate_fidelity
from circle_rotate.metrics.vbench_sc import get_frames as get_frames_vbench, calculate_vbench_sc
from circle_rotate.metrics.motion import calculate_motion

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    parser = argparse.ArgumentParser(description="Unified evaluation for all metrics")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames to sample")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    parser.add_argument("--no_lpips", action="store_true", help="Skip LPIPS calculation")
    args = parser.parse_args()

    results = {}

    # 1. Subject Drift (YOLO + SAM)
    print("\n[1/5] Evaluating Subject Drift (YOLO + SAM)...")
    try:
        yolo, predictor = load_models()
        frames = extract_frames(args.video, args.num_frames)
        drift_metrics = calculate_subject_drift(frames, yolo, predictor)
        results["subject_drift"] = drift_metrics
        print(f"  Subject Flow: {drift_metrics['subject_flow']:.4f}")
        print(f"  Global Flow:  {drift_metrics['global_flow']:.4f}")
        print(f"  Drift:        {drift_metrics['drift']:.4f}")
    except Exception as e:
        print(f"  Error: {e}")
        results["subject_drift"] = None

    # 2. CLIP-I
    print("\n[2/5] Evaluating CLIP-I...")
    try:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        frames = get_frames(args.video, args.num_frames)
        clip_metrics = calculate_clip_i(frames, clip_model, clip_processor)
        results["clip_i"] = clip_metrics
        print(f"  First-Last: {clip_metrics['first_last']:.4f}")
        print(f"  Global Avg: {clip_metrics['global_avg']:.4f}")
        print(f"  Frame Acc:  {clip_metrics['frame_acc']:.4f}")
        del clip_model, clip_processor
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  Error: {e}")
        results["clip_i"] = None

    # 3. Fidelity (PSNR/SSIM/LPIPS)
    print("\n[3/5] Evaluating Fidelity (PSNR/SSIM/LPIPS)...")
    try:
        lpips_model = None
        if not args.no_lpips:
            import lpips
            lpips_model = lpips.LPIPS(net='alex').to(DEVICE).eval()
        fidelity_metrics = calculate_fidelity(args.video, lpips_model)
        results["fidelity"] = fidelity_metrics
        print(f"  PSNR:  {fidelity_metrics['psnr']:.4f}")
        print(f"  SSIM:  {fidelity_metrics['ssim']:.4f}")
        if fidelity_metrics['lpips'] is not None:
            print(f"  LPIPS: {fidelity_metrics['lpips']:.4f}")
        if lpips_model:
            del lpips_model
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"  Error: {e}")
        results["fidelity"] = None

    # 4. VBench Subject Consistency
    print("\n[4/5] Evaluating VBench Subject Consistency...")
    try:
        dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        dino_model = dino_model.to(DEVICE).eval()
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        frames = get_frames_vbench(args.video, args.num_frames)
        vbench_score = calculate_vbench_sc(frames, dino_model, transform)
        results["vbench_sc"] = vbench_score
        print(f"  VBench-SC: {vbench_score:.4f}")
        del dino_model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  Error: {e}")
        results["vbench_sc"] = None

    # 5. Motion Metrics
    print("\n[5/5] Evaluating Motion Metrics...")
    try:
        motion_metrics = calculate_motion(args.video)
        results["motion"] = motion_metrics
        print(f"  Motion Magnitude: {motion_metrics['motion_magnitude']:.4f}")
        print(f"  Dynamic Degree:   {motion_metrics['dynamic_degree']:.4f}")
    except Exception as e:
        print(f"  Error: {e}")
        results["motion"] = None

    # Summary
    print("\n" + "=" * 50)
    print("Evaluation Complete!")
    print("=" * 50)

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    return results


if __name__ == "__main__":
    main()
