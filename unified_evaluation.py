#!/usr/bin/env python3
"""
统一评估脚本 - 整合所有评估功能

用法:
    python unified_evaluation.py --metric psnr --input_dir <路径>
    python unified_evaluation.py --metric all --input_dir <路径>
    python unified_evaluation.py --metric motion_hallucination --input_dir <路径>
"""

import argparse
import os
import sys
import json
import numpy as np

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.video_utils import get_video_frames, get_videos_from_dir, load_image, resize_to_match, extract_frames
from utils.metrics_utils import calc_psnr_ssim, calc_optical_flow, calc_epipolar_error, paired_ttest


DEVICE = "cuda"


def calc_lpips(img1, img2, lpips_model):
    """计算LPIPS"""
    import torch
    img2 = resize_to_match(img2, img1.shape)
    t1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1
    t2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1
    with torch.no_grad():
        return lpips_model(t1.to(DEVICE), t2.to(DEVICE)).item()


def calc_clip_i(frames, clip_model, clip_processor):
    """计算CLIP-I"""
    import torch
    from PIL import Image
    pil_frames = [Image.fromarray(f) for f in frames]
    inputs = clip_processor(images=pil_frames, return_tensors="pt", padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
        features = features / features.norm(p=2, dim=-1, keepdim=True)
    first_last = torch.dot(features[0], features[-1]).item()
    global_avg = torch.mm(features, features[0].unsqueeze(1)).squeeze().mean().item()
    frame_acc = torch.nn.functional.cosine_similarity(features[:-1], features[1:]).mean().item()
    return first_last, global_avg, frame_acc


def evaluate_psnr_ssim(input_dir, first_frame, last_frame, output_file):
    """评估PSNR/SSIM"""
    first_cond = load_image(first_frame)
    last_cond = load_image(last_frame)
    videos = get_videos_from_dir(input_dir)
    results = []
    for vp in videos:
        frames = get_video_frames(vp, 16)
        if len(frames) < 2:
            continue
        psnr_f, ssim_f = calc_psnr_ssim(first_cond, frames[0])
        psnr_l, ssim_l = calc_psnr_ssim(last_cond, frames[-1])
        results.append({
            'video': os.path.basename(vp),
            'psnr_first': psnr_f, 'ssim_first': ssim_f,
            'psnr_last': psnr_l, 'ssim_last': ssim_l
        })
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    return results


def evaluate_motion_hallucination(ours_dir, baseline_dir, target_size, output_file):
    """评估运动幻觉指标"""
    ours_videos = get_videos_from_dir(ours_dir)
    base_videos = get_videos_from_dir(baseline_dir)

    ours_flow, base_flow = [], []
    ours_epi, base_epi = [], []

    for ov, bv in zip(ours_videos, base_videos):
        of = extract_frames(ov, target_size)
        bf = extract_frames(bv, target_size)
        if not of or not bf:
            continue
        ours_flow.append(calc_optical_flow(of)[0])
        base_flow.append(calc_optical_flow(bf)[0])
        epi_o, _ = calc_epipolar_error(of)
        epi_b, _ = calc_epipolar_error(bf)
        if epi_o > 0 and epi_b > 0:
            ours_epi.append(epi_o)
            base_epi.append(epi_b)

    results = {}
    if ours_flow:
        results['flow'] = paired_ttest(ours_flow, base_flow, False)
    if ours_epi:
        results['epipolar'] = paired_ttest(ours_epi, base_epi, False)

    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    return results


def main():
    parser = argparse.ArgumentParser(description='统一评估脚本')
    parser.add_argument('--metric', type=str, required=True,
                        choices=['psnr', 'ssim', 'lpips', 'clip_i', 'motion_hallucination', 'all'],
                        help='要计算的指标')
    parser.add_argument('--input_dir', type=str, help='输入视频目录')
    parser.add_argument('--baseline_dir', type=str, help='基线视频目录')
    parser.add_argument('--first_frame', type=str, help='首帧图片路径')
    parser.add_argument('--last_frame', type=str, help='尾帧图片路径')
    parser.add_argument('--output', type=str, help='输出文件路径')
    parser.add_argument('--target_size', type=str, default='720,1280', help='目标尺寸 h,w')
    args = parser.parse_args()

    target_size = tuple(map(int, args.target_size.split(',')))

    if args.metric in ['psnr', 'ssim']:
        if not args.first_frame or not args.last_frame:
            print("错误: psnr/ssim需要 --first_frame 和 --last_frame")
            return
        results = evaluate_psnr_ssim(args.input_dir, args.first_frame, args.last_frame, args.output)
        print(f"评估完成，共 {len(results)} 个视频")

    elif args.metric == 'motion_hallucination':
        if not args.baseline_dir:
            print("错误: motion_hallucination需要 --baseline_dir")
            return
        results = evaluate_motion_hallucination(args.input_dir, args.baseline_dir, target_size, args.output)
        print("运动幻觉评估完成")

    print("完成!")


if __name__ == "__main__":
    main()
