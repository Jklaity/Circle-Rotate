#!/usr/bin/env python3
"""
统一可视化脚本 - 整合所有可视化功能

用法:
    python unified_visualization.py --type eccv_comparison --output <路径>
    python unified_visualization.py --type optical_flow --input_video <路径>
    python unified_visualization.py --type trajectory --category banana
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2


def visualize_optical_flow(video_path, output_path):
    """可视化光流"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if len(frames) < 2:
        print("视频帧数不足")
        return

    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    flow_vis = []

    for i in range(1, len(frames)):
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros_like(frames[i])
        hsv[..., 1] = 255
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        flow_vis.append(rgb)
        prev_gray = curr_gray

    # 保存中间帧
    mid = len(flow_vis) // 2
    cv2.imwrite(output_path, flow_vis[mid])
    print(f"光流可视化已保存到: {output_path}")


def visualize_xt_slice(video_path, output_path, y_pos=None):
    """生成X-T切片"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        print("无法读取视频")
        return

    h, w = frames[0].shape[:2]
    if y_pos is None:
        y_pos = h // 2

    xt_slice = np.array([f[y_pos, :, :] for f in frames])
    plt.figure(figsize=(12, 4))
    plt.imshow(xt_slice)
    plt.xlabel('X')
    plt.ylabel('Time')
    plt.title(f'X-T Slice at y={y_pos}')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"X-T切片已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='统一可视化脚本')
    parser.add_argument('--type', type=str, required=True,
                        choices=['optical_flow', 'xt_slice', 'trajectory', 'eccv_comparison'],
                        help='可视化类型')
    parser.add_argument('--input_video', type=str, help='输入视频路径')
    parser.add_argument('--output', type=str, help='输出路径')
    parser.add_argument('--y_pos', type=int, help='X-T切片的y位置')
    args = parser.parse_args()

    if args.type == 'optical_flow':
        if not args.input_video:
            print("错误: optical_flow需要 --input_video")
            return
        output = args.output or 'optical_flow.png'
        visualize_optical_flow(args.input_video, output)

    elif args.type == 'xt_slice':
        if not args.input_video:
            print("错误: xt_slice需要 --input_video")
            return
        output = args.output or 'xt_slice.png'
        visualize_xt_slice(args.input_video, output, args.y_pos)

    print("完成!")


if __name__ == "__main__":
    main()
