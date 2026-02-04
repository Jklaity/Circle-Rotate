#!/usr/bin/env python3
"""
统一数据处理脚本 - 整合所有数据处理功能

用法:
    python unified_data_processing.py --task extract_frames --dataset co3d --category banana
    python unified_data_processing.py --task merge_pdf --input_dir <路径>
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from pathlib import Path


def extract_co3d_frames(category, sequence, output_dir, num_frames=16):
    """从CO3D数据集提取帧"""
    co3d_dir = Path('/data/sharedpvc/ComfyUI/jk_work/jk/dataset/co3d')
    img_dir = co3d_dir / category / sequence / 'images'

    if not img_dir.exists():
        print(f"目录不存在: {img_dir}")
        return

    frames = sorted(img_dir.glob('frame*.jpg'))
    if not frames:
        print("未找到帧文件")
        return

    os.makedirs(output_dir, exist_ok=True)
    indices = np.linspace(0, len(frames)-1, num_frames, dtype=int)

    for i, idx in enumerate(indices):
        src = frames[idx]
        dst = os.path.join(output_dir, f'frame_{i:02d}.jpg')
        img = cv2.imread(str(src))
        cv2.imwrite(dst, img)

    print(f"已提取 {num_frames} 帧到: {output_dir}")


def merge_pdfs(input_dir, output_path):
    """合并PDF文件"""
    try:
        from PyPDF2 import PdfMerger
    except ImportError:
        print("需要安装 PyPDF2: pip install PyPDF2")
        return

    pdfs = sorted(Path(input_dir).glob('*.pdf'))
    if not pdfs:
        print("未找到PDF文件")
        return

    merger = PdfMerger()
    for pdf in pdfs:
        merger.append(str(pdf))
    merger.write(output_path)
    merger.close()
    print(f"已合并 {len(pdfs)} 个PDF到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='统一数据处理脚本')
    parser.add_argument('--task', type=str, required=True,
                        choices=['extract_frames', 'merge_pdf'],
                        help='处理任务')
    parser.add_argument('--category', type=str, help='CO3D类别')
    parser.add_argument('--sequence', type=str, help='CO3D序列')
    parser.add_argument('--input_dir', type=str, help='输入目录')
    parser.add_argument('--output', type=str, help='输出路径')
    parser.add_argument('--num_frames', type=int, default=16, help='帧数')
    args = parser.parse_args()

    if args.task == 'extract_frames':
        if not args.category or not args.sequence:
            print("错误: extract_frames需要 --category 和 --sequence")
            return
        output = args.output or f'./{args.category}_{args.sequence}'
        extract_co3d_frames(args.category, args.sequence, output, args.num_frames)

    elif args.task == 'merge_pdf':
        if not args.input_dir:
            print("错误: merge_pdf需要 --input_dir")
            return
        output = args.output or 'merged.pdf'
        merge_pdfs(args.input_dir, output)

    print("完成!")


if __name__ == "__main__":
    main()
