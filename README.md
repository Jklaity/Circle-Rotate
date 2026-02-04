# Unified Scripts for Video Evaluation and Visualization

[English](#english) | [中文](#中文)

---

## English

### Overview

A collection of unified scripts for video generation quality evaluation, visualization, and data processing. This toolkit consolidates multiple scattered scripts into three main unified scripts with command-line interfaces.

### Features

- **Unified Evaluation**: PSNR, SSIM, LPIPS, CLIP-I, FVD, motion hallucination metrics
- **Unified Visualization**: Optical flow, X-T slices, trajectory plots
- **Unified Data Processing**: Frame extraction, PDF merging

### Installation

```bash
# Clone the repository
cd jk_work/jk/gitcr

# Install dependencies
pip install numpy opencv-python matplotlib scipy scikit-image
pip install torch torchvision  # For deep learning metrics
pip install lpips transformers  # For LPIPS and CLIP
```

### Project Structure

```
gitcr/
├── unified_evaluation.py      # Unified evaluation script
├── unified_visualization.py   # Unified visualization script
├── unified_data_processing.py # Unified data processing script
├── utils/
│   ├── __init__.py
│   ├── video_utils.py         # Video I/O utilities
│   ├── metrics_utils.py       # Metric calculation utilities
│   └── plot_utils.py          # Plotting utilities
└── README.md
```

### Usage

#### 1. Evaluation

```bash
# PSNR/SSIM evaluation
python unified_evaluation.py --metric psnr \
    --input_dir /path/to/videos \
    --first_frame /path/to/first.png \
    --last_frame /path/to/last.png \
    --output results.json

# Motion hallucination evaluation
python unified_evaluation.py --metric motion_hallucination \
    --input_dir /path/to/ours \
    --baseline_dir /path/to/baseline \
    --output motion_stats.json
```

#### 2. Visualization

```bash
# Optical flow visualization
python unified_visualization.py --type optical_flow \
    --input_video /path/to/video.mp4 \
    --output flow.png

# X-T slice
python unified_visualization.py --type xt_slice \
    --input_video /path/to/video.mp4 \
    --output xt_slice.png
```

#### 3. Data Processing

```bash
# Extract frames from CO3D
python unified_data_processing.py --task extract_frames \
    --category banana \
    --sequence 610_96665_193716 \
    --output ./frames

# Merge PDFs
python unified_data_processing.py --task merge_pdf \
    --input_dir /path/to/pdfs \
    --output merged.pdf
```

### API Reference

#### unified_evaluation.py

| Argument | Description |
|----------|-------------|
| `--metric` | Metric type: `psnr`, `ssim`, `lpips`, `clip_i`, `motion_hallucination`, `all` |
| `--input_dir` | Input video directory |
| `--baseline_dir` | Baseline video directory (for comparison) |
| `--first_frame` | First frame image path |
| `--last_frame` | Last frame image path |
| `--output` | Output file path |
| `--target_size` | Target size (h,w), default: `720,1280` |

#### unified_visualization.py

| Argument | Description |
|----------|-------------|
| `--type` | Visualization type: `optical_flow`, `xt_slice`, `trajectory`, `eccv_comparison` |
| `--input_video` | Input video path |
| `--output` | Output file path |
| `--y_pos` | Y position for X-T slice |

#### unified_data_processing.py

| Argument | Description |
|----------|-------------|
| `--task` | Task type: `extract_frames`, `merge_pdf` |
| `--category` | CO3D category |
| `--sequence` | CO3D sequence ID |
| `--input_dir` | Input directory |
| `--output` | Output path |
| `--num_frames` | Number of frames, default: `16` |

### License

MIT License

---

## 中文

### 概述

视频生成质量评估、可视化和数据处理的统一脚本集合。本工具包将多个分散的脚本整合为三个主要的统一脚本，提供命令行接口。

### 功能特性

- **统一评估**: PSNR、SSIM、LPIPS、CLIP-I、FVD、运动幻觉指标
- **统一可视化**: 光流、X-T切片、轨迹图
- **统一数据处理**: 帧提取、PDF合并

### 安装

```bash
# 进入目录
cd jk_work/jk/gitcr

# 安装依赖
pip install numpy opencv-python matplotlib scipy scikit-image
pip install torch torchvision  # 深度学习指标
pip install lpips transformers  # LPIPS和CLIP
```

### 使用方法

#### 1. 评估

```bash
# PSNR/SSIM评估
python unified_evaluation.py --metric psnr \
    --input_dir /视频目录 \
    --first_frame /首帧.png \
    --last_frame /尾帧.png \
    --output results.json

# 运动幻觉评估
python unified_evaluation.py --metric motion_hallucination \
    --input_dir /ours目录 \
    --baseline_dir /基线目录 \
    --output motion_stats.json
```

#### 2. 可视化

```bash
# 光流可视化
python unified_visualization.py --type optical_flow \
    --input_video /视频.mp4 \
    --output flow.png

# X-T切片
python unified_visualization.py --type xt_slice \
    --input_video /视频.mp4 \
    --output xt_slice.png
```

#### 3. 数据处理

```bash
# 从CO3D提取帧
python unified_data_processing.py --task extract_frames \
    --category banana \
    --sequence 610_96665_193716 \
    --output ./frames

# 合并PDF
python unified_data_processing.py --task merge_pdf \
    --input_dir /pdf目录 \
    --output merged.pdf
```

### 许可证

MIT License
