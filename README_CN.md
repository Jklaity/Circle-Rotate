<p align="center">
  <a href="README.md">English</a> | <a href="README_CN.md">中文</a>
</p>

<p align="center">
  <h1 align="center">隐式运动对齐：面向刚体视频生成的数据驱动实证研究</h1>
  <p align="center">
    <a href="https://github.com/Jklaity"><strong>陈嘉康</strong></a>
    ·
    <strong>曾舒婷</strong>
  </p>
  <p align="center">
    <a href="https://arxiv.org/abs/xxxx.xxxxx">
      <img src='https://img.shields.io/badge/arXiv-论文-red?style=flat&logo=arxiv' alt='arXiv'>
    </a>
    <a href="https://github.com/Jklaity/Circle-Rotate">
      <img src='https://img.shields.io/badge/GitHub-代码-black?style=flat&logo=github' alt='GitHub'>
    </a>
    <a href="https://huggingface.co/jk1741391802/circle-rotate-lora">
      <img src='https://img.shields.io/badge/HuggingFace-模型-yellow?style=flat&logo=huggingface' alt='Model'>
    </a>
    <a href="#许可证">
      <img src='https://img.shields.io/badge/License-Apache%202.0-blue.svg' alt='License'>
    </a>
  </p>
</p>

<p align="center">
  <img src="assets/teaser.gif" width="100%">
</p>

## 亮点

- **81% 主体漂移降低**：相比基线方法（4.02 vs. 21.27 像素/帧）
- **数据驱动方法**：无需显式 3D 姿态监督
- **轻量级微调**：仅 0.7% LoRA 参数（100M / 14B）
- **强泛化能力**：在 CO3D、Mip-NeRF 360 等 5 个数据集上验证

## 摘要

在图像到视频（I2V）生成中实现精确的相机控制，传统上依赖于显式的 3D 姿态监督（如 CameraCtrl），这带来了昂贵的标注成本和复杂的架构设计。本文探索了一条不同的路径：**在没有任何显式 3D 先验的情况下，仅通过高质量的数据对齐，能否让先进的 I2V 模型"涌现"出刚体控制能力？**

我们采用极简的"数据驱动"策略：构建包含 2,168 个几何对齐视频的 **Circle-Rotate** 基准数据集，并仅使用轻量级 LoRA 适配器（0.7% 参数开销）微调 Wan2.2 I2V 模型。我们的方法相比 I2V 基线，将主体漂移降低了 **81%**，同时保持高保真的生成质量。

## 动态

- **[2025.02.04]** 预训练权重发布！

## 方法概述

<p align="center">
  <img src="assets/framework.png" width="100%">
</p>

我们的方法包含两个关键组件：

1. **Circle-Rotate 数据集**：2,168 个具有隐式几何对齐的专业制作视频
2. **双阶段 LoRA 微调**：针对几何和纹理的频率解耦适配

## 安装

```bash
# 克隆仓库
git clone https://github.com/Jklaity/Circle-Rotate.git
cd Circle-Rotate

# 创建 conda 环境
conda create -n circle-rotate python=3.10 -y
conda activate circle-rotate

# 安装依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## 快速开始

### 下载预训练权重

```bash
# 从 HuggingFace 下载 LoRA 权重
huggingface-cli download jk1741391802/circle-rotate-lora --local-dir ./checkpoints
```

### 推理

```python
from inference import CircleRotateInference

# 初始化模型
model = CircleRotateInference(
    base_model="Wan2.2-I2V-14B",
    lora_high="checkpoints/circle_rotate_h.safetensors",
    lora_low="checkpoints/circle_rotate_l.safetensors"
)

# 从首尾帧生成视频
video = model.generate(
    first_frame="examples/first.png",
    last_frame="examples/last.png",
    prompt="一辆银色轿车，相机平滑地向左环绕"
)
```

## Circle-Rotate 数据集

<p align="center">
  <img src="assets/dataset.png" width="80%">
</p>

Circle-Rotate 数据集包含 **2,168 个高质量视频**，具有隐式几何对齐特性。每个视频展示了相机围绕静态主体的平滑圆周运动，提供一致的刚体旋转模式，无需显式3D姿态标注。

从 HuggingFace 下载：[circle-rotate-dataset](https://huggingface.co/datasets/jk1741391802/circle-rotate-dataset)

## 实验结果

### 可视化对比

与基线方法相比，我们的方法使主体呈现**更静态的效果**，**主体位移更少**。主体保持稳定，相机平滑地围绕其旋转。

<table width="560">
<tr>
<td width="380" align="center"><b>其他方法</b></td>
<td width="180" align="center"><b>我们的方法</b></td>
</tr>
<tr>
<td align="center">
  <table width="360">
    <tr>
      <td align="center"><img src="assets/demo1_base.gif" width="180"></td>
      <td align="center"><img src="assets/demo1_cameractrl.gif" width="180"></td>
    </tr>
  </table>
</td>
<td align="center"><img src="assets/demo1_ours.gif" width="180"></td>
</tr>
<tr>
<td align="center"><img src="assets/cat_others.gif" width="180"></td>
<td align="center"><img src="assets/cat_ours.gif" width="180"></td>
</tr>
<tr>
<td align="center"><img src="assets/person_others.gif" width="180"></td>
<td align="center"><img src="assets/person_ours.gif" width="180"></td>
</tr>
<tr>
<td align="center"><img src="assets/demo2_base.gif" width="180"></td>
<td align="center"><img src="assets/demo2_ours.gif" width="180"></td>
</tr>
<tr>
<td align="center"><img src="assets/demo3_base.gif" width="180"></td>
<td align="center"><img src="assets/demo3_ours.gif" width="180"></td>
</tr>
<tr>
<td align="center"><img src="assets/demo4_base.gif" width="180"></td>
<td align="center"><img src="assets/demo4_ours.gif" width="180"></td>
</tr>
<tr>
<td align="center"><img src="assets/demo5_base.gif" width="180"></td>
<td align="center"><img src="assets/demo5_ours.gif" width="180"></td>
</tr>
<tr>
<td align="center"><img src="assets/demo6_base.gif" width="180"></td>
<td align="center"><img src="assets/demo6_ours.gif" width="180"></td>
</tr>
<tr>
<td align="center"><img src="assets/demo7_base.gif" width="180"></td>
<td align="center"><img src="assets/demo7_ours.gif" width="180"></td>
</tr>
<tr>
<td align="center"><img src="assets/demo8_base.gif" width="180"></td>
<td align="center"><img src="assets/demo8_ours.gif" width="180"></td>
</tr>
<tr>
<td align="center"><img src="assets/demo9_base.gif" width="180"></td>
<td align="center"><img src="assets/demo9_ours.gif" width="180"></td>
</tr>
<tr>
<td align="center"><img src="assets/demo10_base.gif" width="180"></td>
<td align="center"><img src="assets/demo10_ours.gif" width="180"></td>
</tr>
<tr>
<td align="center"><img src="assets/demo11_base.gif" width="180"></td>
<td align="center"><img src="assets/demo11_ours.gif" width="180"></td>
</tr>
<tr>
<td align="center"><img src="assets/demo13_base.gif" width="180"></td>
<td align="center"><img src="assets/demo13_ours.gif" width="180"></td>
</tr>
</table>

## 训练

我们使用 [musubi-tuner](https://github.com/kohya-ss/musubi-tuner) 进行 LoRA 训练。

```bash
# 训练低噪声 LoRA
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 \
    src/musubi_tuner/wan_train_network.py \
    --task i2v-A14B \
    --dit models/wan2.2_i2v_low_noise_14B_fp16.safetensors \
    --dataset_config datasets/circle/circle.toml \
    --vae models/Wan2.1_VAE.pth \
    --sdpa --fp8_base \
    --optimizer_type adamw8bit \
    --learning_rate 2e-4 \
    --gradient_checkpointing \
    --gradient_checkpointing_cpu_offload \
    --max_data_loader_n_workers 1 \
    --persistent_data_loader_workers \
    --network_module networks.lora_wan \
    --network_dim 32 \
    --timestep_sampling shift \
    --discrete_flow_shift 8.0 \
    --max_train_epochs 100 \
    --save_every_n_epochs 1 \
    --seed 42 \
    --output_dir ./outputs/low \
    --output_name circle_rotate_l \
    --mixed_precision fp16 \
    --blocks_to_swap 30 \
    --vae_cache_cpu \
    --t5 models/models_t5_umt5-xxl-enc-bf16.pth \
    --sample_prompts prompts.txt \
    --sample_every_n_epochs 1 \
    --sample_at_first
```

```bash
# 训练高噪声 LoRA
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 \
    src/musubi_tuner/wan_train_network.py \
    --task i2v-A14B \
    --dit models/wan2.2_i2v_high_noise_14B_fp16.safetensors \
    --dataset_config datasets/circle/circle.toml \
    --vae models/Wan2.1_VAE.pth \
    --sdpa --fp8_base \
    --optimizer_type adamw8bit \
    --learning_rate 2e-4 \
    --gradient_checkpointing \
    --gradient_checkpointing_cpu_offload \
    --max_data_loader_n_workers 1 \
    --persistent_data_loader_workers \
    --network_module networks.lora_wan \
    --network_dim 32 \
    --timestep_sampling shift \
    --discrete_flow_shift 8.0 \
    --max_train_epochs 100 \
    --save_every_n_epochs 1 \
    --seed 42 \
    --output_dir ./outputs/high \
    --output_name circle_rotate_h \
    --mixed_precision fp16 \
    --blocks_to_swap 30 \
    --vae_cache_cpu \
    --t5 models/models_t5_umt5-xxl-enc-bf16.pth \
    --sample_prompts prompts.txt \
    --sample_every_n_epochs 1 \
    --sample_at_first
```

## 评估

一键运行所有评估指标：

```bash
python scripts/eval_all.py --video path/to/video.mp4 --output results.json
```

**参数选项：**
- `--num_frames`：采样帧数（默认：16）
- `--no_lpips`：跳过 LPIPS 计算
- `--output`：保存结果到 JSON 文件

**包含指标：**
| 指标 | 描述 |
|------|------|
| Subject Drift | 使用 YOLO + SAM 的主体位置漂移 |
| CLIP-I | 基于 CLIP 嵌入的语义一致性 |
| Fidelity | 首尾帧间的 PSNR, SSIM, LPIPS |
| VBench-SC | 使用 DINO 特征的主体一致性 |
| Motion | 运动幅度和动态程度 |

## 致谢

本工作基于以下优秀的开源项目：
- [Wan2.1](https://github.com/Wan-Video/Wan2.1) - 基础 I2V 模型
- [musubi-tuner](https://github.com/kohya-ss/musubi-tuner) - LoRA 训练框架

## 许可证

本项目采用 Apache License 2.0 许可证 - 详见 [LICENSE](LICENSE) 文件。
