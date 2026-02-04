"""视频处理工具函数"""

import cv2
import numpy as np
from PIL import Image


def load_image(path):
    """加载图像为numpy数组 (RGB)"""
    return np.array(Image.open(path).convert("RGB"))


def get_video_frames(video_path, num_frames=16):
    """读取视频帧 (均匀采样)"""
    cap = cv2.VideoCapture(video_path)
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    total = len(all_frames)
    if total == 0:
        return []
    if total <= num_frames:
        return all_frames

    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    return [all_frames[i] for i in indices]


def resize_to_match(img, target_shape):
    """调整图像尺寸"""
    if img.shape[:2] != target_shape[:2]:
        return cv2.resize(img, (target_shape[1], target_shape[0]))
    return img


def extract_frames(video_path, target_size, num_frames=16):
    """提取视频帧并调整尺寸"""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            if (h, w) != target_size:
                frame = cv2.resize(frame, (target_size[1], target_size[0]))
            frames.append(frame)
    cap.release()
    return frames


def get_videos_from_dir(dir_path):
    """从目录获取所有mp4视频"""
    import os
    videos = []
    if not os.path.exists(dir_path):
        return videos
    for fname in sorted(os.listdir(dir_path)):
        if fname.endswith('.mp4'):
            videos.append(os.path.join(dir_path, fname))
    return videos
