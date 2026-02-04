"""
Fidelity Metrics (First-Last Frame Comparison)

Metrics:
- PSNR: Peak Signal-to-Noise Ratio
- SSIM: Structural Similarity Index
- LPIPS: Learned Perceptual Image Patch Similarity
"""

import cv2
import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_first_last_frames(video_path):
    """Extract first and last frames from video"""
    cap = cv2.VideoCapture(video_path)

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if len(frames) < 2:
        return None, None

    first = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)
    last = cv2.cvtColor(frames[-1], cv2.COLOR_BGR2RGB)
    return first, last


def calc_psnr(img1, img2):
    """Calculate PSNR between two images"""
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)


def calc_ssim(img1, img2):
    """Calculate SSIM between two images (grayscale)"""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY).astype(float)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY).astype(float)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu1 = cv2.GaussianBlur(gray1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(gray2, (11, 11), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(gray1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(gray2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(gray1 * gray2, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return float(ssim_map.mean())


def calc_lpips(img1, img2, lpips_model):
    """Calculate LPIPS between two images"""
    def to_tensor(img):
        img = img.astype(np.float32) / 255.0
        img = img * 2 - 1  # normalize to [-1, 1]
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        return img.to(DEVICE)

    t1 = to_tensor(img1)
    t2 = to_tensor(img2)

    with torch.no_grad():
        dist = lpips_model(t1, t2)

    return float(dist.item())


def calculate_fidelity(video_path, lpips_model=None):
    """
    Calculate fidelity metrics between first and last frame

    Args:
        video_path: Path to video file
        lpips_model: LPIPS model (optional, if None LPIPS won't be calculated)

    Returns:
        dict with keys: psnr, ssim, lpips (if model provided)
    """
    first, last = get_first_last_frames(video_path)

    if first is None or last is None:
        return {"psnr": 0.0, "ssim": 0.0, "lpips": 1.0}

    results = {
        "psnr": calc_psnr(first, last),
        "ssim": calc_ssim(first, last),
    }

    if lpips_model is not None:
        results["lpips"] = calc_lpips(first, last, lpips_model)
    else:
        results["lpips"] = None

    return results
