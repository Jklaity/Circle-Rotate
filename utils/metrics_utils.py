"""指标计算工具函数"""

import cv2
import numpy as np
from scipy import stats
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def calc_psnr_ssim(img1, img2):
    """计算PSNR和SSIM"""
    from .video_utils import resize_to_match
    img2 = resize_to_match(img2, img1.shape)
    p = psnr(img1, img2, data_range=255)
    s = ssim(img1, img2, channel_axis=2, data_range=255)
    return p, s


def calc_optical_flow(frames):
    """计算光流幅度"""
    if len(frames) < 2:
        return 0.0, 0.0
    flow_mags = []
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    for i in range(1, len(frames)):
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        flow_mags.append(np.mean(mag))
        prev_gray = curr_gray
    return np.mean(flow_mags), np.std(flow_mags)


def sampson_distance(F, pts1, pts2):
    """计算Sampson距离"""
    n = pts1.shape[0]
    pts1_h = np.hstack([pts1, np.ones((n, 1))])
    pts2_h = np.hstack([pts2, np.ones((n, 1))])
    Fx1 = (F @ pts1_h.T).T
    Ftx2 = (F.T @ pts2_h.T).T
    x2tFx1 = np.sum(pts2_h * Fx1, axis=1)
    denom = Fx1[:, 0]**2 + Fx1[:, 1]**2 + Ftx2[:, 0]**2 + Ftx2[:, 1]**2
    denom = np.maximum(denom, 1e-8)
    return (x2tFx1 ** 2) / denom


def calc_epipolar_error(frames):
    """计算极线误差"""
    if len(frames) < 2:
        return -1, -1

    frame0 = frames[0]
    gray0 = cv2.cvtColor(frame0, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(nfeatures=2000)
    kp0, desc0 = sift.detectAndCompute(gray0, None)

    if desc0 is None or len(kp0) < 10:
        return -1, -1

    errors, inliers = [], []
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    for t in range(1, len(frames)):
        gray_t = cv2.cvtColor(frames[t], cv2.COLOR_RGB2GRAY)
        kp_t, desc_t = sift.detectAndCompute(gray_t, None)

        if desc_t is None or len(kp_t) < 10:
            continue

        matches = flann.knnMatch(desc0, desc_t, k=2)
        good = [m for m, n in matches if len([m, n]) == 2 and m.distance < 0.7 * n.distance]

        if len(good) < 8:
            continue

        pts0 = np.float32([kp0[m.queryIdx].pt for m in good])
        pts_t = np.float32([kp_t[m.trainIdx].pt for m in good])

        F, mask = cv2.findFundamentalMat(pts0, pts_t, cv2.FM_RANSAC, 3.0, 0.99)
        if F is None or F.shape != (3, 3):
            continue

        sampson = sampson_distance(F, pts0, pts_t)
        errors.append(np.mean(np.sqrt(sampson)))
        inliers.append(np.sum(mask) / len(mask) if mask is not None else 0)

    if not errors:
        return -1, -1
    return np.mean(errors), np.mean(inliers)


def paired_ttest(ours, baseline, higher_better=True):
    """配对t检验"""
    ours = np.array(ours)
    baseline = np.array(baseline)
    t_stat, p_value = stats.ttest_rel(ours, baseline)
    diff = ours - baseline
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0
    if higher_better:
        ours_better = np.mean(ours) > np.mean(baseline)
    else:
        ours_better = np.mean(ours) < np.mean(baseline)
    return {
        'ours_mean': np.mean(ours),
        'ours_std': np.std(ours),
        'baseline_mean': np.mean(baseline),
        'baseline_std': np.std(baseline),
        't_stat': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'ours_better': ours_better,
        'significant': p_value < 0.05
    }
