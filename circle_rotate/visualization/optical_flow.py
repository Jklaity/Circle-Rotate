"""
Optical Flow Visualization

Generates optical flow heatmap between video frames.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def extract_frames(video_path, frame_indices=None):
    """Extract specific frames from video"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_indices is None:
        frame_indices = [0, total_frames - 1]

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            frames.append(None)
    cap.release()
    return frames


def compute_optical_flow(frame1, frame2):
    """Compute optical flow magnitude between two frames"""
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    return mag, flow


def visualize_flow(flow_mag, output_path, title="Optical Flow", vmax=50):
    """Generate optical flow heatmap visualization"""
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(flow_mag, cmap='jet', vmin=0, vmax=vmax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(f"Mean Flow: {flow_mag.mean():.2f}", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Flow Magnitude', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return output_path
