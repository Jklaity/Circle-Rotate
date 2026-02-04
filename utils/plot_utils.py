"""绑图工具函数"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def setup_plot_style():
    """设置绘图样式"""
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Liberation Serif', 'Times New Roman']
    plt.rcParams['font.size'] = 10


def save_figure(fig, output_path, dpi=300):
    """保存图片"""
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0.01)
    plt.close(fig)


GT_COLOR = '#d62728'    # 红色
OURS_COLOR = '#1f77b4'  # 蓝色
