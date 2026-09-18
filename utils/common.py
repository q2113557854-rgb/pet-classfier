"""通用工具：随机种子、设备选择、matplotlib 中文字体配置。

说明：本文件为指南目录结构的最小补充（共用工具），其余模块严格按指南划分。
"""
from __future__ import annotations

import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """固定所有随机源，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """优先使用 CUDA GPU。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_cjk_font() -> bool:
    """配置 matplotlib 中文字体（Windows 用微软雅黑/黑体）。

    返回 True 表示中文字体可用；返回 False 时调用方应改用英文标签，
    避免出现 DejaVu Sans 缺字导致的 "□□" 乱码（Day2 曾踩坑，见报告第 4 节）。
    """
    candidates = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
        "WenQuanYi Zen Hei", "PingFang SC", "Arial Unicode MS",
    ]
    available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False
