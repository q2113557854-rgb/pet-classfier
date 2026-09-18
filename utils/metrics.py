"""评估指标与可视化：Top-1/Top-5 准确率、Macro-F1、混淆矩阵、训练曲线。"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import f1_score

from utils.common import setup_cjk_font


def topk_accuracy(
    output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1, 5)
) -> List[float]:
    """计算 Top-k 准确率（%），返回与 topk 顺序一致的列表。"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            results.append(float(correct_k.mul_(100.0 / batch_size).item()))
        return results


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算 Macro-F1（各类 F1 的算术平均，对类别不平衡更稳健）。"""
    return float(f1_score(y_true, y_pred, average="macro"))


def plot_curves_from_csv(csv_path: str, save_path: str, title: str = "") -> None:
    """根据 train.py 产出的训练日志 CSV 绘制 Loss / Acc 收敛曲线。"""
    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = df["epoch"]

    axes[0].plot(epochs, df["tr_loss"], "o-", label="Train Loss", color="#1f77b4")
    axes[0].plot(epochs, df["va_loss"], "s-", label="Val Loss", color="#d62728")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curve")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, df["tr_acc"], "o-", label="Train Acc", color="#1f77b4")
    axes[1].plot(epochs, df["va_acc"], "s-", label="Val Acc", color="#d62728")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curve")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[save] 训练曲线 -> {save_path}")


def _pretty_name(name: str) -> str:
    """品种名转展示名：american_bulldog -> American Bulldog。"""
    return " ".join(part.capitalize() for part in name.split("_"))


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_dir: str,
    top_k: int = 3,
) -> List[Tuple[int, int, int]]:
    """绘制 37 类混淆矩阵（全图 + 最易混淆区域局部放大图）。

    返回最易混淆的 top_k 对 (真实类 i, 预测类 j, 数量)。
    """
    setup_cjk_font()
    cm = sk_confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 全图
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels([_pretty_name(n) for n in class_names], rotation=90, fontsize=5)
    ax.set_yticklabels([_pretty_name(n) for n in class_names], fontsize=5)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (37 breeds)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=3.5, color="black")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    full_path = save_dir / "confusion_matrix_full.png"
    fig.savefig(full_path, dpi=150)
    plt.close(fig)
    print(f"[save] 混淆矩阵全图 -> {full_path}")

    # 找出最易混淆的 off-diagonal 对
    off = cm.copy()
    np.fill_diagonal(off, 0)
    flat = off.ravel()
    top_indices = np.argsort(flat)[::-1][:top_k]
    top_pairs = []
    for idx in top_indices:
        if flat[idx] == 0:
            continue
        i, j = divmod(int(idx), cm.shape[1])
        top_pairs.append((i, j, int(flat[idx])))
    print("[info] 最易混淆的品种对（真实类 -> 预测类）:")
    for i, j, c in top_pairs:
        print(f"  {_pretty_name(class_names[i])} -> {_pretty_name(class_names[j])}: {c} 张")

    # 局部放大图：取 top 混淆涉及的类别子集
    involved = sorted({i for p in top_pairs for i in (p[0], p[1])})
    sub = cm[np.ix_(involved, involved)]
    fig, ax = plt.subplots(figsize=(max(4, len(involved) * 1.6), max(4, len(involved) * 1.6)))
    ax.imshow(sub, cmap="Blues")
    labels = [_pretty_name(class_names[k]) for k in involved]
    ax.set_xticks(range(len(involved)))
    ax.set_yticks(range(len(involved)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (most confused region)")
    for ii in range(len(involved)):
        for jj in range(len(involved)):
            ax.text(jj, ii, int(sub[ii, jj]), ha="center", va="center", fontsize=10)
    fig.tight_layout()
    zoom_path = save_dir / "confusion_matrix_zoom.png"
    fig.savefig(zoom_path, dpi=150)
    plt.close(fig)
    print(f"[save] 混淆矩阵局部放大 -> {zoom_path}")

    return top_pairs
