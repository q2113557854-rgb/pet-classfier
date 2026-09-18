"""Grad-CAM 可视化工具（基于 ResNet-18 layer4 最后一层卷积输出的梯度）。"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from utils.common import setup_cjk_font


class GradCAM:
    """简化版 Grad-CAM：对 target_layer 输出做前向/反向钩子，生成热力图。"""

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model.eval()
        # ResNet-18 默认取 layer4 的最后一个 BasicBlock（输出 [B, 512, 7, 7]）
        self.target_layer = target_layer if target_layer is not None else model.layer4[-1]
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._fh = self.target_layer.register_forward_hook(self._save_activation)
        self._bh = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """输入 [1,3,224,224]，返回归一化到 [0,1] 的 224x224 热力图。"""
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1.0
        output.backward(gradient=one_hot)

        # 通道权重 = 梯度的全局平均池化；CAM = ReLU(Σ w_c * A_c)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)          # [1,512,1,1]
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))  # [1,1,7,7]
        cam = F.interpolate(
            cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def remove(self) -> None:
        self._fh.remove()
        self._bh.remove()


def _denormalize(tensor: torch.Tensor) -> np.ndarray:
    """将 ImageNet 归一化后的张量还原为 0-255 的 RGB 数组。"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def visualize_trio(
    model: nn.Module,
    image_tensor: torch.Tensor,
    cam: np.ndarray,
    true_label: int,
    pred_label: int,
    class_names,
    save_path: str,
    title_prefix: str = "",
) -> None:
    """保存"原图 / 热力图 / 叠加图"三栏对比图。"""
    setup_cjk_font()
    cjk_ok = setup_cjk_font()
    if cjk_ok:
        true_str, pred_str = class_names[true_label], class_names[pred_label]
        verdict = "正确" if true_label == pred_label else "错误"
        title = f"{title_prefix}{verdict} | True: {true_str} | Pred: {pred_str}"
    else:
        true_str, pred_str = class_names[true_label], class_names[pred_label]
        verdict = "correct" if true_label == pred_label else "wrong"
        title = f"{title_prefix}{verdict} | True: {true_str} | Pred: {pred_str}"

    original = _denormalize(image_tensor.squeeze(0))

    # 热力图渲染（jet colormap）
    heatmap = plt.cm.jet(cam)[:, :, :3] * 255.0

    # 叠加图：热力图 50% 透明度混合
    overlay = (0.5 * original + 0.5 * heatmap).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    axes[0].imshow(original)
    axes[0].set_title("Original" if not cjk_ok else "原图")
    axes[0].axis("off")
    axes[1].imshow(heatmap.astype(np.uint8), cmap="jet")
    axes[1].set_title("Grad-CAM" if not cjk_ok else "Grad-CAM 热力图")
    axes[1].axis("off")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay" if not cjk_ok else "叠加")
    axes[2].axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] Grad-CAM 三栏图 -> {save_path}")
