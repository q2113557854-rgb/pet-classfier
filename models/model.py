"""模型定义：预训练 ResNet-18 + 37 类分类头替换（支持冻结骨干）。"""
from __future__ import annotations

import torch
from torch import nn
from torchvision import models


def build_model(
    num_classes: int = 37,
    freeze_backbone: bool = False,
    pretrained: bool = True,
) -> nn.Module:
    """构建 ResNet-18 分类模型。

    - 使用 ImageNet 预训练权重（weights="DEFAULT"）；
    - 替换最后一层全连接层，输出 num_classes（默认 37）；
    - freeze_backbone=True 时冻结除 fc 外的全部参数（消融实验用）。
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    in_features = model.fc.in_features  # ResNet-18: 512
    model.fc = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False
        print(f"[info] 已冻结骨干网络，仅训练 fc 层（{in_features} -> {num_classes}）")

    return model


def get_optimizer(
    model: nn.Module,
    optimizer_name: str = "adamw",
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    momentum: float = 0.9,
):
    """构造优化器：AdamW(lr=1e-4, wd=0.01) 或 SGD(lr=1e-3, momentum=0.9)。"""
    if optimizer_name.lower() == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
