"""数据加载、70/15/15 分层划分与 Transform。

设计要点（对应考核指南 Day1 检查点）：
1. 训练集：Resize(256) + RandomCrop(224) + RandomHorizontalFlip + Normalize；
2. 验证集 / 测试集：Resize(256) + CenterCrop(224) + Normalize，
   严禁混入随机翻转 / 随机裁剪，避免验证测试环境引入随机干扰（数据泄漏）；
3. 固定随机种子 seed=42，保证划分可复现。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# torchvision 兼容：新版本将 OxfordIIITPet 改名为 OxfordPets（旧名可能被移除）
try:
    _PetDataset = datasets.OxfordIIITPet
except AttributeError:  # torchvision >= 0.20 部分版本
    _PetDataset = datasets.OxfordPets

# Oxford-IIIT Pet 官方 37 个品种（字母序）。正常会从 list.txt 动态解析，此处仅作兜底。
_CANONICAL_BREEDS: List[str] = [
    "Abyssinian", "american_bulldog", "american_pit_bull_terrier",
    "basset_hound", "beagle", "bengal", "birman", "bombay", "boxer",
    "british_shorthair", "chihuahua", "egyptian_mau", "english_cocker_spaniel",
    "english_setter", "german_shorthaired", "great_pyrenees", "havanese",
    "japanese_chin", "keeshond", "leonberger", "maine_coon",
    "miniature_pinscher", "newfoundland", "persian", "pomeranian", "pug",
    "ragdoll", "russian_blue", "saint_bernard", "samoyed", "scottish_terrier",
    "shiba_inu", "siamese", "sphynx", "staffordshire_bull_terrier",
    "wheaten_terrier", "yorkshire_terrier",
]


def get_transform(train: bool = True) -> transforms.Compose:
    """返回训练 / 评估用的 Transform（评估集固定 CenterCrop，无随机增强）。"""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    if train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])


def _parse_breeds_from_list_txt(data_root: str) -> List[str] | None:
    """从官方 annotations/list.txt 解析品种名（按字母序），保证与 label 编号一致。"""
    candidates = [
        Path(data_root) / "oxford-iiit-pet" / "annotations" / "list.txt",
        Path(data_root) / "list.txt",
    ]
    for path in candidates:
        if path.exists():
            breeds: set[str] = set()
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    # 图片名形如 "Abyssinian_1" / "american_bulldog_101"
                    breeds.add(re.sub(r"_\d+$", "", parts[0]))
            return sorted(breeds)
    return None


def get_class_names(data_root: str) -> List[str]:
    """返回 37 个品种名列表（与数据集 label 编号一一对应）。"""
    breeds = _parse_breeds_from_list_txt(data_root)
    if breeds is not None:
        if len(breeds) == 37:
            return breeds
        print(f"[warn] list.txt 解析出 {len(breeds)} 个品种（应为 37），回退到内置列表")
    return list(_CANONICAL_BREEDS)


class _TransformedSubset(Subset):
    """对 Subset 施加（在读取时）Transform，避免"训练集增强泄漏到验证集"的问题。"""

    def __init__(self, dataset, indices, transform=None):
        super().__init__(dataset, indices)
        self.transform = transform

    def __getitem__(self, idx: int):
        img, label = super().__getitem__(idx)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __getitems__(self, indices):
        # 新版 PyTorch DataLoader 批量取数走 __getitems__；必须逐元素应用 transform
        return [self.__getitem__(idx) for idx in indices]


def build_datasets(
    data_root: str = "datasets",
    val_ratio: float = 0.15,
    split_mode: str = "70_15_15",
    seed: int = 42,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset, List[str]]:
    """构建 (train, val, test) 三个数据集与类别名。

    划分协议（指南要求 70% / 15% / 15% 分层抽样）：
    - split_mode="70_15_15"：对官方 trainval(3680) 做 70/15/15 分层划分（2576/552/552）；
    - split_mode="official_test"：对 trainval 做 85/15 划分（3128/552），测试集用官方 test(3669)，
      与官方数据集设计一致（Day1 十轮版本采用的协议）。
    """
    train_transform = get_transform(train=True)
    eval_transform = get_transform(train=False)

    # 先以 transform=None 加载原始数据集，再按子集分别施加 Transform
    trainval_raw = _PetDataset(
        root=data_root, split="trainval", download=True, transform=None
    )
    n = len(trainval_raw)
    labels = np.array([trainval_raw[i][1] for i in range(n)], dtype=np.int64)
    indices = np.arange(n)

    if split_mode == "official_test":
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        train_idx, val_idx = next(sss.split(indices, labels))
        train_idx, val_idx = train_idx.tolist(), val_idx.tolist()

        test_raw = _PetDataset(
            root=data_root, split="test", download=True, transform=None
        )
        test_ds = _TransformedSubset(test_raw, list(range(len(test_raw))), eval_transform)
    else:
        # 70 / 15 / 15：先从 trainval 分出 70% 训练，剩余 30% 再等分给验证与测试
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
        train_idx, rest_idx = next(sss1.split(indices, labels))
        rest_labels = labels[rest_idx]
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        val_idx, test_idx = next(sss2.split(rest_idx, rest_labels))
        train_idx, val_idx, test_idx = (
            train_idx.tolist(), val_idx.tolist(), test_idx.tolist(),
        )

        test_ds = _TransformedSubset(trainval_raw, test_idx, eval_transform)

    train_ds = _TransformedSubset(trainval_raw, train_idx, train_transform)
    val_ds = _TransformedSubset(trainval_raw, val_idx, eval_transform)

    class_names = get_class_names(data_root)
    return train_ds, val_ds, test_ds, class_names


def build_loaders(
    train_ds,
    val_ds,
    test_ds,
    batch_size: int = 32,
    num_workers: int = 2,
):
    """构建 DataLoader：训练集 shuffle=True，验证/测试集 shuffle=False。

    允许传入 None（只构建需要的 loader，例如评估脚本只需测试集）。
    """
    def _loader(ds, shuffle: bool):
        if ds is None:
            return None
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=True,
        )

    train_loader = _loader(train_ds, shuffle=True)
    val_loader = _loader(val_ds, shuffle=False)
    test_loader = _loader(test_ds, shuffle=False)
    return train_loader, val_loader, test_loader
