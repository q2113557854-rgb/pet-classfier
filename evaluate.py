"""评估与可视化独立脚本：测试集指标、混淆矩阵、Grad-CAM、训练曲线。

用法示例：
    python evaluate.py --checkpoint checkpoints/baseline/best_model.pth
    python evaluate.py --checkpoint checkpoints/baseline/best_model.pth --curves_csv outputs/logs/baseline.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data.dataset import build_datasets, build_loaders, get_class_names
from models.model import build_model
from utils.common import get_device, set_seed
from utils.gradcam import GradCAM, visualize_trio
from utils.metrics import (macro_f1, plot_confusion_matrix, plot_curves_from_csv,
                           topk_accuracy)


def parse_args():
    parser = argparse.ArgumentParser(description="评估与可视化")
    parser.add_argument("--checkpoint", type=str, required=True, help="best_model.pth 路径")
    parser.add_argument("--data_root", type=str, default="datasets")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_mode", type=str, default=None,
                        choices=["70_15_15", "official_test"],
                        help="默认读取 checkpoint 中保存的配置")
    parser.add_argument("--output_dir", type=str, default="outputs", help="可视化产物输出目录")
    parser.add_argument("--curves_csv", type=str, default="", help="可选：传入训练日志 CSV 绘制曲线")
    return parser.parse_args()


@torch.no_grad()
def full_eval(model, test_loader, device, num_classes):
    """返回 y_true / y_pred 与 Top-1/Top-5/Macro-F1。"""
    model.eval()
    preds, labels = [], []
    top1_sum = top5_sum = 0.0
    n = 0
    for images, targets in test_loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        t1, t5 = topk_accuracy(outputs, targets)
        top1_sum += t1 * images.size(0)
        top5_sum += t5 * images.size(0)
        n += images.size(0)
        preds.append(outputs.argmax(dim=1).cpu())
        labels.append(targets.cpu())
    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(labels).numpy()
    return (y_true, y_pred, top1_sum / n, top5_sum / n,
            macro_f1(y_true, y_pred))


@torch.no_grad()
def find_correct_wrong(model, test_loader, device):
    """从测试集中各找一张：预测正确的图与预测错误的图。"""
    model.eval()
    correct_sample = wrong_sample = None
    for images, targets in test_loader:
        images, targets = images.to(device), targets.to(device)
        preds = model(images).argmax(dim=1)
        for i in range(images.size(0)):
            img = images[i : i + 1]
            t, p = int(targets[i].item()), int(preds[i].item())
            if t == p and correct_sample is None:
                correct_sample = (img, t, p)
            elif t != p and wrong_sample is None:
                wrong_sample = (img, t, p)
            if correct_sample is not None and wrong_sample is not None:
                return correct_sample, wrong_sample
    return correct_sample, wrong_sample


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"使用设备: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("config", {})
    num_classes = cfg.get("num_classes", 37)
    freeze_backbone = cfg.get("freeze_backbone", False)
    split_mode = args.split_mode or cfg.get("split_mode", "70_15_15")
    print(f"加载 checkpoint: {args.checkpoint} | 配置={cfg}")

    _, _, test_ds, class_names = build_datasets(
        data_root=args.data_root, split_mode=split_mode, seed=args.seed
    )
    _, _, test_loader = build_loaders(
        None, None, test_ds, args.batch_size, args.num_workers
    )

    model = build_model(num_classes, freeze_backbone=freeze_backbone).to(device)
    model.load_state_dict(ckpt["state_dict"])

    # 1) 测试集指标 + 混淆矩阵
    y_true, y_pred, top1, top5, f1 = full_eval(model, test_loader, device, num_classes)
    print(f"测试集 Top-1: {top1:.4f} | Top-5: {top5:.4f} | Macro-F1: {f1:.4f}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    top_pairs = plot_confusion_matrix(
        y_true, y_pred, class_names, save_dir=str(out_dir / "evaluate"), top_k=3
    )

    # 2) Grad-CAM：正确 / 错误各一张
    correct_sample, wrong_sample = find_correct_wrong(model, test_loader, device)
    cam = GradCAM(model)
    for sample, suffix, prefix in (
        (correct_sample, "correct", "正确案例 "),
        (wrong_sample, "wrong", "错误案例 "),
    ):
        if sample is None:
            print(f"[warn] 未找到 {suffix} 样本")
            continue
        img, t, p = sample
        heatmap = cam.generate(img.to(device), class_idx=p)
        save_path = out_dir / "evaluate" / f"gradcam_{suffix}.png"
        visualize_trio(
            model, img, heatmap, t, p, class_names,
            save_path=str(save_path), title_prefix=prefix,
        )
    cam.remove()

    # 3) 可选：绘制训练曲线
    if args.curves_csv and Path(args.curves_csv).exists():
        plot_curves_from_csv(
            args.curves_csv,
            str(out_dir / "evaluate" / "curves.png"),
            title=Path(args.curves_csv).stem,
        )

    # 汇总保存
    result = {
        "checkpoint": args.checkpoint, "split_mode": split_mode,
        "test_top1": top1, "test_top5": top5, "test_macro_f1": f1,
        "top_confused_pairs": [
            {"true": class_names[i], "pred": class_names[j], "count": c}
            for i, j, c in top_pairs
        ],
    }
    result_path = out_dir / "evaluate" / "evaluation_summary.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] 评估汇总 -> {result_path}")
    print("\nDay3 评估产物：混淆矩阵全图/局部图、gradcam_correct.png、gradcam_wrong.png 已生成。")


if __name__ == "__main__":
    main()
