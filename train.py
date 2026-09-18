"""训练主入口：支持 argparse 参数控制（epochs / lr / label_smoothing / freeze_backbone 等）。

用法示例：
    python train.py --epochs 10 --run_name baseline
    python train.py --epochs 10 --label_smoothing 0.1 --run_name label_smooth0.1
    python train.py --epochs 10 --freeze_backbone --run_name freeze_backbone
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from data.dataset import build_datasets, build_loaders
from models.model import build_model, get_optimizer
from utils.common import get_device, set_seed
from utils.metrics import macro_f1, topk_accuracy


def parse_args():
    parser = argparse.ArgumentParser(description="Oxford-IIIT Pet 细粒度分类训练")
    parser.add_argument("--data_root", type=str, default="datasets", help="数据集根目录")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4, help="AdamW 默认 1e-4")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="消融1：标签平滑系数")
    parser.add_argument("--freeze_backbone", action="store_true", help="消融2：冻结骨干只训 FC")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--split_mode", type=str, default="70_15_15",
                        choices=["70_15_15", "official_test"], help="70/15/15 分层划分或官方 test 划分")
    parser.add_argument("--run_name", type=str, default="", help="实验名（默认按配置自动生成）")
    parser.add_argument("--output_root", type=str, default=".", help="checkpoints/runs/outputs 的父目录")
    parser.add_argument("--num_classes", type=int, default=37)
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int, topk=(1, 5)):
    """在验证/测试集上计算 Top-1/Top-5 与 Macro-F1。"""
    model.eval()
    all_preds, all_labels = [], []
    top1_sum = top5_sum = 0.0
    n_total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        t1, t5 = topk_accuracy(outputs, labels, topk=topk)
        top1_sum += t1 * images.size(0)
        top5_sum += t5 * images.size(0)
        n_total += images.size(0)
        all_preds.append(outputs.argmax(dim=1).cpu())
        all_labels.append(labels.cpu())
    top1 = top1_sum / n_total
    top5 = top5_sum / n_total
    f1 = macro_f1(
        torch.cat(all_labels).numpy(), torch.cat(all_preds).numpy()
    )
    return top1, top5, f1


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()                      # 每个 batch 必须清空上一步梯度
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        n += images.size(0)
    # 统一返回百分数（0~100），与验证集 va_acc 保持一致
    return total_loss / n, correct / n * 100.0


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"使用设备: {device}")

    # 实验名：未指定时按配置自动生成
    if not args.run_name:
        if args.freeze_backbone:
            args.run_name = "freeze_backbone"
        elif args.label_smoothing > 0:
            args.run_name = f"label_smooth{args.label_smoothing:g}"
        else:
            args.run_name = "baseline"
    print(f"实验: {args.run_name} | split_mode={args.split_mode} | epochs={args.epochs}")

    # 数据
    train_ds, val_ds, test_ds, class_names = build_datasets(
        data_root=args.data_root, split_mode=args.split_mode, seed=args.seed
    )
    train_loader, val_loader, test_loader = build_loaders(
        train_ds, val_ds, test_ds, args.batch_size, args.num_workers
    )
    imgs, _ = next(iter(train_loader))
    print(f"Batch shape: {tuple(imgs.shape)}")     # Day1 检查点: [32, 3, 224, 224]
    print(f"训练集: {len(train_ds)} 验证集: {len(val_ds)} 测试集: {len(test_ds)}")

    # 模型 / 损失 / 优化器
    model = build_model(args.num_classes, freeze_backbone=args.freeze_backbone).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = get_optimizer(model, args.optimizer, args.lr, args.weight_decay, args.momentum)

    # 输出目录
    output_root = Path(args.output_root)
    ckpt_dir = output_root / "checkpoints" / args.run_name
    log_dir = output_root / "runs" / args.run_name
    csv_dir = output_root / "outputs" / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    best_val_acc, best_epoch = 0.0, -1
    csv_path = csv_dir / f"{args.run_name}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["epoch", "tr_loss", "tr_acc", "va_loss", "va_acc"])

    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        # 验证集损失
        model.eval()
        total_loss, n = 0.0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                total_loss += criterion(outputs, labels).item() * images.size(0)
                n += images.size(0)
        va_loss = total_loss / n
        va_acc = evaluate(model, val_loader, device, args.num_classes)[0]

        # TensorBoard + CSV 记录（Loss 原始值；Acc 统一为百分数 0~100）
        writer.add_scalar("Loss/train", tr_loss, epoch)
        writer.add_scalar("Loss/val", va_loss, epoch)
        writer.add_scalar("Acc/train", tr_acc, epoch)
        writer.add_scalar("Acc/val", va_acc, epoch)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, f"{tr_loss:.6f}", f"{tr_acc:.6f}", f"{va_loss:.6f}", f"{va_acc:.6f}"])

        marker = ""
        if va_acc > best_val_acc:
            best_val_acc, best_epoch = va_acc, epoch
            torch.save(
                {"state_dict": model.state_dict(),
                 "config": {"num_classes": args.num_classes,
                            "freeze_backbone": args.freeze_backbone,
                            "label_smoothing": args.label_smoothing,
                            "split_mode": args.split_mode,
                            "seed": args.seed,
                            "run_name": args.run_name}},
                ckpt_dir / "best_model.pth",
            )
            marker = " ✅ 保存最优模型"
        print(f"ep {epoch:2d} | tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} | "
              f"va_loss={va_loss:.4f} va_acc={va_acc:.4f}{marker}")

    writer.close()
    elapsed = time.time() - t_start
    print(f"\n训练结束！耗时 {elapsed/60:.1f} 分钟，TensorBoard 日志: {log_dir}")

    # 用最优模型在测试集上评估
    ckpt = torch.load(ckpt_dir / "best_model.pth", map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    test_top1, test_top5, test_f1 = evaluate(model, test_loader, device, args.num_classes)
    print(f"最佳验证准确率: {best_val_acc:.4f}（第 {best_epoch} 轮）")
    print(f"测试集 Top-1: {test_top1:.4f} | Top-5: {test_top5:.4f} | Macro-F1: {test_f1:.4f}")

    summary = {
        "run_name": args.run_name, "best_val_acc": best_val_acc, "best_epoch": best_epoch,
        "test_top1": test_top1, "test_top5": test_top5, "test_macro_f1": test_f1,
        "epochs": args.epochs, "label_smoothing": args.label_smoothing,
        "freeze_backbone": args.freeze_backbone, "split_mode": args.split_mode,
        "seed": args.seed, "elapsed_min": round(elapsed / 60, 2),
    }
    summary_path = output_root / "outputs" / f"{args.run_name}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] 实验汇总 -> {summary_path}")


if __name__ == "__main__":
    main()
