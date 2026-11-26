"""Train the paired cervical radiograph classifier."""

import argparse
import hashlib
import json
import math
import random
from itertools import chain
from pathlib import Path

import torch
from filelock import FileLock, Timeout

from model import DualResNetClassifier
from paired_data import GRADE_NAMES, load_partitions
from workflow import (
    CHECKPOINT_FORMAT, add_runtime_args, make_grad_scaler, make_loader, read_checkpoint,
    run_epoch, save_checkpoint, select_device, write_json,
)


DEFAULTS = {
    "batch_size": 8, "image_size": 224, "roi_margin": 0.15, "lr": 3e-4,
    "backbone_lr": None, "weight_decay": 1e-4, "seed": 42, "weight_by_grade": True,
}


def build_parser():
    parser = argparse.ArgumentParser(description="用原片与分割掩膜训练颈椎四级分类模型")
    parser.add_argument("--data-root", type=Path, required=True)
    splits = parser.add_mutually_exclusive_group()
    splits.add_argument("--split-csv", type=Path, help="按患者划分的 image_path,patient_id,split 清单")
    splits.add_argument("--train-only", action="store_true", help="直接读取数据根目录的等级文件夹进行试跑")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/cervical"))
    parser.add_argument("--epochs", type=int, default=30, help="训练总轮数")
    parser.add_argument("--image-size", type=int, help="全图和局部的输入边长，默认 224")
    parser.add_argument("--roi-margin", type=float, help="分割框每侧外扩比例，默认 0.15")
    parser.add_argument("--lr", type=float, help="分类头学习率，默认 3e-4")
    parser.add_argument("--backbone-lr", type=float, help="骨干学习率，预训练微调默认是分类头的 1/10")
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--weight-by-grade", action=argparse.BooleanOptionalAction, default=None,
                        help="按训练分区等级频次加权，默认开启")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=None,
                        help="新训练默认从 TorchVision ImageNet 权重初始化两条骨干")
    parser.add_argument("--global-weights", type=Path, help="本地 TorchVision ResNet34 ImageNet 原始权重")
    parser.add_argument("--local-weights", type=Path, help="本地 TorchVision ResNet18 ImageNet 原始权重")
    parser.add_argument("--resume", action="store_true", help="从输出目录的 last.pt 恢复")
    add_runtime_args(parser, batch_size=None)
    return parser


def _settings(args, previous):
    settings = DEFAULTS.copy()
    if previous is not None:
        settings.update(previous["settings"])
    for key in DEFAULTS:
        value = getattr(args, key)
        if value is not None:
            if previous is not None and value != settings[key]:
                raise ValueError(f"--resume requires the saved {key}={settings[key]}")
            settings[key] = value
    if settings["backbone_lr"] is None:
        has_transfer_weights = args.pretrained is not False or args.global_weights or args.local_weights
        settings["backbone_lr"] = settings["lr"] * (0.1 if has_transfer_weights else 1.0)
    if settings["batch_size"] < 1 or settings["image_size"] < 64:
        raise ValueError("--batch-size must be positive and --image-size must be at least 64")
    if not 0 <= settings["roi_margin"] <= 1:
        raise ValueError("--roi-margin must be between 0 and 1")
    if not 0 <= settings["seed"] < 2**32:
        raise ValueError("--seed must be in [0, 2^32)")
    for name in ("lr", "backbone_lr", "weight_decay"):
        value = settings[name]
        if not math.isfinite(value) or value < 0 or (name != "weight_decay" and value == 0):
            raise ValueError(f"invalid {name}: {value}")
    return settings


def _split_digest(partitions):
    rows = sorted(
        (split, sample.key, sample.mask.name, sample.grade, sample.patient_id)
        for split, samples in partitions.items() for sample in samples
    )
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode("utf-8")).hexdigest()


def _validation_score(metrics):
    return metrics["macro_f1"], -metrics["loss"]


def _write_reports(output, state, history):
    if "val" in history[-1]:
        earlier_scores = (_validation_score(row["val"]) for row in history[:-1])
        previous_best = max(earlier_scores, default=(-1.0, -math.inf))
        if _validation_score(history[-1]["val"]) > previous_best:
            best = {key: state[key] for key in ("format", "grades", "settings", "epoch", "model")}
            save_checkpoint(output / "best.pt", best)
    write_json(output / "history.json", history)


def _train(args, partitions, device):
    output = args.output_dir
    if not args.resume and any((output / name).exists() for name in ("last.pt", "best.pt", "history.json")):
        raise ValueError(f"run outputs already exist in {output}; use --resume or a new directory")
    previous = read_checkpoint(output / "last.pt") if args.resume else None
    digest = _split_digest(partitions)
    if previous is not None and previous.get("split_digest") != digest:
        raise ValueError("--resume requires the same split assignments and relative paths")
    settings = _settings(args, previous)
    history = previous["history"] if previous is not None else []
    start_epoch = history[-1]["epoch"] + 1 if history else 1

    if previous is not None:
        _write_reports(output, previous, history)
    if start_epoch > args.epochs:
        print(f"already completed {start_epoch - 1} epochs")
        return

    torch.manual_seed(settings["seed"])
    auto_pretrained = previous is None and args.pretrained is not False
    if previous is None:
        default_source = "ImageNet" if auto_pretrained else "random"
        print(f"initializing global ResNet34={args.global_weights or default_source} "
              f"local ResNet18={args.local_weights or default_source}", flush=True)
    model = DualResNetClassifier(
        num_classes=len(GRADE_NAMES), pretrained=auto_pretrained,
        global_weights=args.global_weights, local_weights=args.local_weights,
    ).to(device)
    optimizer = torch.optim.AdamW([
        {"params": chain(model.global_backbone.parameters(), model.local_backbone.parameters()),
         "lr": settings["backbone_lr"]},
        {"params": model.classifier.parameters(), "lr": settings["lr"]},
    ], weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    scaler = make_grad_scaler(device)
    if previous is not None:
        model.load_state_dict(previous["model"], strict=True)
        optimizer.load_state_dict(previous["optimizer"])
        scheduler.load_state_dict(previous["scheduler"])
        if previous["scaler"]:
            scaler.load_state_dict(previous["scaler"])
        del previous

    print(f"device={device}  amp={scaler.is_enabled()}  grades={list(GRADE_NAMES)}  output={output}", flush=True)
    train_counts = None
    for name, samples in partitions.items():
        counts = torch.bincount(torch.tensor([sample.grade for sample in samples]), minlength=len(GRADE_NAMES))
        print(f"{name}: {len(samples)} pairs, grade_counts={counts.tolist()}", flush=True)
        if name == "train":
            train_counts = counts.float()
    grade_weights = None
    if settings["weight_by_grade"]:
        factors = train_counts.clamp_min(1).rsqrt().masked_fill(train_counts == 0, 0.0)
        grade_weights = factors / ((train_counts * factors).sum() / train_counts.sum())
        print(f"train_grade_weights={grade_weights.tolist()}", flush=True)
        grade_weights = grade_weights.to(device)

    generator = torch.Generator()
    train_loader = make_loader(partitions["train"], settings, settings["batch_size"], args.workers,
                               device, training=True, generator=generator)
    val_loader = None
    if "val" in partitions:
        val_loader = make_loader(partitions["val"], settings, settings["batch_size"], args.workers, device)
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_seed = settings["seed"] + epoch
        random.seed(epoch_seed)
        torch.manual_seed(epoch_seed)
        generator.manual_seed(epoch_seed)
        row = {
            "epoch": epoch,
            "train": run_epoch(model, train_loader, device, optimizer=optimizer, scaler=scaler,
                               grade_weights=grade_weights),
        }
        if val_loader is not None:
            row["val"] = run_epoch(model, val_loader, device)
            scheduler.step(row["val"]["loss"])
        history.append(row)
        state = {
            "format": CHECKPOINT_FORMAT, "grades": list(GRADE_NAMES), "settings": settings,
            "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "split_digest": digest, "history": history,
        }
        save_checkpoint(output / "last.pt", state)
        _write_reports(output, state, history)
        scores = "  ".join(
            f"{name}_loss={metrics['loss']:.4f} {name}_macro_f1={metrics['macro_f1']:.4f}"
            for name, metrics in row.items() if name != "epoch"
        )
        print(f"epoch={epoch}/{args.epochs}  {scores}", flush=True)


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.epochs < 1 or args.workers < 0:
        parser.error("--epochs must be positive and --workers must be nonnegative")
    if args.resume and (args.pretrained is not None or args.global_weights or args.local_weights):
        parser.error("--resume loads last.pt; use initialization options for a new run")
    for path in (args.global_weights, args.local_weights):
        if path is not None and not path.is_file():
            parser.error(f"pretrained weights not found: {path}")

    try:
        device = select_device(args.device)
        names = ("train",) if args.train_only else ("train", "val")
        partitions = load_partitions(args.data_root, names, split_csv=args.split_csv, train_only=args.train_only)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with FileLock(str(args.output_dir / ".training.lock"), timeout=0):
            _train(args, partitions, device)
    except Timeout:
        parser.exit(2, f"error: another training process is using {args.output_dir}\n")
    except (ValueError, OSError, RuntimeError, FloatingPointError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
