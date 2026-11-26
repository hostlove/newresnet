"""Evaluate a trained classifier on a labeled validation or test split."""

import argparse
import json
from pathlib import Path

from paired_data import GRADE_NAMES, load_partitions
from workflow import add_runtime_args, inference_model, make_loader, run_epoch, select_device, write_json


def main():
    parser = argparse.ArgumentParser(description="评估颈椎四级分类的验证或测试结果")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    add_runtime_args(parser)
    args = parser.parse_args()
    if args.output is not None and args.output.suffix.lower() != ".json":
        parser.error("--output must use a .json file")
    try:
        device = select_device(args.device)
        samples = load_partitions(args.data_root, (args.split,), split_csv=args.split_csv)[args.split]
        model, settings = inference_model(args.checkpoint, device)
        loader = make_loader(samples, settings, args.batch_size, args.workers, device)
        metrics = {"split": args.split, "grades": list(GRADE_NAMES), **run_epoch(model, loader, device)}
        if args.output is not None:
            write_json(args.output, metrics)
        print(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, OSError, RuntimeError, FloatingPointError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
