"""Predict the grade of one paired cervical radiograph."""

import argparse
import json
from pathlib import Path

from paired_data import prediction_samples
from workflow import PREDICTION_COLUMNS, inference_model, make_loader, prediction_rows, select_device


def main():
    parser = argparse.ArgumentParser(description="预测一张原片的等级，自动配对同目录的 _seg.png")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    try:
        if not args.image.is_file():
            raise ValueError(f"original image not found: {args.image}")
        samples = prediction_samples(args.image)
        device = select_device(args.device)
        model, settings = inference_model(args.checkpoint, device)
        loader = make_loader(samples, settings, 1, 0, device)
        result = dict(zip(PREDICTION_COLUMNS, next(prediction_rows(model, loader, device))))
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, OSError, RuntimeError, FloatingPointError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
