"""Write paired cervical radiograph predictions to a CSV."""

import argparse
import csv
from pathlib import Path

from paired_data import prediction_samples
from workflow import (
    PREDICTION_COLUMNS, add_runtime_args, atomic_output, inference_model, make_loader,
    prediction_rows, select_device,
)


def main():
    parser = argparse.ArgumentParser(description="批量预测原片和分割掩膜对")
    parser.add_argument("--input", type=Path, required=True, help="原片文件或配对目录")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    add_runtime_args(parser)
    args = parser.parse_args()
    if args.output.suffix.lower() != ".csv":
        parser.error("--output must use a .csv file")
    try:
        samples = prediction_samples(args.input)
        device = select_device(args.device)
        model, settings = inference_model(args.checkpoint, device)
        loader = make_loader(samples, settings, args.batch_size, args.workers, device)
        with atomic_output(args.output) as stream:
            writer = csv.writer(stream)
            writer.writerow(PREDICTION_COLUMNS)
            writer.writerows(prediction_rows(model, loader, device))
        print(f"wrote {len(samples)} predictions to {args.output}")
    except (ValueError, OSError, RuntimeError, FloatingPointError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
