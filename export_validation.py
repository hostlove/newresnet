"""Export one audit row per validation image for a single checkpoint."""

import argparse
import csv
from pathlib import Path

from model import DualResNetClassifier
from paired_data import GRADE_NAMES, load_partitions
from workflow import PREDICTION_COLUMNS, add_runtime_args, make_loader, prediction_rows, read_checkpoint, select_device


AUDIT_COLUMNS = (
    "checkpoint_epoch", "patient_id", "image_path", "true_grade", "true_label",
    *PREDICTION_COLUMNS[1:], "grade_error",
)


def main():
    parser = argparse.ArgumentParser(description="用固定断点导出验证集的逐片抽查表")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, required=True, help="原实验使用的患者划分清单")
    parser.add_argument("--checkpoint", type=Path, required=True, help="固定的 best.pt 或其副本")
    parser.add_argument("--output", type=Path, required=True, help="新建的抽查 CSV 路径")
    add_runtime_args(parser)
    args = parser.parse_args()
    if args.output.suffix.lower() != ".csv":
        parser.error("--output must use a .csv file")
    if args.output.exists():
        parser.error(f"output already exists; choose a new CSV path: {args.output}")

    try:
        device = select_device(args.device)
        # samples = load_partitions(args.data_root, ("val",), split_csv=args.split_csv)["val"]
        # samples = load_partitions(args.data_root, ("train",), split_csv=args.split_csv)["train"]
        samples = load_partitions(args.data_root, ("test",), split_csv=args.split_csv)["test"]
        samples_by_path = {sample.key: sample for sample in samples}

        checkpoint = read_checkpoint(args.checkpoint)
        epoch = checkpoint.get("epoch")
        if not isinstance(epoch, int) or epoch < 1:
            raise ValueError(f"checkpoint has no valid training epoch: {args.checkpoint}")
        model = DualResNetClassifier(num_classes=len(GRADE_NAMES))
        model.load_state_dict(checkpoint["model"], strict=True)
        loader = make_loader(samples, checkpoint["settings"], args.batch_size, args.workers, device)
        del checkpoint
        model.to(device).eval()

        rows = []
        for image_path, predicted_grade, predicted_label, *probabilities in prediction_rows(model, loader, device):
            sample = samples_by_path[image_path]
            rows.append((
                epoch, sample.patient_id, image_path, sample.grade, GRADE_NAMES[sample.grade],
                predicted_grade, predicted_label, *probabilities, abs(predicted_grade - sample.grade),
            ))

        args.output.parent.mkdir(parents=True, exist_ok=True)
        stream = args.output.open("x", encoding="utf-8-sig", newline="")
        try:
            with stream:
                writer = csv.writer(stream)
                writer.writerow(AUDIT_COLUMNS)
                writer.writerows(rows)
        except BaseException:
            args.output.unlink(missing_ok=True)
            raise

        print(f"checkpoint={args.checkpoint}  epoch={epoch}  split=val  images={len(rows)}")
        print(f"wrote validation audit CSV to {args.output}")
    except (ValueError, OSError, RuntimeError, FloatingPointError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
