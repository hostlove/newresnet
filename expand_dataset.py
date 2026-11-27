"""Predict grades for new image-mask pairs and extend a patient split manifest."""

import argparse
import csv
import hashlib
import io
import re
import shutil
import tempfile
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

from filelock import FileLock, Timeout

from dataset_manifest import SPLIT_COLUMNS, read_manifest
from paired_data import GRADE_NAMES, SPLITS, load_partitions, prediction_samples
from split_dataset import _patient_ids
from workflow import (
    add_runtime_args, atomic_output, inference_model, make_loader, prediction_rows, select_device,
)


DEFAULT_PATIENT_REGEX = r"^[^_]+_([A-Za-z]{2}[0-9]{12})_[0-9]+$"


def _read_manifest(root, manifest):
    original = manifest.read_bytes()
    parsed = list(read_manifest(io.StringIO(original.decode("utf-8-sig"), newline=""), manifest))
    rows = [row.csv_row() for row in parsed]
    present = {row.split for row in parsed}
    partitions = load_partitions(root, tuple(split for split in SPLITS if split in present), split_csv=manifest)
    return original, rows, partitions


def _image_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.digest()


def _new_samples(samples, patient_ids, partitions, skipped):
    held_out = {sample.patient_id for split in ("val", "test") for sample in partitions.get(split, ())}
    eligible = [sample for sample in samples if patient_ids[sample.key] not in held_out]
    skipped["held_out_patient"] = len(samples) - len(eligible)
    if not eligible:
        return []

    sizes = {sample.image.stat().st_size for sample in eligible}
    seen = {
        _image_digest(sample.image)
        for existing in partitions.values()
        for sample in existing
        if sample.image.stat().st_size in sizes
    }
    candidates = []
    for sample in eligible:
        digest = _image_digest(sample.image)
        if digest in seen:
            skipped["duplicate_image"] += 1
            continue
        seen.add(digest)
        candidates.append(sample)
    return candidates


def _predict(args, candidates, skipped):
    if not candidates:
        return []
    device = select_device(args.device)
    model, settings = inference_model(args.checkpoint, device)
    loader = make_loader(candidates, settings, args.batch_size, args.workers, device)
    by_path = {sample.key: sample for sample in candidates}
    accepted = []
    for key, grade, _, *probabilities in prediction_rows(model, loader, device):
        confidence = probabilities[grade]
        if confidence < args.min_confidence:
            skipped["low_confidence"] += 1
            continue
        runner_up = max(probabilities[:grade] + probabilities[grade + 1:])
        if confidence - runner_up < args.min_margin:
            skipped["low_margin"] += 1
            continue
        accepted.append((by_path[key], grade))
    return accepted


def _append_manifest(root, pseudo_root, manifest, original, rows, output, accepted, patient_ids):
    with ExitStack() as rollback:
        additions = []
        if accepted:
            pseudo_root.mkdir(parents=True, exist_ok=True)
            batch = Path(tempfile.mkdtemp(prefix="batch-", dir=pseudo_root)).resolve()
            if batch.parent != pseudo_root:
                raise ValueError(f"批次目录不在指定的伪标签目录下：{batch}")
            # Nothing in this new directory is referenced until the CSV is committed.
            rollback.callback(shutil.rmtree, batch)
            for sample, grade in accepted:
                folder = batch / Path(sample.key).parent / str(grade)
                folder.mkdir(parents=True, exist_ok=True)
                image = folder / sample.image.name
                shutil.copy2(sample.image, image)
                shutil.copy2(sample.mask, folder / sample.mask.name)
                additions.append((image.relative_to(root).as_posix(), patient_ids[sample.key], "train", grade, None))

        if manifest.read_bytes() != original:
            raise ValueError(f"运行期间划分表已更新，请用最新清单重新运行：{manifest}")
        if output != manifest and output.exists():
            raise ValueError(f"{output} 已存在，请将它用作 --split-csv 或指定新的 --output")
        if additions or output != manifest:
            with atomic_output(output) as stream:
                writer = csv.writer(stream)
                writer.writerow(SPLIT_COLUMNS)
                writer.writerows(rows)
                writer.writerows(additions)
        rollback.pop_all()


def _expand(args):
    root = args.data_root.resolve()
    if not root.is_dir():
        raise ValueError(f"数据根目录不存在：{root}")
    manifest = args.split_csv.resolve()
    output = (args.output or root / "splits_expanded.csv").resolve()
    if output.suffix.lower() != ".csv":
        raise ValueError("--output 需要使用 .csv 文件")
    if output != manifest and output.exists():
        raise ValueError(f"{output} 已存在，请将它用作 --split-csv 或指定新的 --output")

    pseudo_root = (root / "pseudo_labeled").resolve()
    if pseudo_root == root or not pseudo_root.is_relative_to(root):
        raise ValueError("伪标签目录必须位于数据根目录内")
    source = args.input.resolve()
    if pseudo_root.is_relative_to(source) or source.is_relative_to(pseudo_root):
        raise ValueError("--input 请指定待推理原片或新数据目录")
    for option, value in (("--min-confidence", args.min_confidence), ("--min-margin", args.min_margin)):
        if not 0 <= value <= 1:
            raise ValueError(f"{option} 需要在 0 到 1 之间")

    with FileLock(root / ".expand_dataset.lock", timeout=0):
        original, rows, partitions = _read_manifest(root, manifest)
        incoming = prediction_samples(source, require_png_pairs=True)
        patient_ids = _patient_ids(args, incoming)
        skipped = Counter()
        candidates = _new_samples(incoming, patient_ids, partitions, skipped)
        accepted = _predict(args, candidates, skipped)
        _append_manifest(root, pseudo_root, manifest, original, rows, output, accepted, patient_ids)

    counts = Counter(grade for _, grade in accepted)
    print(f"min_confidence={args.min_confidence:g}  min_margin={args.min_margin:g}")
    print(f"input_pairs={len(incoming)}  added_train={len(accepted)}  "
          f"held_out_patient={skipped['held_out_patient']}  duplicate_image={skipped['duplicate_image']}  "
          f"low_confidence={skipped['low_confidence']}  low_margin={skipped['low_margin']}")
    print(f"added_grade_counts={[counts[grade] for grade in range(len(GRADE_NAMES))]}  "
          f"total_rows={len(rows) + len(accepted)}")
    print(f"split_csv={output}")


def main():
    parser = argparse.ArgumentParser(description="推理新原片与掩膜，按预测等级归档并追加训练清单")
    parser.add_argument("--input", type=Path, required=True, help="待推理的原片文件或原片与掩膜配对目录")
    parser.add_argument("--data-root", type=Path, required=True, help="当前训练使用的数据根目录")
    parser.add_argument("--split-csv", type=Path, required=True, help="当前的 image_path,patient_id,split 划分清单")
    parser.add_argument("--checkpoint", type=Path, required=True, help="已训练的 best.pt")
    parser.add_argument("--output", type=Path, help="合并后的 CSV，默认写到数据根目录下的 splits_expanded.csv")
    parser.add_argument("--min-confidence", type=float, default=0.60,
                        help="最高类别概率下限，默认 %(default)s")
    parser.add_argument("--min-margin", type=float, default=0.30,
                        help="最高与次高类别概率差的下限，默认 %(default)s")
    patients = parser.add_mutually_exclusive_group()
    patients.add_argument("--patient-regex", help="提取患者编号的正则，包含一个捕获组")
    patients.add_argument("--patient-csv", type=Path, help="新样本的 image_path,patient_id 表，路径相对于 --input")
    add_runtime_args(parser)
    args = parser.parse_args()
    if args.patient_regex is None and args.patient_csv is None:
        args.patient_regex = DEFAULT_PATIENT_REGEX
    try:
        _expand(args)
    except (OSError, ValueError, RuntimeError, Timeout, re.error) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
