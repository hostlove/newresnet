"""Generate patient IDs from filenames and patient-disjoint splits for cervical grading."""

import argparse
import csv
import math
import random
import re
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path, PurePosixPath


def _read_patient_ids(path, image_paths):
    patient_ids = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"image_path", "patient_id"}.issubset(reader.fieldnames or ()):
            raise ValueError("患者表需要 image_path,patient_id 两列")
        for row in reader:
            location = f"{path}:{reader.line_num}"
            path_text = (row["image_path"] or "").strip().replace("\\", "/")
            image_path = PurePosixPath(path_text).as_posix()
            patient_id = (row["patient_id"] or "").strip()
            if image_path not in image_paths or not patient_id:
                raise ValueError(f"{location} 需要相对于数据根目录的原片路径和非空患者编号")
            if image_path in patient_ids:
                raise ValueError(f"{location} 原片在患者表中重复出现 {image_path}")
            patient_ids[image_path] = patient_id
    missing = sorted(image_paths - patient_ids.keys())
    if missing:
        raise ValueError(f"患者表缺少 {len(missing)} 张原片的编号，例如 {missing[0]}")
    return patient_ids


def _patient_ids(args, cases):
    if args.patient_csv is not None:
        return _read_patient_ids(args.patient_csv, {case.key for case in cases})
    if args.patient_regex is not None:
        pattern = re.compile(args.patient_regex)
        if pattern.groups != 1:
            raise ValueError("--patient-regex 应包含一个捕获患者编号的括号组")
        patient_ids = {}
        for case in cases:
            match = pattern.search(case.image.stem)
            patient_id = (match.group(1) or "").strip() if match else ""
            if not patient_id:
                raise ValueError(f"文件名未匹配到患者编号 {case.key}")
            patient_ids[case.key] = patient_id
        return patient_ids
    return {case.key: f"P{index:06d}" for index, case in enumerate(cases, start=1)}


def _assign_patients(patient_grades, proportions, seed):
    """Assign whole patients, prioritizing rare grades and matching split proportions."""
    splits = tuple(proportions)
    patient_count = len(patient_grades)
    if patient_count < len(splits):
        raise ValueError(f"有 {patient_count} 位患者，至少需要 {len(splits)} 位才能填满所选分区")

    patients_per_grade = Counter(grade for counts in patient_grades.values() for grade in counts)
    images_per_grade = Counter()
    for counts in patient_grades.values():
        images_per_grade.update(counts)

    patients = sorted(patient_grades)
    random.Random(seed).shuffle(patients)
    patients.sort(key=lambda patient: (
        min(patients_per_grade[grade] for grade in patient_grades[patient]),
        -len(patient_grades[patient]),
        -patient_grades[patient].total(),
    ))

    assigned_patients = Counter()
    assigned_grades = {split: Counter() for split in splits}
    assignment = {}

    def cost(split, counts):
        # Score the change in squared share errors, omitting terms common to every split.
        target = proportions[split]
        patient_error = (assigned_patients[split] / patient_count - target) / patient_count
        grade_error = sum(
            (assigned_grades[split][grade] / images_per_grade[grade] - target)
            * count / images_per_grade[grade]
            for grade, count in counts.items()
        )
        return patient_error + grade_error

    for index, patient in enumerate(patients):
        counts = patient_grades[patient]
        empty_splits = tuple(split for split in splits if assigned_patients[split] == 0)
        # Reserve one patient for each empty split when the supply is tight.
        candidates = empty_splits if patient_count - index == len(empty_splits) else splits
        unseen_in_train = any(assigned_grades["train"][grade] == 0 for grade in counts)
        if "train" in candidates and unseen_in_train:
            split = "train"
        else:
            split = min(candidates, key=lambda candidate: cost(candidate, counts))
        assignment[patient] = split
        assigned_patients[split] += 1
        assigned_grades[split].update(counts)
    return assignment


def _write_csv(path, columns, rows, rollback):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        rollback.callback(path.unlink, missing_ok=True)
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


def generate_splits(args):
    from cervical.data import GRADE_NAMES, SPLITS, load_splits

    train_ratio, val_ratio, test_ratio = args.ratios
    total_ratio = sum(args.ratios)
    if not math.isfinite(total_ratio) or train_ratio <= 0 or val_ratio <= 0 or test_ratio < 0:
        raise ValueError("--ratios 需要为训练和验证指定正数，测试可以设为 0")
    proportions = {split: ratio / total_ratio for split, ratio in zip(SPLITS, args.ratios) if ratio > 0}

    root = args.data_root.resolve()
    cases = load_splits(root, ("train",), train_only=True)["train"]
    cases.sort(key=lambda case: case.key)
    patient_ids = _patient_ids(args, cases)
    patient_grades = defaultdict(Counter)
    for case in cases:
        patient_grades[patient_ids[case.key]][case.grade] += 1
    assignment = _assign_patients(patient_grades, proportions, args.seed)

    output = (args.output or root / "splits.csv").absolute()
    if output.exists():
        raise ValueError(f"{output} 已存在，请用 --output 指定新文件")
    patient_output = None
    if args.patient_csv is None:
        patient_output = (args.patients_output or root / "patients.csv").absolute()
        if patient_output.resolve() == output.resolve():
            raise ValueError("患者表和划分表需要使用不同的输出路径")

    patient_rows = sorted(patient_ids.items())
    with ExitStack() as rollback:
        if patient_output is not None:
            if patient_output.exists():
                existing = _read_patient_ids(patient_output, patient_ids.keys())
                if existing != patient_ids:
                    raise ValueError(f"{patient_output} 的患者编号与本次提取结果不同，请用 --patients-output 指定新文件")
            else:
                _write_csv(patient_output, ("image_path", "patient_id"), patient_rows, rollback)

        split_rows = ((image_path, patient_id, assignment[patient_id]) for image_path, patient_id in patient_rows)
        _write_csv(output, ("image_path", "patient_id", "split"), split_rows, rollback)
        partitions = load_splits(root, tuple(proportions), split_csv=output)
        rollback.pop_all()

    if patient_output is not None:
        print(f"患者表 {patient_output}")
    print(f"已写入 {output}")
    print(f"等级顺序 {', '.join(GRADE_NAMES)}")
    for split, samples in partitions.items():
        patients = {case.patient_id for case in samples}
        counts = Counter(case.grade for case in samples)
        grade_counts = [counts[grade] for grade in range(len(GRADE_NAMES))]
        print(f"{split}  患者 {len(patients)} 位，原片 {len(samples)} 张，各级片数 {grade_counts}")


def main():
    parser = argparse.ArgumentParser(description="从颈椎数据目录生成患者表并按患者划分数据")
    parser.add_argument("--data-root", type=Path, required=True, help="包含 0/1/2/3 或中文等级目录的数据根目录")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--patient-csv", type=Path, help="包含 image_path,patient_id 的患者对应表")
    source.add_argument("--patient-regex", help="从原片文件名提取患者编号并生成 patients.csv，正则包含一个捕获组")
    source.add_argument("--one-image-per-patient", action="store_true", help="确认每位患者只有一张原片后使用")
    parser.add_argument("--ratios", type=float, nargs=3, metavar=("TRAIN", "VAL", "TEST"), default=(70, 15, 15),
                        help="训练、验证、测试的目标比例，默认 70 15 15")
    parser.add_argument("--seed", type=int, default=42, help="划分随机种子，默认 42")
    parser.add_argument("--patients-output", type=Path, help="生成的患者表路径，默认写入数据根目录的 patients.csv")
    parser.add_argument("--output", type=Path, help="输出文件，默认写入数据根目录的 splits.csv")
    args = parser.parse_args()
    if args.patient_csv is not None and args.patients_output is not None:
        parser.error("--patients-output 用于从目录生成患者表，请选择 --patient-regex 或 --one-image-per-patient")
    try:
        generate_splits(args)
    except (OSError, ValueError, re.error) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
