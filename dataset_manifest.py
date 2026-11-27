"""Read patient split manifests with explicit grades and optional true labels."""

import csv
import posixpath
from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath


GRADE_NAMES = ("正常", "轻微", "中度", "重度")
GRADE_DIRS = {name: grade for grade, name in enumerate(GRADE_NAMES)}
GRADE_DIRS.update({str(grade): grade for grade in range(len(GRADE_NAMES))})
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "val", "test")
SPLIT_COLUMNS = ("image_path", "patient_id", "split", "grade", "true_grade")


@dataclass(frozen=True)
class ManifestRow:
    image_path: str
    patient_id: str
    split: str
    grade: int
    true_grade: int | None
    line: int

    def csv_row(self, *, split=None):
        return (
            self.image_path, self.patient_id, self.split if split is None else split,
            self.grade, self.true_grade,
        )


def _grade(value, field, location):
    text = (value or "").strip()
    if not text:
        return None
    if text not in GRADE_DIRS:
        raise ValueError(f"{location}: {field} 需要为 0 到 3 或对应等级名称")
    return GRADE_DIRS[text]


def read_manifest(stream, source, *, require_split=True):
    reader = csv.DictReader(stream)
    required = SPLIT_COLUMNS[:3] if require_split else SPLIT_COLUMNS[:2]
    if not set(required).issubset(reader.fieldnames or ()):
        raise ValueError(f"{source}: CSV 需要 {','.join(required)} 列")
    has_truth_column = "true_grade" in reader.fieldnames
    for row in reader:
        location = f"{source}:{reader.line_num}"
        path_text = (row["image_path"] or "").strip().replace("\\", "/")
        path = PurePosixPath(posixpath.normpath(path_text))
        patient = (row["patient_id"] or "").strip()
        split = (row.get("split") or "").strip().lower()
        if (not path_text or not patient or path.is_absolute() or PureWindowsPath(path_text).drive
                or not path.parts or path.parts[0] == ".." or path.suffix.casefold() not in IMAGE_SUFFIXES
                or path.name.casefold().endswith("_seg.png")):
            raise ValueError(f"{location}: 需要相对原片路径和非空患者编号")
        if (require_split or split) and split not in SPLITS:
            raise ValueError(f"{location}: split 需要为 train、val 或 test")

        grade = _grade(row.get("grade"), "grade", location)
        if grade is None:
            grade = GRADE_DIRS.get(path.parent.name)
        true_grade = _grade(row.get("true_grade"), "true_grade", location)
        if not has_truth_column and path.parts[0].casefold() != "pseudo_labeled":
            true_grade = grade
        if true_grade is not None:
            grade = true_grade
        if grade is None:
            raise ValueError(f"{location}: 请填写 grade、true_grade 或使用等级目录")
        yield ManifestRow(path.as_posix(), patient, split, grade, true_grade, reader.line_num)
