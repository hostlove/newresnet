"""Pair cervical radiographs with segmentation PNGs and build aligned inputs."""

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


GRADE_NAMES = ("正常", "轻微", "中度", "重度")
GRADE_DIRS = {name: grade for grade, name in enumerate(GRADE_NAMES)}
GRADE_DIRS.update({str(grade): grade for grade in range(len(GRADE_NAMES))})
SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg"}
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Sample:
    image: Path
    mask: Path
    key: str
    grade: int | None = None
    patient_id: str | None = None


def _is_mask(path):
    return path.name.casefold().endswith("_seg.png")


def _pairs(folder, only_stem=None):
    images, masks = {}, {}
    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.casefold() in IMAGE_SUFFIXES:
            key, indexed = path.stem.casefold(), images
        elif _is_mask(path):
            key, indexed = path.stem[:-4].casefold(), masks
        else:
            continue
        if only_stem is not None and key != only_stem:
            continue
        if key in indexed:
            raise ValueError(f"duplicate image or mask for the same stem: {indexed[key]} and {path}")
        indexed[key] = path
    for key in sorted(images.keys() | masks.keys()):
        if key not in masks:
            raise ValueError(f"missing <stem>_seg.png for {images[key]}")
        if key not in images:
            raise ValueError(f"segmentation PNG has no matching JPG/JPEG: {masks[key]}")
        yield images[key], masks[key]


def _grade_directory(root, directory):
    if not directory.is_dir():
        raise ValueError(f"missing split directory: {directory}")
    samples = []
    for folder in sorted(directory.iterdir()):
        if not folder.is_dir():
            continue
        if folder.name not in GRADE_DIRS:
            if any(p.is_file() and (p.suffix.casefold() in IMAGE_SUFFIXES or _is_mask(p))
                   for p in folder.iterdir()):
                raise ValueError(f"unrecognized grade folder: {folder}")
            continue
        for image, mask in _pairs(folder):
            samples.append(Sample(image, mask, image.relative_to(root).as_posix(), GRADE_DIRS[folder.name]))
    return samples


def _manifest_splits(root, manifest):
    partitions = {split: [] for split in SPLITS}
    patient_splits, pair_indexes = {}, {}
    seen_images = set()
    with Path(manifest).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"image_path", "patient_id", "split"}.issubset(reader.fieldnames or ()):
            raise ValueError("split CSV requires image_path, patient_id, and split columns")
        for row in reader:
            location = f"{manifest}:{reader.line_num}"
            text = (row["image_path"] or "").strip().replace("\\", "/")
            patient = (row["patient_id"] or "").strip()
            split = (row["split"] or "").strip().lower()
            if not text or Path(text).is_absolute() or not patient or split not in SPLITS:
                raise ValueError(f"{location}: expected a relative JPG/JPEG path, patient ID and train/val/test")
            image = (root / text).resolve()
            if not image.is_relative_to(root) or not image.is_file() or image.suffix.casefold() not in IMAGE_SUFFIXES:
                raise ValueError(f"{location}: original image not found under data root: {text}")
            if image in seen_images:
                raise ValueError(f"{location}: duplicate original image: {text}")
            if patient in patient_splits and patient_splits[patient] != split:
                raise ValueError(f"{location}: the same patient appears in multiple splits")

            folder = image.parent
            if folder.name not in GRADE_DIRS:
                raise ValueError(f"{location}: original image must be inside a grade folder")
            if folder not in pair_indexes:
                pair_indexes[folder] = dict(_pairs(folder))
            partitions[split].append(Sample(
                image, pair_indexes[folder][image], image.relative_to(root).as_posix(),
                GRADE_DIRS[folder.name], patient,
            ))
            patient_splits[patient] = split
            seen_images.add(image)
    return partitions


def load_partitions(data_root, names, *, split_csv=None, train_only=False):
    root = Path(data_root).resolve()
    if not root.is_dir():
        raise ValueError(f"data root does not exist: {root}")
    if split_csv is not None:
        partitions = _manifest_splits(root, split_csv)
    else:
        partitions = {name: _grade_directory(root, root if train_only else root / name) for name in names}

    selected = {}
    seen_images = set()
    for name in names:
        samples = sorted(partitions[name], key=lambda sample: sample.key)
        if not samples:
            raise ValueError(f"split {name!r} has no labeled pairs")
        for sample in samples:
            image = sample.image.resolve()
            if image in seen_images:
                raise ValueError(f"duplicate original image across selected samples: {sample.key}")
            seen_images.add(image)
        selected[name] = samples
    return selected


def prediction_samples(source):
    root = Path(source).resolve()
    if root.is_file() and root.suffix.casefold() in IMAGE_SUFFIXES:
        return [Sample(image, mask, image.name) for image, mask in _pairs(root.parent, root.stem.casefold())]
    if not root.is_dir():
        raise ValueError(f"prediction input must be a JPG/JPEG or a directory of pairs: {root}")
    folders = sorted({path.parent for path in root.rglob("*") if path.is_file()
                      and (path.suffix.casefold() in IMAGE_SUFFIXES or _is_mask(path))})
    samples = [Sample(image, mask, image.relative_to(root).as_posix())
               for folder in folders for image, mask in _pairs(folder)]
    if not samples:
        raise ValueError(f"no JPG/JPEG and segmentation PNG pairs under {root}")
    return samples


def _binary_mask(segmentation):
    if "A" in segmentation.getbands():
        channel = segmentation.getchannel("A")
    elif "transparency" in segmentation.info:
        channel = segmentation.convert("RGBA").getchannel("A")
    elif segmentation.mode in ("1", "L"):
        channel = segmentation.convert("L")
    else:
        raise ValueError("expected a segmentation PNG with alpha or a binary grayscale PNG")
    levels = [value for value, count in enumerate(channel.histogram()) if count]
    if len(levels) != 2 or levels[0] != 0:
        raise ValueError("mask must have zero background and exactly one positive foreground value")
    return channel.point([0] + [255] * 255)


def _letterbox(image, size, resampling):
    width, height = image.size
    scale = min(size / width, size / height)
    fitted = (max(1, round(width * scale)), max(1, round(height * scale)))
    canvas = Image.new(image.mode, (size, size))
    canvas.paste(image.resize(fitted, resampling), ((size - fitted[0]) // 2, (size - fitted[1]) // 2))
    return canvas


def prepare_pair(original, segmentation, image_size, roi_margin, training=False):
    if original.size != segmentation.size:
        raise ValueError(f"original size {original.size} differs from mask size {segmentation.size}")
    original = original.convert("RGB")
    mask = _binary_mask(segmentation)
    left, top, right, bottom = mask.getbbox()
    dx = math.ceil((right - left) * roi_margin)
    dy = math.ceil((bottom - top) * roi_margin)
    box = (max(0, left - dx), max(0, top - dy), min(original.width, right + dx), min(original.height, bottom + dy))

    global_rgb = _letterbox(original, image_size, Image.Resampling.BILINEAR)
    roi_rgb = _letterbox(original.crop(box), image_size, Image.Resampling.BILINEAR)
    roi_mask = _letterbox(mask.crop(box), image_size, Image.Resampling.NEAREST)
    if roi_mask.getbbox() is None:
        raise ValueError("mask disappears at the current input size")

    if training:
        brightness, contrast = (random.uniform(0.90, 1.10) for _ in range(2))
        global_rgb = ImageEnhance.Contrast(ImageEnhance.Brightness(global_rgb).enhance(brightness)).enhance(contrast)
        roi_rgb = ImageEnhance.Contrast(ImageEnhance.Brightness(roi_rgb).enhance(brightness)).enhance(contrast)
    global_image = TF.normalize(TF.to_tensor(global_rgb), IMAGE_MEAN, IMAGE_STD)
    local_rgb = TF.normalize(TF.to_tensor(roi_rgb), IMAGE_MEAN, IMAGE_STD)
    local_image = torch.cat((local_rgb, TF.to_tensor(roi_mask)), dim=0)
    return global_image, local_image


class PairedDataset(Dataset):
    def __init__(self, samples, image_size, roi_margin, training=False):
        self.samples = samples
        self.image_size = image_size
        self.roi_margin = roi_margin
        self.training = training

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        try:
            with Image.open(sample.image) as original, Image.open(sample.mask) as segmentation:
                global_image, local_image = prepare_pair(original, segmentation, self.image_size,
                                                         self.roi_margin, self.training)
        except (OSError, ValueError) as exc:
            raise ValueError(f"invalid pair {sample.key}: {exc}") from exc
        item = {"global_image": global_image, "local_image": local_image, "image_path": sample.key}
        if sample.grade is not None:
            item["grade"] = sample.grade
        return item
