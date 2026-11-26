"""Shared training, checkpoint, and inference helpers for paired classification."""

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from model import DualResNetClassifier
from paired_data import GRADE_NAMES, PairedDataset


CHECKPOINT_FORMAT = "paired-cervical-dual-resnet-v1"
PREDICTION_COLUMNS = (
    "image_path", "predicted_grade", "predicted_label", "p_normal", "p_mild", "p_moderate", "p_severe",
)


def add_runtime_args(parser, batch_size=8):
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=batch_size)
    parser.add_argument("--workers", type=int, default=0)


def select_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable in this PyTorch environment")
    return torch.device(name)


def make_loader(samples, settings, batch_size, workers, device, *, training=False, generator=None):
    if batch_size < 1 or workers < 0:
        raise ValueError("--batch-size must be positive and --workers must be nonnegative")
    dataset = PairedDataset(samples, settings["image_size"], settings["roi_margin"], training=training)
    return DataLoader(dataset, batch_size=batch_size, num_workers=workers, shuffle=training,
                      pin_memory=device.type == "cuda", generator=generator)


def make_grad_scaler(device):
    enabled = device.type == "cuda"
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _model_inputs(batch, device):
    return (batch["global_image"].to(device, non_blocking=True),
            batch["local_image"].to(device, non_blocking=True))


def summarize(confusion, loss):
    counts = confusion.to(dtype=torch.float64, device="cpu")
    total = counts.sum()
    support, predicted = counts.sum(dim=1), counts.sum(dim=0)
    correct = counts.diag()
    f1 = 2 * correct / (support + predicted).clamp_min(1)
    grades = torch.arange(len(GRADE_NAMES), dtype=torch.float64)
    distance = (grades[:, None] - grades).abs()
    expected = support[:, None] * predicted / total
    expected_error = (expected * distance.square()).sum().item()
    observed_error = (counts * distance.square()).sum().item()
    return {
        "samples": int(total.item()), "loss": loss,
        "accuracy": (correct.sum() / total).item(), "macro_f1": f1.mean().item(),
        "recall_by_grade": [hit / count if count else None for hit, count in zip(correct.tolist(), support.tolist())],
        "mae": ((counts * distance).sum() / total).item(),
        "qwk": 1 - observed_error / expected_error if expected_error else None,
        "confusion_matrix": confusion.cpu().tolist(),
    }


def run_epoch(model, loader, device, *, optimizer=None, scaler=None, grade_weights=None):
    training = optimizer is not None
    mixed_precision = training and scaler.is_enabled()
    model.train(training)
    n_classes = len(GRADE_NAMES)
    confusion = torch.zeros(n_classes * n_classes, dtype=torch.long, device=device)
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)

    with torch.set_grad_enabled(training):
        for batch in loader:
            labels = batch["grade"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=mixed_precision):
                logits = model(*_model_inputs(batch, device))
            losses = F.cross_entropy(logits.float(), labels, reduction="none")
            if grade_weights is not None:
                losses = losses * grade_weights[labels]
            loss = losses.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite classification loss")
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            predictions = logits.argmax(dim=1)
            confusion += torch.bincount(n_classes * labels + predictions, minlength=n_classes * n_classes)
            loss_sum += losses.detach().double().sum()

    matrix = confusion.reshape(n_classes, n_classes).cpu()
    return summarize(matrix, loss_sum.item() / len(loader.dataset))


@contextmanager
def atomic_output(path, *, binary=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = tempfile.NamedTemporaryFile(mode="wb" if binary else "w", encoding=None if binary else "utf-8",
                                         newline=None if binary else "", dir=path.parent, delete=False)
    try:
        with stream:
            yield stream
        os.replace(stream.name, path)
    finally:
        Path(stream.name).unlink(missing_ok=True)


def write_json(path, value):
    with atomic_output(path) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def save_checkpoint(path, state):
    with atomic_output(path, binary=True) as stream:
        torch.save(state, stream)


def read_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(state, dict) or state.get("format") != CHECKPOINT_FORMAT
            or state.get("grades") != list(GRADE_NAMES)):
        raise ValueError(f"expected a paired cervical checkpoint produced by train.py: {path}")
    return state


def inference_model(checkpoint, device):
    state = read_checkpoint(checkpoint)
    model = DualResNetClassifier(num_classes=len(GRADE_NAMES))
    model.load_state_dict(state["model"], strict=True)
    return model.to(device).eval(), state["settings"]


def prediction_rows(model, loader, device):
    with torch.inference_mode():
        for batch in loader:
            logits = model(*_model_inputs(batch, device))
            probabilities = logits.softmax(dim=1).cpu()
            if not torch.isfinite(probabilities).all():
                raise FloatingPointError("non-finite class probability")
            for path, grade, scores in zip(batch["image_path"], probabilities.argmax(dim=1).tolist(),
                                           probabilities.tolist()):
                yield path, grade, GRADE_NAMES[grade], *scores
