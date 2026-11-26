"""Populate the Torch Hub cache for the two classification backbones."""

import argparse
from pathlib import Path

import torch

from model import DualResNetClassifier


def main():
    parser = argparse.ArgumentParser(description="预取双路 ResNet 的 ImageNet 预训练权重")
    parser.add_argument("--global-weights", type=Path, help="本地 ResNet34 权重文件")
    parser.add_argument("--local-weights", type=Path, help="本地 ResNet18 权重文件")
    args = parser.parse_args()
    try:
        DualResNetClassifier(pretrained=True, global_weights=args.global_weights, local_weights=args.local_weights)
        print(f"backbone weights ready, download cache: {Path(torch.hub.get_dir()) / 'checkpoints'}")
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
