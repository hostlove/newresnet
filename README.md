# 颈椎原片与分割掩膜四级分类

全局 ResNet34 编码整张原片。局部 ResNet18 接收颈椎 ROI 的 RGB 像素和对齐的二值掩膜通道。两路各输出 512 维特征，拼接后预测正常、轻微、中度和重度。全局与局部特征融合参考了 X 光分类中的双路 CNN 方法。[Guan 等人的 AG-CNN](https://arxiv.org/abs/1801.09927)

## 数据

同一目录下的 `case01.JPG` 与 `case01_seg.png` 构成一对，要求尺寸一致。分割图支持含二值 Alpha 通道的 PNG 和二值灰度 PNG。四级目录使用 `0/1/2/3` 或 `正常/轻微/中度/重度`。

```text
datasets/
├── 0/
│   ├── case01.JPG
│   └── case01_seg.png
├── 1/
├── 2/
├── 3/
└── splits.csv
```

`splits.csv` 使用 `image_path,patient_id,split` 三列。原片路径相对于数据根目录，同一患者的影像使用同一个编号。已有上一版流程生成的清单可以直接传给 `--split-csv`。如果已按患者分好 `datasets/train/0…3`、`datasets/val/0…3` 和 `datasets/test/0…3`，运行命令时省略 `--split-csv`。

训练、评估与预测共用预处理。ROI 默认在掩膜框每侧外扩 15%，局部原片与掩膜使用相同裁剪、缩放尺寸和补边位置。图像用双线性插值，掩膜用最近邻插值。训练时对两路图像施加相同强度的轻度亮度与对比度扰动。

## 环境与训练

使用 Python 3.10+ 和相互兼容的 PyTorch、TorchVision。GPU 服务器使用支持 CUDA 的 PyTorch。以下命令在 `new_resnet` 目录执行。

```bash
python -m pip install -r requirements.txt
python train.py --data-root ../datasets --split-csv ../datasets/splits.csv --device cuda --epochs 30 --output-dir runs/cervical_rps
```

新训练默认用 ImageNet 权重，并在首次运行时下载所需文件。提前下载可运行 `python load_weights.py`，文件默认在用户目录的 `.cache/torch/hub/checkpoints/`。也可在训练命令里加 `--global-weights /path/resnet34.pth --local-weights /path/resnet18.pth` 使用两个本地 TorchVision 权重字典。CUDA 训练默认使用混合精度，分类头与骨干使用不同学习率。

新训练使用 RPS 序数损失。每张片在三个等级边界上计算预测累计概率与真实累计标签的平方差，并按训练分区的等级频次加权。验证使用不加权的 RPS 调节学习率。`runs/cervical_rps/last.pt` 可续训，`best.pt` 取验证集 MAE 最低的轮次，MAE 相同时取 QWK 较高的一轮。`history.json` 保存 RPS、MAE、QWK、Macro-F1、各级召回率、混淆矩阵和跨两级及以上的错判比例 `error_rate_ge_2`。

两路都从预训练权重初始化的新实验默认使用 `--fine-tune last-block`。训练更新两路最后一个残差块、局部四通道输入卷积和分类头，骨干 BatchNorm 使用固定的运行统计。新实验用真实等级构造 RPS 目标。旧断点续训沿用保存的交叉熵配置和选模规则。评估旧断点也会输出不加权的 `rps`，可与新实验在同一分区对照。

```bash
python train.py --data-root ../datasets --split-csv ../datasets/splits.csv --device cuda --output-dir runs/cervical_rps --resume --epochs 50
```

用本地样例快速检查流程时，可使用以下命令。

```bash
python train.py --data-root ../datatests --train-only --no-pretrained --device cpu --image-size 64 --batch-size 2 --epochs 1 --output-dir runs/smoke
```

## 评估和预测

```bash
python evaluate.py --data-root ../datasets --split-csv ../datasets/splits.csv --split test --checkpoint runs/cervical_rps/best.pt --output runs/cervical_rps/test.json
python predict.py --image ../unlabeled/case01.JPG --checkpoint runs/cervical_rps/best.pt
python batch_predict.py --input ../unlabeled --checkpoint runs/cervical_rps/best.pt --output runs/cervical_rps/predictions.csv
```

预测目录仍使用原片与 `_seg.png` 配对。单张预测输出等级及四级概率。批量预测为每张原片写一行 CSV。
