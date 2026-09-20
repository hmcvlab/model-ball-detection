# Ball Detection Benchmark

[![Paper](https://img.shields.io/badge/Paper-10.1007%2Fs42979--026--04976--9-blue)](https://link.springer.com/article/10.1007/s42979-026-04976-9)
[![Dataset](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18988561-blue)](https://doi.org/10.5281/zenodo.18988561)

This repository contains the code to reproduce the circle detection benchmark
reported in:

> **MuFoRa: A Multimodal Dataset and the Impact of Adverse Weather on Camera and
> LiDAR**\
> Valentino Behret, Regina Kushtanova, Simon Weber, Julian Wiesner, Thomas
> Helmer, Frank Palme\
> *SN Computer Science*, 7, 442 (2026)\
> https://doi.org/10.1007/s42979-026-04976-9

The benchmark compares a classical Hough-transform-based circle detector with
24 deep learning models (torchvision detection models, YOLO variants, RT-DETR
and DETR) on the **Accurate Balls Detection** dataset.

## Dataset

The models are trained and evaluated on the
[Accurate Balls Detection](https://zenodo.org/records/18988561) dataset
(CC-BY-4.0). It contains 759 images with highly accurate ball annotations in
COCO format, split into train (559), valid (126) and holdout/test (74) sets.

Download and extract `accurate-ball-detection.zip` from Zenodo. The scripts
expect the dataset at `/mnt/data/datasets/accurate-balls/` by default (see
`ball_detector/aux.py`), i.e.:

```
/mnt/data/datasets/accurate-balls/
├── images/
├── train.coco.json
├── valid.coco.json
└── holdout.coco.json
```

All paths can be overridden via command line arguments — run any script with
`--help` for details.

## Installation

Python >= 3.10 is required. Install the package in editable mode:

```bash
pip install -e .
```

This pulls in `torch`, `torchvision` and a
[custom DETR fork](https://github.com/behretv/detr) as dependencies.

## Repository structure

- `ball_detector/` — library code (COCO dataset handling, model builders,
  training loop, Hough baseline, drawing utilities)
- `scripts/` — entry points for training, evaluation and benchmark generation
- `tests/` — unit tests (`pytest`)

## Usage

### Training

```bash
# torchvision detection models (e.g. Faster R-CNN, RetinaNet, FCOS, ...)
python scripts/train_torch.py --torch-model fasterrcnn_resnet50_fpn_v2 --augment

# YOLO / RT-DETR (Ultralytics)
python scripts/train_yolo.py --model yolov5s.pt --dir-dataset /mnt/data/datasets/accurate-balls

# DETR
python scripts/train_detr.py --dataset /mnt/data/datasets/accurate-balls
```

### Evaluation

```bash
python scripts/evaluate_torch.py --torch-model fasterrcnn_resnet50_fpn_v2
python scripts/evaluate_yolo.py --yolo-model yolov5s   # or --file-model path/to/checkpoint
python scripts/evaluate_hough.py
```

Each evaluation script runs inference on the holdout split and appends metrics
(mAP, AP per size, inference time) to a benchmark CSV file.

### Benchmark table

```bash
python scripts/table_benchmark.py
```

Aggregates all benchmark CSVs and exports the results table in TeX format (as
used in the paper).

## Development

```bash
make format   # code formatting (docker)
make lint     # linting (docker)
make test     # run unit tests (docker)
```

## Citation

If you use this code or the dataset in your research, please cite:

```bibtex
@article{Behret2026,
  author   = {Behret, Valentino and Kushtanova, Regina and Weber, Simon and
              Wiesner, Julian and Helmer, Thomas and Palme, Frank},
  title    = {MuFoRa: A Multimodal Dataset and the Impact of Adverse Weather
              on Camera and LiDAR},
  journal  = {SN Computer Science},
  year     = {2026},
  volume   = {7},
  number   = {5},
  pages    = {442},
  doi      = {10.1007/s42979-026-04976-9},
  url      = {https://doi.org/10.1007/s42979-026-04976-9},
}
```

## License

This project is released under the [MIT License](LICENSE).
