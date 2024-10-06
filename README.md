# LED Defects Detection

This repository contains scripts to prepare datasets and train YOLO models for detecting defects in LEDs. The dataset preparation is tailored for YOLO family training, and training/testing is performed using the YOLO framework.

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/agbld/led-defects-detection.git
   cd led-defects-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Preparation

Use `prepare_dataset.py` to prepare a dataset in YOLO format from a COCO JSON file.

### Example Usage

* Prepare a dataset for all classes:
  ```bash
  python prepare_dataset.py --coco_json ./data/annotations/instance.json --images_dir ./data/images --yolo_dir ./yolo_dataset --train_ratio 0.8
  ```

* Prepare a dataset for specific classes:
  ```bash
  python prepare_dataset.py --coco_json ./data/annotations/instance.json --images_dir ./data/images --yolo_dir ./yolo_dataset --train_ratio 0.8 --included_classes led particle flip Particle_Big marked
  ```

Arguments:
- `--coco_json`: Path to COCO format JSON.
- `--images_dir`: Directory containing images.
- `--yolo_dir`: Directory to save YOLO formatted dataset.
- `--train_ratio`: Ratio for train-validation split (default: 0.8).
- `--included_classes`: Classes to include (default: `['led', 'particle', 'flip', ...]`).

## Model Training and Evaluation

Use `main.py` to train and evaluate the YOLO model.

### Example Usage
```bash
python main.py --data_config ./yolo_dataset/data.yaml --model yolo11x --epochs 50 --batch_size 16 --run_name experiment1
```

Arguments:
- `--data_config`: Path to the dataset config YAML.
- `--model`: YOLO model type (default: `yolo11x`).
- `--epochs`: Number of training epochs (default: 50).
- `--batch_size`: Training batch size (default: 16).
- `--run_name`: Custom run name.

## Helper Scripts

See inlined documentation in the scripts for more information.

- [`analyze_results.py`](./analyze_results.py): Generate PDF reports of runs.
- [`data/visualize_dataset.py`](./data/visualize_dataset.py): Render annotations on images and export to `data/vis_images/`.
- [`data/preview_defects.py`](./data/preview_defects.py): Copy images with specific defects to `data/.tmp/`.