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

## Experiments

This section contains the results of the experiments conducted during the development of this project. GitHub commits are provided for each experiment to ensure reproducibility.

### Class Sweep

**GitHub Commit**: [cd675400b8f78e49c93cddb925846edf51a1ab2f](https://github.com/agbld/led-defects-detection/commit/cd675400b8f78e49c93cddb925846edf51a1ab2f)

**Objective**:
- An ablation study or an EDA to observe each class training outcomes.
- See the upper bound performance of the model for each class.

**Results**:
- Labels distribution <br>
    <div style="display: flex; justify-content: center;">
      <img src="./assets/class_distribution.png" alt="Labels Distribution" width="70%">
    </div>

- Label "information" <br>
  `led` and `marked` classes have the most samples. However, consider the "information" we get from annotations, `led` and `marked` classes, especially the `marked` class has almost no information. From the following figures, we can see the shape and position of it almost never changes. Things are similar for the `led` class (but not as severe as `marked`).
  <div style="display: flex; justify-content: space-around;">
    <img src="./assets/marked_annotation.jpg" alt="Marked Annotation" width="48%">
    <img src="./assets/led_annotation.jpg" alt="LED Annotation" width="48%">
  </div>

- Label "information" (continued) <br>
  On the other hand, `particle`, `Particle_Big` and `flip` have MUCH more "information". See following figures:
  <div style="display: flex; justify-content: space-around;">
    <img src="./assets/particle_annotation.jpg" alt="Particle Annotation" width="31%">
    <img src="./assets/Particle_Big_annotation.jpg" alt="Particle Big Annotation" width="31%">
    <img src="./assets/flip_annotation.jpg" alt="Flip Annotation" width="31%">
  </div>

- Insufficient data <br>
  For the last two classes, `tilt` and `led_ng`, we have very few samples (< 10). This also leads to the stability problem during training. See the following figures:
    <div style="display: flex; justify-content: center;">
      <img src="./assets/class_sweep_box_loss.png" alt="Class Sweep box-loss plot" width="70%">
    </div>

- Metrics <br>
  According to the following figures, we can see that YOLO could fit very well on `led`, `marked`, and `flip` classes (assuming `led` and `marked` are necessary). The `Particle_Big` has good convergence. The `particle` couldn't go better after around 120 epochs. Since we have more sample for `particle` than `Particle_Big`, the underfitting might be due to the labeling quality (eg. defect definition).
  <div style="display: flex; justify-content: space-around;">
    <img src="./assets/class_sweep_P.png" alt="Class Sweep P" width="48%">
    <img src="./assets/class_sweep_R.png" alt="Class Sweep R" width="48%">
  </div>

- **Conclusion** <br>
  - `led` and `marked` classes have almost no information. Consider removing them.
  - `particle`, `Particle_Big` and `flip` classes have much more information.
  - `tilt` and `led_ng` classes have insufficient data. Consider getting more samples or removing them.

**Method**:
- Train the model with each class separately. Specifically, use `prepare_dataset.py` to prepare a dataset for each class. Then, train the model with `main.py` for each class.
- Use `yolo11x` model for all experiments.
- Use `--epochs 500` to ensure convergence.

### Training as Validation

**GitHub Commit**: N/A

**Objective**:
- Use the training set as the validation set to see is there any unreasonable labelings that model couldn't even overfit.
- This is just a very ir-rigorous sanity check. If model could find any minor patterns to identify each specific sample, it should be able to overfit the training set.

**Results**:
- Confusion matrix <br>
  <div style="display: flex; justify-content: space-around;">
    <img src="./assets/train_is_val_confusion_matrix_normalized.png" alt="Training as Validation Confusion Matrix Normalized" width="45%">
    <img src="./assets/train_is_val_confusion_matrix.png" alt="Training as Validation Confusion Matrix" width="45%">
  </div>
- Most of the classes seem to be learned well.

### Model Sweep

**GitHub Commit**: [3fd972925ad585b568fb641629f41f9b4e2537e9](https://github.com/agbld/led-defects-detection/commit/3fd972925ad585b568fb641629f41f9b4e2537e9)

**Objective**:
- Find the best model for current dataset.

**Results**:
- Performance table:
  ![Model Sweep Performance](./assets/model_sweep_performance.png)
- According to precision, recall, mAPs, all tested models have **similar performance**.
- Later models (*yolov9e*, *yolov10x*, *yolo11x*) have very slightly better performance than older models (*yolov5xu*, *yolov8x*).

**Method**:
- Use only those "major" classes that have enough samples to eliminate the data insufficiency problem. Which are: `led`, `particle`, `flip`, `Particle_Big`, `marked`.
- Tested models:
  - `yolov5xu (97.2M)`
  - `yolov8x (68.2M)`
  - `yolov9e (58.1M)`
  - `yolov10x (29.5M)`
  - `yolo11x (56.9M)`