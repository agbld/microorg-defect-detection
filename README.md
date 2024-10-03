# YOLOv8 Custom Object Detection

This project focuses on preparing a custom object detection dataset in YOLO format and training a YOLOv8 model. The dataset is originally in COCO format, and we convert it into YOLO format for training.

## Project Structure
- `prepare_dataset.py`: Converts COCO annotations to YOLO format and splits the dataset into training and validation sets.
- `main.py`: Handles training and evaluation of the YOLOv8 model.
- `data/`: Contains the original images and annotations in COCO format.
- `yolo_dataset/`: Generated YOLO dataset and annotations.

## Prerequisites

- Python 3.x
- Install the required dependencies:
  ```bash
  pip install ultralytics
  ```

## Dataset Preparation

To prepare the dataset in YOLO format, run the following command:
```bash
python prepare_dataset.py
```

This will:
- Convert the COCO annotations to YOLO format.
- Split the dataset into training and validation sets (default split is 80% for training).
- Create the necessary folders and save images and annotations in the YOLO format.
- Generate the `data.yaml` file required for YOLOv8 training.

## Training the YOLOv8 Model

To train the YOLOv8 model, run the following command:
```bash
python main.py
```

This will:
- Load the YOLOv8 model (`yolo11n.pt`).
- Train the model for 50 epochs using the prepared dataset.
- Save the best weights to `runs/detect/{run_name}/weights/best.pt`.

## Validation

Once training is complete, the model will automatically run validation using the best weights from training. The results will be printed to the console, including the mAP (mean Average Precision).