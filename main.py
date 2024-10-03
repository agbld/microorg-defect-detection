from ultralytics import YOLO
from datetime import datetime

# Path to the dataset configuration YAML file
DATA_CONFIG_PATH = './yolo_dataset/data.yaml'

# Get current timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Custom folder name for the run, including the timestamp
train_name = f"train_{timestamp}"
val_name = f"val_{timestamp}"

# Train the YOLOv8 model
def train_model():
    # Load the YOLOv8 model
    model = YOLO('yolo11x.pt')

    # Train the model
    model.train(
        data=DATA_CONFIG_PATH,   # Pointing to the data.yaml file
        epochs=50,               # Number of training epochs
        imgsz=320,               # Image size
        batch=16,                # Batch size
        name=train_name,           # Custom run name
    )

    print("Training complete!")

# Test the model on the validation set
def test_model():
    best_weights = f'runs/detect/{train_name}/weights/best.pt'

    model = YOLO(best_weights)  # Use the best weights from training

    # Perform inference on the validation set
    results = model.val(data=DATA_CONFIG_PATH,
                        name=val_name)

    # Print mAP results
    print(f"Validation Results:\n{results}")

if __name__ == '__main__':
    train_model()
    test_model()
