from ultralytics import YOLO
from datetime import datetime
import argparse

# Define the command line arguments
parser = argparse.ArgumentParser(description='Train a YOLO model')
parser.add_argument('--data_config', type=str, default='./yolo_dataset/data.yaml', help='Path to the data.yaml file')
parser.add_argument('--model', type=str, default='yolo11x', help='YOLO model to use')
parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--run_name', type=str, default='', help='Custom run name')
args = parser.parse_args()

# Path to the dataset configuration YAML file
DATA_CONFIG_PATH = args.data_config

MODEL = args.model
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
RUN_NAME = args.run_name

# Get current timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Custom folder name for the run, including the timestamp
train_name = f"{RUN_NAME}_train_{timestamp}"
val_name = f"{RUN_NAME}_val_{timestamp}"

# Train the YOLO model
def train_model():
    # Load the YOLO model
    model = YOLO(f'models/{MODEL}.pt')

    # Train the model
    model.train(
        data=DATA_CONFIG_PATH,   # Pointing to the data.yaml file
        epochs=EPOCHS,           # Number of training epochs from argparse
        imgsz=320,               # Image size
        batch=BATCH_SIZE,        # Batch size from argparse
        name=train_name,         # Custom run name
    )

    # Save args to the folder
    with open(f'runs/detect/{train_name}/cli-args.txt', 'w') as f:
        f.write(str(args))

    print("Training complete!")

if __name__ == '__main__':
    train_model()