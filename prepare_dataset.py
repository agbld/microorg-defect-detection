import json
import os
import shutil
import random

# Paths for the input and output data
COCO_JSON_PATH = './data/annotations/instance.json'
IMAGES_DIR = './data/images'
YOLO_DIR = './yolo_dataset'

TRAIN_RATIO = 0.8  # Percentage of data to use for training

# Create directories for YOLO dataset
os.makedirs(f'{YOLO_DIR}/train/images', exist_ok=True)
os.makedirs(f'{YOLO_DIR}/train/labels', exist_ok=True)
os.makedirs(f'{YOLO_DIR}/val/images', exist_ok=True)
os.makedirs(f'{YOLO_DIR}/val/labels', exist_ok=True)

# Load COCO JSON
with open(COCO_JSON_PATH) as f:
    coco_data = json.load(f)

# Convert COCO annotations to YOLO format
def convert_bbox_coco_to_yolo(image_width, image_height, bbox):
    x_min, y_min, width, height = bbox
    x_center = (x_min + width / 2) / image_width
    y_center = (y_min + height / 2) / image_height
    width /= image_width
    height /= image_height
    return x_center, y_center, width, height

# Create a dictionary for image_id to file_name lookup
image_dict = {image['id']: image for image in coco_data['images']}

# Split annotations into train/val sets
annotations = coco_data['annotations']
random.shuffle(annotations)
split_idx = int(len(annotations) * TRAIN_RATIO)
train_annotations = annotations[:split_idx]
val_annotations = annotations[split_idx:]

# Function to save YOLO annotations
def save_yolo_annotation(annotations, subset):
    for annotation in annotations:
        image_info = image_dict[annotation['image_id']]
        image_name = image_info['file_name']
        image_width = image_info['width']
        image_height = image_info['height']
        category_id = annotation['category_id'] - 1  # YOLO class index starts from 0

        # Convert COCO bbox to YOLO bbox format
        yolo_bbox = convert_bbox_coco_to_yolo(image_width, image_height, annotation['bbox'])

        # Create label file
        label_file_path = f'{YOLO_DIR}/{subset}/labels/{os.path.splitext(image_name)[0]}.txt'
        with open(label_file_path, 'a') as label_file:
            label_file.write(f"{category_id} " + " ".join(map(str, yolo_bbox)) + "\n")

        # Copy corresponding image to YOLO folder
        src_image_path = os.path.join(IMAGES_DIR, image_name)
        dst_image_path = f'{YOLO_DIR}/{subset}/images/{image_name}'
        shutil.copyfile(src_image_path, dst_image_path)

# Save training annotations
save_yolo_annotation(train_annotations, 'train')

# Save validation annotations
save_yolo_annotation(val_annotations, 'val')

# Create the data.yaml file for YOLOv8
def create_yaml_file():
    yaml_content = f"""
train: {YOLO_DIR}/train/images
val: {YOLO_DIR}/val/images

nc: {len(coco_data['categories'])}  # Number of classes
names: {[category['name'] for category in coco_data['categories']]}  # Class names
"""

    yaml_path = os.path.join(YOLO_DIR, 'data.yaml')
    with open(yaml_path, 'w') as yaml_file:
        yaml_file.write(yaml_content)
    
    print(f"data.yaml created at {yaml_path}")

# Create data.yaml
create_yaml_file()

print("Dataset preparation complete!")