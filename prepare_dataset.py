import json
import os
import shutil
import random
from collections import defaultdict
import argparse

# Define the command line arguments
parser = argparse.ArgumentParser(description='Prepare a dataset for YOLOv5 training')
parser.add_argument('--coco_json', type=str, default='./data/annotations/instance.json', help='Path to the COCO JSON file')
parser.add_argument('--images_dir', type=str, default='./data/images', help='Path to the directory containing images')
parser.add_argument('--yolo_dir', type=str, default='./yolo_dataset', help='Path to the YOLO dataset directory')
parser.add_argument('--train_ratio', type=float, default=0.8, help='Percentage of data to use for training')
parser.add_argument('--included_classes', type=str, nargs='+', default=['led', 'particle', 'flip', 'Particle_Big', 'marked', 'tilt', 'led_ng'], help='List of classes to include')
args = parser.parse_args()

# Paths for the input and output data
COCO_JSON_PATH = args.coco_json
IMAGES_DIR = args.images_dir
YOLO_DIR = os.path.abspath(args.yolo_dir)

TRAIN_RATIO = args.train_ratio  # Percentage of data to use for training

# Specify the classes you want to include (by their names)
INCLUDED_CLASSES = args.included_classes

# If the YOLO dataset directory already exists, delete it
if os.path.exists(YOLO_DIR):
    shutil.rmtree(YOLO_DIR)

# Create directories for YOLO dataset
os.makedirs(f'{YOLO_DIR}/train/images', exist_ok=True)
os.makedirs(f'{YOLO_DIR}/train/labels', exist_ok=True)
os.makedirs(f'{YOLO_DIR}/val/images', exist_ok=True)
os.makedirs(f'{YOLO_DIR}/val/labels', exist_ok=True)

# Load COCO JSON
with open(COCO_JSON_PATH) as f:
    coco_data = json.load(f)

# Filter categories to include only specified classes
included_category_ids = [category['id'] for category in coco_data['categories'] if category['name'] in INCLUDED_CLASSES]
included_categories = [category for category in coco_data['categories'] if category['id'] in included_category_ids]

# Create a mapping from category_id to new index (since YOLO classes start from 0)
category_id_to_index = {cat_id: idx for idx, cat_id in enumerate(included_category_ids)}

# Filter annotations to include only specified classes
filtered_annotations = [anno for anno in coco_data['annotations'] if anno['category_id'] in included_category_ids]

# Create a dictionary for image_id to file_name and other info
image_dict = {image['id']: image for image in coco_data['images']}

# Build a mapping from image_id to list of category indices it contains
image_id_to_categories = defaultdict(set)
for anno in filtered_annotations:
    image_id_to_categories[anno['image_id']].add(category_id_to_index[anno['category_id']])

# Prepare a list of images with their associated class indices
image_list = []
for image_id, categories in image_id_to_categories.items():
    image_list.append({'image_id': image_id, 'categories': list(categories)})

# Function for stratified splitting at image level
def stratified_split(image_list, train_ratio):
    # Collect all unique class indices
    all_classes = set()
    for item in image_list:
        all_classes.update(item['categories'])
    all_classes = list(all_classes)
    
    # Initialize train and val lists
    train_images = []
    val_images = []
    
    # Create a mapping from class index to images containing that class
    class_to_images = defaultdict(list)
    for item in image_list:
        for class_idx in item['categories']:
            class_to_images[class_idx].append(item)
    
    # Shuffle the images for randomness
    random.shuffle(image_list)
    
    # Assign images to train or val sets
    class_counts = defaultdict(int)
    for item in image_list:
        # Calculate current distribution
        train_class_counts = {cls: sum([cls in img['categories'] for img in train_images]) for cls in all_classes}
        val_class_counts = {cls: sum([cls in img['categories'] for img in val_images]) for cls in all_classes}
        
        # Decide where to put the image to balance the class distributions
        place_in_train = True
        for cls in item['categories']:
            if train_class_counts[cls] / max(1, train_class_counts[cls] + val_class_counts[cls]) > train_ratio:
                place_in_train = False
                break
        
        if place_in_train and len(train_images) / len(image_list) < train_ratio:
            train_images.append(item)
        else:
            val_images.append(item)
    
    return train_images, val_images

# Perform stratified split
train_images, val_images = stratified_split(image_list, TRAIN_RATIO)

# Convert COCO annotations to YOLO format
def convert_bbox_coco_to_yolo(image_width, image_height, bbox):
    x_min, y_min, width, height = bbox
    x_center = (x_min + width / 2) / image_width
    y_center = (y_min + height / 2) / image_height
    width /= image_width
    height /= image_height
    return x_center, y_center, width, height

# Function to save YOLO annotations
def save_yolo_annotations(image_set, subset):
    for item in image_set:
        image_info = image_dict[item['image_id']]
        image_name = image_info['file_name']
        image_width = image_info['width']
        image_height = image_info['height']
        
        # Get all annotations for this image
        annos = [anno for anno in filtered_annotations if anno['image_id'] == item['image_id']]
        
        # Skip images without annotations (after filtering)
        if not annos:
            continue
        
        # Create label file
        label_file_path = f'{YOLO_DIR}/{subset}/labels/{os.path.splitext(image_name)[0]}.txt'
        with open(label_file_path, 'w') as label_file:
            for anno in annos:
                category_id = category_id_to_index[anno['category_id']]  # Adjusted class index
                yolo_bbox = convert_bbox_coco_to_yolo(image_width, image_height, anno['bbox'])
                label_file.write(f"{category_id} " + " ".join(map(str, yolo_bbox)) + "\n")
        
        # Copy corresponding image to YOLO folder
        src_image_path = os.path.join(IMAGES_DIR, image_name)
        dst_image_path = f'{YOLO_DIR}/{subset}/images/{image_name}'
        shutil.copyfile(src_image_path, dst_image_path)

# Save training annotations
save_yolo_annotations(train_images, 'train')

# Save validation annotations
save_yolo_annotations(val_images, 'val')

# Create the data.yaml file for YOLOv8
def create_yaml_file():
    yaml_content = f"""
train: {YOLO_DIR}/train/images
val: {YOLO_DIR}/val/images

nc: {len(included_categories)}  # Number of classes
names: {[category['name'] for category in included_categories]}  # Class names
"""

    yaml_path = os.path.join(YOLO_DIR, 'data.yaml')
    with open(yaml_path, 'w') as yaml_file:
        yaml_file.write(yaml_content)
    
    print(f"data.yaml created at {yaml_path}")

# Create data.yaml
create_yaml_file()

print("Dataset preparation complete!")
