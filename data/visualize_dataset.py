"""
This script visualizes the annotations in the dataset by drawing bounding boxes and labels on the images.

Example usage:
* `python visualize_dataset.py`
"""

import json
import os
import cv2
import matplotlib.pyplot as plt
import random

# Define paths
json_file = './original/annotations/instance.json'
images_folder = './original/images'
output_folder = './visualized'

# Load the JSON data
with open(json_file, 'r') as f:
    data = json.load(f)

# Create output folder if not exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Map category IDs to names
category_map = {cat['id']: cat['name'] for cat in data['categories']}

# Assign a random color for each category (in BGR for OpenCV)
color_map = {cat_id: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) 
             for cat_id in category_map}

# Process each image
for img_info in data['images']:
    img_id = img_info['id']
    img_filename = img_info['file_name']
    
    # Load image
    img_path = os.path.join(images_folder, img_filename)
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Image {img_filename} not found, skipping...")
        continue

    # Get all annotations for this image
    annotations = [ann for ann in data['annotations'] if ann['image_id'] == img_id]

    # Draw each bounding box and label on the image
    for ann in annotations:
        bbox = ann['bbox']  # [x, y, width, height]
        category_id = ann['category_id']
        label = category_map[category_id]  # Get the label name from the category map
        color = color_map[category_id]  # Get the color assigned to this category
        
        # Extract coordinates and dimensions
        x, y, w, h = map(int, bbox)
        
        # Draw bounding box with the color assigned to the category
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        
        # Put label text (either id or name)
        label_text = f'{label} ({category_id})'
        cv2.putText(img, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save the visualized image
    output_path = os.path.join(output_folder, img_filename)
    cv2.imwrite(output_path, img)
    print(f"Saved visualized image to {output_path}")

print("Visualization completed!")
