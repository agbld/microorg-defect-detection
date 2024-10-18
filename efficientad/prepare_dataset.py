import json
import os
import shutil
import random
from collections import defaultdict
import argparse
import matplotlib.pyplot as plt
import numpy as np

# Define the command line arguments
parser = argparse.ArgumentParser(description='Prepare a dataset for EfficientAD training')
parser.add_argument('--normal_dir', type=str, default='../data/original/normal/A', help='Path to the directory containing images. This will go through all the images in the directory including subdirectories.')
parser.add_argument('--anomalous_dir', type=str, default='../data/original/images', help='Path to the directory containing anomalies images. This will go through all the images in the directory including subdirectories.')
parser.add_argument('--train_ratio', type=float, default=0.8, help='Percentage of normal data to use for training. All the anomalies will be used for validation.')
parser.add_argument('--output_dir', type=str, default='./efficientad_dataset', help='Path to the EfficientAD dataset directory')
args = parser.parse_args()

# Paths for the input and output data
NORMAL_DIR = args.normal_dir
ANOMALOUS_DIR = args.anomalous_dir
TRAIN_RATIO = args.train_ratio
OUTPUT_DIR = os.path.abspath(args.output_dir)

# If the EfficientAD dataset directory already exists, delete it
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

# Create directories for EfficientAD dataset
os.makedirs(f'{OUTPUT_DIR}/train/normal', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/test/normal', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/test/anomalous', exist_ok=True)

def get_img_path_list(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                file_list.append(os.path.join(root, file))
    return file_list

normal_img_paths = get_img_path_list(NORMAL_DIR)
anomalous_img_paths = get_img_path_list(ANOMALOUS_DIR)

# Split the normal data into training and validation sets
random.shuffle(normal_img_paths)
train_normal_img_paths = normal_img_paths[:int(len(normal_img_paths) * TRAIN_RATIO)]
val_normal_img_paths = normal_img_paths[int(len(normal_img_paths) * TRAIN_RATIO):]

# Copy the normal images to the EfficientAD dataset directory
for img_path in train_normal_img_paths:
    shutil.copy(img_path, f'{OUTPUT_DIR}/train/normal')
for img_path in val_normal_img_paths:
    shutil.copy(img_path, f'{OUTPUT_DIR}/test/normal')

# Copy the anomalous images to the EfficientAD dataset directory
for img_path in anomalous_img_paths:
    shutil.copy(img_path, f'{OUTPUT_DIR}/test/anomalous')

print("Dataset preparation complete!")
