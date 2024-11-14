#%%
import pandas as pd
import os
import cv2
import shutil
from glob import glob

mismatch_samples_path = 'output/1/anomaly_maps/custom/test/mismatch_samples.csv'
visualized_folder = '../data/visualized'
map_folder = 'output/1/anomaly_maps/custom/test'
output_folder = 'output/1/anomaly_maps/custom/test/mismatch_samples_visualized'
target_class = 'good'

# Check if the output folder exists, if not create it, else clear it
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
else:
    shutil.rmtree(output_folder)
    os.makedirs(output_folder)

#%%
# load mismatch_samples.csv as a DataFrame
mismatch_samples = pd.read_csv(mismatch_samples_path)

# Find all image from 'image_name' column where 'defect_class' is target_class
target_samples = mismatch_samples[mismatch_samples['defect_class'] == target_class]

# Get all paths of anomaly maps from `map_folder` (search including sub dir) that match the image name in `target_samples`
# load all visualized images from `visualized_folder` that match the image name in `target_samples`
anomaly_map_paths = []
visualized_image_paths = []
for index, row in target_samples.iterrows():
    image_name = row['image_name']

    # Find all anomaly maps that match the image name
    anomaly_map_path = glob(os.path.join(map_folder, f'**/{image_name}*'))

    # Find all visualized images that match the image name
    visualized_image_path = glob(os.path.join(visualized_folder, f'{image_name}*'))

    # If one of the image is not found, skip the image
    if len(anomaly_map_path) == 0 or len(visualized_image_path) == 0:
        print(f"Image {image_name} not found")
        continue

    anomaly_map_paths.extend(anomaly_map_path)
    visualized_image_paths.extend(visualized_image_path)

#%%
# Combine the anomaly_map with visualized_image into a single image. Export the combined image to the `output_folder`
for anomaly_map_path, visualized_image_path in zip(anomaly_map_paths, visualized_image_paths):
    anomaly_map = cv2.imread(anomaly_map_path)
    visualized_image = cv2.imread(visualized_image_path)
    
    # Combine the two images horizontally
    combined_image = cv2.hconcat([anomaly_map, visualized_image])
    
    # Save the combined image
    output_path = os.path.join(output_folder, os.path.basename(visualized_image_path))
    cv2.imwrite(output_path, combined_image)
    # print(f"Saved combined image to {output_path}")

#%%