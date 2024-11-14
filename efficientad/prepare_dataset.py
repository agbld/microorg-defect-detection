#%%
# Import libraries
import os
import shutil
import json
import pandas as pd
from collections import defaultdict
from pathlib import Path
import argparse
import csv

#%%
# Create a parser

parser = argparse.ArgumentParser(description='Prepare the dataset for training and testing.')
parser.add_argument('--annotations', type=str, default='original/annotations/instance.json', help='Path to the instance.json file.')
parser.add_argument('--labeled_dir', type=str, default='original/images/', help='Path to the image directory with annotations.')
parser.add_argument('--unlabeled_dir', type=str, default='original/normal/B/', help='Path to the unlabeled (normal) image directory, including all subdirectories.')
parser.add_argument('--output_dir', type=str, default='leddd', help='Path to the output directory.')
parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of training images to total images.')
args = parser.parse_args()

# Print the parsed arguments in a formatted manner
print("\nParsed arguments:")
for arg, value in vars(args).items():
    print(f"--{arg} = {value}")
print()

ANNOTATIONS = args.annotations
LABELED_DIR = args.labeled_dir
UNLABELED_DIR = args.unlabeled_dir
OUTPUT_DIR = args.output_dir
TRAIN_RATIO = args.train_ratio

#%%
# Create defect_img_df

# Load the instance.json file
with open(ANNOTATIONS, 'r') as f:
    data = json.load(f)

# Create a dictionary to store defects for each image
image_defects = defaultdict(list)

# Iterate through annotations to collect category_ids for each image
for annotation in data['annotations']:
    image_id = annotation['image_id']
    category_id = annotation['category_id']
    image_defects[image_id].append(category_id)

# Prepare the DataFrame data
df_data = []
for image in data['images']:
    image_id = image['id']
    file_name = image['file_name']
    defects = image_defects.get(image_id, [])  # Get defects or an empty list if none
    df_data.append({'image_path': file_name, 'defects': defects})

# Convert to DataFrame
anonnated_img_df = pd.DataFrame(df_data)

# Convert 'image_path' to Path objects for consistency
anonnated_img_df['image_path'] = anonnated_img_df['image_path'].apply(lambda x: Path(x))

# Create a defect_img_df, which contains images with any defects other than 1 and 5
defect_img_df = anonnated_img_df[anonnated_img_df['defects'].apply(lambda x: any(defect not in [1, 5] for defect in x))]

# Create normal_defect_img_df, which contains images that have only defects 1 and/or 5
normal_defect_img_df = anonnated_img_df[anonnated_img_df['defects'].apply(lambda x: len(x) > 0 and all(defect in [1, 5] for defect in x))]

# Count the occurrence of each defect, skipping defects 1 and 5
defect_count = defaultdict(int)
for defects in defect_img_df['defects']:
    for defect in defects:
        if defect not in [1, 5]:
            defect_count[defect] += 1

#%%
# Create normal_img_df

normal_img_dir = UNLABELED_DIR

# Get all image paths in the normal image directory including subdirectories
normal_img_paths = []
for path in Path(normal_img_dir).rglob('*.jpg'):
    normal_img_paths.append(path)

# Create a DataFrame for normal images
normal_img_df = pd.DataFrame(normal_img_paths, columns=['full_image_path'])

# Add an empty 'defects' column to normal_img_df
normal_img_df['defects'] = [[] for _ in range(len(normal_img_df))]

# Prepare 'full_image_path' in normal_defect_img_df
defect_img_dir = Path(LABELED_DIR)
normal_defect_img_df = normal_defect_img_df.copy()
normal_defect_img_df.loc[:, 'full_image_path'] = normal_defect_img_df['image_path'].apply(lambda x: defect_img_dir / x)

# Combine normal images from normal_img_df and normal_defect_img_df
combined_normal_img_df = pd.concat([normal_img_df, normal_defect_img_df[['full_image_path', 'defects']]], ignore_index=True)

#%%
# Construct the custom_dataset directory

custom_dataset_dir = Path(OUTPUT_DIR)

# If the custom_dataset directory already exists, delete it
if custom_dataset_dir.exists():
    shutil.rmtree(custom_dataset_dir)

train_ratio = TRAIN_RATIO

# Create the custom_dataset directory
custom_dataset_dir.mkdir(parents=True, exist_ok=True)

# Create the train and test directories
train_dir = custom_dataset_dir / 'train'
test_dir = custom_dataset_dir / 'test'
train_dir.mkdir(exist_ok=True)
test_dir.mkdir(exist_ok=True)

# Create the good and defect directories in train and test
good_train_dir = train_dir / 'good'
good_test_dir = test_dir / 'good'
good_train_dir.mkdir(exist_ok=True)
good_test_dir.mkdir(exist_ok=True)

defect_test_dirs = {}
for defect in defect_count.keys():
    defect_test_dirs[defect] = test_dir / f'defect_type_{defect}'
    defect_test_dirs[defect].mkdir(exist_ok=True)

# Shuffle the combined normal images
combined_normal_img_df = combined_normal_img_df.sample(frac=1).reset_index(drop=True)

# Split into train and test
normal_img_count = len(combined_normal_img_df)
normal_img_train_count = int(normal_img_count * train_ratio)
normal_img_train_df = combined_normal_img_df.iloc[:normal_img_train_count]
normal_img_test_df = combined_normal_img_df.iloc[normal_img_train_count:]

# Copy the normal images to the train and test directories. Copy, not move.
for i, row in normal_img_train_df.iterrows():
    image_path = row['full_image_path']
    image_name = os.path.basename(image_path)
    new_image_path = good_train_dir / image_name
    try:
        shutil.copy(image_path, new_image_path)
    except FileNotFoundError:
        print(f'cp {image_path} {new_image_path}')

for i, row in normal_img_test_df.iterrows():
    image_path = row['full_image_path']
    image_name = os.path.basename(image_path)
    new_image_path = good_test_dir / image_name
    try:
        shutil.copy(image_path, new_image_path)
    except FileNotFoundError:
        print(f'cp {image_path} {new_image_path}')

# Prepare 'full_image_path' in defect_img_df
defect_img_df = defect_img_df.copy()
defect_img_df.loc[:, 'full_image_path'] = defect_img_df['image_path'].apply(lambda x: defect_img_dir / x)

# Copy the defect images to the test directories. Copy, not move.
for i, row in defect_img_df.iterrows():
    image_path = row['full_image_path']
    image_name = os.path.basename(image_path)
    defects = row['defects']
    for defect in defects:
        if defect not in [1, 5] and defect in defect_test_dirs:
            new_image_path = defect_test_dirs[defect] / image_name
            try:
                shutil.copy(image_path, new_image_path)
            except FileNotFoundError:
                print(f'cp {image_path} {new_image_path}')

#%%
# Print the number of images in each directory.
print(f'Number of images in {good_train_dir}: {len(list(good_train_dir.glob("*.jpg")))}')
print(f'Number of images in {good_test_dir}: {len(list(good_test_dir.glob("*.jpg")))}')
for defect, defect_test_dir in defect_test_dirs.items():
    print(f'Number of images in {defect_test_dir}: {len(list(defect_test_dir.glob("*.jpg")))}')

# Export the counts to a CSV file.
counts = [
    {'directory': str(good_train_dir), 'count': len(list(good_train_dir.glob("*.jpg")))},
    {'directory': str(good_test_dir), 'count': len(list(good_test_dir.glob("*.jpg")))}
]

for defect, defect_test_dir in defect_test_dirs.items():
    counts.append({'directory': str(defect_test_dir), 'count': len(list(defect_test_dir.glob("*.jpg")))})

csv_output_path = custom_dataset_dir / 'image_counts.csv'
with open(csv_output_path, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['directory', 'count'])
    writer.writeheader()
    writer.writerows(counts)

#%%
