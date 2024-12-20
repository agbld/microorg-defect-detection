import os
import json
import shutil
import argparse
from pathlib import Path
from PIL import Image

def convert_to_coco(dataset_dir, output_dir):
    # Ensure paths are Path objects
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)

    # Define COCO format structure
    coco_data = {
        "info": {
            "description": "Custom Dataset",
            "url": "",
            "version": "1.0",
            "year": 2024,
            "contributor": "Generated by convert_to_coco.py",
            "date_created": "2024-01-01"
        },
        "licenses": [
            {
                "id": 1,
                "name": "Unknown",
                "url": ""
            }
        ],
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Category mapping
    categories = {}
    annotation_id = 1
    image_id = 1

    # Prepare output directories
    coco_images_dir = output_dir / "images"
    coco_annotations_dir = output_dir / "annotations"
    coco_images_dir.mkdir(parents=True, exist_ok=True)
    coco_annotations_dir.mkdir(parents=True, exist_ok=True)

    # Iterate over dataset
    for class_dir in dataset_dir.iterdir():
        if not class_dir.is_dir():
            continue
        category_name = class_dir.name

        # Add category to COCO format
        if category_name not in categories:
            category_id = len(categories) + 1
            categories[category_name] = category_id
            coco_data["categories"].append({
                "id": category_id,
                "name": category_name
            })

        for image_file in class_dir.glob("*.*"):
            if image_file.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp", ".gif"]:
                continue

            # Add class name as prefix to avoid duplicate names
            new_file_name = f"{category_name}_{image_file.stem}.jpg"
            image_dest = coco_images_dir / new_file_name

            # Convert image to .jpg
            try:
                with Image.open(image_file) as img:
                    img = img.convert("RGB")  # Ensure compatibility with .jpg
                    img.save(image_dest, "JPEG")
            except Exception as e:
                print(f"Failed to process {image_file}: {e}")
                continue

            # Get image dimensions
            with Image.open(image_dest) as img:
                width, height = img.size

            # Add image entry
            coco_data["images"].append({
                "id": image_id,
                "file_name": new_file_name,
                "width": width,
                "height": height
            })

            # Process corresponding .txt file
            annotation_file = image_file.with_suffix(".txt")
            if annotation_file.exists():
                with open(annotation_file, "r") as af:
                    for line in af:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue

                        class_id, x_center, y_center, bbox_width, bbox_height = map(float, parts)
                        category_id = categories[category_name]

                        # Convert to COCO bbox format
                        x = (x_center - bbox_width / 2) * width
                        y = (y_center - bbox_height / 2) * height
                        w = bbox_width * width
                        h = bbox_height * height

                        # Add annotation entry
                        coco_data["annotations"].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": category_id,
                            "bbox": [x, y, w, h],
                            "segmentation": [],
                            "area": w * h,
                            "iscrowd": 0
                        })
                        annotation_id += 1

            image_id += 1

    # Save COCO JSON
    coco_annotation_path = coco_annotations_dir / "instance.json"
    with open(coco_annotation_path, "w") as coco_file:
        json.dump(coco_data, coco_file, indent=4)

    print(f"COCO dataset created at: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert custom dataset to COCO format.")
    parser.add_argument("dataset_dir", type=str, help="Path to the custom dataset root directory.")
    parser.add_argument("output_dir", type=str, help="Path to the output directory for the COCO dataset.")
    args = parser.parse_args()

    convert_to_coco(args.dataset_dir, args.output_dir)
