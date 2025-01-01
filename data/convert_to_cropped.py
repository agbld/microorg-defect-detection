import os
import json
import cv2

json_file = './original-top10/annotations/instance.json'
images_folder = './original-top10/images'
output_root = './cropped-top10'
os.makedirs(os.path.join(output_root, 'images'), exist_ok=True)
os.makedirs(os.path.join(output_root, 'annotations'), exist_ok=True)

with open(json_file, 'r') as f:
    data = json.load(f)

new_data = {
    "images": [],
    "annotations": [],
    "categories": data["categories"]
}

img_id_counter = 0
ann_id_counter = 0

for img_info in data["images"]:
    img_id = img_info["id"]
    file_name = img_info["file_name"]
    path = os.path.join(images_folder, file_name)
    img = cv2.imread(path)
    if img is None:
        continue
    ann_list = [a for a in data["annotations"] if a["image_id"] == img_id]
    for i, ann in enumerate(ann_list):
        x, y, w, h = map(int, ann["bbox"])
        crop = img[y:y+h, x:x+w]
        if crop.size == 0:
            continue
        img_id_counter += 1
        new_fname = f"{os.path.splitext(file_name)[0]}_{i}.jpg"
        cv2.imwrite(os.path.join(output_root, 'images', new_fname), crop)
        new_data["images"].append({
            "id": img_id_counter,
            "file_name": new_fname,
            "width": crop.shape[1],
            "height": crop.shape[0]
        })
        ann_id_counter += 1
        new_data["annotations"].append({
            "id": ann_id_counter,
            "image_id": img_id_counter,
            "category_id": ann["category_id"],
            "bbox": [0, 0, w, h],
            "iscrowd": ann.get("iscrowd", 0)
        })

with open(os.path.join(output_root, 'annotations', 'instance.json'), 'w') as f:
    json.dump(new_data, f)

print("Done.")
