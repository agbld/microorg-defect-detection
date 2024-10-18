import argparse
import os
from ultralytics import YOLO
import cv2
import yaml
from pathlib import Path
import numpy as np
from tqdm import tqdm
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Error analysis for YOLO model on a dataset')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights file')
    parser.add_argument('--data_config', type=str, required=True, help='Path to data.yaml file')
    parser.add_argument('--split', type=str, choices=['train', 'val'], default='val', help='Dataset split to analyze')
    parser.add_argument('--run_name', type=str, default='', help='Custom run name for analysis')
    parser.add_argument('--save_dir', type=str, default='./runs/detect/', help='Directory to save analysis results')
    parser.add_argument('--classes', type=str, nargs='+', default=None, help='List of class names to filter')
    args = parser.parse_args()
    return args

def load_dataset(data_config_path, split):
    with open(data_config_path, 'r') as f:
        data = yaml.safe_load(f)
    if split not in data:
        raise ValueError(f"Split '{split}' not found in data.yaml")
    dataset_path = data[split]
    class_names = data['names']
    return dataset_path, class_names

def get_image_paths(dataset_path):
    if os.path.isfile(dataset_path):
        with open(dataset_path, 'r') as f:
            image_paths = f.read().strip().splitlines()
    elif os.path.isdir(dataset_path):
        image_paths = [str(p) for p in Path(dataset_path).glob('**/*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    else:
        raise ValueError(f"Invalid dataset path: {dataset_path}")
    return image_paths

def load_labels(label_path):
    labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                labels.append([class_id, x_center, y_center, width, height])
    return labels

def yolo_to_bbox(yolo_label, img_width, img_height):
    class_id, x_center, y_center, width, height = yolo_label
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2
    return [int(x_min), int(y_min), int(x_max), int(y_max), int(class_id)]

def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = interArea / float(box1Area + box2Area - interArea + 1e-6)
    return iou

def match_predictions(pred_boxes, gt_boxes, iou_threshold=0.5):
    matches = []
    unmatched_preds = []
    unmatched_gts = []
    used_gt = set()
    for pred_box in pred_boxes:
        best_iou = 0
        best_gt_idx = -1
        for idx, gt_box in enumerate(gt_boxes):
            if idx in used_gt:
                continue
            iou = compute_iou(pred_box[:4], gt_box[:4])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx
        if best_iou >= iou_threshold and pred_box[4] == gt_boxes[best_gt_idx][4]:
            matches.append((pred_box, gt_boxes[best_gt_idx]))
            used_gt.add(best_gt_idx)
        else:
            unmatched_preds.append(pred_box)
    for idx, gt_box in enumerate(gt_boxes):
        if idx not in used_gt:
            unmatched_gts.append(gt_box)
    return matches, unmatched_preds, unmatched_gts

def visualize_results(image, pred_boxes, gt_boxes, class_names, save_path):
    img = image.copy()
    
    # Ground Truth: Green solid line
    for box in gt_boxes:
        x_min, y_min, x_max, y_max, class_id = box
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
        cv2.putText(img, f'GT: {class_names[int(class_id)]}', (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Predictions: Red dotted line
    for box in pred_boxes:
        x_min, y_min, x_max, y_max, class_id = box
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1, lineType=cv2.LINE_AA)
        cv2.putText(img, f'Pred: {class_names[int(class_id)]}', (x_min, y_min - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    cv2.imwrite(save_path, img)

def filter_classes(boxes, class_names, desired_classes):
    desired_class_ids = [class_names.index(cls) for cls in desired_classes]
    return [box for box in boxes if box[4] in desired_class_ids]

def main():
    args = parse_args()
    weights = args.weights
    data_config_path = args.data_config
    split = args.split
    run_name = args.run_name
    save_dir = args.save_dir
    desired_classes = args.classes

    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_name = f"{run_name}_inference_{timestamp}" if run_name else f"inference_{timestamp}"
    save_dir = os.path.join(save_dir, analysis_name)
    os.makedirs(save_dir, exist_ok=True)

    # Save args to the folder
    with open(os.path.join(save_dir, 'cli-args.txt'), 'w') as f:
        f.write(str(args))

    # Load model
    model = YOLO(weights)

    # Load dataset
    dataset_path, class_names = load_dataset(data_config_path, split)
    image_paths = get_image_paths(dataset_path)

    if desired_classes:
        for cls in desired_classes:
            if cls not in class_names:
                raise ValueError(f"Class '{cls}' not found in dataset class names")

    total_images = len(image_paths)
    total_matches = 0
    total_false_positives = 0
    total_false_negatives = 0

    per_class_stats = {i: {'tp': 0, 'fp': 0, 'fn': 0} for i in range(len(class_names))}

    for img_path in tqdm(image_paths, desc='Processing images'):
        img = cv2.imread(img_path)
        img_height, img_width = img.shape[:2]

        # Load ground truth labels
        label_path = os.path.join(os.path.dirname(os.path.dirname(img_path)), 'labels', os.path.basename(os.path.splitext(img_path)[0]) + '.txt')
        gt_labels = load_labels(label_path)
        gt_boxes = [yolo_to_bbox(lbl, img_width, img_height) for lbl in gt_labels]

        # Filter ground truth boxes for desired classes
        if desired_classes:
            gt_boxes = filter_classes(gt_boxes, class_names, desired_classes)

        # Run model to get predictions
        results = model(img, imgsz=img_width, verbose=False)
        pred_boxes = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                    class_id = int(box.cls)
                    pred_boxes.append([x_min, y_min, x_max, y_max, class_id])

        # Filter predicted boxes for desired classes
        if desired_classes:
            pred_boxes = filter_classes(pred_boxes, class_names, desired_classes)

        # Compare predictions with ground truth
        matches, unmatched_preds, unmatched_gts = match_predictions(pred_boxes, gt_boxes)

        # Always update totals and per-class stats
        total_matches += len(matches)
        total_false_positives += len(unmatched_preds)
        total_false_negatives += len(unmatched_gts)

        # Update per-class stats
        for match in matches:
            pred_box, gt_box = match
            class_id = pred_box[4]
            per_class_stats[class_id]['tp'] += 1
        for pred_box in unmatched_preds:
            class_id = pred_box[4]
            per_class_stats[class_id]['fp'] += 1
        for gt_box in unmatched_gts:
            class_id = gt_box[4]
            per_class_stats[class_id]['fn'] += 1

        # Generate visualizations only for incorrect samples
        if unmatched_preds or unmatched_gts:
            vis_save_path = os.path.join(save_dir, os.path.basename(img_path))
            visualize_results(img, pred_boxes, gt_boxes, class_names, vis_save_path)


    # Generate report
    total_detections = total_matches + total_false_positives
    precision = total_matches / total_detections if total_detections > 0 else 0
    recall = total_matches / (total_matches + total_false_negatives) if (total_matches + total_false_negatives) > 0 else 0

    print(f"Total images processed: {total_images}")
    print(f"Incorrect samples visualized: {total_matches + total_false_positives + total_false_negatives}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    # Save per-class stats
    with open(os.path.join(save_dir, 'per_class_stats.csv'), 'w') as f:
        f.write('Class_ID,Class_Name,TP,FP,FN,Precision,Recall\n')
        for class_id, stats in per_class_stats.items():
            tp = stats['tp']
            fp = stats['fp']
            fn = stats['fn']
            class_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            class_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f.write(f"{class_id},{class_names[class_id]},{tp},{fp},{fn},{class_precision:.4f},{class_recall:.4f}\n")

    print(f"Analysis complete! Results saved to {save_dir}")

if __name__ == '__main__':
    main()
