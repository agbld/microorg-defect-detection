# eval.py
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import os
from tqdm import tqdm
from common import get_autoencoder, get_pdn_small, get_pdn_medium, \
    ImageFolderWithPath
from utils import predict, default_transform, seed, on_gpu, out_channels, image_size
from sklearn.metrics import roc_auc_score
from PIL import Image
from tabulate import tabulate
import tifffile
import matplotlib
import csv

def get_eval_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='mvtec_ad',
                        choices=['mvtec_ad', 'mvtec_loco', 'custom'])
    parser.add_argument('-s', '--subdataset', default='bottle',
                        help='One of 15 sub-datasets of Mvtec AD or 5 '
                             'sub-datasets of Mvtec LOCO. Ignored if dataset is custom.')
    parser.add_argument('--mvtec_ad_path',
                        default='./dataset/original/mvtec_anomaly_detection',
                        help='Downloaded Mvtec AD dataset')
    parser.add_argument('--mvtec_loco_path',
                        default='./mvtec_loco_anomaly_detection',
                        help='Downloaded Mvtec LOCO dataset')
    parser.add_argument('-c', '--custom_dataset_path',
                        default='./custom_dataset',
                        help='Path to your custom dataset')
    parser.add_argument('-m', '--model_size', default='small',
                        choices=['small', 'medium'])
    parser.add_argument('-w', '--weights_dir', default='output/1/trainings/mvtec_ad/bottle',
                        help='Directory containing trained models and parameters')
    parser.add_argument('-o', '--output_dir', default='output/1')
    parser.add_argument('-T', '--threshold', type=int, default=80, help='Threshold for anomaly detection. From 0 to 255. Default is 80.')
    parser.add_argument('-f', '--map_format', default='tiff', choices=['tiff', 'jpg'],
                        help='Format to save anomaly maps as')
    config = parser.parse_args()
    return config

@torch.no_grad()
def test(test_set, teacher, student, autoencoder, teacher_mean, teacher_std,
         q_st_start, q_st_end, q_ae_start, q_ae_end, test_output_dir=None,
         desc='Running inference',
         map_format='tiff',
         threshold=80):
    y_true = []
    y_score = []
    y_class = []
    
    # List to store mismatch samples
    mismatches = []

    for image, target, path in tqdm(test_set, desc=desc):
        orig_width = image.width
        orig_height = image.height
        image = default_transform(image)
        image = image[None]
        if on_gpu:
            image = image.cuda()
        map_combined, map_st, map_ae = predict(
            image=image, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
            q_ae_start=q_ae_start, q_ae_end=q_ae_end)
        map_combined = torch.nn.functional.pad(map_combined, (4, 4, 4, 4))
        map_combined = torch.nn.functional.interpolate(
            map_combined, (orig_height, orig_width), mode='bilinear')
        map_combined = map_combined[0, 0].cpu().numpy()

        # Scale anomaly map to [0, 255]
        map_combined = map_combined * 255
        map_combined = map_combined.astype(np.int16)

        defect_class = os.path.basename(os.path.dirname(path))
        img_nm = os.path.split(path)[1].split('.')[0]
        if not os.path.exists(os.path.join(test_output_dir, defect_class)):
            os.makedirs(os.path.join(test_output_dir, defect_class))

        if map_format == 'tiff':
            anomaly_map_path = os.path.join(test_output_dir, defect_class, img_nm + '.tiff')
            tifffile.imwrite(anomaly_map_path, map_combined)
        elif map_format == 'jpg':
            original_image = Image.open(path).convert('RGB')
            gradient_softness = threshold / 3
            map_combined_normalized = (map_combined - threshold) / gradient_softness
            map_combined_normalized = 1 / (1 + np.exp(-map_combined_normalized))
            colormap = matplotlib.colormaps['jet']
            anomaly_map_image = colormap(map_combined_normalized)
            anomaly_map_image = (anomaly_map_image[:, :, :3] * 255).astype(np.uint8)
            anomaly_map_image = Image.fromarray(anomaly_map_image)
            anomaly_map_image = anomaly_map_image.resize((original_image.width, original_image.height))
            combined_image_width = original_image.width + anomaly_map_image.width
            combined_image_height = original_image.height
            combined_image = Image.new('RGB', (combined_image_width, combined_image_height))
            combined_image.paste(original_image, (0, 0))
            combined_image.paste(anomaly_map_image.convert('RGB'), (original_image.width, 0))
            combined_image_path = os.path.join(test_output_dir, defect_class, img_nm + '.jpg')
            combined_image.save(combined_image_path)
        else:
            raise ValueError("Invalid map format specified. Use 'tiff' or 'jpg'.")

        y_true_image = 0 if defect_class == 'good' else 1
        y_score_image = np.max(map_combined)
        y_true.append(y_true_image)
        y_score.append(y_score_image)
        y_class.append(defect_class)

        # Collect mismatch samples (where prediction does not match ground truth)
        if (y_true_image == 0 and y_score_image > threshold) or (y_true_image == 1 and y_score_image <= threshold):
            mismatches.append([defect_class, img_nm, y_score_image])

    # Calculate metrics for each defect class
    defect_classes = set(y_class)
    class_metrics = []
    for defect_class in defect_classes:
        class_indices = [i for i, cls in enumerate(y_class) if cls == defect_class]
        y_true_class = [y_true[i] for i in class_indices]
        y_score_class = [y_score[i] for i in class_indices]
        
        accuracy_class = np.mean(np.array(y_true_class) == (np.array(y_score_class) > threshold))

        if defect_class == 'good':
            precision_class = np.sum((np.array(y_true_class) == 0) & (np.array(y_score_class) <= threshold)) / np.sum(np.array(y_score_class) <= threshold)
            recall_class = np.sum((np.array(y_true_class) == 0) & (np.array(y_score_class) <= threshold)) / np.sum(np.array(y_true_class) == 0)
        else:
            precision_class = np.sum((np.array(y_true_class) == 1) & (np.array(y_score_class) > threshold)) / np.sum(np.array(y_score_class) > threshold)
            recall_class = np.sum((np.array(y_true_class) == 1) & (np.array(y_score_class) > threshold)) / np.sum(np.array(y_true_class) == 1)

        num_samples_class = len(y_true_class)
        
        class_metrics.append([defect_class, accuracy_class, precision_class, recall_class, num_samples_class])
    
    # Print class metrics as a table
    headers = ["Class", "Accuracy", "Precision", "Recall", "Num Samples"]
    class_metrics.sort(key=lambda x: x[0])  # Sort by Class
    print()
    print(tabulate(class_metrics, headers=headers, floatfmt=".4f"))

    # Calculate overall metrics
    accuracy = np.mean(np.array(y_true) == (np.array(y_score) > threshold))
    precision = np.sum((np.array(y_true) == 1) & (np.array(y_score) > threshold)) / np.sum(np.array(y_score) > threshold)
    recall = np.sum((np.array(y_true) == 1) & (np.array(y_score) > threshold)) / np.sum(np.array(y_true) == 1)
    num_samples = len(y_true)
    
    # Print overall metrics as a table
    overall_metrics = [["Overall", accuracy, precision, recall, num_samples]]
    print()
    print(tabulate(overall_metrics, headers=headers, floatfmt=".4f"))

    auc = roc_auc_score(y_true=y_true, y_score=y_score)
    
    # Export mismatches to a CSV file
    mismatch_csv_path = os.path.join(test_output_dir, 'mismatch_samples.csv')
    with open(mismatch_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['defect_class', 'image_name', 'y_score'])  # Write headers
        writer.writerows(mismatches)  # Write mismatch samples

    return auc * 100

def main():
    config = get_eval_argparse()

    if config.dataset == 'mvtec_ad':
        dataset_path = config.mvtec_ad_path
    elif config.dataset == 'mvtec_loco':
        dataset_path = config.mvtec_loco_path
    elif config.dataset == 'custom':
        dataset_path = config.custom_dataset_path
    else:
        raise Exception('Unknown config.dataset')

    # Create output directory
    if config.dataset == 'custom':
        test_output_dir = os.path.join(config.output_dir, 'anomaly_maps', 'custom', 'test')
    else:
        test_output_dir = os.path.join(config.output_dir, 'anomaly_maps',
                                       config.dataset, config.subdataset, 'test')
    if not os.path.exists(test_output_dir):
        os.makedirs(test_output_dir)

    # Load test data
    if config.dataset == 'mvtec_ad' or config.dataset == 'mvtec_loco':
        test_set = ImageFolderWithPath(
            os.path.join(dataset_path, config.subdataset, 'test'))
    elif config.dataset == 'custom':
        test_set = ImageFolderWithPath(
            os.path.join(dataset_path, 'test'))
    else:
        raise Exception('Unknown config.dataset')

    # Create models
    if config.model_size == 'small':
        teacher = get_pdn_small(out_channels)
        student = get_pdn_small(2 * out_channels)
    elif config.model_size == 'medium':
        teacher = get_pdn_medium(out_channels)
        student = get_pdn_medium(2 * out_channels)
    else:
        raise Exception('Unknown model size')
    autoencoder = get_autoencoder(out_channels)

    # Load trained model weights
    teacher.load_state_dict(torch.load(os.path.join(config.weights_dir, 'teacher_final.pth'), map_location='cpu'))
    student.load_state_dict(torch.load(os.path.join(config.weights_dir, 'student_final.pth'), map_location='cpu'))
    autoencoder.load_state_dict(torch.load(os.path.join(config.weights_dir, 'autoencoder_final.pth'), map_location='cpu'))

    # Load teacher normalization parameters
    teacher_mean = torch.load(os.path.join(config.weights_dir, 'teacher_mean.pt'))
    teacher_std = torch.load(os.path.join(config.weights_dir, 'teacher_std.pt'))

    # Load quantiles
    quantiles = torch.load(os.path.join(config.weights_dir, 'quantiles.pt'))
    q_st_start = quantiles['q_st_start']
    q_st_end = quantiles['q_st_end']
    q_ae_start = quantiles['q_ae_start']
    q_ae_end = quantiles['q_ae_end']

    # Move models to GPU if available
    if on_gpu:
        teacher.cuda()
        student.cuda()
        autoencoder.cuda()

    teacher.eval()
    student.eval()
    autoencoder.eval()

    # Run evaluation
    auc = test(
        test_set=test_set, teacher=teacher, student=student,
        autoencoder=autoencoder, teacher_mean=teacher_mean,
        teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
        q_ae_start=q_ae_start, q_ae_end=q_ae_end,
        test_output_dir=test_output_dir, desc='Final inference',
        map_format=config.map_format,
        threshold=config.threshold)
    print(f'Final image AUC: {auc:.4f}')

if __name__ == '__main__':
    main()
