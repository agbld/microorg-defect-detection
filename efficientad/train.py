# train.py
import numpy as np
import torch
torch.set_float32_matmul_precision('medium')
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import itertools
import os
import random
from tqdm import tqdm
from common import get_autoencoder, get_pdn_small, get_pdn_medium, \
    ImageFolderWithoutTarget, InfiniteDataloader, ImageFolderWithPath
from utils import predict, map_normalization, teacher_normalization, default_transform, train_transform, seed, on_gpu, out_channels, image_size

# for evaluation
from sklearn.metrics import roc_auc_score
from PIL import Image
from tabulate import tabulate
import tifffile
import matplotlib
import csv

def get_train_argparse():
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
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('-m', '--model_size', default='small',
                        choices=['small', 'medium'])
    parser.add_argument('-w', '--weights', default='models/teacher_small.pth')
    parser.add_argument('-i', '--imagenet_train_path',
                        default='none',
                        help='Set to "none" to disable ImageNet '
                            'pretraining penalty. Or see README.md to '
                            'download ImageNet and set to ImageNet path')
    parser.add_argument('-o', '--output_dir', default='output/1')
    # parser.add_argument('--eval_epochs', nargs='+', type=int, default=[1, 5, 10, 20, 50, 100, 200, 500],
    #                     help='List of epochs at which to evaluate the model')
    config = parser.parse_args()
    return config

def main():
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    config = get_train_argparse()

    if config.dataset == 'mvtec_ad':
        dataset_path = config.mvtec_ad_path
    elif config.dataset == 'mvtec_loco':
        dataset_path = config.mvtec_loco_path
    elif config.dataset == 'custom':
        dataset_path = config.custom_dataset_path
    else:
        raise Exception('Unknown config.dataset')

    pretrain_penalty = True
    if config.imagenet_train_path == 'none':
        pretrain_penalty = False

    # Create output directory
    if config.dataset == 'custom':
        train_output_dir = os.path.join(config.output_dir, 'trainings', 'custom')
    else:
        train_output_dir = os.path.join(config.output_dir, 'trainings',
                                        config.dataset, config.subdataset)

    # Remove existing directories if they exist
    if os.path.exists(train_output_dir):
        import shutil
        shutil.rmtree(train_output_dir)
    os.makedirs(train_output_dir)

    # Load data
    if config.dataset == 'mvtec_ad' or config.dataset == 'mvtec_loco':
        if config.dataset == 'mvtec_ad':
            # MVTec AD dataset recommends 10% validation set
            full_train_set = ImageFolderWithoutTarget(
                os.path.join(dataset_path, config.subdataset, 'train'),
                transform=transforms.Lambda(train_transform))
            train_size = int(0.9 * len(full_train_set))
            validation_size = len(full_train_set) - train_size
            rng = torch.Generator().manual_seed(seed)
            train_set, validation_set = torch.utils.data.random_split(
                full_train_set, [train_size, validation_size], generator=rng)
        elif config.dataset == 'mvtec_loco':
            full_train_set = ImageFolderWithoutTarget(
                os.path.join(dataset_path, config.subdataset, 'train'),
                transform=transforms.Lambda(train_transform))
            train_set = full_train_set
            validation_set = ImageFolderWithoutTarget(
                os.path.join(dataset_path, config.subdataset, 'validation'),
                transform=transforms.Lambda(train_transform))
        else:
            raise Exception('Unknown config.dataset')
    elif config.dataset == 'custom':
        # Load custom training data (normal images only)
        full_train_set = ImageFolderWithoutTarget(
            os.path.join(dataset_path, 'train'),
            transform=transforms.Lambda(train_transform))
        # Split into train and validation sets
        train_size = int(0.9 * len(full_train_set))
        validation_size = len(full_train_set) - train_size
        rng = torch.Generator().manual_seed(seed)
        train_set, validation_set = torch.utils.data.random_split(
            full_train_set, [train_size, validation_size], generator=rng)
    else:
        raise Exception('Unknown config.dataset')

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    train_loader_infinite = InfiniteDataloader(train_loader)
    validation_loader = DataLoader(validation_set, batch_size=config.batch_size)

    if pretrain_penalty:
        # Load pretraining data for penalty
        penalty_transform = transforms.Compose([
            transforms.Resize((2 * image_size, 2 * image_size)),
            transforms.RandomGrayscale(0.3),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,
                                                                  0.225])
        ])
        penalty_set = ImageFolderWithoutTarget(config.imagenet_train_path,
                                               transform=penalty_transform)
        penalty_loader = DataLoader(penalty_set, batch_size=4, shuffle=True,
                                    num_workers=4, pin_memory=True)
        penalty_loader_infinite = InfiniteDataloader(penalty_loader)
    else:
        penalty_loader_infinite = itertools.repeat(None)

    # Create models
    if config.model_size == 'small':
        teacher = get_pdn_small(out_channels)
        student = get_pdn_small(2 * out_channels)
    elif config.model_size == 'medium':
        teacher = get_pdn_medium(out_channels)
        student = get_pdn_medium(2 * out_channels)
    else:
        raise Exception('Unknown model size')
    state_dict = torch.load(config.weights, map_location='cpu')
    teacher.load_state_dict(state_dict)
    autoencoder = get_autoencoder(out_channels)

    # Move models to GPU if available
    if on_gpu:
        teacher.cuda()
        student.cuda()
        autoencoder.cuda()

    # Freeze teacher model
    teacher.eval()
    student.train()
    autoencoder.train()

    # Compute teacher normalization
    teacher_mean, teacher_std = teacher_normalization(teacher, train_loader)

    optimizer = torch.optim.Adam(itertools.chain(student.parameters(),
                                                 autoencoder.parameters()),
                                 lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=int(0.95 * config.epochs * len(train_loader)), gamma=0.1)
    num_epochs = config.epochs
    tqdm_obj = tqdm(range(num_epochs))
    for epoch in tqdm_obj:
        for iteration, (image_st, image_ae), image_penalty in zip(
                range(len(train_loader)), train_loader_infinite, penalty_loader_infinite):
            if on_gpu:
                image_st = image_st.cuda()
                image_ae = image_ae.cuda()
                if image_penalty is not None:
                    image_penalty = image_penalty.cuda()
            with torch.no_grad():
                teacher_output_st = teacher(image_st)
                teacher_output_st = (teacher_output_st - teacher_mean) / teacher_std
            student_output_st = student(image_st)[:, :out_channels]
            distance_st = (teacher_output_st - student_output_st) ** 2
            d_hard = torch.quantile(distance_st, q=0.999)
            loss_hard = torch.mean(distance_st[distance_st >= d_hard])

            if image_penalty is not None:
                student_output_penalty = student(image_penalty)[:, :out_channels]
                loss_penalty = torch.mean(student_output_penalty**2)
                loss_st = loss_hard + loss_penalty
            else:
                loss_st = loss_hard

            ae_output = autoencoder(image_ae)
            with torch.no_grad():
                teacher_output_ae = teacher(image_ae)
                teacher_output_ae = (teacher_output_ae - teacher_mean) / teacher_std
            student_output_ae = student(image_ae)[:, out_channels:]
            distance_ae = (teacher_output_ae - ae_output)**2
            distance_stae = (ae_output - student_output_ae)**2
            loss_ae = torch.mean(distance_ae)
            loss_stae = torch.mean(distance_stae)
            loss_total = loss_st + loss_ae + loss_stae

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            scheduler.step()

            if iteration % 10 == 0:
                tqdm_obj.set_description(
                    f"Epoch: {epoch + 1}/{num_epochs} - Iteration: {iteration + 1}/{len(train_loader)} - Current loss: {loss_total.item():.4f}")
                
            # if epoch + 1 in config.eval_epochs:

            #     @torch.no_grad()
            #     def test(test_set, teacher, student, autoencoder, teacher_mean, teacher_std,
            #             q_st_start, q_st_end, q_ae_start, q_ae_end, test_output_dir=None,
            #             desc='Running inference',
            #             map_format='tiff',
            #             threshold=80):
            #         y_true = []
            #         y_score = []
            #         y_class = []
                    
            #         # List to store mismatch samples
            #         mismatches = []

            #         for image, target, path in tqdm(test_set, desc=desc):
            #             orig_width = image.width
            #             orig_height = image.height
            #             image = default_transform(image)
            #             image = image[None]
            #             if on_gpu:
            #                 image = image.cuda()
            #             map_combined, map_st, map_ae = predict(
            #                 image=image, teacher=teacher, student=student,
            #                 autoencoder=autoencoder, teacher_mean=teacher_mean,
            #                 teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
            #                 q_ae_start=q_ae_start, q_ae_end=q_ae_end)
            #             map_combined = torch.nn.functional.pad(map_combined, (4, 4, 4, 4))
            #             map_combined = torch.nn.functional.interpolate(
            #                 map_combined, (orig_height, orig_width), mode='bilinear')
            #             map_combined = map_combined[0, 0].cpu().numpy()

            #             # Scale anomaly map to [0, 255]
            #             map_combined = map_combined * 255
            #             map_combined = map_combined.astype(np.int16)

            #             defect_class = os.path.basename(os.path.dirname(path))
            #             img_nm = os.path.split(path)[1].split('.')[0]
            #             if not os.path.exists(os.path.join(test_output_dir, defect_class)):
            #                 os.makedirs(os.path.join(test_output_dir, defect_class))

            #             if map_format == 'tiff':
            #                 anomaly_map_path = os.path.join(test_output_dir, defect_class, img_nm + '.tiff')
            #                 tifffile.imwrite(anomaly_map_path, map_combined)
            #             elif map_format == 'jpg':
            #                 original_image = Image.open(path).convert('RGB')
            #                 gradient_softness = threshold / 3
            #                 map_combined_normalized = (map_combined - threshold) / gradient_softness
            #                 map_combined_normalized = 1 / (1 + np.exp(-map_combined_normalized))
            #                 colormap = matplotlib.colormaps['jet']
            #                 anomaly_map_image = colormap(map_combined_normalized)
            #                 anomaly_map_image = (anomaly_map_image[:, :, :3] * 255).astype(np.uint8)
            #                 anomaly_map_image = Image.fromarray(anomaly_map_image)
            #                 anomaly_map_image = anomaly_map_image.resize((original_image.width, original_image.height))
            #                 combined_image_width = original_image.width + anomaly_map_image.width
            #                 combined_image_height = original_image.height
            #                 combined_image = Image.new('RGB', (combined_image_width, combined_image_height))
            #                 combined_image.paste(original_image, (0, 0))
            #                 combined_image.paste(anomaly_map_image.convert('RGB'), (original_image.width, 0))
            #                 combined_image_path = os.path.join(test_output_dir, defect_class, img_nm + '.jpg')
            #                 combined_image.save(combined_image_path)
            #             else:
            #                 raise ValueError("Invalid map format specified. Use 'tiff' or 'jpg'.")

            #             y_true_image = 0 if defect_class == 'good' else 1
            #             y_score_image = np.max(map_combined)
            #             y_true.append(y_true_image)
            #             y_score.append(y_score_image)
            #             y_class.append(defect_class)

            #             # Collect mismatch samples (where prediction does not match ground truth)
            #             if (y_true_image == 0 and y_score_image > threshold) or (y_true_image == 1 and y_score_image <= threshold):
            #                 mismatches.append([defect_class, img_nm, y_score_image])

            #         # Calculate metrics for each defect class
            #         defect_classes = set(y_class)
            #         class_metrics = []
            #         for defect_class in defect_classes:
            #             class_indices = [i for i, cls in enumerate(y_class) if cls == defect_class]
            #             y_true_class = [y_true[i] for i in class_indices]
            #             y_score_class = [y_score[i] for i in class_indices]
                        
            #             accuracy_class = np.mean(np.array(y_true_class) == (np.array(y_score_class) > threshold))

            #             if defect_class == 'good':
            #                 precision_class = np.sum((np.array(y_true_class) == 0) & (np.array(y_score_class) <= threshold)) / np.sum(np.array(y_score_class) <= threshold)
            #                 recall_class = np.sum((np.array(y_true_class) == 0) & (np.array(y_score_class) <= threshold)) / np.sum(np.array(y_true_class) == 0)
            #             else:
            #                 precision_class = np.sum((np.array(y_true_class) == 1) & (np.array(y_score_class) > threshold)) / np.sum(np.array(y_score_class) > threshold)
            #                 recall_class = np.sum((np.array(y_true_class) == 1) & (np.array(y_score_class) > threshold)) / np.sum(np.array(y_true_class) == 1)

            #             num_samples_class = len(y_true_class)
                        
            #             class_metrics.append([defect_class, accuracy_class, precision_class, recall_class, num_samples_class])
                    
            #         # Print class metrics as a table
            #         headers = ["Class", "Accuracy", "Precision", "Recall", "Num Samples"]
            #         class_metrics.sort(key=lambda x: x[0])  # Sort by Class
            #         print()
            #         print(tabulate(class_metrics, headers=headers, floatfmt=".4f"))

            #         # Calculate overall metrics
            #         accuracy = np.mean(np.array(y_true) == (np.array(y_score) > threshold))
            #         precision = np.sum((np.array(y_true) == 1) & (np.array(y_score) > threshold)) / np.sum(np.array(y_score) > threshold)
            #         recall = np.sum((np.array(y_true) == 1) & (np.array(y_score) > threshold)) / np.sum(np.array(y_true) == 1)
            #         num_samples = len(y_true)
                    
            #         # Print overall metrics as a table
            #         overall_metrics = [["Overall", accuracy, precision, recall, num_samples]]
            #         print()
            #         print(tabulate(overall_metrics, headers=headers, floatfmt=".4f"))

            #         auc = roc_auc_score(y_true=y_true, y_score=y_score)
                    
            #         # Export mismatches to a CSV file
            #         mismatch_csv_path = os.path.join(test_output_dir, 'mismatch_samples.csv')
            #         with open(mismatch_csv_path, mode='w', newline='') as file:
            #             writer = csv.writer(file)
            #             writer.writerow(['defect_class', 'image_name', 'y_score'])  # Write headers
            #             writer.writerows(mismatches)  # Write mismatch samples

            #         return auc * 100
                
            #     # Compute map normalization parameters
            #     q_st_start, q_st_end, q_ae_start, q_ae_end = map_normalization(
            #         validation_loader=validation_loader, teacher=teacher, student=student,
            #         autoencoder=autoencoder, teacher_mean=teacher_mean,
            #         teacher_std=teacher_std, desc='Final map normalization')
                
            #     # eval config
            #     test_set = ImageFolderWithPath(os.path.join(dataset_path, 'test'))
            #     test_output_dir = os.path.join(train_output_dir, f'epoch_{epoch + 1}')
            #     threshold = 15

            #     # Run evaluation
            #     auc = test(
            #         test_set=test_set,
            #         teacher=teacher, student=student,
            #         autoencoder=autoencoder, teacher_mean=teacher_mean,
            #         teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
            #         q_ae_start=q_ae_start, q_ae_end=q_ae_end,
            #         test_output_dir=test_output_dir, desc='evaluating',
            #         map_format='jpg',
            #         threshold=threshold)
            #     print(f'Final image AUC: {auc:.4f}')

        # Save models periodically
        if epoch % 10 == 0 and epoch > 0:
            torch.save(student.state_dict(), os.path.join(train_output_dir,
                                                         f'student_epoch_{epoch}.pth'))
            torch.save(autoencoder.state_dict(), os.path.join(train_output_dir,
                                                             f'autoencoder_epoch_{epoch}.pth'))

    # Save final models
    torch.save(teacher.state_dict(), os.path.join(train_output_dir, 'teacher_final.pth'))
    torch.save(student.state_dict(), os.path.join(train_output_dir, 'student_final.pth'))
    torch.save(autoencoder.state_dict(), os.path.join(train_output_dir, 'autoencoder_final.pth'))

    # Save teacher normalization parameters
    torch.save(teacher_mean, os.path.join(train_output_dir, 'teacher_mean.pt'))
    torch.save(teacher_std, os.path.join(train_output_dir, 'teacher_std.pt'))

    # Compute and save map normalization parameters
    q_st_start, q_st_end, q_ae_start, q_ae_end = map_normalization(
        validation_loader=validation_loader, teacher=teacher, student=student,
        autoencoder=autoencoder, teacher_mean=teacher_mean,
        teacher_std=teacher_std, desc='Final map normalization')
    torch.save({'q_st_start': q_st_start, 'q_st_end': q_st_end,
                'q_ae_start': q_ae_start, 'q_ae_end': q_ae_end},
               os.path.join(train_output_dir, 'quantiles.pt'))

if __name__ == '__main__':
    main()
