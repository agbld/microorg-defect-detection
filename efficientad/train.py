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
    ImageFolderWithoutTarget, InfiniteDataloader
from utils import predict, map_normalization, teacher_normalization, default_transform, train_transform, seed, on_gpu, out_channels, image_size

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
