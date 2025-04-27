import random
from random import sample
import argparse
import numpy as np
import os
import pickle
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
from skimage import morphology
from skimage.segmentation import mark_boundaries
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import matplotlib

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import densenet121, resnet18, wide_resnet50_2
from torchvision.transforms import InterpolationMode
import datasets.mvtec as mvtec

# device setup
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser('PaDiM')
    parser.add_argument('--data_path', type=str, default='./datasets')
    parser.add_argument('--save_path', type=str, default='./MvTec_result_Purposed_Model')
    parser.add_argument('--arch', type=str, choices=['resnet18', 'wide_resnet50_2', 'densenet121'], default='densenet121')
    return parser.parse_args()

def find_layer(module, name):
    """Recursively searches for a layer by name in a module."""
    current_name = name.split('.')[0]
    remaining_name = '.'.join(name.split('.')[1:])

    for child_name, child_module in module.named_children():
        if child_name == current_name:
            if not remaining_name:
                return child_module
            else:
                if isinstance(child_module, torch.nn.Module):
                    return find_layer(child_module, remaining_name)
                else:
                    return None
    return None

def main():
    args = parse_args()
    print(f"Running with arguments: {args}")
    print(f"Using device: {device}")

    # Model Loading
    if args.arch == 'resnet18':
        model = resnet18(weights='IMAGENET1K_V1')
        t_d = 448
        d = 100
        feature_layer_ids = ['layer1', 'layer2', 'layer3']
    elif args.arch == 'wide_resnet50_2':
        model = wide_resnet50_2(weights='IMAGENET1K_V1')
        t_d = 1792
        d = 550
        feature_layer_ids = ['layer1', 'layer2', 'layer3']
    elif args.arch == 'densenet121':
        model = densenet121(weights='IMAGENET1K_V1')
        feature_layer_ids = ['features.denseblock1', 'features.denseblock2', 'features.denseblock3']
        c1 = 256
        c2 = 512
        c3 = 1024
        t_d = c1 + c2 + c3
        d = int(t_d * (550.0 / 1792.0))
    
    model.to(device)
    model.eval()

    # Set seeds
    random.seed(1024)
    torch.manual_seed(1024)
    if use_cuda:
        torch.cuda.manual_seed_all(1024)

    if d > t_d:
        print(f"Warning: Reduced dimension d ({d}) is greater than total dimension t_d ({t_d}). Setting d = t_d.")
        d = t_d
    if d <= 0:
         raise ValueError(f"Reduced dimension 'd' must be positive, but got {d}")
    
    idx = torch.tensor(sample(range(0, t_d), d)).to(device)

    # Create save directories
    os.makedirs(args.save_path, exist_ok=True)
    temp_save_dir = os.path.join(args.save_path, f'temp_{args.arch}')
    os.makedirs(temp_save_dir, exist_ok=True)

    # Setup plots
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    fig_img_rocauc = ax[0]
    fig_pixel_rocauc = ax[1]

    total_roc_auc = []
    total_pixel_roc_auc = []

    # Global list to store hook outputs
    outputs = []
    def hook(module, input, output):
        outputs.append(output.cpu().detach())

    for class_name in mvtec.CLASS_NAMES:
        print(f"\n--- Processing class: {class_name} ---")

        # Hook Registration
        registered_hooks = []
        layer_names_for_outputs = []
        print(f"Registering hooks for layers: {feature_layer_ids}")
        for layer_id in feature_layer_ids:
            layer = find_layer(model, layer_id)
            if layer is not None:
                print(f"  Registering hook for: {layer_id}")
                registered_hooks.append(layer.register_forward_hook(hook))
                simple_key = f'layer{len(layer_names_for_outputs)+1}'
                layer_names_for_outputs.append(simple_key)
            else:
                raise RuntimeError(f"Error: Layer '{layer_id}' not found in model {args.arch}.")

        if len(registered_hooks) != 3:
            raise RuntimeError(f"Expected 3 hooks to be registered, but only got {len(registered_hooks)}.")

        # Data Loading
        train_dataset = mvtec.MVTecDataset(args.data_path, class_name=class_name, is_train=True)
        train_dataloader = DataLoader(train_dataset, batch_size=32, pin_memory=True)
        test_dataset = mvtec.MVTecDataset(args.data_path, class_name=class_name, is_train=False)
        test_dataloader = DataLoader(test_dataset, batch_size=32, pin_memory=True)

        train_outputs_dict = OrderedDict([(key, []) for key in layer_names_for_outputs])
        test_outputs_dict = OrderedDict([(key, []) for key in layer_names_for_outputs])

        # Training Phase
        train_feature_filepath = os.path.join(temp_save_dir, f'train_{class_name}.pkl')
        if not os.path.exists(train_feature_filepath):
            print(f"Extracting training features for {class_name}...")
            outputs = []
            for (x, _, _) in tqdm(train_dataloader, desc=f"Extracting train features | {class_name}"):
                with torch.no_grad():
                    _ = model(x.to(device))
                if len(outputs) != len(train_outputs_dict.keys()):
                     raise RuntimeError(f"Hook error: Expected {len(train_outputs_dict.keys())} outputs, got {len(outputs)}")
                for k, v in zip(train_outputs_dict.keys(), outputs):
                    train_outputs_dict[k].append(v)
                outputs = []

            for k, v in train_outputs_dict.items():
                train_outputs_dict[k] = torch.cat(v, 0)
                print(f"  Layer {k} train features shape: {train_outputs_dict[k].shape}")

            embedding_vectors = train_outputs_dict[layer_names_for_outputs[0]]
            for layer_key in layer_names_for_outputs[1:]:
                embedding_vectors = embedding_concat(embedding_vectors, train_outputs_dict[layer_key])
            print(f"  Concatenated train embedding shape: {embedding_vectors.shape}")

            if embedding_vectors.device != idx.device:
                 idx = idx.to(embedding_vectors.device)
            embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
            print(f"  Selected train embedding shape: {embedding_vectors.shape}")

            B, C, H, W = embedding_vectors.size()
            embedding_vectors = embedding_vectors.view(B, C, H * W).numpy()
            mean = np.mean(embedding_vectors, axis=0)
            cov = np.zeros((C, C, H * W), dtype=np.float64)
            I = np.identity(C, dtype=np.float64)
            print("  Calculating covariance matrices...")
            for i in tqdm(range(H * W), desc="Covariance Calculation"):
                cov[:, :, i] = np.cov(embedding_vectors[:, :, i], rowvar=False) + 0.01 * I

            train_outputs_params_dict = {'mean': mean, 'cov': cov}
            print(f"  Saving features to {train_feature_filepath}")
            try:
                with open(train_feature_filepath, 'wb') as f:
                    pickle.dump(train_outputs_params_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            except OverflowError:
                 print("\nError: Cannot serialize - covariance data too large.")
                 for h in registered_hooks: h.remove()
                 return
        else:
            print('Load train set feature from: %s' % train_feature_filepath)
            try:
                with open(train_feature_filepath, 'rb') as f:
                    train_outputs_params_dict = pickle.load(f)
            except Exception as e:
                print(f"Error loading pickled file {train_feature_filepath}: {e}")
                for h in registered_hooks: h.remove()
                return

        # Testing Phase
        print(f"Extracting testing features for {class_name}...")
        gt_list = []
        gt_mask_list = []
        test_imgs = []
        test_outputs_dict = OrderedDict([(key, []) for key in layer_names_for_outputs])
        outputs = []

        for (x, y, mask) in tqdm(test_dataloader, desc=f"Extracting test features | {class_name}"):
            test_imgs.extend(x.cpu().numpy())
            gt_list.extend(y.cpu().numpy())
            gt_mask_list.extend(mask.cpu().numpy())
            with torch.no_grad():
                _ = model(x.to(device))
            if len(outputs) != len(test_outputs_dict.keys()):
                 raise RuntimeError(f"Hook error: Expected {len(test_outputs_dict.keys())} outputs, got {len(outputs)}")
            for k, v in zip(test_outputs_dict.keys(), outputs):
                test_outputs_dict[k].append(v)
            outputs = []

        for k, v in test_outputs_dict.items():
            test_outputs_dict[k] = torch.cat(v, 0)
            print(f"  Layer {k} test features shape: {test_outputs_dict[k].shape}")

        embedding_vectors = test_outputs_dict[layer_names_for_outputs[0]]
        for layer_key in layer_names_for_outputs[1:]:
            embedding_vectors = embedding_concat(embedding_vectors, test_outputs_dict[layer_key])
        print(f"  Concatenated test embedding shape: {embedding_vectors.shape}")

        if embedding_vectors.device != idx.device:
            idx = idx.to(embedding_vectors.device)
        embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
        print(f"  Selected test embedding shape: {embedding_vectors.shape}")

        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W).numpy()
        dist_list = []
        mean_params = train_outputs_params_dict['mean']
        cov_params = train_outputs_params_dict['cov']
        print("  Calculating Mahalanobis distances...")
        for i in tqdm(range(H * W), desc="Mahalanobis Calculation"):
            mean = mean_params[:, i]
            cov = cov_params[:, :, i]
            try:
                inv_cov = np.linalg.inv(cov)
                dist = [mahalanobis(sample[:, i], mean, inv_cov) for sample in embedding_vectors]
            except np.linalg.LinAlgError:
                dist = np.zeros(embedding_vectors.shape[0])
            dist_list.append(dist)

        dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

        # Upsample score map
        dist_list = torch.tensor(dist_list)
        try:
             original_input_size = test_dataset.cropsize
        except AttributeError:
             original_input_size = 224
        score_map = F.interpolate(dist_list.unsqueeze(1), size=original_input_size, 
                                 mode='bilinear', align_corners=False).squeeze().numpy()

        # Apply gaussian smoothing
        print("  Applying Gaussian smoothing...")
        if score_map.ndim == 2:
             score_map = np.expand_dims(score_map, axis=0)

        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)

        # Normalize scores
        max_score = score_map.max()
        min_score = score_map.min()
        if max_score <= min_score:
            scores = np.zeros_like(score_map)
        else:
            scores = (score_map - min_score) / (max_score - min_score)

        # Image-level ROC AUC
        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
        gt_list = np.asarray(gt_list)
        if len(np.unique(gt_list)) > 1:
            fpr, tpr, _ = roc_curve(gt_list, img_scores)
            img_roc_auc = roc_auc_score(gt_list, img_scores)
            total_roc_auc.append(img_roc_auc)
            print('Image ROCAUC: %.3f' % (img_roc_auc))
            fig_img_rocauc.plot(fpr, tpr, label=f'{class_name} img: {img_roc_auc:.3f}')

        # Automatic Thresholding using Otsu
        flat_scores = scores.flatten()
        threshold = 0.5
        try:
            if np.var(flat_scores) > 1e-8:
                 threshold = threshold_otsu(flat_scores)
                 print(f"Otsu threshold: {threshold:.6f}")
            else:
                 threshold = np.mean(flat_scores)
        except Exception as e:
            print(f"Error during Otsu threshold calculation: {e}. Using default threshold 0.5.")
            threshold = 0.5

        # Pixel-level ROC AUC
        gt_mask = np.asarray(gt_mask_list).astype(np.uint8)
        if len(np.unique(gt_mask)) > 1:
            fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
            per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
            total_pixel_roc_auc.append(per_pixel_rocauc)
            print('Pixel ROCAUC: %.3f' % (per_pixel_rocauc))
            fig_pixel_rocauc.plot(fpr, tpr, label=f'{class_name} pixel: {per_pixel_rocauc:.3f}')

        # Plotting
        pictures_save_dir = os.path.join(args.save_path, f'pictures_{args.arch}')
        os.makedirs(pictures_save_dir, exist_ok=True)
        plot_fig(np.array(test_imgs), scores, gt_mask_list, threshold, pictures_save_dir, class_name)

        # Remove hooks
        print("Removing hooks...")
        for h in registered_hooks:
            h.remove()
        registered_hooks = []

    # Final Averaging and Plot Saving
    if total_roc_auc:
        mean_img_roc = np.mean(total_roc_auc)
        print('\nAverage Image ROCAUC: %.3f' % mean_img_roc)
        fig_img_rocauc.title.set_text('Average image ROCAUC: %.3f' % mean_img_roc)
    else:
        print('\nNo image ROCAUC scores to average.')
        fig_img_rocauc.title.set_text('Average image ROCAUC: N/A')
    fig_img_rocauc.legend(loc="lower right")

    if total_pixel_roc_auc:
        mean_pixel_roc = np.mean(total_pixel_roc_auc)
        print('Average Pixel ROCAUC: %.3f' % mean_pixel_roc)
        fig_pixel_rocauc.title.set_text('Average pixel ROCAUC: %.3f' % mean_pixel_roc)
    else:
         print('\nNo pixel ROCAUC scores to average.')
         fig_pixel_rocauc.title.set_text('Average pixel ROCAUC: N/A')
    fig_pixel_rocauc.legend(loc="lower right")

    fig.tight_layout()
    fig_save_path = os.path.join(args.save_path, 'roc_curve.png')
    fig.savefig(fig_save_path, dpi=100)
    print(f"ROC curve plot saved to {fig_save_path}")

def plot_fig(test_img, scores, gts, threshold, save_dir, class_name):
    """Plots the original image, ground truth, heatmap, predicted mask, and segmentation."""
    gts = np.asarray(gts)
    num = len(scores)
    max_s = scores.max()
    min_s = scores.min()
    vmax = max_s * 255. if max_s > min_s else 255.0
    vmin = min_s * 255.

    for i in tqdm(range(num), desc=f"Plotting {class_name}", leave=False):
        img = test_img[i]
        if img.shape[0] == 3 and img.ndim == 3:
             img = img.transpose(1, 2, 0)
        img_denorm = denormalization(img)

        gt = gts[i]
        if gt.ndim == 3 and gt.shape[0] == 1:
            gt = gt.squeeze(0)
        elif gt.ndim == 3 and gt.shape[-1] == 1:
             gt = gt.squeeze(-1)
        elif gt.ndim != 2:
            gt_h, gt_w = scores[i].shape[:2] if scores[i].ndim >= 2 else (224, 224)
            gt = np.zeros((gt_h, gt_w), dtype=np.uint8)

        heat_map = scores[i] * 255
        pred_mask = scores[i].copy()
        pred_mask[pred_mask > threshold] = 1
        pred_mask[pred_mask <= threshold] = 0
        pred_mask = pred_mask.astype(np.uint8)

        try:
            kernel = morphology.disk(4)
            pred_mask_morph = morphology.opening(pred_mask, kernel)
            pred_mask_vis = pred_mask_morph * 255
        except Exception as e:
            pred_mask_vis = pred_mask * 255

        try:
             vis_img = mark_boundaries(img_denorm, pred_mask_morph.astype(bool), color=(1, 0, 0), mode='thick')
        except ValueError as e:
             vis_img = img_denorm
        except Exception as e:
             vis_img = img_denorm

        fig_img, ax_img = plt.subplots(1, 5, figsize=(12, 3))
        fig_img.suptitle(f'{class_name} - Image {i:03d}', fontsize=10)
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

        titles = ['Image', 'GroundTruth', 'Heatmap', 'Pred Mask (Otsu)', 'Segmentation']
        images_to_plot = [img_denorm, gt, heat_map, pred_mask_vis, vis_img]
        cmaps = [None, 'gray', 'jet', 'gray', None]

        for idx, (ax_i, title, img_data, cmap) in enumerate(zip(ax_img, titles, images_to_plot, cmaps)):
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
            ax_i.set_title(title, fontsize=8)
            if idx == 2:
                 im = ax_i.imshow(heat_map, cmap=cmap, norm=norm)
                 ax_i.imshow(img_denorm, cmap='gray', alpha=0.3, interpolation='none')
            else:
                 im = ax_i.imshow(img_data, cmap=cmap)

        left, bottom, width, height = 0.92, 0.15, 0.015, 0.7
        cbar_ax = fig_img.add_axes([left, bottom, width, height])
        if ax_img[2].images:
             mappable = ax_img[2].images[0]
             cb = plt.colorbar(mappable, cax=cbar_ax)
             cb.ax.tick_params(labelsize=8)
             font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 8}
             cb.set_label('Anomaly Score', fontdict=font)
        else:
             cbar_ax.set_visible(False)

        fig_path = os.path.join(save_dir, f"{class_name}_{i:03d}.png")
        try:
             fig_img.savefig(fig_path, dpi=100, bbox_inches='tight')
        except Exception as e:
             print(f"Error saving figure {fig_path}: {e}")
        plt.close(fig_img)

def denormalization(x):
    """Denormalizes image tensor assuming input is (H, W, C) numpy array."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    if x.ndim != 3 or x.shape[-1] != 3:
         if x.ndim == 3 and x.shape[0] == 3:
             x = x.transpose(1, 2, 0)
         else:
              if x.dtype == np.float32 or x.dtype == np.float64:
                   return (x * 255.).clip(0, 255).astype(np.uint8)
              return x.astype(np.uint8)

    x_denorm = (((x * std) + mean) * 255.).clip(0, 255).astype(np.uint8)
    return x_denorm

def embedding_concat(x, y):
    """Concatenates features from two layers by resizing the deeper feature map."""
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    y_resized = F.interpolate(y, size=(H1, W1), mode='bilinear', align_corners=False)
    z = torch.cat((x, y_resized), dim=1)
    return z

if __name__ == '__main__':
    main()