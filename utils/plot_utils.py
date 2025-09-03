import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import get_cmap
import os
from pathlib import Path
import scipy.stats


def create_custom_colormap():
    """
    Creates a custom colormap for segmentation visualization.
    """
    nipy_spectral = get_cmap('nipy_spectral', 12)
    colors = [nipy_spectral(i) for i in range(12)]
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(boundaries=np.arange(13) - 0.5, ncolors=12)
    return cmap, norm

def plot_random_slice(volume1, pred, gt, save_base_path, z_slice):
    """
    Plots a single slice of the volume, predicted mask, and ground truth.

    Args:
        volume1 (np.ndarray): The input image volume.
        pred (np.ndarray): The predicted segmentation mask.
        gt (np.ndarray): The ground truth segmentation mask.
        save_base_path (str): Base path to save the image (e.g., 'patientX').
                              The slice number will be appended.
        z_slice (int): The z-index of the slice to plot.
    """
    save_path = f"{save_base_path}_slice_{z_slice}.png"
    cmap, norm = create_custom_colormap()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Volume
    axes[0].imshow(np.rot90(volume1[:, :, z_slice]), cmap='gray')
    axes[0].set_title('Volume')
    axes[0].axis('off')

    # Predicted
    axes[1].imshow(np.rot90(volume1[:, :, z_slice]), cmap='gray')
    axes[1].imshow(np.rot90(pred[:, :, z_slice]), cmap=cmap, norm=norm, alpha=0.2)
    axes[1].set_title('Predicted')
    axes[1].axis('off')

    # Ground Truth
    axes[2].imshow(np.rot90(volume1[:, :, z_slice]), cmap='gray')
    axes[2].imshow(np.rot90(gt[:, :, z_slice]), cmap=cmap, norm=norm, alpha=0.2)
    axes[2].set_title('Ground Truth')
    axes[2].axis('off')

    plt.savefig(save_path)
    plt.close('all')



def plot_uncertainty(volume1, pred, gt, prob, save_base_path, z_slice):
    """
    Plots a single slice of the volume, predicted mask, and ground truth.

    Args:
        volume1 (np.ndarray): The input image volume.
        pred (np.ndarray): The predicted segmentation mask.
        gt (np.ndarray): The ground truth segmentation mask.
        save_base_path (str): Base path to save the image (e.g., 'patientX').
                              The slice number will be appended.
        z_slice (int): The z-index of the slice to plot.
    """
    save_path = os.path.join(save_base_path,f"{save_base_path}_slice_{z_slice}.png")
    cmap, norm = create_custom_colormap()


    # Entropy over the class axis (axis=-1)
    epsilon = 1e-10  # or 1e-8
    mean_probs_safe = np.clip(prob, epsilon, 1.0)

    # Compute entropy safely
    predictive_entropy = scipy.stats.entropy(mean_probs_safe, axis=0)  # Shape: (H, W)


    fig, axes = plt.subplots(1, 4, figsize=(15, 5))

    # Volume
    axes[0].imshow(np.rot90(volume1[:, :, z_slice]), cmap='gray')
    axes[0].set_title('Volume')
    axes[0].axis('off')

    # Predicted
    axes[1].imshow(np.rot90(volume1[:, :, z_slice]), cmap='gray')
    axes[1].imshow(np.rot90(pred[:, :, z_slice]), cmap=cmap, norm=norm, alpha=0.3)
    axes[1].set_title('Predicted')
    axes[1].axis('off')

    # Ground Truth
    axes[2].imshow(np.rot90(volume1[:, :, z_slice]), cmap='gray')
    axes[2].imshow(np.rot90(gt[:, :, z_slice]), cmap=cmap, norm=norm, alpha=0.3)
    axes[2].set_title('Ground Truth')
    axes[2].axis('off')

    # Uncertainty
    axes[3].imshow(np.rot90(volume1[:, :, z_slice]), cmap='gray')
    axes[3].imshow(np.rot90(predictive_entropy[:, :, z_slice]), cmap='jet',alpha=0.45)
    axes[3].set_title('Uncertainty Map')
    axes[3].axis('off')

    plt.savefig(save_path)
    plt.close('all')