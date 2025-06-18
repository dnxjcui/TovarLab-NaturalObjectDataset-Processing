#!/usr/bin/env python3
"""
utils.py

Common utility functions and constants for fMRI data processing and analysis.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
from pathlib import Path
import nibabel as nib
import sys
from scipy.stats import wasserstein_distance
import itertools
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

# Constants
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# Common regions used across analyses
VISUAL_REGIONS = ["V1", "V2", "V3", "V4", "V8", "PIT", "FFC", "VVC", 
                 "VMV1", "VMV2", "VMV3", "LO1", "LO2", "LO3"]

# Common parameters
DEFAULT_SESSION = 'imagenet01'
DEFAULT_TASK = 'imagenet'
DEFAULT_RUN_COUNT = 10
DEFAULT_SUBJECT_COUNT = 30

def get_region_roi_indices(region):
    """
    Get indices of ROIs for a specific region.
    
    Args:
        region (str): Name of the brain region
        
    Returns:
        tuple: Indices where ROI data equals 1
    """
    from NOD_fmri.validation.nod_utils import get_roi_data
    os.chdir(os.path.join(BASE_DIR, 'NOD_fmri', 'validation')) 
    roi_indices = np.where(get_roi_data(None, region) == 1)
    os.chdir(BASE_DIR) 
    return roi_indices

def load_beta_weights_with_roi_mask(beta_file, roi_file=None):
    """
    Load beta weights for a specific run, optionally masked by an ROI.
    
    Args:
        beta_file (str): Path to beta weights file
        roi_file (str, optional): Path to ROI mask file
        
    Returns:
        numpy.ndarray: Beta weights data, optionally masked by ROI
    """
    beta_img = nib.load(beta_file)
    beta_data = beta_img.get_fdata()
    
    if roi_file is not None:
        roi_img = nib.load(roi_file)
        roi_data = roi_img.get_fdata()
        masked_beta = beta_data[:, roi_data[0] > 0]
    else:
        masked_beta = beta_data
    
    return masked_beta

def get_beta_data(data_dir, subject, session, task, run):
    """
    Get beta data for a specific subject, session, task, and run.
    
    Args:
        data_dir (str): Base directory containing the data
        subject (str): Subject identifier
        session (str): Session identifier
        task (str): Task identifier
        run (str): Run identifier
        
    Returns:
        numpy.ndarray: Beta weights data
        
    Raises:
        FileNotFoundError: If beta file doesn't exist
    """
    beta_path = os.path.join(data_dir, 'derivatives', 'ciftify', subject, 
                            'results', f'ses-{session}_task-{task}_{run}', 
                            f'ses-{session}_task-{task}_{run}_beta.dscalar.nii')
    
    if not os.path.exists(beta_path):
        raise FileNotFoundError(f"Beta file does not exist: {beta_path}")
    
    beta_img = nib.load(beta_path)
    return beta_img.get_fdata()

def compute_rdm_from_betas(beta_weights, metric='correlation'):
    """
    Compute Representational Dissimilarity Matrix from beta weights.
    
    Args:
        beta_weights (numpy.ndarray): Beta weights data
        metric (str): Distance metric to use (default: 'correlation')
        
    Returns:
        numpy.ndarray: RDM matrix
    """
    rdm = squareform(pdist(beta_weights, metric=metric))
    return rdm

def plot_rdm_heatmap(rdm_square, distance_metric, labels, save_dir, 
                     fname='region_region_rdm_heatmap.eps', 
                     title='Region-Region RDM Heatmap', 
                     clim=(0.0, 1.0), fontsize=None):
    """
    Plot and save the RDM heatmap.
    
    Args:
        rdm_square (numpy.ndarray): Square RDM matrix
        distance_metric (str): Name of the distance metric used
        labels (list): Labels for the axes
        save_dir (str): Directory to save the plot
        fname (str): Filename for the saved plot
        title (str): Plot title
        clim (tuple): Color limits for the heatmap
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(rdm_square, cmap='plasma', interpolation='none')
    
    if clim is not None:
        im.set_clim(clim[0], clim[1])
    
    # plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)

    # colorbar
    cbar = plt.colorbar(im, ax=ax)

    if fontsize is not None:
        plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=45, ha='right', fontsize=fontsize)
        plt.yticks(ticks=np.arange(len(labels)), labels=labels, fontsize=fontsize)

        heatmap_path = os.path.join(save_dir, fname.split('.')[0] + f'_{fontsize}pt.' + fname.split('.')[-1])
        plt.savefig(heatmap_path, dpi=300, transparent=True)
        print(f"Heatmap saved at {heatmap_path}")

        plt.close(fig)
        return

    for fontsize in range(20, 10, -1):
        plt.xticks(ticks=np.arange(len(labels)), labels=labels, 
                  rotation=45, ha='right', fontsize=fontsize)
        plt.yticks(ticks=np.arange(len(labels)), labels=labels, 
                  fontsize=fontsize)

        heatmap_path = os.path.join(save_dir, 
                                  fname.split('.')[0] + f'_{fontsize}pt.' + 
                                  fname.split('.')[-1])
        # plt.savefig(heatmap_path, dpi=300, transparent=True)
        plt.savefig(heatmap_path, dpi=300, transparent=True)
        print(f"Heatmap saved at {heatmap_path}")
        

    plt.close(fig)

def get_labels(data_dir, subject, session, task, run):
    """
    Get labels for a specific subject, session, task, and run.
    
    Args:
        data_dir (str): Base directory containing the data
        subject (str): Subject identifier
        session (str): Session identifier
        task (str): Task identifier
        run (str): Run identifier
        
    Returns:
        tuple: (label_categories, fnames) - category names and filenames
        
    Raises:
        FileNotFoundError: If labels file doesn't exist
    """
    labels_path = os.path.join(data_dir, 'derivatives', 'ciftify', subject, 
                               'results', f'ses-{session}_task-{task}_{run}', 
                               f'ses-{session}_task-{task}_{run}_label.txt')
    
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file does not exist: {labels_path}")
    
    with open(labels_path, 'r') as f:
        labels = f.read().strip().split('\n')
    
    # Original format as label/label_number 
    label_categories = [(label.split('/'))[0] for label in labels] 
    label_categories = [(label.split('_'))[0] for label in label_categories]

    fnames = [(label.split('/'))[1 if len(label.split('/')) == 2 else 0] for label in labels] 
    
    return label_categories, fnames

def plot_mds_visualization(rdm, all_regions, save_dir):
    """
    Plot MDS visualization of the RDM and save the figure.
    
    Args:
        rdm (numpy.ndarray): RDM matrix
        all_regions (list): List of region names
        save_dir (str): Directory to save the plot
    """
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coords = mds.fit_transform(rdm)

    plt.figure(figsize=(10, 8))
    plt.scatter(coords[:, 0], coords[:, 1], marker='o', color='blue')
    
    for i, region in enumerate(all_regions):
        plt.annotate(region, (coords[i, 0], coords[i, 1]), fontsize=12)

    plt.title('MDS of RDM')
    plt.xlabel('MDS Dimension 1')
    plt.ylabel('MDS Dimension 2')
    plt.grid(True)
    
    save_path = os.path.join(save_dir, 'mds_plot.png')
    plt.savefig(save_path)
    plt.close()

def compute_wasserstein_distance_matrix(all_fmri, regions, plot_1d_vectors=False):
    """ 
    Compute a distance matrix using Wasserstein distance between regions.
    This function computes the Wasserstein distance between regions based on their flattened fMRI data. 
    It assumes that all_fmri is a dictionary where keys are region names and values are 3D numpy arrays of shape [stimuli, voxels, subjects].
    
    Args:
        all_fmri (dict): Dictionary with region names as keys and 3D arrays as values
        regions (list): List of region names to compare
        plot_1d_vectors (bool): Whether to return 1D vectors instead of distance matrix
        
    Returns:
        numpy.ndarray or dict: Distance matrix or dictionary of 1D vectors
    """
    n = len(regions)
    D = np.zeros((n, n))

    if plot_1d_vectors:
        all_1d_vectors = {}

    for i, j in itertools.combinations(range(n), 2):
        r1, r2 = regions[i], regions[j]
        data1 = all_fmri[r1]  # [stimuli, voxels, subjects]
        data2 = all_fmri[r2]
        vec1 = data1.reshape(1, -1).flatten()
        vec2 = data2.reshape(1, -1).flatten()        

        if not plot_1d_vectors:
            dist = wasserstein_distance(vec1, vec2)
            D[i, j] = D[j, i] = dist
        else:
            all_1d_vectors[r1] = vec1
            all_1d_vectors[r2] = vec2
    if not plot_1d_vectors:
        return D
    else:
        return all_1d_vectors 

def compute_crossval_mapping_distance_matrix(all_fmri, regions, alpha=1.0, n_splits=5):
    """ 
    Compute a distance matrix using cross-validation mapping between regions.
    This function computes the distance between regions based on the cross-validated prediction error of a Ridge regression model.
    
    Args:
        all_fmri (dict): Dictionary with region names as keys and 3D arrays as values
        regions (list): List of region names to compare
        alpha (float): Ridge regression regularization parameter
        n_splits (int): Number of cross-validation splits
        
    Returns:
        numpy.ndarray: Distance matrix based on cross-validated mapping errors
    """
    n = len(regions)
    D = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        r1, r2 = regions[i], regions[j]
        data1 = all_fmri[r1]
        data2 = all_fmri[r2]
        stimuli, vox1, subs = data1.shape
        errors = []
        for s in range(subs):
            X = data1[:, :, s]
            Y = data2[:, :, s]
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            for train_idx, test_idx in kf.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                Y_train, Y_test = Y[train_idx], Y[test_idx]
                model = Ridge(alpha=alpha)
                model.fit(X_train, Y_train)
                Y_pred = model.predict(X_test)
                error = np.mean(np.linalg.norm(Y_pred - Y_test, axis=1))
                errors.append(error)
        D[i, j] = D[j, i] = np.mean(errors)
    return D 