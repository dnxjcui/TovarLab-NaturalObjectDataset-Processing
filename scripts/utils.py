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
import pickle

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

def plot_heatmap(data_matrix, labels, save_dir, 
                 fname='region_region_rdm_heatmap.eps', 
                 title='Region-Region RDM Heatmap', 
                 clim=(0.0, 1.0), fontsize=None):
    """
    Plot and save a heatmap for either square RDM matrices or rectangular matrices.
    
    Args:
        data_matrix (numpy.ndarray): Square RDM matrix or rectangular matrix
        labels (list): Labels for the axes (for square) or tuple of (y_labels, x_labels) for rectangular
        save_dir (str): Directory to save the plot
        fname (str): Filename for the saved plot
        title (str): Plot title
        clim (tuple): Color limits for the heatmap
        fontsize (int): Font size for labels
    """
    rows, cols = data_matrix.shape
    is_square = (rows == cols)
    
    # Handle diagonal fill for square matrices only
    if is_square:
        np.fill_diagonal(data_matrix, np.nan)

    # Determine appropriate figure size based on matrix dimensions
    if is_square:
        fig, ax = plt.subplots(figsize=(10, 8))
        aspect = 'equal'
    else:
        # For rectangular matrices, adjust figure size dynamically
        aspect_ratio = cols / rows
        if aspect_ratio > 2:  # Wide matrix
            fig, ax = plt.subplots(figsize=(12, 6))
        elif aspect_ratio < 0.5:  # Tall matrix
            fig, ax = plt.subplots(figsize=(8, 12))
        else:  # Moderate rectangle
            fig, ax = plt.subplots(figsize=(10, 8))
        aspect = 'auto'

    im = ax.imshow(data_matrix, cmap='plasma', interpolation='none', aspect=aspect)

    if clim is not None:
        im.set_clim(clim[0], clim[1])
        vmin, vmax = clim[0], clim[1]
    else:
        # Get the actual data range (excluding NaNs)
        vmin = np.nanmin(data_matrix)
        vmax = np.nanmax(data_matrix)
        im.set_clim(vmin, vmax)

    os.makedirs(save_dir, exist_ok=True)

    # Handle labels for square vs rectangular matrices
    if is_square:
        x_labels = y_labels = labels
    else:
        if isinstance(labels, tuple) and len(labels) == 2:
            y_labels, x_labels = labels
        else:
            raise ValueError(f"Labels must be a tuple of length 2: {labels}")

    colorbar = True
    if fontsize is not None:
        plt.xticks(ticks=np.arange(len(x_labels)), labels=x_labels, rotation=45, ha='right', fontsize=fontsize)
        plt.yticks(ticks=np.arange(len(y_labels)), labels=y_labels, fontsize=fontsize)

        heatmap_path = os.path.join(save_dir, fname.split('.')[0] + f'_{fontsize}pt.' + fname.split('.')[-1])

        # for file in os.listdir(save_dir):
        #     # if file.endswith('.eps') or file.endswith('.svg') or file.endswith('.png'):
        #     if file.split('_')[2] == 'wasserstein':
        #         os.remove(os.path.join(save_dir, file))

        if colorbar:
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_ticks([vmin, vmax])
            cbar.set_ticklabels(['Low Distance', 'High Distance'], fontsize=fontsize)

        if fname.split('.')[-1] == 'eps':
            plt.savefig(heatmap_path, dpi=300, transparent=True, format='eps')
        elif fname.split('.')[-1] == 'svg':
            plt.savefig(heatmap_path, dpi=300, transparent=True, format='svg')
        else:
            plt.savefig(heatmap_path, dpi=300, transparent=True)
        print(f"Heatmap saved at {heatmap_path}")

        plt.close(fig)
        return

    for fontsize in range(20, 10, -1):
        plt.xticks(ticks=np.arange(len(x_labels)), labels=x_labels, 
                  rotation=45, ha='right', fontsize=fontsize)
        plt.yticks(ticks=np.arange(len(y_labels)), labels=y_labels, 
                  fontsize=fontsize)

        heatmap_path = os.path.join(save_dir, 
                                  fname.split('.')[0] + f'_{fontsize}pt.' + 
                                  fname.split('.')[-1])
        if colorbar and fontsize == 20:
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_ticks([vmin, vmax])
            cbar.set_ticklabels(['Low Distance', 'High Distance'], fontsize=fontsize)

        if fname.split('.')[-1] == 'eps':
            plt.savefig(heatmap_path, dpi=300, transparent=True, format='eps')
        elif fname.split('.')[-1] == 'svg':
            plt.savefig(heatmap_path, dpi=300, transparent=True, format='svg')
        else:
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

def plot_mds_visualization(rdm, all_regions, save_dir, fname='mds_plot.png', fontsize=12):
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
        plt.annotate(region, (coords[i, 0], coords[i, 1]), fontsize=fontsize)

    # plt.title('MDS of RDM', )
    plt.xlabel('MDS Dimension 1', fontsize=fontsize)
    plt.ylabel('MDS Dimension 2', fontsize=fontsize)
    plt.grid(True)
    
    save_path = os.path.join(save_dir, fname)
    if fname.split('.')[-1] == 'png':
        plt.savefig(save_path, dpi=300, transparent=True)
    elif fname.split('.')[-1] == 'eps':
        plt.savefig(save_path, dpi=300, transparent=True, format='eps')
    elif fname.split('.')[-1] == 'svg':
        plt.savefig(save_path, dpi=300, transparent=True, format='svg')
    else:
        plt.savefig(save_path, dpi=300, transparent=True)

    print(f"MDS plot saved at {save_path}")

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


def get_fmri_data(data_dir, regions=None, subject_n=None, session=None, task=None, run_n=None, 
                  handle_missing_files=False):
    """
    Load and consolidate fMRI data across subjects, runs, and regions.
    
    Args:
        data_dir (str): Base directory containing the fMRI data
        regions (list, optional): List of brain regions to process. Defaults to VISUAL_REGIONS
        subject_n (int, optional): Number of subjects to process. Defaults to DEFAULT_SUBJECT_COUNT
        session (str, optional): Session identifier. Defaults to DEFAULT_SESSION
        task (str, optional): Task identifier. Defaults to DEFAULT_TASK
        run_n (int, optional): Number of runs to process. Defaults to DEFAULT_RUN_COUNT
        handle_missing_files (bool): Whether to handle missing files gracefully. Defaults to False
        
    Returns:
        dict: Dictionary with region names as keys and 3D arrays as values.
              Each array has shape (num_conditions, num_voxels, num_subjects)
              
    Raises:
        FileNotFoundError: If data directory doesn't exist or required files are missing
    """
    # Set defaults
    if regions is None:
        regions = VISUAL_REGIONS
    if subject_n is None:
        subject_n = DEFAULT_SUBJECT_COUNT
    if session is None:
        session = DEFAULT_SESSION
    if task is None:
        task = DEFAULT_TASK
    if run_n is None:
        run_n = DEFAULT_RUN_COUNT
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    # Create cache directory
    cache_dir = os.path.join(BASE_DIR, 'outputs', 'fmri_data')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create cache filename based on parameters
    regions_str = "_".join(sorted(regions))
    handle_missing_str = "with_missing" if handle_missing_files else "no_missing"
    cache_filename = f"fmri_data_sub{subject_n}_run{run_n}_{session}_{task}_{handle_missing_str}_{hash(regions_str) % 10000:04d}.pkl"
    cache_path = os.path.join(cache_dir, cache_filename)
    
    # Check if cached data exists
    if os.path.exists(cache_path):
        print(f"Loading cached fMRI data from: {cache_filename}")
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            print(f"Successfully loaded cached data with {len(cached_data)} regions")
            return cached_data
        except Exception as e:
            print(f"Warning: Failed to load cache file {cache_filename}: {e}")
            print("Proceeding with fresh computation...")
    
    print(f"Computing fMRI data from scratch (will cache as: {cache_filename})...")

    all_runs = [f"run-{i+1}" for i in range(run_n)]
    all_subjects = [f"sub-{i+1:02d}" for i in range(subject_n)]

    # Get ROI indices for each region
    print("Getting ROI indices for regions...")
    region_roi_indices = {}
    for region in regions:
        roi_indices = get_region_roi_indices(region)
        print(f"Region: {region}, Indices shape: {roi_indices[0].shape}")
        region_roi_indices[region] = roi_indices    

    # Initialize data structure
    all_fmri = {region: None for region in regions}
    prev_categories = None
    
    # Process each subject
    for subject_idx, subject in enumerate(all_subjects):
        print(f"Processing subject: {subject} ({subject_idx + 1}/{len(all_subjects)})")
        all_beta_data = None
        all_categories = []
        all_fnames = []

        # Fetch all data from runs for this subject and consolidate
        for run in all_runs:
            try:
                beta_data = get_beta_data(data_dir, subject, session, task, run)
                categories, fnames = get_labels(data_dir, subject, session, task, run)

                num_conditions, _ = beta_data.shape
                assert num_conditions == len(categories), f"Mismatch: {num_conditions} conditions vs {len(categories)} labels"

                if all_beta_data is None:
                    all_beta_data = beta_data
                else:
                    all_beta_data = np.append(all_beta_data, beta_data, axis=0)
                
                all_categories.extend(categories)
                all_fnames.extend(fnames)
                
            except FileNotFoundError as e:
                if handle_missing_files:
                    print(f"Warning: Skipping {subject} {run} due to missing file: {e}")
                    continue
                else:
                    raise e
        
        if all_beta_data is None:
            if handle_missing_files:
                print(f"Warning: No data found for {subject}, skipping")
                continue
            else:
                raise FileNotFoundError(f"No data found for {subject}")

        # Process each region for this subject
        for region in regions:
            print(f"Processing region: {region} for subject: {subject}")
            roi_indices = region_roi_indices[region]
            roi_beta_data = all_beta_data[:, roi_indices[0]]
            
            # Sort data by categories for consistency across subjects
            unique_categories = sorted(set(all_categories))
            label_to_index = {label: i for i, label in enumerate(unique_categories)}
            sorted_indices = sorted(range(len(all_categories)), key=lambda i: label_to_index[all_categories[i]])
            sorted_categories = [all_categories[i] for i in sorted_indices]
            sorted_activations = roi_beta_data[sorted_indices, :]

            # Ensure labels are consistent across subjects
            if prev_categories is not None:
                for i, label in enumerate(sorted_categories):
                    if i < len(prev_categories) and label != prev_categories[i]:
                        if handle_missing_files:
                            print(f"Warning: Label mismatch for {subject} in {region}: {label} vs {prev_categories[i]}")
                        else:
                            raise ValueError(f"Label mismatch for {subject} in {region}: {label} vs {prev_categories[i]}")
            prev_categories = sorted_categories

            # Reshape and stack data across subjects
            sorted_activations_reshaped = sorted_activations.reshape(sorted_activations.shape[0], -1, 1)
            
            if all_fmri[region] is None:
                all_fmri[region] = sorted_activations_reshaped
            else:
                all_fmri[region] = np.concatenate([
                    all_fmri[region], 
                    sorted_activations_reshaped
                ], axis=2)
                print(f"Shape of all_fmri[{region}]: {all_fmri[region].shape}")

    print("fMRI data loading completed.")
    
    # Save to cache
    print(f"Saving fMRI data to cache: {cache_filename}")
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(all_fmri, f)
        print(f"Successfully cached fMRI data ({len(all_fmri)} regions)")
    except Exception as e:
        print(f"Warning: Failed to save cache file {cache_filename}: {e}")
    
    return all_fmri

 