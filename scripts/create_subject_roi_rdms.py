#!/usr/bin/env python3
"""
create_subject_roi_rdms.py

Create Representational Dissimilarity Matrices (RDMs) for each ROI using beta weights.

Usage:
  python create_subject_roi_rdms.py --data_dir ds004496 --output_dir derivatives/subject_roi_rdms --task imagenet01
"""
import os
import glob
import argparse
import numpy as np
from scipy.spatial.distance import pdist, squareform
import nibabel as nib
from tqdm import tqdm
import sys
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from NOD_fmri.validation.nod_utils import get_roi_data
from utils import (
    get_region_roi_indices, 
    load_beta_weights_with_roi_mask, 
    compute_rdm_from_betas, 
    get_beta_data,
    get_labels,
    plot_heatmap,
    VISUAL_REGIONS,
    DEFAULT_SESSION,
    DEFAULT_TASK,
    DEFAULT_RUN_COUNT,
    DEFAULT_SUBJECT_COUNT
)


def parse_args():
    parser = argparse.ArgumentParser(description="Create RDMs for each ROI using beta weights.")
    parser.add_argument('--data_dir', type=str, required=True, 
                       help='Path to the BIDS data directory (e.g., ds004496)')
    parser.add_argument('--output_dir', type=str, required=True, 
                       help='Directory to save the RDMs')
    parser.add_argument('--task', type=str, default='imagenet',
                       help='Task to create RDMs for (default: imagenet)')
    return parser.parse_args()


def plot_rdm_no_background(rdm_square, subject, session, region, distance_metric, save_dir):
    """
    Plot and save the RDM heatmap with no background (transparent, no axes).
    This is a specialized version for the specific styling needs of this script.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(rdm_square, cmap='viridis', interpolation='none')
    im.set_clim(0.0, 1.0)
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    background_dir = os.path.join(save_dir, subject, 'figures', 'rdm_no_background')
    if not os.path.exists(background_dir):
        os.makedirs(background_dir, exist_ok=True)
    
    heatmap_path = os.path.join(background_dir, f'{region}_roi_rdm_heatmap.png')
    
    # Remove axes and ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Create a border around the heatmap
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(4)
        spine.set_color('black')

    plt.savefig(heatmap_path, dpi=70, transparent=True)
    print(f"No-background heatmap saved at {heatmap_path}")
    plt.close(fig)


def main():
    regions = VISUAL_REGIONS
    subject_n = DEFAULT_SUBJECT_COUNT
    session = DEFAULT_SESSION
    task = DEFAULT_TASK
    run_n = DEFAULT_RUN_COUNT
    data_dir = "C:/Users/BrainInspired/Documents/GitHub/Seeds/NOD"
    save_dir = "C:/Users/BrainInspired/Documents/GitHub/Seeds/Nick_RDMs/outputs"
    distance_metric = 'cosine'
    visualize = True

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    all_runs = [f"run-{i+1}" for i in range(run_n)]
    all_subjects = [f"sub-{i+1:02d}" for i in range(subject_n)]

    region_roi_indices = {}
    for region in regions:
        roi_indices = get_region_roi_indices(region)
        
        print(f"Region: {region}, Indices shape: {roi_indices[0].shape}")
        region_roi_indices[region] = roi_indices    

    prev_categories = None
    # os.chdir(os.path.join(BASE_DIR, '../'))
    for subject in all_subjects:
        print(f"Processing subject: {subject}")
        all_beta_data = None
        all_categories = []
        all_fnames = []

        # fetch all the data from the runs for the subject and consolidate
        for run in all_runs:
            beta_data = get_beta_data(data_dir, subject, session, task, run)
            categories, fnames = get_labels(data_dir, subject, session, task, run)

            num_conditions, _ = beta_data.shape
            assert num_conditions == len(categories)

            if all_beta_data is None:
                all_beta_data = beta_data
            else:
                all_beta_data = np.append(all_beta_data, beta_data, axis=0)
            
            all_categories.extend(categories)
            all_fnames.extend(fnames)
        
        # for each region, find the active voxels and compute the RDM
        for region in regions:
            print(f"Processing region: {region} for subject: {subject} . . .")
            roi_indices = region_roi_indices[region]

            roi_beta_data = all_beta_data[:, roi_indices[0]]
            
            unique_categories = sorted(set(all_categories))
            label_to_index = {label: i for i, label in enumerate(unique_categories)}
            sorted_indices = sorted(range(len(all_categories)), key=lambda i: label_to_index[all_categories[i]])
            sorted_categories = [all_categories[i] for i in sorted_indices]

            sorted_activations = roi_beta_data[sorted_indices, :]
            sorted_fnames = [all_fnames[i] for i in sorted_indices]
            full_fpaths = [os.path.join(category, fname) for category, fname in zip(sorted_categories, sorted_fnames)]

            rdm_square = compute_rdm_from_betas(sorted_activations, metric=distance_metric)
            
            if visualize:
                # Use the specialized no-background version for specific styling needs
                plot_rdm_no_background(rdm_square, subject, session, region, distance_metric, save_dir)
                
                # Also create a standard version using utils.py function
                standard_save_dir = os.path.join(save_dir, subject, 'figures', 'standard')
                plot_heatmap(
                    rdm_square, 
                    [], # no labels for cleaner look
                    standard_save_dir, 
                    fname=f'{region}_roi_rdm_standard.png',
                    title=f'{subject} {region} RDM',
                    clim=(0.0, 1.0),
                    fontsize=12
                )
            
            # save out RDM as a vector
            rdm_vector = squareform(rdm_square)
            rdm_vector_path = os.path.join(save_dir, subject, f'{region}_roi_rdm_vector.npy')
            if not os.path.exists(os.path.join(save_dir, subject)):
                os.makedirs(os.path.join(save_dir, subject), exist_ok=True)
            np.save(rdm_vector_path, rdm_vector)
            # save out all the fnames as a text file
            fnames_path = os.path.join(save_dir, subject, f'filenames.txt')
            with open(fnames_path, 'w') as f:
                for fpath in full_fpaths:
                    f.write(f"{fpath}\n")

            # make sure labels are sorted the same and consistent across subjects
            if prev_categories is not None:
                for i, label in enumerate(sorted_categories):
                    if label != prev_categories[i]:
                        print(f"Label mismatch for {subject} in {region}: {label} vs {prev_categories[i]}")
                        raise ValueError("Labels do not match across runs.")     
            prev_categories = sorted_categories

        quit()


if __name__ == "__main__":
    main() 