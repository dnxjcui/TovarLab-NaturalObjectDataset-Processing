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


def parse_args():
    parser = argparse.ArgumentParser(description="Create RDMs for each ROI using beta weights.")
    parser.add_argument('--data_dir', type=str, required=True, 
                       help='Path to the BIDS data directory (e.g., ds004496)')
    parser.add_argument('--output_dir', type=str, required=True, 
                       help='Directory to save the RDMs')
    parser.add_argument('--task', type=str, default='imagenet',
                       help='Task to create RDMs for (default: imagenet)')
    return parser.parse_args()


def get_roi_indices(region):
    """Get indices of ROIs; calls on get_roi_data and simply abstracts it."""
    os.chdir(os.path.join(BASE_DIR, 'NOD_fmri', 'validation')) 
    roi_indices = np.where(get_roi_data(None, region) == 1)
    os.chdir(BASE_DIR) 
    return roi_indices


def load_beta_weights(beta_file, roi_file=None):
    """Load beta weights for a specific run, optionally masked by an ROI."""
    beta_img = nib.load(beta_file)
    beta_data = beta_img.get_fdata()
    
    if roi_file is not None:
        roi_img = nib.load(roi_file)
        roi_data = roi_img.get_fdata()
        # Mask beta data with ROI
        masked_beta = beta_data[:, roi_data[0] > 0]
    else:
        masked_beta = beta_data
    
    return masked_beta


def compute_rdm(beta_weights, metric='correlation'):
    """Compute RDM from beta weights using correlation distance."""
    rdm = squareform(pdist(beta_weights, metric=metric))
    return rdm


def get_beta_data(data_dir, subject, session, task, run):
    """Get beta data for a specific subject, session, task, and run."""
    beta_path = os.path.join(data_dir, 'derivatives', 'ciftify', subject, 
                             'results', f'ses-{session}_task-{task}_{run}', 
                             f'ses-{session}_task-{task}_{run}_beta.dscalar.nii')
    
    if not os.path.exists(beta_path):
        raise FileNotFoundError(f"Beta file does not exist: {beta_path}")
    
    beta_img = nib.load(beta_path)
    return beta_img.get_fdata()


def get_labels(data_dir, subject, session, task, run):
    """Get labels for a specific subject, session, task, and run."""
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


def plot_rdm(rdm_square, subject, session, region, distance_metric, save_dir):
    """Plot and save the RDM heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(rdm_square, cmap='viridis', interpolation='none')
    # plt.colorbar(im, ax=ax, label=f'{distance_metric} Distance')
    im.set_clim(0.0, 1.0)
    
    # ax.set_title(f'{subject}, {session}, {region}')
    plt.tight_layout()
    
    # if not os.path.exists(os.path.join(save_dir, subject, 'figures')):
    #     os.makedirs(os.path.join(save_dir, subject, 'figures'), exist_ok=True)
    
    # heatmap_path = os.path.join(save_dir, subject, 'figures', f'{region}_roi_rdm_heatmap.png')
    # plt.savefig(heatmap_path, dpi=300)
    if not os.path.exists(os.path.join(save_dir, subject, 'figures', 'rdm_no_background')):
        os.makedirs(os.path.join(save_dir, subject, 'figures', 'rdm_no_background'), exist_ok=True)
    
    heatmap_path = os.path.join(save_dir, subject, 'figures', 'rdm_no_background', f'{region}_roi_rdm_heatmap.png')
    # axes off
    ax.axis('off')
    # ticks off
    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig(heatmap_path, dpi=70, transparent=True)
    print(f"Heatmap saved at {heatmap_path}")

    plt.close(fig)


def main():
    regions = ["V1", "V2", "V3", "V4", "V8", "PIT", "FFC", "VVC", "VMV1", "VMV2", "VMV3", "LO1", "LO2", "LO3"]
    subject_n = 30
    session = 'imagenet01'
    task = 'imagenet'
    run_n = 10
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
        roi_indices = get_roi_indices(region)
        
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

            rdm_square = compute_rdm(sorted_activations, metric=distance_metric)
            
            if visualize:
                plot_rdm(rdm_square, subject, session, region, distance_metric, save_dir)
            
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