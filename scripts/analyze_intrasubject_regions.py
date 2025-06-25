#!/usr/bin/env python3
"""
analyze_subject_rdms.py

Create MDS plots for subject-level RDMs across different regions of the brain.

Usage:
  python analyze_subject_rdms.py --rdm_dir outputs/ --subject_n 1
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
from pathlib import Path
import sys
from typing import Dict, List
from utils import (
    compute_rdm_from_betas,
    plot_mds_visualization,
    plot_heatmap,
    get_region_roi_indices,
    get_beta_data,
    get_labels,
    compute_wasserstein_distance_matrix,
    VISUAL_REGIONS,
    DEFAULT_SESSION,
    DEFAULT_TASK,
    DEFAULT_RUN_COUNT,
    DEFAULT_SUBJECT_COUNT
)


def load_rdms(rdm_dir, subject_n) -> Dict[str, np.ndarray]:
    """
    Load RDMs for a specific subject from the specified directory.
    """
    rdm_files = list(Path(rdm_dir).glob(f'sub-{subject_n:02d}/*.npy'))
    if not rdm_files:
        raise FileNotFoundError(f"No RDM files found in {rdm_dir} for subject {subject_n}.")
    
    rdms = {}
    
    for rdm_file in rdm_files:
        region = rdm_file.stem.split('_')[0]  # Extract region name from filename
        rdms[region] = np.load(rdm_file)
    
    return rdms


# Removed duplicated functions - now using imports from utils.py


def comparison_of_rdms():
    """
    Compare RDMs across different regions of the brain for each subject.
    """
    rdm_dir = 'C:/Users/BrainInspired/Documents/GitHub/NaturalObjectDataset-Processing/Nick_RDMs/outputs'

    # hard coding to maintain order
    regions = VISUAL_REGIONS

    for subject_n in range(1, 31):    
        rdms = load_rdms(rdm_dir, subject_n)
        rdm_data = np.array([rdms[region] for region in regions])

        rdm = compute_rdm_from_betas(rdm_data, metric='correlation') # region-region RDM

        save_dir = os.path.join(rdm_dir, f'sub-{subject_n:02d}', 'mds_plots')
        os.makedirs(save_dir, exist_ok=True)

        all_regions = list(rdms.keys())
        plot_mds_visualization(rdm, all_regions, save_dir)

        plot_heatmap(rdm, all_regions, save_dir)

        print(f"MDS plot saved to {save_dir}")


def comparison_of_voxels():
    """
    Compare voxel activations across different regions within each subject using Wasserstein distance.
    This function loads voxel activations directly and performs region-region comparison 
    limited to WITHIN each subject, not across ALL subjects.
    """
    
    regions = VISUAL_REGIONS
    
    session = DEFAULT_SESSION
    task = DEFAULT_TASK
    run_n = DEFAULT_RUN_COUNT
    data_dir = "C:/Users/BrainInspired/Documents/GitHub/NaturalObjectDataset-Processing/NOD/"
    rdm_dir = "C:/Users/BrainInspired/Documents/GitHub/NaturalObjectDataset-Processing/Nick_RDMs/outputs"

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    all_runs = [f"run-{i+1}" for i in range(run_n)]
    
    # Get ROI indices for all regions once
    region_roi_indices = {}
    for region in regions:
        roi_indices = get_region_roi_indices(region)
        print(f"Region: {region}, Indices shape: {roi_indices[0].shape}")
        region_roi_indices[region] = roi_indices    

    # Process each subject separately to get intrasubject region comparisons  
    # for subject_idx in range(1, DEFAULT_SUBJECT_COUNT + 1):
    for subject_idx in range(1, 2): # for testing
        subject = f"sub-{subject_idx:02d}"
        print(f"Processing subject: {subject}")
        
        # Create save directory for this subject
        save_dir = os.path.join(rdm_dir, subject, 'voxel_comparisons')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        # Initialize data structure for this subject only
        subject_fmri = {region: None for region in regions}
        prev_categories = None
        
        all_beta_data = None
        all_categories = []
        all_fnames = []

        # Fetch all data from runs for this subject and consolidate
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

        if all_beta_data is None:
            print(f"Warning: No data found for {subject}, skipping...")
            continue
            
        # Process each region for this subject
        for region in regions:
            print(f"Processing region: {region} for subject: {subject}")
            roi_indices = region_roi_indices[region]

            roi_beta_data = all_beta_data[:, roi_indices[0]]
            
            unique_categories = sorted(set(all_categories))
            label_to_index = {label: i for i, label in enumerate(unique_categories)}
            sorted_indices = sorted(range(len(all_categories)), key=lambda i: label_to_index[all_categories[i]])
            sorted_categories = [all_categories[i] for i in sorted_indices]

            # Reshape to [stimuli, voxels, 1] format (single subject)
            sorted_activations = roi_beta_data[sorted_indices, :]
            sorted_activations = np.array(sorted_activations).reshape(sorted_activations.shape[0], -1, 1)

            if prev_categories is not None:
                for i, label in enumerate(sorted_categories):
                    if label != prev_categories[i]:
                        print(f"Label mismatch for {subject} in {region}: {label} vs {prev_categories[i]}")
                        raise ValueError("Labels do not match across runs.")     
            prev_categories = sorted_categories

            subject_fmri[region] = sorted_activations
            print(f"Shape of subject_fmri[{region}]: {subject_fmri[region].shape}")
    
        # Compute Wasserstein distance matrix for this subject only
        print(f"Computing Wasserstein distance matrix for {subject}")

        distance_matrix = compute_wasserstein_distance_matrix(subject_fmri, regions)

        plot_mds_visualization(distance_matrix, regions, save_dir, fname='mds_plot.svg', fontsize=20)
        
        # Plot and save the region-region distance matrix for this subject
        plot_heatmap(
            distance_matrix, 
            regions, 
            save_dir, 
            fname=f'{subject}_intrasubject_wasserstein_distance_matrix.svg', 
            title=f'Intrasubject Wasserstein Distance Matrix for {subject}', 
            clim=None,
            fontsize=20
        )
        
        # Also save the matrix as numpy array
        # np.save(os.path.join(save_dir, f'{subject}_wasserstein_distance_matrix.npy'), distance_matrix)
        
        print(f"Saved distance matrix for {subject} to {save_dir}")
            

if __name__ == "__main__":
    # comparison_of_rdms()  # RDM-based comparison
    comparison_of_voxels()  # Voxel-based comparison within subjects