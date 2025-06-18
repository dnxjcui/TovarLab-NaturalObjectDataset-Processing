#!/usr/bin/env python3
"""
analyze_intersubject_regions.py

Create MDS plots for subject-level RDMs across different regions of the brain.

Usage:
  python analyze_intersubject_regions.py --rdm_dir outputs/ --region V1
"""
import argparse
import numpy as np
import os
from analyze_intrasubject_regions import load_rdms
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from utils import (
    compute_rdm_from_betas, 
    plot_rdm_heatmap,
    get_beta_data, 
    get_labels, 
    get_region_roi_indices,
    compute_wasserstein_distance_matrix,
    compute_crossval_mapping_distance_matrix,
    VISUAL_REGIONS,
    DEFAULT_SESSION,
    DEFAULT_TASK,
    DEFAULT_RUN_COUNT,
    DEFAULT_SUBJECT_COUNT
)


def rdm_correlation(a, b, distance_metric='pearson'):
    """
    Calculate the intersubject correlation between two RDMs.
    This function assumes that a and b are 1D arrays representing the flattened RDMs.
    """
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    
    if distance_metric == 'pearson':
        return pearsonr(a, b)[0]
    else:
        raise ValueError(f"Unsupported distance metric: {distance_metric}. Only 'pearson' is supported.")


def intersubject_correlation(target_rdm, other_rdms, distance_metric='pearson'):
    """
    Calculate the intersubject correlation between a target RDM and a mean derived from a set of other RDMs.
    This function assumes that target_rdm and other_rdms are 1D arrays representing the flattened RDMs.
    """
    mean_other_rdms = np.mean(other_rdms, axis=0)

    if mean_other_rdms.shape != target_rdm.shape:
        raise ValueError(f"Shape mismatch: target_rdm {target_rdm.shape} vs mean_other_rdm {mean_other_rdms.shape}")
    return rdm_correlation(target_rdm, mean_other_rdms, distance_metric=distance_metric)


def plot_rdm_side_bar_graph(rdm_square, distance_metric, labels, save_dir, fname='region_region_rdm_heatmap.png', title='Region-Region RDM Heatmap'):
    """
    Plot and save the RDM heatmap. Also plot an additional bar graph on the side corresponding to the rank of the mean of each row, where highest mean = highest rank.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5), gridspec_kw={'width_ratios': [3, 1]})
    im = ax1.imshow(rdm_square, cmap='viridis', interpolation='none')
    plt.colorbar(im, ax=ax1, label=f'{distance_metric} Distance')
    
    ax1.set_xticks([])
    ax1.set_yticks([])

    ## calculate bar plot stuff ##
    mean_rdm = np.nanmean(rdm_square, axis=1)
    ranked_indices = np.argsort(mean_rdm)[::-1] + 1    
    y_positions = np.arange(rdm_square.shape[0])

    ## plot the bar plot ##
    ax2.barh(y_positions, mean_rdm, align='center')
    ax2.set_yticks(y_positions)
    ax2.invert_yaxis()
    ax2.set_xlabel('Ranked mean of row values')
    ax2.set_yticklabels([])
    ax2.set_ylim(-0.5, rdm_square.shape[0] - 0.5)
    
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    
    # Auto-scale x-axis to show differences better
    min_val = np.nanmin(mean_rdm)
    max_val = np.nanmax(mean_rdm)
    range_val = max_val - min_val
    
    # Add 5% padding on each side of the range
    padding = range_val * 0.05
    ax2.set_xlim(min_val - padding, max_val + padding)

    ## save plot ##
    plt.tight_layout()
    heatmap_path = os.path.join(save_dir, fname)
    os.makedirs(save_dir, exist_ok=True)

    plt.savefig(heatmap_path, dpi=300)
    print(f"Heatmap saved at {heatmap_path}")

    plt.close(fig)
    

def make_similarity_matrices(plot=True):
    """
    Compute intersubject RDMs for each region and plot them.
    """
    rdm_dir = 'C:/Users/BrainInspired/Documents/GitHub/NaturalObjectDataset-Processing/Nick_RDMs/outputs'
    save_dir = os.path.join(rdm_dir, 'intersubject-RDMs')
    regions = VISUAL_REGIONS

    subjects = [ int(dir.split('-')[1]) for dir in os.listdir(rdm_dir) if dir.startswith('sub-') ]
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    rdm_per_region = {}
    for region in regions:
        print(f"Analyzing region: {region}")

        all_rdms = [] # all rdms for a specific region across subjects
        for subject in subjects:
            rdms = load_rdms(rdm_dir, subject)
            all_rdms.append(rdms[region])

        all_rdms = np.array(all_rdms)

        subject_subject_rdm = compute_rdm_from_betas(all_rdms, metric='correlation')

        assert(all_rdms.shape[0] == subject_subject_rdm.shape[0])

        np.fill_diagonal(subject_subject_rdm, np.nan)

        if plot:
            plot_rdm_side_bar_graph(subject_subject_rdm, 'Correlation', subjects, save_dir, fname=f'{region}_subject_subject_rdm_heatmap.png', title=f'Scaled Subject-Subject RDM Heatmap for {region}')
        
        rdm_per_region[region] = subject_subject_rdm
    return rdm_per_region


def compare_similarity_matrices():
    """ 
    Compare similarity matrices across regions. This function computes the RDM 
    for each region, flattens them, and then computes a single RDM across all regions.
    """
    rdm_dir = "C:/Users/BrainInspired/Documents/GitHub/NaturalObjectDataset-Processing/Nick_RDMs/outputs"
    save_dir = os.path.join(rdm_dir, 'intersubject-RDMs')

    rdm_per_region = make_similarity_matrices(plot=False)

    all_rdms = None
    for _, rdm in rdm_per_region.items():
        np.fill_diagonal(rdm, 0)
        flattened_rdm = squareform(rdm)

        flattened_rdm = flattened_rdm.reshape(1, -1) 

        if all_rdms is None:
            all_rdms = flattened_rdm
        all_rdms = np.concatenate((all_rdms, flattened_rdm), axis=0)

        print(f"Shape of all_rdms after concatenation: {all_rdms.shape}")

    # now we compute the RDM per region
    rdm = compute_rdm_from_betas(all_rdms, metric='correlation')

    # plot the RDM for all regions
    all_regions = list(rdm_per_region.keys())
    plot_rdm_heatmap(rdm, 'Correlation', all_regions, save_dir, fname='all_regions_rdm2rdm.png', title='RDM-RDM RDM Heatmap for All Regions')








def compare_all_fMRI(): 
    """ 
    Loads in and then compares all fMRI voxel data across subjects and regions using Gromov-Wasserstein distance to 
    compute and save a single region-region distance matrix. 
    """

    regions = VISUAL_REGIONS
    
    subject_n = DEFAULT_SUBJECT_COUNT
    session = DEFAULT_SESSION
    task = DEFAULT_TASK
    run_n = DEFAULT_RUN_COUNT
    data_dir = "C:/Users/BrainInspired/Documents/GitHub/NaturalObjectDataset-Processing/NOD/"
    save_dir = "C:/Users/BrainInspired/Documents/GitHub/NaturalObjectDataset-Processing/Nick_RDMs/outputs/intersubject-figures"

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    all_runs = [f"run-{i+1}" for i in range(run_n)]
    all_subjects = [f"sub-{i+1:02d}" for i in range(subject_n)]

    region_roi_indices = {}
    for region in regions:
        roi_indices = get_region_roi_indices(region)
        
        print(f"Region: {region}, Indices shape: {roi_indices[0].shape}")
        region_roi_indices[region] = roi_indices    

    all_fmri = {
        region : None for region in regions
    }
    prev_categories = None
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
        
        for region in regions:
            print(f"Processing region: {region} for subject: {subject} . . .")
            roi_indices = region_roi_indices[region]

            roi_beta_data = all_beta_data[:, roi_indices[0]]
            
            unique_categories = sorted(set(all_categories))
            label_to_index = {label: i for i, label in enumerate(unique_categories)}
            sorted_indices = sorted(range(len(all_categories)), key=lambda i: label_to_index[all_categories[i]])
            sorted_categories = [all_categories[i] for i in sorted_indices]

            sorted_activations = roi_beta_data[sorted_indices, :]
            sorted_activations = np.array(sorted_activations).reshape(sorted_activations.shape[0], -1, 1)

            # make sure labels are sorted the same and consistent across subjects
            if prev_categories is not None:
                for i, label in enumerate(sorted_categories):
                    if label != prev_categories[i]:
                        print(f"Label mismatch for {subject} in {region}: {label} vs {prev_categories[i]}")
                        raise ValueError("Labels do not match across runs.")     
            prev_categories = sorted_categories

            if all_fmri[region] is None:
                all_fmri[region] = sorted_activations
            else:
                all_fmri[region] = np.append(all_fmri[region], sorted_activations, axis=2)
                print(f"Shape of all_fmri[{region}]: {all_fmri[region].shape}")
    
    # for figure making 
    plot_1d_vectors = False
    if plot_1d_vectors:
        all_1d_vectors = compute_wasserstein_distance_matrix(all_fmri, regions, plot_1d_vectors=True)
        for region, vec in all_1d_vectors.items():
            plt.figure(figsize=(100, 1))

            # crop vector to the first 1000 elements for better visualization
            vec = vec[:1000]
            plt.imshow(vec.reshape(1, -1), aspect='auto', cmap='viridis')

            # turn off axes
            plt.axis('off')

            if not os.path.exists(os.path.join(save_dir, "1dvecs")):
                os.makedirs(os.path.join(save_dir, "1dvecs"), exist_ok=True)
            plt.savefig(os.path.join(save_dir, "1dvecs", f'{region}_1d_vector.png'), transparent=True, dpi=300)
            plt.close()
    else:
        grid = compute_wasserstein_distance_matrix(all_fmri, regions)
        plot_rdm_heatmap(grid, 'Wasserstein Distance', regions, save_dir, fname='wasserstein_distance_matrix.png', title='Wasserstein Distance Matrix for All Regions', clim=None)
    
    # grid = compute_crossval_mapping_distance_matrix(all_fmri, regions)
    # plot_rdm(grid, 'Cross-Validated Linear Mapping Distance', regions, save_dir, fname='crossval_mapping_distance_matrix.png', title='Cross-Validated Linear Mapping Distance Matrix for All Regions', clim=None)
    # print(f"Shape of grid: {grid.shape}")
    

if __name__ == "__main__":
    # compare_similarity_matrices()
    compare_all_fMRI()
