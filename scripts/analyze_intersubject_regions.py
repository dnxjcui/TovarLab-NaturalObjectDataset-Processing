#!/usr/bin/env python3
"""
analyze_intersubject_regions.py

Create MDS plots for subject-level RDMs across different regions of the brain.

Usage:
  python analyze_intersubject_regions.py --rdm_dir outputs/ --region V1
"""
import argparse
import time
import numpy as np
import os
from analyze_intrasubject_regions import load_rdms
from scipy.stats import pearsonr, wasserstein_distance
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from utils import (
    compute_rdm_from_betas, 
    plot_heatmap,
    get_beta_data, 
    get_labels, 
    get_region_roi_indices,
    compute_wasserstein_distance_matrix,
    compute_crossval_mapping_distance_matrix,
    get_fmri_data,
    clear_fmri_cache,
    list_fmri_cache,
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
    plot_heatmap(rdm, all_regions, save_dir, fname='all_regions_rdm2rdm.png', title='RDM-RDM RDM Heatmap for All Regions')


def compare_all_fMRI(): 
    """ 
    Loads in and then compares all fMRI voxel data across subjects and regions using Gromov-Wasserstein distance to 
    compute and save a single region-region distance matrix. 
    """
    data_dir = "C:/Users/BrainInspired/Documents/GitHub/NaturalObjectDataset-Processing/NOD/"
    save_dir = "C:/Users/BrainInspired/Documents/GitHub/NaturalObjectDataset-Processing/Nick_RDMs/outputs/intersubject-figures"
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Load fMRI data using the utility function
    all_fmri = get_fmri_data(data_dir, handle_missing_files=False)
    regions = list(all_fmri.keys())
    
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
        plot_heatmap(grid, regions, save_dir, fname='wasserstein_distance_matrix.png', title='Wasserstein Distance Matrix for All Regions', clim=None)
    
    # save grid
    np.save(os.path.join(save_dir, 'wasserstein_distance_matrix.npy'), grid)

    # grid = compute_crossval_mapping_distance_matrix(all_fmri, regions)
    # plot_rdm(grid, 'Cross-Validated Linear Mapping Distance', regions, save_dir, fname='crossval_mapping_distance_matrix.png', title='Cross-Validated Linear Mapping Distance Matrix for All Regions', clim=None)
    # print(f"Shape of grid: {grid.shape}")


def track_downstream_regions():
    """
    Attempts to answer: if a subject has a low correlation in early regions (e.g. V1) from the population, 
    does this translate to a low correlation in later regions (e.g. V2, LO3, etc.)?
    The final output should be some heatmap with dimension subject x region, that tracks the distance 
    between the subject's fMRI data and the population's fMRI data.
    """
    tic = time.time()
    data_dir = "C:/Users/BrainInspired/Documents/GitHub/NaturalObjectDataset-Processing/NOD/"
    save_dir = "C:/Users/BrainInspired/Documents/GitHub/NaturalObjectDataset-Processing/Nick_RDMs/outputs/intersubject-figures"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Load fMRI data using the utility function with error handling enabled
    all_fmri = get_fmri_data(data_dir, handle_missing_files=True)
    regions = list(all_fmri.keys())

    # Initialize distance matrices
    # Determine number of subjects from the data shape
    num_subjects = list(all_fmri.values())[0].shape[2] if all_fmri else DEFAULT_SUBJECT_COUNT
    num_regions = len(regions)
    
    # Matrix 1: Distance to population (excluding subject)
    population_distances = np.zeros((num_subjects, num_regions))
    
    # Matrix 2: Distance to population mean (excluding subject)
    mean_population_distances = np.zeros((num_subjects, num_regions))

    # Compute distances for each subject and region
    for region_idx, region in enumerate(regions):
        print(f"Computing distances for region: {region}")
        region_data = all_fmri[region]  # Shape: (num_conditions, num_voxels, num_subjects)
        
        for subject_idx in range(min(num_subjects, region_data.shape[2])):
            # Get subject's data
            subject_data = region_data[:, :, subject_idx].flatten()
            
            # Get population data excluding current subject
            other_subjects_mask = np.ones(region_data.shape[2], dtype=bool)
            other_subjects_mask[subject_idx] = False
            population_data = region_data[:, :, other_subjects_mask]

            assert(population_data.shape[2] == num_subjects - 1)
            
            # Distance to population (concatenated data from all other subjects)
            population_flattened = population_data.reshape(1, -1).flatten()
            population_distances[subject_idx, region_idx] = wasserstein_distance(
                subject_data, population_flattened
            )
            
            # Distance to population mean (excluding subject)
            population_mean = np.mean(population_data, axis=2).flatten()
            mean_population_distances[subject_idx, region_idx] = wasserstein_distance(
                subject_data, population_mean
            )

    # Save the data matrices for further analysis
    np.save(os.path.join(save_dir, 'subject_population_distances.npy'), population_distances)
    np.save(os.path.join(save_dir, 'subject_mean_population_distances.npy'), mean_population_distances)

    # Create subject labels
    subject_labels = [f"Sub-{i+1:02d}" for i in range(num_subjects)]

    # normalize across subjects [0, 1] (normalize across all rows for each column)
    population_distances = (population_distances - population_distances.min(axis=0)) / (population_distances.max(axis=0) - population_distances.min(axis=0))
    mean_population_distances = (mean_population_distances - mean_population_distances.min(axis=0)) / (mean_population_distances.max(axis=0) - mean_population_distances.min(axis=0))

    # Plot and save heatmaps using the new plot_heatmap function
    plot_heatmap(
        population_distances, 
        (subject_labels, regions),  # Pass as tuple for rectangular matrix
        save_dir, 
        fname='subject_region_distances.eps', 
        title='Subject vs Population Wasserstein Distances',
        clim=None,
        fontsize=20
    )

    plot_heatmap(
        mean_population_distances, 
        (subject_labels, regions),  # Pass as tuple for rectangular matrix
        save_dir, 
        fname='subject_region_mean-population_distances.eps', 
        title='Subject vs Mean Population Wasserstein Distances',
        clim=None,
        fontsize=20
    )

    print(f"Distance matrices saved in {save_dir}")
    print(f"Population distances shape: {population_distances.shape}")
    print(f"Mean population distances shape: {mean_population_distances.shape}")

    toc = time.time()
    print(f"Time taken: {toc - tic} seconds")

    return population_distances, mean_population_distances 
    
    
if __name__ == "__main__":
    # compare_similarity_matrices()
    # compare_all_fMRI()
    track_downstream_regions()
