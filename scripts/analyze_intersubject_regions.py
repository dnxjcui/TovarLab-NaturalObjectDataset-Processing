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
from analyze_subject_rdms import load_rdms, compute_rdm, plot_rdm
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze intersubject RDMs for a specific region.")
    parser.add_argument('--rdm_dir', type=str, required=True, 
                        help='Directory containing the RDM files')
    parser.add_argument('--region', type=str, required=True, 
                        help='Region of interest (e.g., V1, V2)')
    return parser.parse_args()


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


def plot_histogram(data, bins, xaxis, labels, region, save_dir):
    """
    Plot a histogram of the distances of intersubject correlations.
    """

    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, alpha=0.7, color='blue', edgecolor='black')
    plt.title(f'Intersubject Correlation Distance for {region}')
    plt.xlabel(xaxis)
    plt.ylabel('Frequency')
    
    save_path = os.path.join(save_dir, f'{region}_histogram.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Histogram saved to {save_path}")


def plot_rdm_side_bar_graph(rdm_square, distance_metric, labels, save_dir, fname='region_region_rdm_heatmap.png', title='Region-Region RDM Heatmap'):
    """Plot and save the RDM heatmap. Also plot an additional bar graph on the side corresponding to the rank of the mean of each row, where highest mean = highest rank."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5), gridspec_kw={'width_ratios': [3, 1]})
    im = ax1.imshow(rdm_square, cmap='viridis', interpolation='none')
    plt.colorbar(im, ax=ax1, label=f'{distance_metric} Distance')
    
    ax1.set_xticks([])
    ax1.set_yticks([])

    ## calculate bar plot stuff ##
    mean_rdm = np.nanmean(rdm_square, axis=1)

    ranked_indices = np.argsort(mean_rdm)[::-1] + 1    
    y_positions = np.arange(rdm_square.shape[0])

    # scale mean_rdm
    # mean_rdm = (mean_rdm - np.nanmin(mean_rdm)) / (np.nanmax(mean_rdm) - np.nanmin(mean_rdm))

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
    


def histograms():    
    rdm_dir = "outputs/"
    save_dir = os.path.join(rdm_dir, 'intersubject-histograms')
    regions = ["V1", "V2", "V3", "V4", "V8", "PIT", "FFC", "VVC", "VMV1", "VMV2", "VMV3", "LO1", "LO2", "LO3"]

    subjects = [ int(dir.split('-')[1]) for dir in os.listdir(rdm_dir) if dir.startswith('sub-') ]
    # now we plot the z scores as a histogram
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    for region in regions:
        print(f"Analyzing region: {region}")

        all_rdms = [] # all rdms for a specific region across subjects
        for subject in subjects:
            rdms = load_rdms(rdm_dir, subject)
            all_rdms.append(rdms[region])

        distances = []
        for i in range(len(all_rdms)):
            target_rdm = all_rdms[i]

            other_rdms = np.array([ rdm for j, rdm in enumerate(all_rdms) if j != i ])
            
            corr = intersubject_correlation(target_rdm, other_rdms, distance_metric='pearson')

            distances.append(1 - corr)

        plot_histogram(distances, 10, "Distance from Mean", subjects, region, save_dir)


# make similarity matrix of all participants w/ each other
# mean of every row

def make_similarity_matrices(plot=True):
    rdm_dir = "outputs/"
    save_dir = os.path.join(rdm_dir, 'intersubject-RDMs')
    regions = ["V1", "V2", "V3", "V4", "V8", "PIT", "FFC", "VVC", "VMV1", "VMV2", "VMV3", "LO1", "LO2", "LO3"]

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

        subject_subject_rdm = compute_rdm(all_rdms, metric='correlation')

        assert(all_rdms.shape[0] == subject_subject_rdm.shape[0])

        np.fill_diagonal(subject_subject_rdm, np.nan)

        if plot:
            # plot_rdm(subject_subject_rdm, 'Correlation', subjects, save_dir, fname=f'{region}_subject_subject_rdm_heatmap.png', title=f'Scaled Subject-Subject RDM Heatmap for {region}', clim=
            plot_rdm_side_bar_graph(subject_subject_rdm, 'Correlation', subjects, save_dir, fname=f'{region}_subject_subject_rdm_heatmap.png', title=f'Scaled Subject-Subject RDM Heatmap for {region}')
        
        rdm_per_region[region] = subject_subject_rdm
        # quit()
    return rdm_per_region


def compare_similarity_matrices():
    rdm_dir = "outputs/"
    save_dir = os.path.join(rdm_dir, 'intersubject-RDMs')
    regions = ["V1", "V2", "V3", "V4", "V8", "PIT", "FFC", "VVC", "VMV1", "VMV2", "VMV3", "LO1", "LO2", "LO3"]

    rdm_per_region = make_similarity_matrices(plot=False)

    all_rdms = None
    for region, rdm in rdm_per_region.items():
        np.fill_diagonal(rdm, 0)
        flattened_rdm = squareform(rdm)

        flattened_rdm = flattened_rdm.reshape(1, -1) 

        if all_rdms is None:
            all_rdms = flattened_rdm
        all_rdms = np.concatenate((all_rdms, flattened_rdm), axis=0)

        print(f"Shape of all_rdms after concatenation: {all_rdms.shape}")

    # now we compute the RDM per region
    rdm = compute_rdm(all_rdms, metric='correlation')

    # plot the RDM for all regions
    all_regions = list(rdm_per_region.keys())
    plot_rdm(rdm, 'Correlation', all_regions, save_dir, fname='all_regions_rdm2rdm.png', title='RDM-RDM RDM Heatmap for All Regions')


if __name__ == "__main__":
    # histograms()
    compare_similarity_matrices()
