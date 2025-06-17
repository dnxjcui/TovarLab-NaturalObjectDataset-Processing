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


def compute_rdm(beta_weights, metric='correlation'):
    """
    Compute RDM from RDMs using distance metric.
    """
    rdm = squareform(pdist(beta_weights, metric=metric))
    return rdm


def plot_mds(rdm, all_regions, save_dir):
    """
    Plot MDS of the RDM and save the figure.
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


def plot_rdm(rdm_square, distance_metric, labels, save_dir, fname='region_region_rdm_heatmap.png', title='Region-Region RDM Heatmap', clim=(0.0, 1.0)):
    """
    Plot and save the RDM heatmap.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(rdm_square, cmap='viridis', interpolation='none')
    
    if clim is not None:
        im.set_clim(clim[0], clim[1])
    
    plt.tight_layout()
    heatmap_path = os.path.join(save_dir, fname)
    os.makedirs(save_dir, exist_ok=True)

    plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=45, ha='right')
    plt.yticks(ticks=np.arange(len(labels)), labels=labels)

    plt.savefig(heatmap_path, dpi=300, transparent=True)
    print(f"Heatmap saved at {heatmap_path}")

    plt.close(fig)


def main():
    rdm_dir = 'C:/Users/BrainInspired/Documents/GitHub/NaturalObjectDataset-Processing/Nick_RDMs/outputs'

    # hard coding to maintain order
    regions = ["V1", "V2", "V3", "V4", "V8", "PIT", "FFC", "VVC", "VMV1", "VMV2", "VMV3", "LO1", "LO2", "LO3"]

    for subject_n in range(1, 31):    
        rdms = load_rdms(rdm_dir, subject_n)
        rdm_data = np.array([rdms[region] for region in regions])

        rdm = compute_rdm(rdm_data, metric='correlation') # region-region RDM

        save_dir = os.path.join(rdm_dir, f'sub-{subject_n:02d}', 'mds_plots')
        os.makedirs(save_dir, exist_ok=True)

        all_regions = list(rdms.keys())
        plot_mds(rdm, all_regions, save_dir)

        plot_rdm(rdm, 'correlation', all_regions, save_dir)

        print(f"MDS plot saved to {save_dir}")


if __name__ == "__main__":
    main()