#!/usr/bin/env python3
"""
cache_utils.py

Cache management utilities for fMRI data processing.
"""
import os
import glob
import datetime
from pathlib import Path

# Constants
BASE_DIR = Path(__file__).resolve().parent.parent

def clear_fmri_cache(cache_pattern=None):
    """
    Clear cached fMRI data files.
    
    Args:
        cache_pattern (str, optional): Pattern to match for deletion (e.g., '*sub30*'). 
                                     If None, clears all cache files.
    """
    cache_base_dir = os.path.join(BASE_DIR, 'outputs', 'fmri_data')
    
    if not os.path.exists(cache_base_dir):
        print("No fMRI cache directory found.")
        return
    
    if cache_pattern is None:
        cache_pattern = "*.pkl"
    
    cache_files = glob.glob(os.path.join(cache_base_dir, cache_pattern))
    
    if not cache_files:
        print(f"No cache files found matching pattern: {cache_pattern}")
        return
    
    print(f"Found {len(cache_files)} cache files to delete:")
    for cache_file in cache_files:
        print(f"  - {os.path.basename(cache_file)}")
    
    confirm = input("Delete these files? (y/N): ")
    if confirm.lower() in ['y', 'yes']:
        for cache_file in cache_files:
            try:
                os.remove(cache_file)
                print(f"Deleted: {os.path.basename(cache_file)}")
            except Exception as e:
                print(f"Error deleting {cache_file}: {e}")
    else:
        print("Cache deletion cancelled.")


def list_fmri_cache():
    """
    List all cached fMRI data files with their details.
    """
    cache_base_dir = os.path.join(BASE_DIR, 'outputs', 'fmri_data')
    
    if not os.path.exists(cache_base_dir):
        print("No fMRI cache directory found.")
        return
    
    cache_files = glob.glob(os.path.join(cache_base_dir, "*.pkl"))
    
    if not cache_files:
        print("No cached fMRI files found.")
        return
    
    print(f"Found {len(cache_files)} cached fMRI files:")
    print("-" * 80)
    
    for cache_file in sorted(cache_files):
        filename = os.path.basename(cache_file)
        size_mb = os.path.getsize(cache_file) / (1024 * 1024)
        mtime = os.path.getmtime(cache_file)
        mtime_str = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"File: {filename}")
        print(f"  Size: {size_mb:.2f} MB")
        print(f"  Modified: {mtime_str}")
        print()


if __name__ == "__main__":
    print("Cache Management Utilities")
    print("Available functions:")
    print("  - clear_fmri_cache(pattern=None)")
    print("  - list_fmri_cache()")
    print()
    list_fmri_cache() 