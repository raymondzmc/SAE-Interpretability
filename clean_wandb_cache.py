#!/usr/bin/env python3
"""Clean up Wandb cache directories to free up space."""

import os
import shutil
from pathlib import Path

def get_dir_size(path):
    """Get total size of directory in GB."""
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    except (PermissionError, FileNotFoundError):
        pass
    return total / (1024**3)  # Convert to GB

def clean_wandb_cache():
    """Clean up Wandb cache directories."""
    
    # Common Wandb cache locations
    home = Path.home()
    cache_dirs = [
        home / ".config" / "wandb",
        home / ".cache" / "wandb",
        home / "wandb",
        Path("/tmp") / "wandb",
    ]
    
    print("Checking Wandb cache directories...")
    print("=" * 60)
    
    total_size = 0
    existing_dirs = []
    
    for cache_dir in cache_dirs:
        if cache_dir.exists():
            size = get_dir_size(cache_dir)
            total_size += size
            existing_dirs.append((cache_dir, size))
            print(f"{cache_dir}: {size:.2f} GB")
    
    if not existing_dirs:
        print("No Wandb cache directories found.")
        return
    
    print("=" * 60)
    print(f"Total cache size: {total_size:.2f} GB")
    print()
    
    # Ask for confirmation
    response = input("Do you want to clean these directories? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Clean directories
    for cache_dir, size in existing_dirs:
        try:
            # Only clean tmp and cache subdirectories, keep config
            subdirs_to_clean = []
            
            if "config" in str(cache_dir):
                # For config dir, only clean tmp files
                for item in cache_dir.glob("tmp*"):
                    subdirs_to_clean.append(item)
                for item in cache_dir.glob("cache*"):
                    subdirs_to_clean.append(item)
            else:
                # For other dirs, clean everything
                subdirs_to_clean = [cache_dir]
            
            for item in subdirs_to_clean:
                if item.is_file():
                    item.unlink()
                    print(f"Deleted file: {item}")
                elif item.is_dir():
                    shutil.rmtree(item)
                    print(f"Deleted directory: {item}")
            
            print(f"✓ Cleaned {cache_dir}")
            
        except Exception as e:
            print(f"✗ Failed to clean {cache_dir}: {e}")
    
    # Check space after cleaning
    print()
    print("Space after cleaning:")
    print("=" * 60)
    
    new_total = 0
    for cache_dir, _ in existing_dirs:
        if cache_dir.exists():
            size = get_dir_size(cache_dir)
            new_total += size
            print(f"{cache_dir}: {size:.2f} GB")
    
    print("=" * 60)
    print(f"Space freed: {total_size - new_total:.2f} GB")


if __name__ == "__main__":
    clean_wandb_cache() 