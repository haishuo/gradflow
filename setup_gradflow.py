#!/usr/bin/env python3
"""
GradFlow Setup - Smart directory initialization

This script:
1. Checks if proper Forge directories exist
2. Creates symlinks if on Forge system  
3. Creates local directories if not on Forge
4. Does nothing if directories already exist (as files or symlinks)
"""

import os
import sys
from pathlib import Path

def setup_gradflow():
    """Set up GradFlow directory structure intelligently"""
    
    project_root = Path(__file__).parent
    
    # Define the directory mappings
    dirs_to_setup = {
        'data': '/mnt/data/gradflow',
        'artifacts': '/mnt/artifacts/gradflow', 
        'reference-fortran': '/mnt/projects/weno-reference/reference'
    }
    
    print("🚀 Setting up GradFlow directory structure...")
    print(f"📁 Project root: {project_root}")
    
    for local_name, forge_path in dirs_to_setup.items():
        local_path = project_root / local_name
        forge_path_obj = Path(forge_path)
        
        # Check if local directory/symlink already exists
        if local_path.exists():
            if local_path.is_symlink():
                target = local_path.readlink()
                print(f"✅ {local_name}/ already exists (symlink to {target})")
            else:
                print(f"✅ {local_name}/ already exists (local directory)")
            continue
        
        # Try to create symlink to Forge location
        if forge_path_obj.exists():
            try:
                local_path.symlink_to(forge_path)
                print(f"🔗 Created symlink: {local_name}/ → {forge_path}")
            except OSError as e:
                print(f"⚠️  Could not create symlink for {local_name}: {e}")
                # Fallback to local directory
                local_path.mkdir(parents=True, exist_ok=True)
                print(f"📁 Created local directory: {local_name}/")
        else:
            # Forge path doesn't exist, create local directory
            local_path.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created local directory: {local_name}/ (Forge path not found)")
    
    # Create subdirectories in data and artifacts
    data_dir = project_root / 'data'
    artifacts_dir = project_root / 'artifacts'
    
    if data_dir.exists():
        for subdir in ['inputs', 'datasets', 'validation', 'benchmarks']:
            (data_dir / subdir).mkdir(exist_ok=True)
        print("📂 Created data subdirectories")
    
    if artifacts_dir.exists():
        for subdir in ['results', 'models', 'plots', 'benchmarks', 'validation_reports']:
            (artifacts_dir / subdir).mkdir(exist_ok=True)
        print("📂 Created artifacts subdirectories")
    
    print("\n✅ GradFlow setup complete!")
    print("\nDirectory structure:")
    for dir_name in ['data', 'artifacts', 'reference-fortran']:
        path = project_root / dir_name
        if path.exists():
            if path.is_symlink():
                print(f"  {dir_name}/ → {path.readlink()}")
            else:
                print(f"  {dir_name}/ (local)")
        else:
            print(f"  {dir_name}/ (missing)")

if __name__ == "__main__":
    setup_gradflow()
