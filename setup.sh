#!/bin/bash

echo "🚀 Setting up GradFlow - Revolutionary Differentiable CFD"
echo "=========================================================="

# Navigate to project directory
cd /mnt/projects/gradflow

# Create Forge data and artifacts directories
echo "📁 Creating Forge directories..."
mkdir -p /mnt/data/gradflow/{inputs,datasets,validation,benchmarks}
mkdir -p /mnt/artifacts/gradflow/{results,models,plots,benchmarks,validation_reports}

# Create symlinks for development convenience
echo "🔗 Creating development symlinks..."
ln -sf /mnt/data/gradflow ./data
ln -sf /mnt/artifacts/gradflow ./artifacts
ln -sf /mnt/projects/weno-reference/reference ./reference-fortran

# Create .gitignore
echo "📝 Creating .gitignore..."
cat > .gitignore << 'EOF'
# Forge system symlinks (local development only)
data
artifacts
reference-fortran

# Data files (belong in /mnt/data/)
*.dat
*.npz
*.hdf5
*.pkl
*.csv

# Artifacts (belong in /mnt/artifacts/)
results/
models/
plots/
benchmarks/
validation_reports/
checkpoints/

# Python
__pycache__/
*.py[cod]
*.so
*.egg-info/
.pytest_cache/

# Jupyter
.ipynb_checkpoints/
*.ipynb

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
EOF

# Create smart setup script for contributors
echo "🔧 Creating setup_gradflow.py..."
cat > setup_gradflow.py << 'EOF'
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
EOF

chmod +x setup_gradflow.py

# Create README
echo "📖 Creating README.md..."
cat > README.md << 'EOF'
# GradFlow - Revolutionary Differentiable CFD

🚀 **World's First Differentiable WENO Scheme**  
Built on Chi-Wang Shu's proven FORTRAN foundation

## Quick Start

### Automatic Setup
```bash
git clone [your-repo]/gradflow
cd gradflow
python setup_gradflow.py
```

This automatically:
- 🔗 Creates symlinks to Forge directories (if on Forge system)
- 📁 Creates local directories (if not on Forge system)  
- ✅ Does nothing if directories already exist

### Manual Setup (Alternative)
```bash
# On Forge systems
ln -s /mnt/data/gradflow ./data
ln -s /mnt/artifacts/gradflow ./artifacts
ln -s /mnt/projects/weno-reference/reference ./reference-fortran

# On other systems  
mkdir -p data artifacts reference-fortran
```

## Directory Structure

```
gradflow/                    # Code (this repo)
├── gradflow/               # Main Python package
├── data/                   # → /mnt/data/gradflow/ (inputs, datasets)
├── artifacts/              # → /mnt/artifacts/gradflow/ (outputs, models)
├── reference-fortran/      # → WENO reference (validation data)
└── setup_gradflow.py       # Smart directory setup
```

## Development Workflow

```bash
# Work on code
cd /mnt/projects/gradflow
vim gradflow/core/weno.py

# Access data
ls data/inputs/
cp reference-fortran/sod_shock_working/*.dat data/validation/

# Check results  
ls artifacts/results/
python -m gradflow.validation.compare_fortran
```

## Vision
Transform computational fluid dynamics through:
- **Exact gradients** for design optimization  
- **GPU acceleration** with PyTorch tensors
- **Bit-perfect validation** against FORTRAN reference
- **Revolutionary applications**: airfoil optimization, inverse problems

## Status
🏗️ **In Development** - Building the future of CFD

**Next**: Implement differentiable WENO reconstruction in `gradflow/core/`
EOF

# Create Python package structure
echo "🐍 Creating Python package structure..."
mkdir -p gradflow/{core,validation,applications,examples}

# Main package init
cat > gradflow/__init__.py << 'EOF'
"""
GradFlow - Revolutionary Differentiable CFD

World's first differentiable WENO scheme for computational fluid dynamics.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

# Auto-setup directories on import
from pathlib import Path
import os

_project_root = Path(__file__).parent.parent

# Smart path detection
_data_candidates = [
    _project_root / 'data',
    Path('/mnt/data/gradflow'), 
    _project_root / 'local_data'
]

_artifacts_candidates = [
    _project_root / 'artifacts',
    Path('/mnt/artifacts/gradflow'),
    _project_root / 'local_artifacts'  
]

# Find existing directories
DATA_ROOT = next((p for p in _data_candidates if p.exists()), _data_candidates[0])
ARTIFACTS_ROOT = next((p for p in _artifacts_candidates if p.exists()), _artifacts_candidates[0])

# Import main modules
from . import core
from . import validation 
from . import applications

__all__ = ["core", "validation", "applications", "DATA_ROOT", "ARTIFACTS_ROOT"]
EOF

# Create module inits
touch gradflow/core/__init__.py
touch gradflow/validation/__init__.py
touch gradflow/applications/__init__.py
touch gradflow/examples/__init__.py

# Create requirements.txt
echo "📦 Creating requirements.txt..."
cat > requirements.txt << 'EOF'
torch>=2.0.0
numpy>=1.21.0
matplotlib>=3.5.0
scipy>=1.8.0
tqdm>=4.64.0
pytest>=7.0.0
jupyter>=1.0.0
pyyaml>=6.0
h5py>=3.7.0
EOF

# Copy validation data to data directory
echo "📋 Copying reference validation data..."
cp -r /mnt/projects/weno-reference/reference/sod_shock_working /mnt/data/gradflow/validation/

# Create initial input file
echo "📝 Creating sample input configuration..."
cat > /mnt/data/gradflow/inputs/sod_shock_params.yaml << 'EOF'
# GradFlow Input Parameters - Sod Shock Tube
simulation:
  problem_type: "sod_shock_tube"
  grid:
    nx: 100
    ny: 50
    domain: [0.0, 10.0, 0.0, 5.0]
  
  time:
    method: "runge_kutta_3"
    cfl: 0.3
    max_steps: 200
    final_time: 0.2

  initial_conditions:
    left_state:
      density: 1.0
      pressure: 1.0
      velocity: [0.0, 0.0]
    right_state:
      density: 0.125
      pressure: 0.1
      velocity: [0.0, 0.0]
    shock_location: 5.0

validation:
  reference_data: "/mnt/data/gradflow/validation/sod_shock_working/"
  tolerance: 1e-12  # Bit-perfect matching
EOF

# Initialize git repository
if [ ! -d ".git" ]; then
    echo "📝 Initializing git repository..."
    git init
    git config user.name "$(whoami)"
    git config user.email "$(whoami)@forge.local"
fi

# Add files to git (symlinks will be ignored due to .gitignore)
echo "📝 Adding files to git..."
git add .
git commit -m "🚀 Initial GradFlow setup

- Smart Forge directory system with symlinks
- Portable setup script for contributors
- Python package structure ready for development
- Reference validation data integrated
- Template for all future projects

Ready to build the world's first differentiable WENO!"

# Final status
echo ""
echo "🎉 GradFlow setup complete!"
echo "=========================="
echo ""
echo "📁 Project structure:"
echo "  Code:      /mnt/projects/gradflow/"
echo "  Data:      /mnt/data/gradflow/"
echo "  Artifacts: /mnt/artifacts/gradflow/"
echo ""
echo "🔗 Development symlinks:"
ls -la data artifacts reference-fortran
echo ""
echo "✅ Git repository initialized and committed"
echo "✅ Reference validation data copied"
echo "✅ Python package structure created"
echo "✅ Requirements and configurations ready"
echo ""
echo "🚀 Ready to build the future of CFD!"
echo ""
echo "Next steps:"
echo "1. cd /mnt/projects/gradflow"
echo "2. Start implementing gradflow/core/weno.py"
echo "3. Validate against reference-fortran/sod_shock_working/"
echo ""
echo "🌟 This smart directory system can now be used for all future projects!"
