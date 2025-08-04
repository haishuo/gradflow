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
