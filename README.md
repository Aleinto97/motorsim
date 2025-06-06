# MotorOptiX

[![Codecov](https://codecov.io/gh/YOUR-ORG/YOUR-REPO/branch/main/graph/badge.svg)](https://codecov.io/gh/YOUR-ORG/YOUR-REPO)

> **Note:** To get the Codecov badge working:
> 1. After your first successful CI run with Codecov, visit your repository's page on [Codecov.io](https://codecov.io)
> 2. Find the "Embed" or "Badge" option to get your specific badge URL
> 3. Replace `YOUR-ORG` with your GitHub username/organization
> 4. Replace `YOUR-REPO` with your repository name

A Python-based tool for the design and optimization of high-speed Brushless DC (BLDC) and Switched Reluctance (SR) motors for the blower industry.

## System Requirements

- macOS ARM64 (Apple Silicon)
- Xcode Command Line Tools
- Miniforge or Anaconda

## Installation

1. Install Xcode Command Line Tools:
```bash
xcode-select --install
```

2. Install Miniforge (if not already installed):
```bash
curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh -o Miniforge3-MacOSX-arm64.sh
bash Miniforge3-MacOSX-arm64.sh
```

3. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate motoroptix
```

## Environment Details

The environment is configured with the following key components:

- Python 3.10
- Core Scientific Computing: NumPy 2.1.0+, SciPy, Matplotlib
- Geometric Processing: Shapely 2.1.1+, ezdxf 1.4.2+
- Finite Element Analysis: Gmsh 4.13.1+, FEniCSx 0.9.0+
- Optimization: Optuna 4.3.0+, Pydantic 2.11.5+

## Performance Optimization

For optimal performance on Apple Silicon:

1. Ensure all packages are running natively on ARM64 (not through Rosetta2)
2. Use conda-forge channel for native ARM64 builds
3. Consider building FEniCSx from source with native compiler optimizations for maximum performance

## Troubleshooting

If you encounter libtbb linking issues with FEniCSx:

1. Try using the full Anaconda distribution instead of Miniconda
2. Ensure Python version matches the host system
3. Consider rebuilding FEniCSx from source with specific optimization flags

## License

[Add your license information here] 