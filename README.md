<p align="center">
  <img src="docs/einit.png" alt="einit logo" height="320"/>
</p>

<h1 align="center">Fast and Robust ICP Initialization</h1>

<p align="center">
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"/>
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/python-3.6+-blue.svg" alt="Python 3.6+"/>
  </a>
  <a href="https://pypi.org/project/einit/">
    <img src="https://img.shields.io/pypi/v/einit.svg" alt="PyPI"/>
  </a>
  <a href="https://github.com/sashakolpakov/einit/actions/workflows/pylint.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/sashakolpakov/einit/pylint.yml?branch=main&label=CI&logo=github" alt="CI"/>
  </a>
  <a href="https://github.com/sashakolpakov/einit/actions/workflows/deploy_docs.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/sashakolpakov/einit/deploy_docs.yml?branch=main&label=Docs&logo=github" alt="Docs"/>
  </a>
  <a href="https://sashakolpakov.github.io/einit/">
    <img src="https://img.shields.io/website-up-down-green-red/https/sashakolpakov.github.io/einit?label=API%20Documentation" alt="Docs Status"/>
  </a>
</p>



**einit** provides fast and robust initialization for 3D point cloud alignment using ellipsoid analysis. It computes initial transformations by analyzing the ellipsoids of inertia of point clouds and uses KD-tree correspondence recovery for robustness to real-world scenarios.

## Key Features

- **Fast**: < 1ms for 1000 points
- **Accurate**: Often achieves excellent alignment without iterative refinement
- **OpenCV compatible**: Returns standard 4×4 transformation matrices  
- **Robust**: Handles noise, partial overlap, and permuted point clouds
- **Permutation invariant**: Results are identical regardless of point ordering
- **Configurable**: Adjustable parameters for different use cases
- **Simple API**: One function call to get results

## Quick Start

```python
import numpy as np
from einit import ellipsoid_init_icp

# Your point clouds (N x 3 arrays)
src_points = np.random.randn(1000, 3)
dst_points = src_points @ R + t  # Apply some transformation

# Get the transformation matrix
T = ellipsoid_init_icp(src_points, dst_points)
print(T)  # 4x4 homogeneous transformation matrix

# With custom parameters for robustness control
T = ellipsoid_init_icp(
    src_points, dst_points,
    max_correspondence_distance=0.1,  # Maximum distance for valid correspondences
    min_inlier_fraction=0.7,          # Require 70% valid correspondences  
    leafsize=8                        # Smaller KD-tree leaf size
)
```

## Installation

There are multiple installation options. One of them is to install the current release from PyPI:

```bash
pip install einit
```

Another option is to install most recent builds directly from GitHub:

```bash
pip install git+https://github.com/sashakolpakov/einit.git
```

For development or testing:
```bash
pip install "einit[test] @ git+https://github.com/sashakolpakov/einit.git@main"  # Includes matplotlib, pytest
pip install "einit[all] @ git+https://github.com/sashakolpakov/einit.git@main"   # Everything including docs
```

Or clone and install locally:
```bash
git clone https://github.com/sashakolpakov/einit.git
cd einit
pip install -e .[test]  # Editable install with test dependencies
```

## Performance

Real-world performance on test datasets:

| Dataset | Points | RMSE  | Time             |
|---------|--------|-------|------------------|
| Sphere  | 1500   | 0.03  | 0.006 ± 0.002 ms |  
| Cube    | 3375   | 0.02  | 0.010 ± 0.008 ms |
| Bunny   | 992    | 0.02  | 0.047 ± 0.021 ms |

*With 0.01-0.02 standard Gaussian noise and ~ 80% overlap*

## Algorithm

The algorithm works by:

1. **Centering** point clouds at their centroids
2. **Computing** ellipsoids of inertia via eigendecomposition  
3. **Searching** through 8 reflection combinations using KD-tree correspondence recovery
4. **Filtering** correspondences by distance and inlier fraction
5. **Returning** a 4×4 transformation matrix

KD-tree correspondence recovery makes the algorithm robust to point cloud permutations, partial overlaps, and outliers without assuming that points at the same array indices correspond to each other.

## OpenCV Integration

```python
import cv2
from einit import ellipsoid_init_icp

# Get initial transformation
T_init = ellipsoid_init_icp(src, dst)

# Refine alignment with OpenCV 
src_aligned = apply_transform(src, T_init)
retval, T_refined, inliers = cv2.estimateAffine3D(
    src_aligned.astype(np.float32), 
    dst.astype(np.float32)
)
```

## Examples and Testing

### Running Examples

The `examples/` directory contains demonstrations and visualizations:

**Interactive Jupyter Notebook:**
```bash
jupyter notebook examples/visual_tests.ipynb
```
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](
  https://colab.research.google.com/github/sashakolpakov/einit/blob/main/examples/visual_tests.ipynb
)
Comprehensive visual demonstrations including sphere, cube, and Stanford bunny alignments with performance analysis.

**Permutation Invariance Test:**
```bash
python examples/point_reoder_test.py
```
Demonstrates that einit correctly handles randomly permuted point clouds.

**Partial Overlap Test:**
```bash
python examples/rand_overlap_test.py
```
Tests algorithm robustness with randomized partial overlap scenarios using the Stanford bunny.

**Bounding Box Overlap Test:**
```bash
python examples/bbox_overlap_test.py
```
Evaluates performance on the Stanford bunny with geometric bounding box constraints.

> **Note**: Unlike randomized overlaps, this is a known failure mode of the algorithm. Low success rate is expected. 


### Running Tests

The `tests/` directory contains comprehensive test suites validating core functionality:

```bash
# All tests
pytest tests/ -v

# Specific test categories  
pytest tests/test_einit.py -v              # Core algorithm tests
pytest tests/test_integration.py -v        # Integration and robustness tests

# Test permutation invariance specifically
pytest tests/test_einit.py::test_random_permutation_invariance -v
```

**Test Coverage:**
- **Core Algorithm Tests** (`test_einit.py`): Basic functionality, permutation invariance, noise robustness, and Stanford bunny dataset validation
- **Integration Tests** (`test_integration.py`): End-to-end pipeline testing with real-world scenarios

## Documentation

More comprehensive [documentation](https://sashakolpakov.github.io/einit/) is available. 

## Authors

- **Alexander Kolpakov** (University of Austin at Texas)
- **Michael Werman** (Hebrew University of Jerusalem)  
- **Judah Levin** (University of Austin at Texas)

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgement

This work is supported by the Google Cloud Research Award number GCP19980904.

## Citation

Based on the original paper by Kolpakov and Werman:

[![Paper](https://img.shields.io/badge/arXiv-read%20PDF-b31b1b.svg)](https://arxiv.org/abs/2212.05332)

*"An approach to robust ICP initialization"*
