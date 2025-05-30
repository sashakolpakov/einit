# einit: Fast and Robust ICP Initialization

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)

<!-- CI status from GitHub Actions -->
[![CI](https://img.shields.io/github/actions/workflow/status/sashakolpakov/einit/pylint.yml?branch=main&label=CI&logo=github)](https://github.com/sashakolpakov/einit/actions/workflows/pylint.yml) <!-- Docs status from GitHub Actions -->
[![Docs](https://img.shields.io/github/actions/workflow/status/sashakolpakov/einit/deploy_docs.yml?branch=main&label=Docs&logo=github)](https://github.com/sashakolpakov/einit/actions/workflows/deploy_docs.yml) <!-- Docs health via HTTP ping -->
[![Docs](https://img.shields.io/website-up-down-green-red/https/sashakolpakov.github.io/einit?label=API%20Documentation)](https://sashakolpakov.github.io/einit/)



**einit** provides fast and robust initialization for 3D point cloud alignment using ellipsoid analysis. It computes optimal initial transformations by analyzing the ellipsoids of inertia of point clouds, often achieving excellent results without needing iterative refinement.

## Key Features

- ⚡ **Fast**: < 1ms for 1000 points
- 🎯 **Accurate**: The initial alignment is often very good
- 🔧 **OpenCV compatible**: Returns standard 4×4 transformation matrices  
- 🛡️ **Robust**: Handles noise and partial overlap
- 🐍 **Simple API**: One function call to get results

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
```

## Installation

Install directly from GitHub:

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
3. **Searching** through 8 reflection combinations for optimal alignment
4. **Returning** a 4×4 transformation matrix

This provides an excellent initialization that often eliminates the need for iterative refinement.

## OpenCV Integration

```python
import cv2
from einit import ellipsoid_init_icp

# Get initial transformation
T_init = ellipsoid_init_icp(src, dst)

# Optional: refine with OpenCV if needed
src_aligned = apply_transform(src, T_init)
retval, T_refined, inliers = cv2.estimateAffine3D(
    src_aligned.astype(np.float32), 
    dst.astype(np.float32)
)
```

## Testing

Run the test suite:
```bash
pytest tests/ -v -s
```

Visualize failure cases:
```bash
cd tests && python visualize_failures.py
```

## Documentation

More comprehensive [documentation](https://sashakolpakov.github.io/einit/) available here. 

## Authors

- **Alexander Kolpakov** (University of Austin at Texas)
- **Michael Werman** (Hebrew University of Jerusalem)  
- **Judah Levin** (University of Austin at Texas)

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

Based on the original paper by Kolpakov and Werman:

[![Paper](https://img.shields.io/badge/arXiv-read%20PDF-b31b1b.svg)](https://arxiv.org/abs/2212.05332)

*"An approach to robust ICP initialization"*
