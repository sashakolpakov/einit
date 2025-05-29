.. einit documentation master file

einit: Fast and Robust Ellipsoid ICP Initialization
====================================================

.. image:: https://img.shields.io/pypi/v/einit.svg
   :target: https://pypi.org/project/einit/
   :alt: PyPI version

.. image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

**einit** is a Python library that provides fast and robust initialization for Iterative Closest Point (ICP) algorithms using ellipsoid analysis. It computes optimal initial transformations between 3D point clouds by analyzing their ellipsoids of inertia.

Features
--------

- **Fast Initialization**: Compute initial transformations in milliseconds
- **Robust Algorithm**: Handles various point cloud shapes (spheres, ellipsoids, general shapes)
- **OpenCV Integration**: Returns standard 4×4 homogeneous transformation matrices
- **Noise Tolerance**: Maintains performance even with significant noise
- **Partial Overlap**: Works with incomplete point cloud correspondences

Quick Start
-----------

.. code-block:: python

   import numpy as np
   from einit import ellipsoid_init_icp

   # Create source and destination point clouds
   src_points = np.random.randn(100, 3)
   dst_points = src_points @ R + t  # Apply some transformation

   # Compute initial transformation
   T_init = ellipsoid_init_icp(src_points, dst_points)

   # T_init is a 4×4 homogeneous transformation matrix
   # Compatible with OpenCV and other computer vision libraries

Installation
------------

.. code-block:: bash

   pip install einit

Requirements:
- Python ≥ 3.6
- NumPy ≥ 1.15
- OpenCV ≥ 3.4

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Documentation:

   algorithm
   api
   examples
   integration


Algorithm Overview
------------------

The ellipsoid initialization algorithm works by:

1. **Centering**: Both point clouds are centered at their respective centroids
2. **Ellipsoids of Inertia**: Computes ellipsoid matrices and their eigendecompositions
3. **Axis Alignment**: Searches through all 8 possible axis orientations (±1 reflections)
4. **Optimization**: Selects the transformation that minimizes alignment error
5. **Output**: Returns a 4×4 homogeneous transformation matrix


Mathematical Foundation
-----------------------

Given source points :math:`P = \{p_1, p_2, \ldots, p_n\}` and destination points :math:`Q = \{q_1, q_2, \ldots, q_m\}`, the algorithm:

1. Centers the point clouds:
   
   .. math::
      \bar{p} = \frac{1}{n}\sum_{i=1}^n p_i, \quad \bar{q} = \frac{1}{m}\sum_{j=1}^m q_j

   .. math::
      P_c = P - \bar{p}, \quad Q_c = Q - \bar{q}

2. Computes covariance matrices:
   
   .. math::
      E_P = P_c^T P_c, \quad E_Q = Q_c^T Q_c

3. Performs eigendecomposition:
   
   .. math::
      E_P = U_P \Lambda_P U_P^T, \quad E_Q = U_Q \Lambda_Q U_Q^T

4. Finds optimal rotation through reflection search:
   
   .. math::
      R^* = \arg\min_{D \in \{\pm 1\}^{3 \times 3}} \|Q_c - R P_c\|_F

where :math:`R = U_Q D U_P^T` and :math:`D` is a diagonal matrix with ±1 entries.

Performance Characteristics
---------------------------

- **Time Complexity**: O(n) where n is the number of points
- **Space Complexity**: O(1) additional memory  
- **Typical Runtime**: < 1ms for 1000 points
- **Convergence**: Non-iterative, deterministic result

Benchmark Results
~~~~~~~~~~~~~~~~~

Real-world performance with 2% noise and 80% overlap:

- **Sphere (500 points)**: RMSE 0.017 ± 0.014
- **Cube (1728 points)**: RMSE 0.002 ± 0.001  
- **Stanford Bunny (1000 points)**: RMSE 0.004 ± 0.001

The algorithm often **outperforms traditional ICP refinement**, providing excellent results without iteration.

Use Cases
---------

**Computer Vision**
- Point cloud registration
- 3D object alignment
- Structure from motion
- SLAM applications

**Robotics**
- Sensor fusion
- Object pose estimation
- Navigation and mapping

**Scientific Computing**
- Molecular alignment
- Geometric analysis
- Shape matching

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`