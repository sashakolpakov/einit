Examples
========

Basic Usage
-----------

Simple Point Cloud Alignment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from einit import register_ellipsoid

   # Create source points (sphere)
   n = 1000
   phi = np.random.uniform(0, np.pi, n)
   theta = np.random.uniform(0, 2*np.pi, n)
   src = np.column_stack([
       np.sin(phi) * np.cos(theta),
       np.sin(phi) * np.sin(theta),
       np.cos(phi)
   ]) * 5  # radius = 5

   # Create destination by applying known transformation
   R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])  # 90° rotation around Z
   t = np.array([10, 5, 2])  # translation
   dst = src @ R.T + t

   # Compute initial transformation
   T_init = register_ellipsoid(src, dst)
   print("Estimated transformation:")
   print(T_init)

Integration with OpenCV
-----------------------

Using with cv2.estimateAffine3D
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import cv2
   import numpy as np
   from einit import register_ellipsoid

   def apply_transform(pts, T):
       """Apply 4x4 homogeneous transform to points."""
       homo = np.hstack([pts, np.ones((pts.shape[0], 1))])
       return (T @ homo.T).T[:, :3]

   def refine_with_opencv(src, dst, init_T):
       """Refine einit result with OpenCV."""
       src_aligned = apply_transform(src, init_T)
       retval, out, inliers = cv2.estimateAffine3D(
           src_aligned.astype(np.float32),
           dst.astype(np.float32)
       )
       if retval:
           T_refined = np.eye(4)
           T_refined[:3, :4] = out
           return T_refined @ init_T
       return init_T

   # Use einit for initialization, then refine with OpenCV
   T_init = register_ellipsoid(src, dst)
   T_final = refine_with_opencv(src, dst, T_init)

Real-World Data
---------------

Working with Noisy Point Clouds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Add realistic noise to destination points
   noise_std = 0.02
   dst_noisy = dst + np.random.normal(0, noise_std, dst.shape)

   # Algorithm handles noise well
   T_init = register_ellipsoid(src, dst_noisy)
   aligned = apply_transform(src, T_init)

   # Compute alignment quality
   errors = np.linalg.norm(aligned - dst_noisy, axis=1)
   rmse = np.sqrt(np.mean(errors**2))
   print(f"RMSE: {rmse:.4f}")

Partial Overlap Scenarios
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Simulate partial overlap (common in real scanning)
   overlap_ratio = 0.7
   n_overlap = int(len(src) * overlap_ratio)
   
   # Random subset of points
   indices = np.random.choice(len(src), n_overlap, replace=False)
   src_partial = src[indices]
   dst_partial = dst_noisy[indices]

   # Algorithm works with partial data
   T_init = register_ellipsoid(src_partial, dst_partial)

Visualization
-------------

3D Plotting with Matplotlib
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt
   from mpl_toolkits.mplot3d import Axes3D

   def plot_alignment(src, dst, aligned):
       fig = plt.figure(figsize=(15, 5))
       
       # Original source
       ax1 = fig.add_subplot(131, projection='3d')
       ax1.scatter(src[:, 0], src[:, 1], src[:, 2], c='red', alpha=0.6)
       ax1.set_title('Source')
       
       # Target destination
       ax2 = fig.add_subplot(132, projection='3d')
       ax2.scatter(dst[:, 0], dst[:, 1], dst[:, 2], c='blue', alpha=0.6)
       ax2.set_title('Destination')
       
       # Overlay aligned source with destination
       ax3 = fig.add_subplot(133, projection='3d')
       ax3.scatter(dst[:, 0], dst[:, 1], dst[:, 2], c='blue', alpha=0.4, label='Target')
       ax3.scatter(aligned[:, 0], aligned[:, 1], aligned[:, 2], c='red', alpha=0.6, label='Aligned')
       ax3.set_title('Alignment Result')
       ax3.legend()
       
       plt.tight_layout()
       plt.show()

   # Use the visualization
   aligned = apply_transform(src, T_init)
   plot_alignment(src, dst, aligned)

Running Examples and Tests
--------------------------

Examples Directory
~~~~~~~~~~~~~~~~~~

The ``einit_examples/`` directory contains several demonstration scripts and notebooks:

**Interactive Jupyter Notebook**

Comprehensive visual demonstrations including sphere, cube, and Stanford bunny alignments with performance analysis:

.. code-block:: bash

   # Launch Jupyter and open the notebook
   jupyter notebook einit_examples/visual_tests.ipynb

**Permutation Invariance Test**

Demonstrates that einit correctly handles randomly permuted point clouds:

.. code-block:: bash

   python einit_examples/point_reoder_test.py

This script shows that einit's ellipsoid-based approach is robust to point ordering changes, achieving identical performance whether points are permuted or not.

**Partial Overlap Test**

Tests algorithm robustness with realistic partial overlap scenarios using Stanford bunny data:

.. code-block:: bash

   python einit_examples/rand_overlap_test.py

**Bounding Box Overlap Test**

Evaluates performance with geometric bounding box constraints:

.. code-block:: bash

   python einit_examples/bbox_overlap_test.py

The notebook includes:

- Interactive visualizations of point cloud alignment
- Step-by-step algorithm walkthrough  
- Performance analysis with different geometric shapes (spheres, cubes, Stanford bunny)
- Timing benchmarks and noise robustness analysis

Running Tests
~~~~~~~~~~~~~

To verify the installation and run comprehensive tests:

.. code-block:: bash

   # Run all tests
   python -m pytest einit_tests/ -v
   
   # Run specific test categories
   python -m pytest einit_tests/test_einit.py -v              # Core algorithm tests
   python -m pytest einit_tests/test_integration.py -v        # Stanford bunny integration test
   
   # Run individual test functions
   python -m pytest einit_tests/test_einit.py::test_basic_functionality -v
   python -m pytest einit_tests/test_einit.py::test_identity_transform -v
   python -m pytest einit_tests/test_einit.py::test_synthetic_shapes_statistical -v
   python -m pytest einit_tests/test_einit.py::test_bunny_cloud_statistical -v

The test suite includes:

- **Core algorithm tests** (``test_einit.py``): Basic functionality, identity transforms, robust statistical analysis on synthetic shapes (spheres and cube surfaces), Stanford bunny dataset validation, noise robustness testing using correlation analysis, and performance scaling verification using log-log regression
- **Integration tests** (``test_integration.py``): Stanford bunny alignment test using the original PLY dataset with partial overlap, noise, and ICP refinement comparison between OpenCV and Open3D methods

The tests use advanced statistical validation methods:

- **Noise robustness**: Verifies that RMSE grows approximately linearly with noise level using correlation coefficients (r > 0.7) rather than hard thresholds
- **Performance scaling**: Uses log-log regression to verify sub-quadratic time complexity (O(n^α) where α < 2.0) across different point cloud sizes
- **Robust testing**: Uses statistical repetition with configurable failure thresholds to handle inherent randomness in point cloud generation

Test results provide detailed statistics including success rates, RMSE distributions, transform errors, correlation analysis, scaling exponents, and performance timing for different geometric shapes and real-world data.