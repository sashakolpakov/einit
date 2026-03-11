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

Using Per-Point Features
------------------------

Per-point features help when the geometry alone is ambiguous — symmetric shapes
like spheres, cubes, or cylinders all have degenerate covariance eigenvalues, so
the plain algorithm is essentially guessing among the 8 candidate rotations.
Passing a feature array alongside the coordinates breaks that degeneracy.

**What counts as a feature?**  Any per-point attribute that is a *surface property*
(i.e. it travels with the point under rigid motion) and varies across the cloud:

- RGB or greyscale colour from an RGBD camera
- LiDAR reflectance / intensity
- Surface normals (the outward normal vector at each point)
- Curvature estimates
- Any custom descriptor

**Minimal example — 1-D intensity:**

.. code-block:: python

   import numpy as np
   from einit import register_ellipsoid

   # src_xyz, dst_xyz: (N, 3) and (M, 3) point arrays
   # src_intensity, dst_intensity: scalar reflectance per point

   # 1-D arrays are accepted directly — no need to reshape to (N, 1)
   T = register_ellipsoid(
       src_xyz, dst_xyz,
       src_features=src_intensity,   # shape (N,) or (N, 1) — both work
       dst_features=dst_intensity,
       feature_weight=0.5,
   )

**Multi-channel example — RGB colour:**

.. code-block:: python

   T = register_ellipsoid(
       src_xyz, dst_xyz,
       src_features=src_rgb,         # shape (N, 3), values in [0, 1]
       dst_features=dst_rgb,
       feature_weight=1.0,
   )

**Choosing feature_weight:**

+-------------------+----------------------------------------------------------+
| ``feature_weight``| Effect                                                   |
+===================+==========================================================+
| ``0.0``           | Geometry only — identical to calling without features.   |
+-------------------+----------------------------------------------------------+
| ``0.1–0.3``       | Light feature influence; good when geometry is already   |
|                   | close to correct and you just want to resolve sign flips.|
+-------------------+----------------------------------------------------------+
| ``0.5–1.0``       | Strong feature influence; recommended for highly         |
|                   | symmetric shapes (spheres, cubes).                       |
+-------------------+----------------------------------------------------------+
| ``> 1.0``         | Features dominate; only useful if the feature signal is  |
|                   | much stronger than the geometric signal.                 |
+-------------------+----------------------------------------------------------+

A quick sweep over ``[0.1, 0.3, 0.5, 1.0]`` on a small validation set is usually
enough to find a good value.  ``feature_weight=1.0`` is a safe default for RGB
colour and LiDAR intensity.

**Flexible API patterns:**

.. code-block:: python

   # Pattern 1: always pass features, weight controls their influence
   def align(src, dst, src_feat=None, dst_feat=None, weight=0.0):
       return register_ellipsoid(
           src, dst,
           src_features=src_feat,
           dst_features=dst_feat,
           feature_weight=weight,
       )

   # Geometry only
   T = align(src, dst)

   # Feature-augmented (weight=0.0 → identical to geometry-only even if
   # feature arrays are passed, so this is always safe to call)
   T = align(src, dst, src_rgb, dst_rgb, weight=1.0)

   # Pattern 2: conditional — only pass features when available
   kwargs = {}
   if src_feat is not None:
       kwargs = dict(src_features=src_feat, dst_features=dst_feat,
                     feature_weight=0.5)
   T = register_ellipsoid(src, dst, **kwargs)

**API guarantees:**

- ``src_features`` and ``dst_features`` must *both* be provided or *neither*.
  Providing only one raises ``ValueError``.
- The two feature arrays must have the **same number of columns** (feature
  dimensionality), but can have different numbers of rows (N ≠ M is fine).
- Setting ``feature_weight=0.0`` gives exactly the same result as omitting
  features entirely — useful for A/B comparisons.
- Features are normalised internally; you do not need to pre-scale them.

Feature-Augmented Registration
-------------------------------

Per-point features such as RGB colour or LiDAR intensity can be passed alongside
the point coordinates to resolve alignment ambiguities that geometry alone cannot
handle (spheres, cubes, cylinders).

.. code-block:: python

   import numpy as np
   from einit import register_ellipsoid

   # --- LiDAR intensity example ---
   # Hemisphere flip: a sphere is geometrically invariant under any rotation.
   # Intensity (high for sunlit top half, low for shadowed bottom half) makes
   # the z-axis special, allowing the algorithm to select the correct sign.

   n = 2000
   rng = np.random.default_rng(0)
   phi   = rng.uniform(0, np.pi, n)
   theta = rng.uniform(0, 2 * np.pi, n)
   src_xyz = np.column_stack([
       np.sin(phi) * np.cos(theta),
       np.sin(phi) * np.sin(theta),
       np.cos(phi),
   ])
   src_intensity = np.where(src_xyz[:, 2] > 0, 0.8, 0.2).reshape(-1, 1)

   # Apply a near-180° rotation around z (hemisphere flip)
   angle = np.pi + rng.uniform(-0.1, 0.1)
   c, s = np.cos(angle), np.sin(angle)
   R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
   dst_xyz = src_xyz @ R.T + rng.uniform(-0.5, 0.5, 3)
   dst_intensity = src_intensity.copy()   # intensity is a surface property

   # Geometry-only often fails for a sphere
   T_geo  = register_ellipsoid(src_xyz, dst_xyz)

   # Feature-augmented resolves the hemisphere ambiguity
   T_feat = register_ellipsoid(
       src_xyz, dst_xyz,
       src_features=src_intensity,
       dst_features=dst_intensity,
       feature_weight=0.5,
   )

   # --- RGB colour example (coloured cube) ---
   # A cube has 48-fold rotational symmetry; geometry-only succeeds ~2% of the
   # time.  Six face colours with different per-axis contrast break this.

   from einit_examples.cube_feature_matching import make_colored_cube

   src_xyz, src_rgb = make_colored_cube(grid_n=15)
   T_true_4x4 = np.eye(4)
   # (apply a known random rotation/translation to src to get dst ...)
   dst_xyz = src_xyz.copy()   # placeholder
   dst_rgb = src_rgb.copy()

   T_feat_cube = register_ellipsoid(
       src_xyz, dst_xyz,
       src_features=src_rgb,
       dst_features=dst_rgb,
       feature_weight=1.0,
   )

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

This script shows that einit's ellipsoid-based approach is robust to point ordering changes in the destination cloud, achieving identical performance whether points are permuted or not.

**Partial Overlap Test**

Tests algorithm robustness with realistic partial overlap scenarios using Stanford bunny data:

.. code-block:: bash

   python einit_examples/rand_overlap_test.py

**Bounding Box Overlap Test**

Evaluates performance with geometric bounding box constraints:

.. code-block:: bash

   python einit_examples/bbox_overlap_test.py

**Feature Matching Diagnostic**

Four-panel diagnostic visualisation for the coloured-cube scenario.  Shows
eigenvalue spectra (geometry-only vs feature-augmented), per-candidate RMSE and
inlier counts for all 8 sign combinations, and side-by-side alignment results:

.. code-block:: bash

   python einit_examples/cube_feature_matching.py

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
   python -m pytest einit_tests/test_integration.py -v        # Integration tests
   python -m pytest einit_tests/test_features.py -v           # Feature-augmented benchmarks

The test suite includes:

- **Core algorithm tests** (``test_einit.py``): Basic functionality, identical point clouds, statistical analysis on synthetic shapes (spheres), and Stanford bunny dataset validation with noise and partial overlap
- **Integration tests** (``test_integration.py``): End-to-end pipeline testing with real-world scenarios
- **Feature tests** (``test_features.py``): Three real-world feature scenarios — Stanford bunny with LiDAR intensity, hemisphere-flip LiDAR sphere, and 48-fold symmetric coloured cube — plus a feature-weight sweep and API-level validation tests

Test results provide detailed statistics including success rates, RMSE distributions, and performance benchmarks for different geometric shapes.