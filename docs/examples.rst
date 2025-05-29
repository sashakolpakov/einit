Examples
========

Basic Usage
-----------

Simple Point Cloud Alignment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from einit import ellipsoid_init_icp

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
   T_init = ellipsoid_init_icp(src, dst)
   print("Estimated transformation:")
   print(T_init)

Integration with OpenCV
-----------------------

Using with cv2.estimateAffine3D
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import cv2
   import numpy as np
   from einit import ellipsoid_init_icp

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
   T_init = ellipsoid_init_icp(src, dst)
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
   T_init = ellipsoid_init_icp(src, dst_noisy)
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
   T_init = ellipsoid_init_icp(src_partial, dst_partial)

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

Failure Case Analysis
~~~~~~~~~~~~~~~~~~~~~

For debugging and analysis, a visualization utility is provided in the test suite:

.. code-block:: bash

   # Run failure visualization (requires matplotlib)
   cd tests
   python visualize_failures.py

This script analyzes the bunny test with configurable parameters and shows visualizations of the worst failure cases, including:

- Algorithm alignment results (aligned source vs. true target)
- Point-wise error distributions with RMSE statistics

You can customize the analysis parameters:

.. code-block:: python

   from tests.visualize_failures import visualize_bunny_failures
   
   # Analyze with higher noise and lower overlap
   visualize_bunny_failures(
       noise_std=0.05,           # Higher noise level
       overlap_fraction=0.6,     # Lower overlap (60%)
       n_points=2000,           # Fewer points for speed
       show_worst=3             # Show top 3 failures
   )