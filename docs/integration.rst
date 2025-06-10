OpenCV Integration
==================

The einit library is designed to work seamlessly with OpenCV and other computer vision libraries. This section covers integration patterns and best practices.

Transformation Matrix Format
----------------------------

einit returns standard 4×4 homogeneous transformation matrices:

.. code-block:: python

   T = np.array([
       [R11, R12, R13, tx],
       [R21, R22, R23, ty], 
       [R31, R32, R33, tz],
       [0,   0,   0,   1 ]
   ])

This format is directly compatible with:
- OpenCV's 3D functions
- Open3D transformations
- PCL (Point Cloud Library)
- Standard robotics libraries

Common Integration Patterns
---------------------------

ICP Initialization
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import cv2
   from einit import register_ellipsoid

   # 1. Get initial transformation with einit
   T_init = register_ellipsoid(src_points, dst_points)

   # 2. Apply initial transformation
   src_aligned = apply_transform(src_points, T_init)

   # 3. Refine with OpenCV ICP
   retval, T_refine, inliers = cv2.estimateAffine3D(
       src_aligned.astype(np.float32),
       dst_points.astype(np.float32),
       confidence=0.99,
       ransacThreshold=0.01
   )

   # 4. Combine transformations
   if retval:
       T_final = np.eye(4)
       T_final[:3, :4] = T_refine
       T_complete = T_final @ T_init
   else:
       T_complete = T_init

RANSAC Integration
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def robust_alignment(src, dst, max_iterations=1000):
       best_T = None
       best_inliers = 0
       
       for _ in range(max_iterations):
           # Sample subset of points
           indices = np.random.choice(len(src), min(100, len(src)), replace=False)
           src_sample = src[indices]
           dst_sample = dst[indices]
           
           # Get transformation
           T_candidate = register_ellipsoid(src_sample, dst_sample)
           
           # Evaluate on full dataset
           aligned = apply_transform(src, T_candidate)
           distances = np.linalg.norm(aligned - dst, axis=1)
           inliers = np.sum(distances < threshold)
           
           if inliers > best_inliers:
               best_inliers = inliers
               best_T = T_candidate
               
       return best_T

Multi-Scale Processing
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def multiscale_alignment(src, dst, scales=[1.0, 0.5, 0.25]):
       T_cumulative = np.eye(4)
       
       for scale in scales:
           # Downsample point clouds
           n_points = int(len(src) * scale)
           indices = np.random.choice(len(src), n_points, replace=False)
           
           src_scale = src[indices]
           dst_scale = dst[indices]
           
           # Apply current transformation
           src_transformed = apply_transform(src_scale, T_cumulative)
           
           # Compute refinement
           T_delta = register_ellipsoid(src_transformed, dst_scale)
           T_cumulative = T_delta @ T_cumulative
           
       return T_cumulative

Performance Optimization
------------------------

Preprocessing for Speed
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def preprocess_for_alignment(points, target_size=1000):
       """Downsample and clean point cloud for faster processing."""
       if len(points) > target_size:
           # Uniform downsampling
           indices = np.linspace(0, len(points)-1, target_size, dtype=int)
           points = points[indices]
       
       # Remove outliers (optional)
       centroid = np.mean(points, axis=0)
       distances = np.linalg.norm(points - centroid, axis=1)
       threshold = np.percentile(distances, 95)  # Keep 95% of points
       mask = distances <= threshold
       
       return points[mask]

Batch Processing
~~~~~~~~~~~~~~~~

.. code-block:: python

   def batch_align_clouds(src_clouds, dst_clouds):
       """Align multiple point cloud pairs efficiently."""
       transformations = []
       
       for src, dst in zip(src_clouds, dst_clouds):
           # Preprocess for consistency
           src_clean = preprocess_for_alignment(src)
           dst_clean = preprocess_for_alignment(dst)
           
           # Compute transformation
           T = register_ellipsoid(src_clean, dst_clean)
           transformations.append(T)
           
       return transformations

Error Handling
--------------

Robust Error Checking
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def safe_alignment(src, dst, fallback='identity'):
       """Perform alignment with comprehensive error handling."""
       try:
           # Input validation
           if src.shape[0] < 4 or dst.shape[0] < 4:
               raise ValueError("Need at least 4 points for alignment")
           
           if src.shape != dst.shape:
               raise ValueError("Source and destination must have same shape")
           
           # Compute transformation
           T = register_ellipsoid(src, dst)
           
           # Validate result
           if not np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-6):
               raise ValueError("Computed rotation matrix is not orthogonal")
           
           return T, True
           
       except Exception as e:
           print(f"Alignment failed: {e}")
           
           if fallback == 'identity':
               return np.eye(4), False
           elif fallback == 'centroid':
               # Align centroids only
               t = np.mean(dst, axis=0) - np.mean(src, axis=0)
               T = np.eye(4)
               T[:3, 3] = t
               return T, False
           else:
               raise

Integration with Other Libraries
================================

Open3D Integration
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import open3d as o3d

   def align_with_open3d(src_pcd, dst_pcd):
       # Convert to numpy arrays
       src_points = np.asarray(src_pcd.points)
       dst_points = np.asarray(dst_pcd.points)
       
       # Get initial transformation
       T_init = register_ellipsoid(src_points, dst_points)
       
       # Refine with Open3D ICP
       result = o3d.pipelines.registration.registration_icp(
           src_pcd, dst_pcd, 
           max_correspondence_distance=0.1,
           init=T_init
       )
       
       return result.transformation

PCL Integration (via python-pcl)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pcl

   def align_with_pcl(src_cloud, dst_cloud):
       # Extract points
       src_points = src_cloud.to_array()
       dst_points = dst_cloud.to_array()
       
       # Get initialization
       T_init = register_ellipsoid(src_points, dst_points)
       
       # Use PCL's ICP with initialization
       icp = src_cloud.make_IterativeClosestPoint()
       converged, transform, estimate, fitness = icp.icp(dst_cloud, T_init)
       
       return transform if converged else T_init