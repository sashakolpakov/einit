Algorithm Details
=================

The ellipsoid initialization algorithm computes initial transformations between 3D point clouds by aligning their principal axes. This section describes the mathematical foundation and implementation.

Overview
--------

The algorithm assumes that aligning 3D objects requires aligning their ellipsoids of inertia. The principal axes of these ellipsoids provide a coordinate system for computing the transformation. The covariance matrices describe the distribution of points along coordinate axes rather than point-to-point relationships.

For point clouds with shape (N, 3), the covariance matrix has shape (3, 3), describing the distribution of mass along coordinate axes rather than the (N, N) Gram matrix that would describe point-to-point relationships.

The implementation uses KD-tree nearest neighbor search to find point correspondences during transformation evaluation. This handles point clouds with different orderings, partial overlaps, and missing correspondences without assuming that points at the same array indices correspond to each other. 

Step-by-Step Breakdown
----------------------

1. Input Validation and Preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The algorithm begins by validating that both input point clouds are (N, 3) arrays representing 3D coordinates:

.. code-block:: python

   if src_points.ndim != 2 or src_points.shape[1] != 3:
       raise ValueError("src_points and dst_points must both be (:,3) arrays")

This ensures that the inputs are properly formatted 3D point clouds with the same number of points.

2. Centroid Computation and Centering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both point clouds are centered at their respective centroids:

.. math::
   \bar{p} = \frac{1}{n}\sum_{i=1}^n p_i, \quad \bar{q} = \frac{1}{m}\sum_{j=1}^m q_j

.. math::
   P_c = P - \bar{p}, \quad Q_c = Q - \bar{q}

This step removes the translational component of the transformation, allowing us to focus on the rotational alignment.

.. code-block:: python

   def _centroid_and_center(pts: np.ndarray):
       c = np.mean(pts, axis=0)
       return pts - c, c

   src_c, cs = _centroid_and_center(src_points)
   dst_c, ct = _centroid_and_center(dst_points)

3. Ellipsoid of Inertia Computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The algorithm computes covariance matrices for both centered point clouds:

.. math::
   E_P = P_c^T P_c, \quad E_Q = Q_c^T Q_c

In the implementation, we write:

.. code-block:: python

   Ep = src_c.T @ src_c  # 3×3 covariance matrix for source
   Eq = dst_c.T @ dst_c  # 3×3 covariance matrix for destination

4. Eigendecomposition
~~~~~~~~~~~~~~~~~~~~~

Each covariance matrix is decomposed into its eigenvalues and eigenvectors:

.. math::
   E_P = U_P \Lambda_P U_P^T, \quad E_Q = U_Q \Lambda_Q U_Q^T

where :math:`U_P, U_Q` are orthogonal matrices containing the eigenvectors (principal axes), and :math:`\Lambda_P, \Lambda_Q` are diagonal matrices of eigenvalues.

.. code-block:: python

   eigp, Up = np.linalg.eigh(Ep)  # eigenvalues and eigenvectors for source
   eigq, Uq = np.linalg.eigh(Eq)  # eigenvalues and eigenvectors for destination

The ``numpy.linalg.eigh`` function is used because the covariance matrices are symmetric and positive semi-definite.

5. Reflection Search and Correspondence Recovery
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The algorithm tests all :math:`2^3 = 8` possible axis orientations since eigendecomposition can produce eigenvectors pointing in either direction along each axis.

For each combination of signs :math:`s_1, s_2, s_3 \in \{-1, +1\}`, we construct a diagonal reflection matrix:

.. math::
   D = \begin{pmatrix}
   s_1 & 0 & 0 \\
   0 & s_2 & 0 \\
   0 & 0 & s_3
   \end{pmatrix}

And compute the corresponding rotation matrix:

.. math::
   R = U_Q U_P^T D U_P U_P^T = U_Q D U_P^T

The implementation uses a KD-tree to find nearest neighbor correspondences for each candidate transformation:

.. code-block:: python

   # Build KD-tree for target points
   kdtree = cKDTree(Q_centered, leafsize=leafsize)
   
   best_error = np.inf
   best_transform = U0
   
   for signs in [[1,1,1], [-1,1,1], [1,-1,1], [1,1,-1],
                 [-1,-1,1], [-1,1,-1], [1,-1,-1], [-1,-1,-1]]:
       U = U0 @ Up @ np.diag(signs) @ Up.T
       P_transformed = P_centered @ U.T
       
       # Find nearest neighbors to establish correspondence
       distances, indices = kdtree.query(P_transformed)
       
       # Filter correspondences by distance threshold
       valid_mask = distances <= max_correspondence_distance
       if np.sum(valid_mask) / len(distances) < min_inlier_fraction:
           continue
           
       # Compute error using valid correspondences only
       error = np.sum(distances[valid_mask]**2)

6. Error Computation and Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each candidate transformation, the algorithm computes the sum of squared distances between transformed source points and their nearest neighbors in the target cloud:

.. math::
   \text{error} = \sum_{i \in \text{valid}} d_i^2

where :math:`d_i` is the distance from transformed source point :math:`i` to its nearest neighbor in the target cloud, and the sum includes only correspondences within the distance threshold.

The transformation with the minimum error and sufficient inlier count is selected as the optimal initialization.

7. Parameter Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The algorithm accepts several parameters for robustness control:

**max_correspondence_distance**: Maximum distance for valid point correspondences. If not specified, the algorithm estimates this as 3 times the median nearest-neighbor distance within the target cloud.

**min_inlier_fraction**: Minimum fraction of points that must have valid correspondences (default 0.5). Transformations with insufficient inliers are rejected.

**leafsize**: KD-tree leaf size parameter affecting search performance (default 16). Smaller values may improve accuracy for small point clouds at the cost of build time.

**positive_only**: When True, restricts the search to only proper rotations (determinant +1) by selecting sign combinations that preserve chirality (default False).

8. Homogeneous Transformation Matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Finally, the optimal rotation and translation are packed into a 4×4 homogeneous transformation matrix:

.. math::
   T = \begin{pmatrix}
   R & t \\
   0^T & 1
   \end{pmatrix}

.. code-block:: python

   T = np.eye(4, dtype=best_R.dtype)
   T[:3, :3] = best_R
   T[:3, 3] = best_t
   return T

Mathematical Properties
-----------------------

Optimality
~~~~~~~~~~

The algorithm finds the globally optimal solution within the constraint of axis-aligned rotations. While this may not be the globally optimal rigid transformation, it provides a good initialization that captures the primary geometric structure of the point clouds.

Complexity Analysis
~~~~~~~~~~~~~~~~~~~

- **Time Complexity**: O(n) where n is the number of points
  - Centroid computation: O(n)
  - Covariance matrices: O(n)
  - Eigendecomposition: O(1) (3×3 matrices)
  - Reflection search: O(1) (8 iterations)
  - Error computation: O(n) per iteration

- **Space Complexity**: O(1) additional memory beyond input storage

Robustness Properties
~~~~~~~~~~~~~~~~~~~~~

The algorithm handles several challenging scenarios:

1. **Scale Invariance**: Uniform scaling of input point clouds does not affect the result
2. **Noise Tolerance**: Moderate noise in point coordinates has limited impact on principal axes computation
3. **Partial Overlap**: Works with point clouds that have different numbers of points, occlusions, and missing correspondences
4. **Permutation Invariance**: Point ordering in the input arrays does not affect the result
5. **Outlier Rejection**: Distance thresholding filters out poor correspondences 


Applications and Use Cases
--------------------------

The ellipsoid initialization algorithm is particularly well-suited for:

1. **ICP Preprocessing**: Providing good initial guesses for ICP algorithms
2. **Multi-Modal Registration**: Aligning point clouds from different sensors
3. **Shape Analysis**: Initial alignment for shape comparison and analysis
4. **Real-Time Applications**: Fast initialization for time-critical applications

Implementation Notes
--------------------

**Numerical Stability**
The implementation uses ``numpy.linalg.eigh`` for eigendecomposition, which is numerically stable for symmetric matrices. The algorithm avoids matrix inversions and uses well-conditioned operations throughout.

**Memory Efficiency**
The algorithm operates primarily on small 3×3 matrices regardless of the input size, making it memory-efficient even for large point clouds.

**Floating Point Precision**
The algorithm preserves the input data type (float32 or float64) throughout the computation, maintaining appropriate numerical precision for the application.