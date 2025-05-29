Algorithm Details
=================

The ellipsoid initialization algorithm provides a robust method for computing initial transformations between 3D point clouds. This section provides a detailed explanation of the mathematical foundation and implementation.

Overview
--------

The algorithm is based on the insight that 3D objects being aligned means that their ellipsoids of inertia being aligned, and that the principal axes of these ellipsoids provide a natural coordinate system for the best alignment. By analyzing the covariance structure of point clouds, we can identify these principal axes and use them to compute optimal initial transformations. However, here the covariance is taken not from point to point, but from one coordinate axis to the other. In this case, if we have an array of shape (N, 3), the shape of the covariance matrix is not (N, N), as it would be for the Gram matrix, but rather (3, 3). The latter describes the distribution of mass (or inertia) of the point cloud along the coordinate axes. 

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

5. Reflection Search
~~~~~~~~~~~~~~~~~~~~

The core innovation of this algorithm is the systematic search through all possible axis orientations. Since eigendecomposition can produce eigenvectors pointing in either direction along each axis, we need to test all :math:`2^3 = 8` possible combinations of orientations.

For each combination of signs :math:`s_1, s_2, s_3 \in \{-1, +1\}`, we construct a diagonal reflection matrix:

.. math::
   D = \begin{pmatrix}
   s_1 & 0 & 0 \\
   0 & s_2 & 0 \\
   0 & 0 & s_3
   \end{pmatrix}

And compute the corresponding rotation matrix:

.. math::
   R = U_Q D U_P^T

The translation is then computed as:

.. math::
   t = \bar{q} - R\bar{p}

.. code-block:: python

   best_err = np.inf
   best_R = None
   best_t = None

   for signs in itertools.product([-1, 1], repeat=3):
       D = np.diag(signs)
       R = Uq @ D @ Up.T
       t = ct - R @ cs
       
       transformed = (R @ src_c.T).T + t
       err = np.linalg.norm(transformed - dst_c, ord=2)
       
       if err < best_err:
           best_err = err
           best_R = R
           best_t = t

6. Error Computation and Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each candidate transformation, we apply it to the centered source points and compute the Frobenius norm of the alignment error:

.. math::
   \text{error} = \|Q_c - R P_c\|_F = \sqrt{\sum_{i,j} (Q_c - R P_c)_{ij}^2}

The transformation with the minimum error is selected as the optimal initialization.

7. Homogeneous Transformation Matrix
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

The algorithm exhibits several desirable robustness properties:

1. **Scale Invariance**: The algorithm is invariant to uniform scaling of the input point clouds
2. **Noise Tolerance**: Reasonable amounts of noise in the point coordinates have minimal impact on the principal axes
3. **Partial Overlap**: The algorithm works even when the point clouds don't have perfect correspondence, have occlusions, discrepancy in the number of points and in their mutual correspondences 


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