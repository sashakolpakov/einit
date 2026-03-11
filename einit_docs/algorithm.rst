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
       raise ValueError("src_points must be (N,3) array")
   if dst_points.ndim != 2 or dst_points.shape[1] != 3:
       raise ValueError("dst_points must be (N,3) array")

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

   Ep = P_centered.T @ P_centered  # 3×3 covariance matrix for source
   Eq = Q_centered.T @ Q_centered  # 3×3 covariance matrix for destination

When per-point features are provided, the covariance is augmented before
eigendecomposition (see `Feature Augmentation`_ below).

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

4b. Feature Augmentation
~~~~~~~~~~~~~~~~~~~~~~~~

When ``src_features``, ``dst_features``, and ``feature_weight > 0`` are supplied,
the spatial covariance is augmented before eigendecomposition:

.. math::
   E_P^{\text{aug}} = P_c^T P_c \;+\; \beta \cdot \frac{\text{tr}(E_P)}{\text{tr}(E_{ff})} \cdot E_{xf} E_{xf}^T

where :math:`E_{xf} = P_c^T F_{\text{scaled}}` is the spatial-feature
cross-covariance (shape 3×k), and :math:`\beta` is ``feature_weight``.

The trace-ratio factor keeps the scale of the feature term commensurate with the
spatial term regardless of the number of points or the dynamic range of the features,
so ``feature_weight=1.0`` means "feature term has the same total variance as the
spatial term".

Two separate normalisations are used for the two algorithmic steps:

* **Covariance step** — global scale (max std across all feature columns). This
  preserves inter-channel contrast ratios, which is what creates distinct
  eigenvalues for asymmetrically coloured objects like a cube with differently
  coloured faces.

* **KD-tree step** — per-column std normalisation, so each feature channel
  contributes equally to the augmented distance regardless of dynamic range.

Features are estimated from ``dst_features`` (the reference cloud).

The augmented KD-tree is built in the space ``[x, y, z, w·f₁, …, w·fₖ]``; inlier
filtering and error scoring remain in spatial (coordinate) units only.

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
   R = U_Q D U_P^T

The implementation uses a KD-tree to find nearest neighbor correspondences for each candidate transformation:

.. code-block:: python

   # Build KD-tree for target points
   kdtree = cKDTree(Q_centered, leafsize=leafsize)
   
   best_error = np.inf
   best_transform = U0
   
   for signs in [[1,1,1], [-1,1,1], [1,-1,1], [1,1,-1],
                 [-1,-1,1], [-1,1,-1], [1,-1,-1], [-1,-1,-1]]:
       D = np.diag(signs)
       U = Uq @ D @ Up.T
       P_transformed = P_centered @ U.T
       
       # Find nearest neighbors to establish correspondence
       distances, indices = kdtree.query(P_transformed)
       
       # Filter correspondences by distance threshold
       valid_mask = distances <= max_correspondence_distance
       if np.sum(valid_mask) / len(distances) < min_inlier_fraction:
           continue
           
       # Compute error using valid correspondences only
       error = np.sum(distances[valid_mask]**2)

6. Candidate Scoring and Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two scoring strategies are available, selected via ``params["scoring"]``.

**Hard scoring** (default, ``"hard"``)

Ranks candidates lexicographically: first by inlier count (src points whose
nearest neighbour in dst falls within ``max_correspondence_distance``), then
by sum of squared distances among ties:

.. math::
   \text{score}_{\text{hard}} = \text{inlier\_count} - \frac{\text{error}}{\text{error} + 1}

The second term is a normalised tie-breaker that stays below 1, so inlier
count always dominates.  This works well when dst is dense (full or near-full
overlap) but degrades when dst is a sparse subsample — missing points reduce
inlier counts for the *correct* candidate, making discrimination noisy.

**Soft scoring** (``"soft"``)

Replaces the binary inlier/outlier decision with a Gaussian kernel:

.. math::
   \text{score}_{\text{soft}} = \sum_{i=1}^{N} \exp\!\left(-\frac{d_i^2}{2\sigma^2}\right), \quad \sigma = \text{max\_correspondence\_distance}

Every src point contributes continuously.  A point whose nearest dst
neighbour is at distance :math:`d \ll \sigma` contributes ≈ 1; at
:math:`d = \sigma` it contributes :math:`e^{-1/2} \approx 0.6`; far points
decay smoothly to zero.  There is no hard cutoff, so the score does not
collapse when dst is sparse.

The ``min_inlier_fraction`` guard is still applied before scoring in both
modes to reject degenerate candidates outright.

**Observed degradation under dst subsampling** (LiDAR sphere, 100 trials each):

.. code-block:: text

   dst ratio    hard    soft    winner
     100%       100%    100%    tie
      80%        92%    100%    soft
      60%        80%    100%    soft
      50%        78%     99%    soft
      40%        75%     92%    soft
      30%        57%     78%    soft
      20%        44%     72%    soft

Soft scoring degrades more gracefully: at 50% subsampling hard is at 78%
while soft stays at 99%, and the gap widens as dst becomes sparser.  At full
overlap both strategies are equivalent.  Use ``"hard"`` for speed-critical
full-overlap pipelines; use ``"soft"`` whenever the destination cloud may be
a partial sample of the scene.

7. Parameter Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

All algorithm control parameters are passed as a single ``params`` dict.
Unspecified keys fall back to ``DEFAULT_PARAMS``::

   from einit import DEFAULT_PARAMS   # inspect defaults

**max_correspondence_distance** (default ``None``)
    Maximum spatial distance for a valid correspondence.  Auto-estimated as
    3× the median nearest-neighbour spacing inside dst when not supplied.
    Always in coordinate units.

**min_inlier_fraction** (default ``0.5``)
    Fraction of src points that must find a neighbour within the distance
    threshold.  Candidates below this fraction are rejected outright.

**leafsize** (default ``16``)
    KD-tree leaf size.  Smaller values can improve accuracy for small clouds
    at the cost of build time.  Typical range: 8–32.

**positive_only** (default ``False``)
    Restrict search to proper rotations (det +1), preventing reflections.
    Recommended when point distributions are spatially biased.

**scoring** (default ``"hard"``)
    Candidate ranking strategy: ``"hard"`` (lexicographic inlier-count then
    RMSE) or ``"soft"`` (Gaussian kernel, see above).

**src_features / dst_features**: Optional per-point feature arrays (shape N×k and M×k respectively). 1-D arrays are accepted and treated as (N, 1). Both must be provided together. Typical features: RGB colour, LiDAR intensity, surface normals.

**feature_weight**: Controls how strongly features influence the alignment (default 0.0 = geometry only). Typical useful range: 0.1–1.0. Setting this to 0.0 gives identical results to passing no features.

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
2. **Noise Tolerance**: RMSE grows approximately linearly with noise level, maintaining robust performance even with significant noise (validated via correlation analysis r > 0.7)
3. **Partial Overlap**: Works with point clouds that have different numbers of points, occlusions, and missing correspondences
4. **Permutation Invariance**: Point ordering in the input arrays does not affect the result
5. **Outlier Rejection**: Distance thresholding filters out poor correspondences
6. **Performance Scaling**: Time complexity verified to be sub-quadratic (O(n^α) where α < 2.0) via log-log regression analysis 


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