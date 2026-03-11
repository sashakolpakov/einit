
"""
einit – Ellipsoid ICP initialization

Provides register_ellipsoid(src, dst) that:
 1. Centers src & dst by their centroids.
 2. Computes ellipsoid matrices; eigen-decomposes to get principal axes.
 3. Searches all 8 diagonal ±1 reflections for best alignment.
 4. Returns a 4×4 homogeneous transform for OpenCV.
"""

import numpy as np
from scipy.spatial import cKDTree

__all__ = ["register_ellipsoid", "barycentered"]


def barycentered(points):
    """Center point cloud around barycenter (N×3 format)"""
    centroid = np.mean(points, axis=0)
    return points - centroid


def register_ellipsoid(src_points, dst_points,
                       src_features=None, dst_features=None,
                       feature_weight=0.0,
                       max_correspondence_distance=None,
                       min_inlier_fraction=0.5, leafsize=16, positive_only=False):
    """
    Compute initial transformation between 3D point clouds using ellipsoid analysis.

    This function computes an initial rigid transformation that aligns the source
    point cloud with the destination point cloud by analyzing their ellipsoids of
    inertia. The algorithm uses KD-tree correspondence recovery to handle point
    clouds with different orderings, partial overlaps, and outliers.

    Parameters
    ----------
    src_points : array_like, shape (N, 3)
        Source point cloud as N×3 array of 3D coordinates.
    dst_points : array_like, shape (M, 3)
        Destination point cloud as M×3 array of 3D coordinates.
        N and M can be different (partial overlap).
    src_features : array_like, shape (N, k), optional
        Per-point feature vectors for the source cloud (e.g. RGB colour, LiDAR
        intensity, surface normals).  1-D arrays of shape (N,) are accepted and
        treated as (N, 1).  If provided, dst_features must also be given.
    dst_features : array_like, shape (M, k), optional
        Per-point feature vectors for the destination cloud.  Must have the same
        number of columns as src_features.
    feature_weight : float, default 0.0
        Controls how strongly features influence the alignment.  0.0 (default)
        gives the original geometry-only behaviour.  Typical useful range: 0.1–1.0.

        Features enter the algorithm in two places:

        1. **Ellipsoid step** – the spatial covariance is augmented by the
           spatial-feature cross-covariance::

               E_aug = P^T P  +  (feature_weight / N) * (P^T F) (P^T F)^T

           This biases the principal axes toward spatial directions where features
           vary most, breaking eigenvalue degeneracy for symmetric shapes (spheres,
           cubes) where geometry alone gives arbitrary axes.

        2. **KD-tree step** – correspondences are searched in the augmented space
           [x, y, z, w·f₁, …, w·fₖ], so that feature similarity guides which
           destination point is selected as nearest neighbour.  Inlier filtering
           and error scoring remain in spatial (coordinate) units.

        Features are normalised to unit standard deviation (per column, estimated
        from dst_features) before scaling, so feature_weight is dimensionless and
        comparable across datasets.
    max_correspondence_distance : float, optional
        Maximum distance for valid point correspondences. Points farther than
        this distance from their nearest neighbors are considered outliers.
        If None (default), automatically estimated as 3× the median nearest-
        neighbor distance within the destination point cloud.
        Always interpreted in spatial (coordinate) units, even when features
        are used.
    min_inlier_fraction : float, default 0.5
        Minimum fraction of source points that must have valid correspondences
        within max_correspondence_distance. Transformations with fewer inliers
        are rejected. Must be between 0 and 1.
    leafsize : int, default 16
        KD-tree leaf size parameter. Affects search performance vs memory usage.
        Smaller values may improve accuracy for small point clouds but increase
        build time. Typical range: 8-32.
    positive_only : bool, default False
        If True, only search proper rotations (determinant +1) by considering
        only sign combinations with an even number of negative values. This
        prevents reflections and ensures chirality preservation. Recommended
        when point distributions are spatially biased (e.g., bounding box overlap).

    Returns
    -------
    T : ndarray, shape (4, 4)
        Homogeneous transformation matrix that transforms src_points to align
        with dst_points. Apply as: dst_aligned = (src @ T[:3,:3].T) + T[:3,3]

    Raises
    ------
    ValueError
        If input arrays don't have shape (N, 3) or (M, 3), if features are
        supplied inconsistently, or if feature_weight is negative.

    Examples
    --------
    Basic usage:

    >>> import numpy as np
    >>> from einit import register_ellipsoid
    >>> src = np.random.randn(100, 3)
    >>> dst = np.random.randn(80, 3)  # Different size OK
    >>> T = register_ellipsoid(src, dst)
    >>> T.shape
    (4, 4)

    With custom parameters:

    >>> T = register_ellipsoid(
    ...     src, dst,
    ...     max_correspondence_distance=0.1,
    ...     min_inlier_fraction=0.7,
    ...     leafsize=8,
    ...     positive_only=True
    ... )

    With per-point features (e.g. LiDAR intensity or RGB colour):

    >>> src_intensity = np.random.rand(100, 1)   # scalar reflectance per point
    >>> dst_intensity = np.random.rand(80, 1)
    >>> T = register_ellipsoid(
    ...     src, dst,
    ...     src_features=src_intensity,
    ...     dst_features=dst_intensity,
    ...     feature_weight=0.3,
    ... )

    Notes
    -----
    The algorithm is permutation-invariant: point ordering in the input arrays
    does not affect the result. It handles partial overlaps, noise, and outliers
    through KD-tree correspondence recovery and distance-based filtering.

    Time complexity is O(N + M log M) where N and M are the number of points
    in the source and destination clouds respectively.
    """
    if src_points.ndim != 2 or src_points.shape[1] != 3:
        raise ValueError("src_points must be (N,3) array")
    if dst_points.ndim != 2 or dst_points.shape[1] != 3:
        raise ValueError("dst_points must be (N,3) array")

    # ------------------------------------------------------------------
    # Validate and prepare optional per-point features
    # ------------------------------------------------------------------
    use_features = False
    if src_features is not None or dst_features is not None:
        if src_features is None or dst_features is None:
            raise ValueError(
                "src_features and dst_features must both be provided, or neither")
        src_features = np.asarray(src_features, dtype=float)
        dst_features = np.asarray(dst_features, dtype=float)
        if src_features.ndim == 1:
            src_features = src_features.reshape(-1, 1)
        if dst_features.ndim == 1:
            dst_features = dst_features.reshape(-1, 1)
        if src_features.shape[0] != src_points.shape[0]:
            raise ValueError(
                "src_features must have the same number of rows as src_points")
        if dst_features.shape[0] != dst_points.shape[0]:
            raise ValueError(
                "dst_features must have the same number of rows as dst_points")
        if src_features.shape[1] != dst_features.shape[1]:
            raise ValueError(
                "src_features and dst_features must have the same number of columns")
        if feature_weight < 0.0:
            raise ValueError("feature_weight must be non-negative")
        use_features = feature_weight > 0.0

    # Center point clouds
    centroid_src = np.mean(src_points, axis=0)
    centroid_dst = np.mean(dst_points, axis=0)
    P_centered = src_points - centroid_src  # N x 3
    Q_centered = dst_points - centroid_dst  # M x 3

    # ------------------------------------------------------------------
    # Compute ellipsoid matrices and eigendecompose.
    #
    # When features are active, the spatial covariance is augmented by
    # the spatial-feature cross-covariance term:
    #
    #   E_aug = P^T P  +  (feature_weight / N) * (P^T F) (P^T F)^T
    #
    # The cross-covariance P^T F  (shape 3×k) captures how much feature
    # variation is attributable to each spatial direction.  Contracting it
    # to a 3×3 matrix and adding it to the spatial covariance biases the
    # principal axes toward spatially informative directions, breaking the
    # eigenvalue degeneracy that plagues symmetric shapes.
    #
    # Dividing by N keeps the scale of the feature term comparable to the
    # spatial term regardless of point count.  Note that centering F is
    # redundant: P_centered^T @ mean(F) = 0 since P is already centered.
    # ------------------------------------------------------------------
    if use_features:
        # ------------------------------------------------------------------
        # Two different normalisations, for two different purposes:
        #
        # (a) COVARIANCE STEP — global scale only (max std across columns).
        #     Preserves the natural contrast ratio between feature channels.
        #     For example, if RGB face colours have R-contrast >> B-contrast,
        #     the X-axis gets a larger eigenvalue boost than the Z-axis,
        #     which is exactly what breaks the cube's geometric degeneracy.
        #     Per-column normalisation would equalise all channels and make
        #     E_xf @ E_xf^T isotropic — defeating the whole purpose.
        #
        # (b) KD-TREE STEP — per-column std normalisation.
        #     Each feature dimension contributes equally to the augmented
        #     KD-tree distance, regardless of its natural dynamic range.
        # ------------------------------------------------------------------
        feat_std_percol = dst_features.std(axis=0)
        feat_std_percol[feat_std_percol < 1e-10] = 1.0   # constant columns

        # (a) Single global scale: max std across all feature columns
        feat_global_scale = feat_std_percol.max()

        src_feat_cov = src_features / feat_global_scale   # (N, k)
        dst_feat_cov = dst_features / feat_global_scale   # (M, k)

        # Spatial-feature cross-covariance  (3 × k)
        E_xf_src = P_centered.T @ src_feat_cov
        E_xf_dst = Q_centered.T @ dst_feat_cov

        # Scale so feature_weight=1.0 means "feature term has same trace
        # (total variance) as the spatial term".  This makes the weight
        # intuitive and independent of N and feature magnitude.
        E_xx_src = P_centered.T @ P_centered
        E_xx_dst = Q_centered.T @ Q_centered
        E_ff_src = E_xf_src @ E_xf_src.T
        E_ff_dst = E_xf_dst @ E_xf_dst.T

        trace_ratio_src = np.trace(E_xx_src) / max(np.trace(E_ff_src), 1e-12)
        trace_ratio_dst = np.trace(E_xx_dst) / max(np.trace(E_ff_dst), 1e-12)

        Ep = E_xx_src + feature_weight * trace_ratio_src * E_ff_src
        Eq = E_xx_dst + feature_weight * trace_ratio_dst * E_ff_dst

        # (b) Per-column scaled features for KD-tree augmentation
        src_feat_scaled = feature_weight * src_features / feat_std_percol
        dst_feat_scaled = feature_weight * dst_features / feat_std_percol
    else:
        Ep = P_centered.T @ P_centered  # 3 x 3
        Eq = Q_centered.T @ Q_centered  # 3 x 3

    _, Up = np.linalg.eigh(Ep)
    _, Uq = np.linalg.eigh(Eq)

    # Initial transformation
    U0 = Uq @ Up.T

    # ------------------------------------------------------------------
    # Determine correspondence distance threshold if not provided.
    # Always computed in spatial-only space so the value stays in
    # coordinate units regardless of whether features are active.
    # ------------------------------------------------------------------
    if max_correspondence_distance is None:
        sample_size = min(1000, Q_centered.shape[0])
        sample_idx = np.random.choice(Q_centered.shape[0], sample_size, replace=False)
        temp_tree = cKDTree(Q_centered[sample_idx], leafsize=leafsize)
        sample_distances, _ = temp_tree.query(Q_centered[sample_idx], k=2)
        median_spacing = np.median(sample_distances[:, 1])
        max_correspondence_distance = 3.0 * median_spacing

    # ------------------------------------------------------------------
    # Build KD-tree for correspondence recovery.
    # When features are active the tree is built in the augmented space
    # [x, y, z, w·f₁, …, w·fₖ] so that feature similarity guides which
    # destination point is selected as nearest neighbour.
    # ------------------------------------------------------------------
    if use_features:
        kdtree = cKDTree(
            np.hstack([Q_centered, dst_feat_scaled]), leafsize=leafsize)
    else:
        kdtree = cKDTree(Q_centered, leafsize=leafsize)

    # Search discrete isometries for best alignment
    best_error = np.inf
    best_transform = U0
    best_inlier_count = 0

    # Choose sign combinations based on positive_only parameter
    if positive_only:
        base_det = np.linalg.det(Uq @ Up.T)
        if base_det > 0:
            sign_combinations = [[1,1,1], [-1,-1,1], [-1,1,-1], [1,-1,-1]]
        else:
            sign_combinations = [[-1,1,1], [1,-1,1], [1,1,-1], [-1,-1,-1]]
    else:
        sign_combinations = [[1,1,1], [-1,1,1], [1,-1,1], [1,1,-1],
                             [-1,-1,1], [-1,1,-1], [1,-1,-1], [-1,-1,-1]]

    for signs in sign_combinations:
        D = np.diag(signs)
        U = Uq @ D @ Up.T
        P_transformed = P_centered @ U.T

        # Find nearest neighbours.
        # With features: query augmented tree, then recover spatial distances
        # separately so inlier filtering and error remain in coordinate units.
        # Without features: standard spatial query.
        if use_features:
            _, nn_indices = kdtree.query(
                np.hstack([P_transformed, src_feat_scaled]))
            distances = np.linalg.norm(
                P_transformed - Q_centered[nn_indices], axis=1)
        else:
            distances, _ = kdtree.query(P_transformed)

        # Filter correspondences by spatial distance threshold
        valid_mask = distances <= max_correspondence_distance
        inlier_count = np.sum(valid_mask)
        inlier_fraction = inlier_count / len(distances)

        if inlier_fraction < min_inlier_fraction:
            continue

        if inlier_count > 0:
            error = np.sum(distances[valid_mask] ** 2)
            is_better = (inlier_count > best_inlier_count or
                         (inlier_count == best_inlier_count and error < best_error))
            if is_better:
                best_error = error
                best_transform = U
                best_inlier_count = inlier_count

    # Pack into 4×4 homogeneous transform
    T = np.eye(4, dtype=best_transform.dtype)
    T[:3, :3] = best_transform
    T[:3, 3] = centroid_dst - best_transform @ centroid_src
    return T
