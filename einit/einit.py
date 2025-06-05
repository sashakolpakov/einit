
"""
einit – Ellipsoid ICP initialization

Provides ellipsoid_init_icp(src, dst) that:
 1. Centers src & dst by their centroids.
 2. Computes ellipsoid matrices; eigen-decomposes to get principal axes.
 3. Searches all 8 diagonal ±1 reflections for best alignment.
 4. Returns a 4×4 homogeneous transform for OpenCV.
"""

import numpy as np
from scipy.spatial import cKDTree

__all__ = ["ellipsoid_init_icp", "barycentered"]


def barycentered(points):
    """Center point cloud around barycenter (N×3 format)"""
    centroid = np.mean(points, axis=0)
    return points - centroid


def ellipsoid_init_icp(src_points, dst_points, max_correspondence_distance=None,
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
    max_correspondence_distance : float, optional
        Maximum distance for valid point correspondences. Points farther than
        this distance from their nearest neighbors are considered outliers.
        If None (default), automatically estimated as 3× the median nearest-
        neighbor distance within the destination point cloud.
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
        If input arrays don't have shape (N, 3) or (M, 3).
    
    Examples
    --------
    Basic usage:
    
    >>> import numpy as np
    >>> from einit import ellipsoid_init_icp
    >>> src = np.random.randn(100, 3)
    >>> dst = np.random.randn(80, 3)  # Different size OK
    >>> T = ellipsoid_init_icp(src, dst)
    >>> T.shape
    (4, 4)
    
    With custom parameters:
    
    >>> T = ellipsoid_init_icp(
    ...     src, dst,
    ...     max_correspondence_distance=0.1,
    ...     min_inlier_fraction=0.7,
    ...     leafsize=8,
    ...     positive_only=True
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

    # Center point clouds
    centroid_src = np.mean(src_points, axis=0)
    centroid_dst = np.mean(dst_points, axis=0)
    P_centered = src_points - centroid_src  # N x 3
    Q_centered = dst_points - centroid_dst  # N x 3

    # Compute ellipsoid matrices and eigendecompose
    Ep = P_centered.T @ P_centered  # 3 x 3
    Eq = Q_centered.T @ Q_centered  # 3 x 3
    _, Up = np.linalg.eigh(Ep)
    _, Uq = np.linalg.eigh(Eq)

    # Initial transformation
    U0 = Uq @ Up.T

    # Determine correspondence distance threshold if not provided
    if max_correspondence_distance is None:
        # Estimate typical point spacing in target cloud
        sample_size = min(1000, Q_centered.shape[0])
        sample_idx = np.random.choice(Q_centered.shape[0], sample_size, replace=False)
        temp_tree = cKDTree(Q_centered[sample_idx], leafsize=leafsize)
        sample_distances, _ = temp_tree.query(Q_centered[sample_idx], k=2)
        median_spacing = np.median(sample_distances[:, 1])
        max_correspondence_distance = 3.0 * median_spacing
    # Search discrete isometries for best alignment
    # Use KD-tree for efficient nearest neighbor search to recover correspondences
    best_error = np.inf
    best_transform = U0
    best_inlier_count = 0
    # Build KD-tree for target points with specified leaf size
    kdtree = cKDTree(Q_centered, leafsize=leafsize)
    # Choose sign combinations based on positive_only parameter
    if positive_only:
        # Determine which parity of negative signs gives positive determinant
        base_det = np.linalg.det(Uq @ Up.T)
        if base_det > 0:
            # Need even number of negative signs
            sign_combinations = [[1,1,1], [-1,-1,1], [-1,1,-1], [1,-1,-1]]
        else:
            # Need odd number of negative signs
            sign_combinations = [[-1,1,1], [1,-1,1], [1,1,-1], [-1,-1,-1]]
    else:
        # All 8 isometries (4 proper rotations + 4 reflections)
        sign_combinations = [[1,1,1], [-1,1,1], [1,-1,1], [1,1,-1],
                           [-1,-1,1], [-1,1,-1], [1,-1,-1], [-1,-1,-1]]
    for signs in sign_combinations:
        D = np.diag(signs)
        U = Uq @ D @ Up.T
        P_transformed = P_centered @ U.T
        # Find nearest neighbors to establish correspondence
        distances, _ = kdtree.query(P_transformed)

        # Filter correspondences by distance threshold
        valid_mask = distances <= max_correspondence_distance
        inlier_count = np.sum(valid_mask)
        inlier_fraction = inlier_count / len(distances)
        # Skip if too few valid correspondences
        if inlier_fraction < min_inlier_fraction:
            continue
        # Compute error using only valid correspondences
        if inlier_count > 0:
            valid_distances = distances[valid_mask]
            error = np.sum(valid_distances**2)
            # Prefer solutions with more inliers, then lower error
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
