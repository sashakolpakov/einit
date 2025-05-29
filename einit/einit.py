
"""
einit – Ellipsoid ICP initialization

Provides ellipsoid_init_icp(src, dst) that:
 1. Centers src & dst by their centroids.
 2. Computes ellipsoid matrices; eigen-decomposes to get principal axes.
 3. Searches all 8 diagonal ±1 reflections for best alignment.
 4. Returns a 4×4 homogeneous transform for OpenCV.
"""

import numpy as np

__all__ = ["ellipsoid_init_icp", "barycentered"]


def barycentered(points):
    """Center point cloud around barycenter (N×3 format)"""
    centroid = np.mean(points, axis=0)
    return points - centroid


def ellipsoid_init_icp(src_points, dst_points):
    """
    Ellipsoid ICP initialization following the original SageMath algorithm
    
    Args:
        src_points: Source point cloud (N x 3)
        dst_points: Destination point cloud (N x 3)
    
    Returns:
        4x4 homogeneous transformation matrix for OpenCV
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

    # Search all 8 discrete isometries for best alignment
    best_error = np.inf
    best_transform = U0
    for signs in [[1,1,1], [-1,1,1], [1,-1,1], [1,1,-1],
                  [-1,-1,1], [-1,1,-1], [1,-1,-1], [-1,-1,-1]]:
        U = U0 @ Up @ np.diag(signs) @ Up.T
        P_transformed = P_centered @ U.T  # Apply rotation to N x 3 points
        error = np.linalg.norm(P_transformed - Q_centered, ord='fro')
        if error < best_error:
            best_error = error
            best_transform = U

    # Pack into 4×4 homogeneous transform
    T = np.eye(4, dtype=best_transform.dtype)
    T[:3, :3] = best_transform
    T[:3, 3] = centroid_dst - best_transform @ centroid_src
    return T
