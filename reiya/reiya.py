"""
reiya – Conformal Registration System

Provides register_conformal(src, dst) that:
 1. Computes Gaussian maps (surface normals) for both point clouds
 2. Projects to complex plane via stereographic projection  
 3. Optimizes Möbius transformation for conformal registration
 4. Returns registration result with transformation parameters
"""

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree
from scipy.optimize import minimize

__all__ = ["register_conformal", "gaussian_map"]


def gaussian_map(points, k_neighbors=20):
    """
    Map 3D point cloud to unit sphere via local surface normals.
    
    Parameters
    ----------
    points : array_like, shape (N, 3)
        3D point cloud coordinates
    k_neighbors : int, default=20
        Number of neighbors to use for local surface normal estimation
        
    Returns
    -------
    sphere_points : ndarray, shape (N, 3)
        Unit normals projected onto sphere
    """
    tree = cKDTree(points)
    sphere_points = []
    
    for i in range(len(points)):
        _, idx = tree.query(points[i], k=k_neighbors)
        neighborhood = points[idx]
        
        # Center the neighborhood for PCA
        centered = neighborhood - np.mean(neighborhood, axis=0)
        
        # Principal Component Analysis to find local surface normal
        pca = PCA(n_components=3)
        pca.fit(centered)
        
        # Normal is the component with smallest variance
        normal = pca.components_[-1]
        
        # Ensure consistent orientation (outward normal)
        centroid = np.mean(points, axis=0)
        if np.dot(normal, points[i] - centroid) < 0:
            normal = -normal
            
        sphere_points.append(normal)
        
    return np.array(sphere_points)


def _stereographic_projection(sphere_points, pole=np.array([0, 0, 1])):
    """Stereographic projection from unit sphere to complex plane."""
    eps = 1e-10
    sphere_points = sphere_points.copy()
    
    dot_products = np.dot(sphere_points, pole)
    too_close = np.abs(dot_products - 1) < eps
    if np.any(too_close):
        sphere_points[too_close] += eps * np.random.randn(np.sum(too_close), 3)
        sphere_points[too_close] /= np.linalg.norm(sphere_points[too_close], axis=1, keepdims=True)
    
    denominators = 1 - np.dot(sphere_points, pole)
    denominators[denominators == 0] = eps
    
    x = sphere_points[:, 0] / denominators
    y = sphere_points[:, 1] / denominators
    
    return x + 1j * y


def _inverse_stereographic(complex_points, pole=np.array([0, 0, 1])):
    """Inverse stereographic projection from complex plane to sphere."""
    x, y = complex_points.real, complex_points.imag
    denom = 1 + x**2 + y**2
    
    sphere_x = 2 * x / denom
    sphere_y = 2 * y / denom  
    sphere_z = (denom - 2) / denom
    
    return np.column_stack([sphere_x, sphere_y, sphere_z])


def _mobius_transform(z, a, b, c, d):
    """Apply Möbius transformation: f(z) = (az + b) / (cz + d)"""
    return (a * z + b) / (c * z + d)


# Global cache for KD-tree optimization
_kdtree_cache = {
    'tree': None,
    'last_params': None,
    'target_2d': None,
    'rebuild_threshold': 0.01  # Rebuild if params change by more than this
}

def _conformal_energy(params, source_complex, target_complex):
    """Compute conformal distortion energy for registration optimization."""
    a = params[0] + 1j * params[1]
    b = params[2] + 1j * params[3] 
    c = params[4] + 1j * params[5]
    
    # Determine d from normalization constraint ad - bc = 1
    d = 1.0
    a = (1 + b * c) / d
    
    # Apply Möbius transformation to source
    transformed_source = _mobius_transform(source_complex, a, b, c, d)
    
    # Convert complex points to 2D for KD-tree
    target_2d = np.column_stack([target_complex.real, target_complex.imag])
    transformed_2d = np.column_stack([transformed_source.real, transformed_source.imag])
    
    # Check if we need to rebuild KD-tree
    rebuild_tree = False
    if _kdtree_cache['tree'] is None or _kdtree_cache['last_params'] is None:
        rebuild_tree = True
    elif _kdtree_cache['target_2d'] is None or not np.array_equal(_kdtree_cache['target_2d'], target_2d):
        rebuild_tree = True  # Target points changed
    else:
        # Check if parameters changed significantly
        param_diff = np.linalg.norm(params - _kdtree_cache['last_params'])
        if param_diff > _kdtree_cache['rebuild_threshold']:
            rebuild_tree = True
    
    # Rebuild KD-tree only when necessary
    if rebuild_tree:
        _kdtree_cache['tree'] = cKDTree(target_2d)
        _kdtree_cache['last_params'] = params.copy()
        _kdtree_cache['target_2d'] = target_2d.copy()
    
    # Find closest point correspondences using cached KD-tree
    distances, closest_indices = _kdtree_cache['tree'].query(transformed_2d)
    
    # Compute registration error
    registration_error = np.mean(distances**2)
    
    # Add regularization
    param_penalty = 0.01 * np.sum(np.abs(params)**2)
    
    return registration_error + param_penalty


def _clear_kdtree_cache():
    """Clear the KD-tree cache for a fresh optimization."""
    global _kdtree_cache
    _kdtree_cache['tree'] = None
    _kdtree_cache['last_params'] = None
    _kdtree_cache['target_2d'] = None

def register_conformal(src_points, dst_points, k_neighbors=20, 
                      max_correspondence_distance=None, min_inlier_fraction=0.5):
    """
    Compute conformal registration between 3D point clouds using Gaussian maps.
    
    This function performs conformal registration by mapping point clouds to the
    unit sphere via their Gaussian maps, then optimizing Möbius transformations
    in the complex plane to align the spherical representations.
    
    Parameters
    ----------
    src_points : array_like, shape (N, 3)
        Source point cloud as N×3 array of 3D coordinates.
    dst_points : array_like, shape (M, 3)
        Destination point cloud as M×3 array of 3D coordinates.
        N and M can be different (partial overlap).
    k_neighbors : int, default=20
        Number of neighbors to use for local surface normal estimation.
        Affects the quality of Gaussian map computation.
    max_correspondence_distance : float, optional
        Maximum distance for valid point correspondences. Currently unused
        but included for API compatibility with einit.
    min_inlier_fraction : float, default=0.5
        Minimum fraction of inlier correspondences. Currently unused
        but included for API compatibility with einit.
    
    Returns
    -------
    result : dict
        Registration result containing:
        - 'transformation_params': Möbius transformation parameters
        - 'registration_error': Final optimization error
        - 'source_sphere': Source Gaussian map
        - 'target_sphere': Target Gaussian map  
        - 'transformed_sphere': Transformed source spherical representation
        - 'optimization_result': Scipy optimization result
    
    Raises
    ------
    ValueError
        If input arrays don't have shape (N, 3) or (M, 3).
    
    Examples
    --------
    Basic usage:
    
    >>> import numpy as np
    >>> from reiya import register_conformal
    >>> src = np.random.randn(100, 3)
    >>> dst = np.random.randn(80, 3)
    >>> result = register_conformal(src, dst)
    >>> result['registration_error'] < 1.0
    True
    
    With custom parameters:
    
    >>> result = register_conformal(
    ...     src, dst,
    ...     k_neighbors=30,
    ...     min_inlier_fraction=0.7
    ... )
    
    Notes
    -----
    The algorithm preserves angular relationships through conformal mappings,
    making it particularly suitable for shape analysis and registration tasks
    where local geometric properties are important.
    
    Time complexity is O(N * k_neighbors * log(k_neighbors) + optimization_cost)
    where N is the number of points in the source cloud.
    """
    if src_points.ndim != 2 or src_points.shape[1] != 3:
        raise ValueError("src_points must be (N,3) array")
    if dst_points.ndim != 2 or dst_points.shape[1] != 3:
        raise ValueError("dst_points must be (N,3) array")
    
    # Clear KD-tree cache for fresh optimization
    _clear_kdtree_cache()
    
    # Compute Gaussian maps (surface normals on unit sphere)
    source_sphere = gaussian_map(src_points, k_neighbors)
    target_sphere = gaussian_map(dst_points, k_neighbors)
    
    # Project to complex plane via stereographic projection
    source_complex = _stereographic_projection(source_sphere)
    target_complex = _stereographic_projection(target_sphere)
    
    # Initial parameters for Möbius transformation (near identity)
    initial_params = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Optimize conformal transformation
    result = minimize(
        _conformal_energy,
        initial_params,
        args=(source_complex, target_complex),
        method='L-BFGS-B',
        options={'maxiter': 1000, 'ftol': 1e-9}
    )
    
    # Extract optimal parameters
    optimal_params = result.x
    a = optimal_params[0] + 1j * optimal_params[1]
    b = optimal_params[2] + 1j * optimal_params[3]
    c = optimal_params[4] + 1j * optimal_params[5]
    d = 1.0
    a = (1 + b * c) / d  # Ensure constraint ad - bc = 1
    
    # Apply optimal transformation
    transformed_source = _mobius_transform(source_complex, a, b, c, d)
    transformed_sphere = _inverse_stereographic(transformed_source)
    
    return {
        'transformation_params': {'a': a, 'b': b, 'c': c, 'd': d},
        'registration_error': result.fun,
        'source_sphere': source_sphere,
        'target_sphere': target_sphere,
        'transformed_sphere': transformed_sphere,
        'optimization_result': result
    }