#!/usr/bin/env python3
"""
Demonstration of einit's permutation invariance property
"""
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from einit import ellipsoid_init_icp, barycentered

def apply_transform(pts, T):
    """Apply 4x4 homogeneous transform to points."""
    N = pts.shape[0]
    homo = np.hstack([pts, np.ones((N, 1))])
    return (T @ homo.T).T[:, :3]

def demo_permutation_invariance():
    """Demonstrate that einit handles permuted point clouds correctly."""
    print("=== EINIT Permutation Invariance Demo ===\n")
    
    # Generate random point cloud (same as in working test)
    np.random.seed(42)
    n_points = 500
    P = np.random.uniform(-2, 2, (n_points, 3))
    
    # Create orthogonal transformation O (random rotation)
    seed = np.random.normal(0.0, 1.0, (3, 3))
    O, _ = np.linalg.qr(seed)
    if np.linalg.det(O) < 0:  # Ensure proper rotation
        O[:, 0] *= -1
        
    # Generate Q = P @ O^T
    Q_clean = P @ O.T
    
    # Add small amount of noise
    noise_std = 0.01
    Q_clean += np.random.normal(0, noise_std, Q_clean.shape)
    
    # Create permuted version of Q
    perm_indices = np.random.permutation(n_points)
    Q_permuted = Q_clean[perm_indices]
    
    # Center both point clouds
    P_centered = barycentered(P)
    Q_permuted_centered = barycentered(Q_permuted)
    Q_clean_centered = barycentered(Q_clean)
    
    print(f"Point cloud size: {n_points} points")
    print(f"Noise level: {noise_std}")
    print(f"Random permutation applied to destination\n")
    
    # Test the algorithm with permuted destination
    T_recovered = ellipsoid_init_icp(P_centered, Q_permuted_centered)
    
    # Apply recovered transform to ORIGINAL P and compare with ORIGINAL Q (no permutation)
    P_aligned = apply_transform(P_centered, T_recovered)
    error_permuted = np.sqrt(np.mean(np.linalg.norm(P_aligned - Q_clean_centered, axis=1)**2))
    
    # For comparison, test without permutation
    T_direct = ellipsoid_init_icp(P_centered, Q_clean_centered)
    P_aligned_direct = apply_transform(P_centered, T_direct)
    error_direct = np.sqrt(np.mean(np.linalg.norm(P_aligned_direct - Q_clean_centered, axis=1)**2))
    
    print("Results:")
    print(f"  Direct alignment RMSE:    {error_direct:.6f}")
    print(f"  Permuted alignment RMSE:  {error_permuted:.6f}")
    print(f"  Performance ratio:        {error_permuted/error_direct:.2f}x")
    
    # Check if transformations are similar
    transform_diff = np.linalg.norm(T_direct - T_recovered, ord='fro')
    print(f"  Transform difference:     {transform_diff:.6f}")
    
    if error_permuted < 0.1 and error_permuted/error_direct < 2.0:
        print("\n✓ SUCCESS: einit handles permuted point clouds well!")
    else:
        print("\n✗ Issue detected with permutation handling")
    
    print(f"\nThis demonstrates that einit's ellipsoid-based approach")
    print(f"is robust to point ordering changes in the destination cloud.")

if __name__ == "__main__":
    demo_permutation_invariance()