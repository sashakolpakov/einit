#!/usr/bin/env python3
"""
Integration test for einit using the Stanford Bunny PLY dataset.
"""
import numpy as np
import pytest
import urllib.request
import tarfile
import io
import timeit

from einit import register_ellipsoid
try:
    from .test_einit import apply_transform, random_rigid_transform
except ImportError:
    from test_einit import apply_transform, random_rigid_transform


def download_stanford_bunny_ply(n_points=50000):
    """Download the Stanford Bunny mesh from PLY archive and extract vertices."""
    url = 'http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz'
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    tf = tarfile.open(fileobj=io.BytesIO(data), mode='r:gz')
    ply_member = None
    for m in tf.getmembers():
        if m.name.endswith('bun_zipper.ply'):
            ply_member = m
            break
    if ply_member is None:
        raise RuntimeError('Bunny PLY not found in archive')
    f = tf.extractfile(ply_member)
    
    # Parse PLY format: x y z confidence intensity
    header_ended = False
    verts = []
    for line in f:
        line = line.decode('utf-8').strip()
        if header_ended:
            parts = line.split()
            if len(parts) >= 5:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                verts.append([x, y, z])
        elif line == 'end_header':
            header_ended = True
    
    pts = np.array(verts)
    
    # Random sample if too many points
    if pts.shape[0] > n_points:
        idx = np.random.choice(pts.shape[0], n_points, replace=False)
        pts = pts[idx]
    return pts


def test_stanford_bunny_alignment():
    """Test bunny alignment using the Stanford PLY dataset with partial overlap and noise."""
    print("\n=== STANFORD BUNNY ALIGNMENT TEST ===")
    
    # Download the Stanford Bunny point cloud
    src_bunny_clean = download_stanford_bunny_ply(n_points=5000)
    print(f"Loaded Stanford bunny with {src_bunny_clean.shape[0]} points")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Apply a random rigid transform and simulate partial overlap + noise
    T_true = random_rigid_transform()
    dst_bunny_clean = apply_transform(src_bunny_clean, T_true)
    
    # Keep 80% of points for partial overlap
    src_mask = np.random.rand(src_bunny_clean.shape[0]) < 0.8
    src_partial = src_bunny_clean[src_mask]
    
    dst_mask = np.random.rand(dst_bunny_clean.shape[0]) < 0.8
    dst_partial = dst_bunny_clean[dst_mask]
    
    # Apply a random permutation
    perm = np.random.permutation(dst_partial.shape[0])
    dst_partial = dst_partial[perm]
    
    # Add small Gaussian noise
    noise = np.random.normal(scale=0.01, size=dst_partial.shape)
    dst_partial += noise
    
    # Run einit algorithm for timing statistics
    timer = timeit.Timer(lambda: register_ellipsoid(src_partial, dst_partial))
    times = np.array(timer.repeat(repeat=100, number=1))
    ms_per_call = times * 1e3
    
    mean_ms = ms_per_call.mean()
    std_ms = ms_per_call.std(ddof=1)
    print(f"\nregister_ellipsoid (bunny): {mean_ms:.3f} ms ± {std_ms:.3f} ms over 100 runs")
    
    # Run einit algorithm for alignment evaluation
    T_recovered = register_ellipsoid(src_partial, dst_partial)
    aligned_bunny = apply_transform(src_bunny_clean, T_recovered)
    
    # Compute alignment metrics
    src_aligned_full = apply_transform(src_bunny_clean, T_recovered)
    dst_clean_full = apply_transform(src_bunny_clean, T_true)
    alignment_rmse = np.sqrt(np.mean(np.linalg.norm(src_aligned_full - dst_clean_full, axis=1)**2))
    
    transform_error = np.linalg.norm(T_recovered - T_true, ord='fro')
    
    pct = 100 * src_partial.shape[0] / src_bunny_clean.shape[0]
    print(f"\nUsing {src_partial.shape[0]} of {src_bunny_clean.shape[0]} points ({pct:.1f}%)")
    print(f"Alignment RMSE: {alignment_rmse:.6f}")
    print(f"Transform error (Frobenius norm): {transform_error:.6f}")
    print(f"Performance: {mean_ms:.3f} ± {std_ms:.3f} ms per registration")
    
    # Validation assertions
    assert T_recovered is not None, "Algorithm should return a valid transformation matrix"
    assert T_recovered.shape == (4, 4), "Transformation matrix should be 4x4"
    assert np.allclose(T_recovered[3, :], [0, 0, 0, 1]), "Bottom row should be [0, 0, 0, 1]"
    assert alignment_rmse < 0.1, f"Alignment RMSE {alignment_rmse:.6f} should be reasonable for noisy Stanford bunny"
    assert mean_ms < 50.0, f"Performance {mean_ms:.3f} ms should be under 50ms for 5000 points"
    
    print("✓ Stanford bunny alignment test passed")


if __name__ == "__main__":
    test_stanford_bunny_alignment()