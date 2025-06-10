import numpy as np
import pytest
import urllib.request
import tarfile
import io
from einit import register_ellipsoid


def apply_transform(pts, T):
    """Apply a 4x4 homogeneous transform T to an (N,3) array of points."""
    N = pts.shape[0]
    homo = np.hstack([pts, np.ones((N, 1))])
    return (T @ homo.T).T[:, :3]


def random_rigid_transform():
    """Generate a random rigid transformation matrix."""
    A = np.random.normal(size=(3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    t = np.random.uniform(-10, 10, size=(3,))
    T = np.eye(4)
    T[:3, :3] = Q
    T[:3, 3] = t
    return T


def random_reflection_transform():
    """Generate a random reflection transformation matrix."""
    signs = np.random.choice([-1, 1], size=3)
    D = np.diag(signs)
    t = np.random.uniform(-5, 5, size=(3,))
    T = np.eye(4)
    T[:3, :3] = D
    T[:3, 3] = t
    return T


def download_stanford_bunny(n_points=50000):
    """Download the Stanford Bunny point cloud from NREL dataset."""
    url = 'https://data.nrel.gov/system/files/153/bunny.csv'
    with urllib.request.urlopen(url) as resp:
        bunny = np.loadtxt(resp, delimiter=',')
    pts = bunny[:, :3]  # only x,y,z columns
    
    # Random sample if too many points
    if pts.shape[0] > n_points:
        idx = np.random.choice(pts.shape[0], n_points, replace=False)
        pts = pts[idx]
    return pts


def test_basic_functionality():
    """Test basic functionality with random data."""
    src = np.random.randn(100, 3)
    dst = np.random.randn(100, 3)
    T = register_ellipsoid(src, dst)
    assert T.shape == (4, 4)
    assert np.allclose(T[3, :], [0, 0, 0, 1])


def test_identity_transform():
    """Test with identical point clouds."""
    src = np.random.randn(50, 3)
    dst = src.copy()
    T = register_ellipsoid(src, dst)
    np.testing.assert_allclose(T, np.eye(4), atol=1e-10)


def test_synthetic_shapes_statistical(noise_std=0.02, n_points=1000):
    """Test synthetic shapes with statistical analysis over multiple runs.
    
    Args:
        noise_std: Standard deviation of Gaussian noise to add to target points
        n_points: Number of points to generate for sphere test
    """
    print(f"\n=== SPHERE STATISTICAL TEST (100 runs) ===")
    print(f"Test parameters: {n_points} points, noise_std={noise_std}")
    
    sphere_transform_errors = []
    sphere_clean_rmses = []
    sphere_successes = 0
    
    for i in range(100):
        np.random.seed(2000 + i)
        
        # Generate sphere
        phi = np.random.uniform(0, np.pi, n_points)
        theta = np.random.uniform(0, 2*np.pi, n_points)
        src_sphere = np.vstack([np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)]).T
        src_sphere *= np.random.uniform(3, 7, size=(n_points,1))
        
        T_true = random_rigid_transform()
        dst_full = apply_transform(src_sphere, T_true)
        noise = np.random.normal(scale=noise_std, size=dst_full.shape)
        dst_noisy = dst_full + noise
        
        # Use 80% of points for overlap
        n_overlap = int(0.8 * n_points)
        indices = np.random.choice(n_points, n_overlap, replace=False)
        
        T_recovered = register_ellipsoid(src_sphere[indices], dst_noisy[indices])
        
        transform_error = np.linalg.norm(T_recovered - T_true, ord='fro')
        
        src_clean_aligned = apply_transform(src_sphere, T_recovered)
        dst_clean = apply_transform(src_sphere, T_true)
        clean_rmse = np.sqrt(np.mean(np.linalg.norm(src_clean_aligned - dst_clean, axis=1)**2))
        
        sphere_transform_errors.append(transform_error)
        sphere_clean_rmses.append(clean_rmse)
        
        if transform_error < 0.05 and clean_rmse < 0.05:
            sphere_successes += 1
    
    print(f"Sphere success rate: {sphere_successes}/100 = {sphere_successes}%")
    print(f"Sphere transform error - Mean: {np.mean(sphere_transform_errors):.4f}, Std: {np.std(sphere_transform_errors):.4f}")
    print(f"Sphere clean RMSE - Mean: {np.mean(sphere_clean_rmses):.4f}, Std: {np.std(sphere_clean_rmses):.4f}")
    
    print(f"\n=== CUBE STATISTICAL TEST (100 runs) ===")
    print(f"Test parameters: cube surface points (15x15 grid per face), noise_std={noise_std}")
    
    cube_transform_errors = []
    cube_clean_rmses = []
    cube_successes = 0
    
    for i in range(100):
        np.random.seed(3000 + i)
        
        # Generate cube surface points 
        grid = np.linspace(-1, 1, 15)
        face = np.array(np.meshgrid(grid, grid)).reshape(2, -1).T
        
        faces = []
        for fixed_axis in range(3):
            for fixed_value in [-1, 1]:
                points = np.insert(face, fixed_axis, fixed_value, axis=1)
                faces.append(points)
        src_cube = np.vstack(faces) * np.array([2,1,0.5])
        
        T_true = random_reflection_transform()
        dst_full = apply_transform(src_cube, T_true)
        noise = np.random.normal(scale=noise_std, size=dst_full.shape)
        dst_noisy = dst_full + noise
        
        # Use 80% of points for overlap
        n_cube_points = src_cube.shape[0]
        n_overlap = int(0.8 * n_cube_points)
        indices = np.random.choice(n_cube_points, n_overlap, replace=False)
        
        T_recovered = register_ellipsoid(src_cube[indices], dst_noisy[indices])
        
        transform_error = np.linalg.norm(T_recovered - T_true, ord='fro')
        
        src_clean_aligned = apply_transform(src_cube, T_recovered)
        dst_clean = apply_transform(src_cube, T_true)
        clean_rmse = np.sqrt(np.mean(np.linalg.norm(src_clean_aligned - dst_clean, axis=1)**2))
        
        cube_transform_errors.append(transform_error)
        cube_clean_rmses.append(clean_rmse)
        
        if transform_error < 0.05 and clean_rmse < 0.05:
            cube_successes += 1
    
    print(f"Cube success rate: {cube_successes}/100 = {cube_successes}%")
    print(f"Cube transform error - Mean: {np.mean(cube_transform_errors):.4f}, Std: {np.std(cube_transform_errors):.4f}")
    print(f"Cube clean RMSE - Mean: {np.mean(cube_clean_rmses):.4f}, Std: {np.std(cube_clean_rmses):.4f}")
    
    # Assertions
    assert sphere_successes >= 80, f"Sphere success rate {sphere_successes}% too low"
    assert cube_successes >= 95, f"Cube success rate {cube_successes}% too low"
    assert np.mean(sphere_transform_errors) < 0.2, f"Mean sphere transform error too high"
    assert np.mean(cube_transform_errors) < 0.05, f"Mean cube transform error too high"


def test_bunny_cloud_statistical(noise_std=0.02, n_points=1500):
    """Test bunny with statistical analysis over multiple runs.
    
    Args:
        noise_std: Standard deviation of Gaussian noise to add to target points
        n_points: Number of points to downsample from Stanford bunny
    """
    print(f"\n=== BUNNY STATISTICAL TEST (100 runs) ===")
    print(f"Test parameters: {n_points} points, noise_std={noise_std}")
    
    # Load bunny once
    src = download_stanford_bunny(n_points=n_points)
    
    transform_errors = []
    clean_rmses = []
    successes = 0
    
    for i in range(100):
        np.random.seed(1000 + i)
        
        T_true = random_rigid_transform()
        dst_full = apply_transform(src, T_true)
        noise = np.random.normal(scale=noise_std, size=dst_full.shape)
        dst_noisy = dst_full + noise
        
        # Use 80% of points for overlap
        n_src_points = src.shape[0]
        n_overlap = int(0.8 * n_src_points)
        indices = np.random.choice(n_src_points, n_overlap, replace=False)
        
        T_recovered = register_ellipsoid(src[indices], dst_noisy[indices])
        
        # Compute errors
        transform_error = np.linalg.norm(T_recovered - T_true, ord='fro')
        
        src_clean_aligned = apply_transform(src, T_recovered)
        dst_clean = apply_transform(src, T_true)
        clean_rmse = np.sqrt(np.mean(np.linalg.norm(src_clean_aligned - dst_clean, axis=1)**2))
        
        transform_errors.append(transform_error)
        clean_rmses.append(clean_rmse)
        
        # Count successes
        if transform_error < 0.08 and clean_rmse < 0.08:
            successes += 1
    
    # Print statistics
    print(f"Success rate: {successes}/100 = {successes}%")
    print(f"Transform error - Mean: {np.mean(transform_errors):.4f}, Std: {np.std(transform_errors):.4f}")
    print(f"Transform error - Min: {np.min(transform_errors):.4f}, Max: {np.max(transform_errors):.4f}")
    print(f"Clean RMSE - Mean: {np.mean(clean_rmses):.4f}, Std: {np.std(clean_rmses):.4f}")
    print(f"Clean RMSE - Min: {np.min(clean_rmses):.4f}, Max: {np.max(clean_rmses):.4f}")
    
    # Should have reasonable success rate
    assert successes >= 70, f"Success rate {successes}% too low"
    assert np.mean(transform_errors) < 2.0, f"Mean transform error {np.mean(transform_errors):.4f} too high"
    assert np.mean(clean_rmses) < 0.2, f"Mean clean RMSE {np.mean(clean_rmses):.4f} too high"