import numpy as np
import pytest
import urllib.request
import tarfile
import io
from reiya import register_conformal


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
    t = np.random.uniform(-0.2, 0.2, size=(3,))  # Smaller translation for bunny
    T = np.eye(4)
    T[:3, :3] = Q
    T[:3, 3] = t
    return T


def download_stanford_bunny(n_points=50000):
    """Download the Stanford Bunny mesh, extract vertices, sample n_points."""
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
    
    # Parse PLY - format is: x y z confidence intensity
    header_ended = False
    verts = []
    for line in f:
        line = line.decode('utf-8').strip()
        if header_ended:
            parts = line.split()
            if len(parts) >= 5:  # Ensure we have all 5 values
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


def test_basic_functionality():
    """Test basic functionality with random data."""
    src = np.random.randn(100, 3)
    dst = np.random.randn(100, 3)
    result = register_conformal(src, dst)
    assert isinstance(result, dict)
    assert 'registration_error' in result
    assert 'transformation_params' in result
    assert result['registration_error'] >= 0


def test_identical_point_clouds():
    """Test with identical point clouds."""
    src = np.random.randn(50, 3)
    dst = src.copy()
    result = register_conformal(src, dst)
    assert result['registration_error'] < 0.1  # Should be very low for identical clouds


def test_synthetic_shapes_statistical(noise_std=0.02, overlap_fraction=0.8, n_points=500):
    """Test synthetic shapes with statistical analysis over multiple runs."""
    print(f"\n=== SPHERE CONFORMAL TEST (50 runs) ===")
    print(f"Test parameters: {n_points} points, noise_std={noise_std}, overlap={overlap_fraction*100:.0f}%")
    
    sphere_errors = []
    sphere_successes = 0
    
    for i in range(50):
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
        mask = np.random.choice([True, False], size=(n_points,), p=[overlap_fraction, 1-overlap_fraction])
        src_o = src_sphere[mask]
        dst_o = dst_noisy[mask]

        min_points = min(len(src_o), len(dst_o))
        src_o = src_o[:min_points]
        dst_o = dst_o[:min_points]
        
        result = register_conformal(src_o, dst_o)
        
        registration_error = result['registration_error']
        sphere_errors.append(registration_error)
        
        # Success if registration error is reasonable
        if registration_error < 0.5:
            sphere_successes += 1
    
    print(f"Sphere success rate: {sphere_successes}/50 = {sphere_successes*2}%")
    print(f"Sphere registration error - Mean: {np.mean(sphere_errors):.4f}, Std: {np.std(sphere_errors):.4f}")
    
    print(f"\n=== CUBE CONFORMAL TEST (50 runs) ===")
    print(f"Test parameters: {12**3} points, noise_std={noise_std}, overlap={overlap_fraction*100:.0f}%")
    
    cube_errors = []
    cube_successes = 0
    
    for i in range(50):
        np.random.seed(3000 + i)
        
        # Generate cube
        grid = np.linspace(-1,1,8)  # Smaller cube for faster testing
        X, Y, Z = np.meshgrid(grid, grid, grid)
        src_cube = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T * np.array([2,1,0.5])
        
        T_true = random_rigid_transform()
        dst_full = apply_transform(src_cube, T_true)
        noise = np.random.normal(scale=noise_std, size=dst_full.shape)
        dst_noisy = dst_full + noise
        mask = np.random.choice([True, False], size=(src_cube.shape[0],), p=[overlap_fraction, 1-overlap_fraction])
        src_o = src_cube[mask]
        dst_o = dst_noisy[mask]

        min_points = min(len(src_o), len(dst_o))
        src_o = src_o[:min_points]
        dst_o = dst_o[:min_points]
        
        result = register_conformal(src_o, dst_o)
        
        registration_error = result['registration_error']
        cube_errors.append(registration_error)
        
        # Success if registration error is reasonable
        if registration_error < 0.5:
            cube_successes += 1
    
    print(f"Cube success rate: {cube_successes}/50 = {cube_successes*2}%")
    print(f"Cube registration error - Mean: {np.mean(cube_errors):.4f}, Std: {np.std(cube_errors):.4f}")
    
    # Assertions - more lenient than einit since conformal registration is different
    assert sphere_successes >= 25, f"Sphere success rate {sphere_successes*2}% too low"
    assert cube_successes >= 20, f"Cube success rate {cube_successes*2}% too low"
    assert np.mean(sphere_errors) < 1.0, f"Mean sphere registration error too high"
    assert np.mean(cube_errors) < 1.0, f"Mean cube registration error too high"


def test_bunny_conformal_registration(noise_std=0.01, overlap_fraction=0.85, n_points=2000):
    """Test bunny with conformal registration analysis over multiple runs."""
    print(f"\n=== BUNNY CONFORMAL TEST (20 runs) ===")
    print(f"Test parameters: {n_points} points, noise_std={noise_std}, overlap={overlap_fraction*100:.0f}%")
    
    # Load bunny once
    src = download_stanford_bunny(n_points=n_points)
    
    registration_errors = []
    successes = 0
    
    for i in range(20):
        np.random.seed(1000 + i)  # Different seed each time
        
        T_true = random_rigid_transform()
        dst = apply_transform(src, T_true)
        noise = np.random.normal(scale=noise_std, size=dst.shape)
        dst_noisy = dst + noise
        mask = np.random.choice([True, False], size=(src.shape[0],), p=[overlap_fraction, 1-overlap_fraction])
        src_o = src[mask]
        dst_o = dst_noisy[mask]

        min_points = min(len(src_o), len(dst_o))
        src_o = src_o[:min_points]
        dst_o = dst_o[:min_points]
        
        result = register_conformal(src_o, dst_o)
        
        # Compute registration error
        registration_error = result['registration_error']
        registration_errors.append(registration_error)
        
        # Count successes - more lenient criteria for conformal registration
        if registration_error < 0.3:
            successes += 1
    
    # Print statistics
    print(f"Success rate: {successes}/20 = {successes*5}%")
    print(f"Registration error - Mean: {np.mean(registration_errors):.4f}, Std: {np.std(registration_errors):.4f}")
    print(f"Registration error - Min: {np.min(registration_errors):.4f}, Max: {np.max(registration_errors):.4f}")
    
    # Should have reasonable success rate - more lenient than einit
    assert successes >= 10, f"Success rate {successes*5}% too low"
    assert np.mean(registration_errors) < 1.0, f"Mean registration error {np.mean(registration_errors):.4f} too high"