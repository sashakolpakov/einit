import numpy as np
import timeit
import urllib.request
import pytest
import open3d as o3d
from scipy.spatial import cKDTree
from colorama import Fore, Style
from einit import register_ellipsoid


def make_robust(n_runs=100, fail_threshold=5):
    def decorator(test_fn):
        print(f"\n{Fore.CYAN}ROBUST {test_fn.__name__}: n_runs = {n_runs}, fail_threshold = {fail_threshold}{Style.RESET_ALL}")
        def wrapped():
            failures = 0
            for _ in range(n_runs):
                try:
                    test_fn()
                except AssertionError as e:
                    print(f"\n{Fore.CYAN}FAILURE: {e}{Style.RESET_ALL}")
                    failures += 1
            assert failures <= fail_threshold, f"{test_fn.__name__} failed {failures}/{n_runs} times"
        return wrapped
    return decorator


def apply_transform(pts, T):
    """Apply a 4×4 homogeneous transformation matrix to an (N, 3) point cloud."""
    N = pts.shape[0]
    homo = np.hstack([pts, np.ones((N, 1))])
    return (T @ homo.T).T[:, :3]


def random_rigid_transform():
    """Generate a random 4×4 rigid transformation matrix (rotation + translation)."""
    A = np.random.normal(size=(3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    t = np.random.uniform(-10, 10, size=(3,))
    T = np.eye(4)
    T[:3, :3] = Q
    T[:3, 3] = t
    return T


def download_stanford_bunny(n_points=50000):
    """Download the Stanford Bunny CSV and return a subset of n_points (x, y, z)."""
    url = 'https://data.nrel.gov/system/files/153/bunny.csv'
    with urllib.request.urlopen(url) as resp:
        bunny = np.loadtxt(resp, delimiter=',')
    pts = bunny[:, :3]
    if pts.shape[0] > n_points:
        idx = np.random.choice(pts.shape[0], n_points, replace=False)
        pts = pts[idx]
    return pts


def test_basic_functionality():
    """Test that register_ellipsoid outputs a proper 4×4 matrix on random clouds."""
    src = np.random.randn(100, 3)
    dst = np.random.randn(100, 3)
    T = register_ellipsoid(src, dst)
    assert T.shape == (4, 4)
    assert np.allclose(T[3, :], [0, 0, 0, 1])


def test_identity_transform():
    """Test that register_ellipsoid recovers identity when source and target match."""
    src = np.random.randn(50, 3)
    dst = src.copy()
    T = register_ellipsoid(src, dst)
    np.testing.assert_allclose(T, np.eye(4), atol=1e-10)


@make_robust(n_runs=100, fail_threshold=10)
def test_random_gaussian_cloud():
    """Test Einit alignment of a Gaussian cloud under transformation + permutation"""
    n_points = 1500
    src = np.random.randn(n_points, 3)
    T_true = random_rigid_transform()
    dst = apply_transform(src, T_true)
    dst = dst[np.random.permutation(n_points)]

    T_est = register_ellipsoid(src, dst)
    aligned = apply_transform(src, T_est)

    # Use cKDTree-based nearest neighbor RMSE
    tree = cKDTree(dst)
    dists, _ = tree.query(aligned)
    rmse = np.sqrt(np.mean(dists**2))

    assert rmse < 0.075, f"Random cloud alignment RMSE too high: {rmse:.4f}"


@make_robust(n_runs=100, fail_threshold=10)
def test_sphere_alignment():
    """Test alignment of spherical data under transformation + permutation + noise."""
    n_points = 1500
    std_noise = 0.02
    src = np.random.randn(n_points, 3)
    src /= np.linalg.norm(src, axis=1, keepdims=True)
    T_true = random_rigid_transform()
    dst = apply_transform(src, T_true)
    dst = dst[np.random.permutation(n_points)] + np.random.normal(0, std_noise, size=(n_points, 3))

    T_est = register_ellipsoid(src, dst)
    aligned = apply_transform(src, T_est)

    # Use cKDTree-based nearest neighbor RMSE
    tree = cKDTree(dst)
    dists, _ = tree.query(aligned)
    rmse = np.sqrt(np.mean(dists**2))

    assert rmse < 0.075, f"Sphere alignment RMSE too high: {rmse:.4f}"


@make_robust(n_runs=100, fail_threshold=10)
def test_cube_alignment():
    """
    Test cube alignment using Open3D point-to-plane ICP refinement.
    This tests uses random transformation + permutation + noise + partial overlap. 

    Uses cKDTree nearest-neighbor RMSE instead of point-to-point RMSE
    to account for symmetry-related ambiguities in cube correspondences.
    """
    grid = np.linspace(-1, 1, 15)
    face = np.array(np.meshgrid(grid, grid)).reshape(2, -1).T
    faces = []
    for axis in range(3):
        for val in [-1, 1]:
            faces.append(np.insert(face, axis, val, axis=1))
    src_clean = np.vstack(faces)

    T_true = random_rigid_transform()
    dst_clean = apply_transform(src_clean, T_true)

    # Simulate partial overlap (80% of points)  
    src_mask = np.random.choice([True, False], size=(src_clean.shape[0],), p=[0.8, 0.2])
    src_partial = src_clean[src_mask]
    
    dst_mask = np.random.choice([True, False], size=(src_clean.shape[0],), p=[0.8, 0.2])
    dst_partial = dst_clean[dst_mask]
    
    # Apply random permutation
    perm = np.random.permutation(dst_partial.shape[0])
    dst_partial = dst_partial[perm]
    
    # Add noise
    dst_partial += np.random.normal(scale=0.02, size=dst_partial.shape)

    T_init = register_ellipsoid(src_partial, dst_partial)
    src_aligned = apply_transform(src_partial, T_init)

    def to_o3d_pc(points):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points)
        pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=6))
        pc.normalize_normals()
        return pc

    src_o3d = to_o3d_pc(src_aligned)
    dst_o3d = to_o3d_pc(dst_partial)

    result = o3d.pipelines.registration.registration_icp(
        src_o3d, dst_o3d, max_correspondence_distance=2.0,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

    T_refined = result.transformation @ T_init
    
    # Evaluate on clean clouds using symmetry-invariant nearest-neighbor RMSE
    src_clean_aligned = apply_transform(src_clean, T_refined)
    tree = cKDTree(dst_clean)
    dists, _ = tree.query(src_clean_aligned, distance_upper_bound=0.5)
    rmse = np.sqrt(np.mean(dists**2))

    assert rmse < 0.05, f"Cube alignment RMSE too high: {rmse:.4f}"


def test_bunny_alignment():
    """
    Test alignment on Stanford Bunny point cloud with partial overlap and noise.
    Uses fixed random seeds for reproducibility and skips the test if download fails.
    """
    n_points = 1000
    std_noise = 0.02
    try:
        src = download_stanford_bunny(n_points=n_points)
    except Exception:
        pytest.skip("Could not download Stanford Bunny dataset")

    T_true = random_rigid_transform()
    dst = apply_transform(src, T_true)

    rng_src = np.random.default_rng(seed=17)
    rng_dst = np.random.default_rng(seed=71)
    rng_perm = np.random.default_rng(seed=123)

    src_mask = rng_src.random(src.shape[0]) < 0.8
    dst_mask = rng_dst.random(dst.shape[0]) < 0.8

    src_partial = src[src_mask]
    dst_partial = dst[dst_mask]
    dst_partial = dst_partial[rng_perm.permutation(dst_partial.shape[0])]
    dst_partial += np.random.normal(0, std_noise, dst_partial.shape)

    T_est = register_ellipsoid(src_partial, dst_partial)
    aligned = apply_transform(src, T_est)

    rmse = np.sqrt(np.mean(np.sum((aligned - dst) ** 2, axis=1)))
    assert rmse < 0.05, f"Bunny alignment RMSE too high: {rmse:.4f}"


@make_robust(n_runs=50, fail_threshold=5)
def test_performance_scaling():
    """
    Test that runtime scales sub-quadratically with point count.
    Uses log-log regression to verify scaling exponent is reasonable.
    """
    sizes = [100, 500, 1000, 2500]
    times = []
    
    for size in sizes:
        # Use fixed seed for reproducible point generation
        np.random.seed(42)
        phi = np.random.uniform(0, np.pi, size)
        theta = np.random.uniform(0, 2*np.pi, size)
        src = np.vstack([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi)
        ]).T * 3
        T_true = random_rigid_transform()
        dst = apply_transform(src, T_true) + np.random.normal(scale=0.02, size=src.shape)
        
        # Time the registration with multiple runs for stability
        timer = timeit.Timer(lambda: register_ellipsoid(src, dst))
        elapsed = min(timer.repeat(repeat=3, number=1)) * 1000  # Convert to ms
        times.append(elapsed)
    
    # Use log-log regression to find scaling exponent
    log_sizes = np.log(sizes)
    log_times = np.log(times)
    
    # Fit line: log(time) = slope * log(size) + intercept
    slope, intercept = np.polyfit(log_sizes, log_times, 1)
    
    # Performance should scale better than quadratic (slope < 2)
    assert slope < 2.0, f"Performance scaling too steep: O(n^{slope:.2f}), should be sub-quadratic"
    
    # Should scale at least linearly (not constant time)
    assert slope > 0.5, f"Performance scaling too flat: O(n^{slope:.2f}), should scale with input size"
    
    # Basic sanity check: largest case should complete in reasonable time
    assert times[-1] < 1000, f"Performance too slow for {sizes[-1]} points: {times[-1]:.1f}ms"


@make_robust(n_runs=50, fail_threshold=5)
def test_noise_robustness():
    """
    Test that RMSE grows approximately linearly with noise level.
    Uses correlation coefficient to verify linear relationship.
    """
    n_points = 500
    noise_levels = [0.01, 0.02, 0.05, 0.1, 0.2]
    rmse_values = []
    
    # Set seed for reproducible point generation within each run
    np.random.seed(42)
    
    # Generate same base geometry for all noise levels
    phi = np.random.uniform(0, np.pi, n_points)
    theta = np.random.uniform(0, 2*np.pi, n_points)
    src = np.vstack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ]).T * 3
    T_true = random_rigid_transform()
    dst_clean = apply_transform(src, T_true)
    
    for noise_std in noise_levels:
        # Add noise to clean transformed points
        dst_noisy = dst_clean + np.random.normal(0, noise_std, size=dst_clean.shape)
        T_est = register_ellipsoid(src, dst_noisy)
        aligned = apply_transform(src, T_est)
        
        # Calculate RMSE using nearest neighbor (more robust for noisy data)
        tree = cKDTree(dst_noisy)
        dists, _ = tree.query(aligned)
        rmse = np.sqrt(np.mean(dists**2))
        rmse_values.append(rmse)
    
    # Check that RMSE grows approximately linearly with noise
    correlation = np.corrcoef(noise_levels, rmse_values)[0, 1]
    assert correlation > 0.7, f"RMSE should correlate strongly with noise level, got r={correlation:.3f}"
    
    # Check that RMSE doesn't grow too fast (slope should be reasonable)
    slope = (rmse_values[-1] - rmse_values[0]) / (noise_levels[-1] - noise_levels[0])
    assert slope < 10, f"RMSE growth too steep: {slope:.2f}"
    
    # Basic sanity check: RMSE should be reasonable even at highest noise
    assert rmse_values[-1] < 2.0, f"RMSE too high at max noise: {rmse_values[-1]:.3f}" 
