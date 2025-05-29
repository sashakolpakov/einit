#!/usr/bin/env python3
"""
Simple integration tests for einit with OpenCV ICP procedures
"""
import numpy as np
import pytest
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

from einit import ellipsoid_init_icp
try:
    from .test_einit import apply_transform, random_rigid_transform, download_stanford_bunny
except ImportError:
    from test_einit import apply_transform, random_rigid_transform, download_stanford_bunny


@pytest.mark.skipif(not OPENCV_AVAILABLE, reason="OpenCV not available")
def test_sphere_einit_quality():
    """Test that einit provides good initialization for sphere data."""
    print(f"\n=== SPHERE EINIT QUALITY TEST ===")
    
    errors = []
    for i in range(10):
        np.random.seed(4000 + i)
        
        # Generate sphere
        phi = np.random.uniform(0, np.pi, 500)
        theta = np.random.uniform(0, 2*np.pi, 500)
        src = np.vstack([np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)]).T * 5
        
        # Apply transformation and add noise
        T_true = random_rigid_transform()
        dst = apply_transform(src, T_true)
        noise = np.random.normal(scale=0.02, size=dst.shape)
        dst_noisy = dst + noise
        
        # Test with partial overlap
        mask = np.random.choice([True, False], size=(500,), p=[0.8, 0.2])
        src_partial = src[mask]
        dst_partial = dst_noisy[mask]
        
        # Run einit
        T_recovered = ellipsoid_init_icp(src_partial, dst_partial)
        src_aligned = apply_transform(src, T_recovered)
        dst_clean = apply_transform(src, T_true)
        error = np.sqrt(np.mean(np.linalg.norm(src_aligned - dst_clean, axis=1)**2))
        errors.append(error)
    
    mean_error = np.mean(errors)
    print(f"Einit RMSE: {mean_error:.6f} ± {np.std(errors):.6f}")
    
    # Einit should achieve excellent alignment
    assert mean_error < 0.03, f"Einit RMSE {mean_error:.6f} should be < 0.03"
    print("✓ Einit provides good sphere alignment (RMSE < 0.03)")


@pytest.mark.skipif(not OPENCV_AVAILABLE, reason="OpenCV not available") 
def test_cube_einit_quality():
    """Test that einit provides good initialization for cube data."""
    print(f"\n=== CUBE EINIT QUALITY TEST ===")
    
    errors = []
    for i in range(10):
        np.random.seed(5000 + i)
        
        # Generate cube
        grid = np.linspace(-1,1,12)
        X, Y, Z = np.meshgrid(grid, grid, grid)
        src = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T * np.array([2,1,0.5])
        
        # Apply transformation and add noise
        T_true = random_rigid_transform()
        dst = apply_transform(src, T_true)
        noise = np.random.normal(scale=0.02, size=dst.shape)
        dst_noisy = dst + noise
        
        # Test with partial overlap
        mask = np.random.choice([True, False], size=(src.shape[0],), p=[0.8, 0.2])
        src_partial = src[mask]
        dst_partial = dst_noisy[mask]
        
        # Run einit
        T_recovered = ellipsoid_init_icp(src_partial, dst_partial)
        src_aligned = apply_transform(src, T_recovered)
        dst_clean = apply_transform(src, T_true)
        error = np.sqrt(np.mean(np.linalg.norm(src_aligned - dst_clean, axis=1)**2))
        errors.append(error)
    
    mean_error = np.mean(errors)
    print(f"Einit RMSE: {mean_error:.6f} ± {np.std(errors):.6f}")
    
    # Einit should achieve excellent alignment for cubes
    assert mean_error < 0.05, f"Einit RMSE {mean_error:.6f} should be < 0.05"
    print("✓ Einit provides good cube alignment (RMSE < 0.05)")


@pytest.mark.skipif(not OPENCV_AVAILABLE, reason="OpenCV not available")
def test_bunny_einit_quality():
    """Test that einit provides good initialization for bunny data."""
    print(f"\n=== BUNNY EINIT QUALITY TEST ===")
    
    # Load bunny
    src = download_stanford_bunny(n_points=1000)
    
    errors = []
    for i in range(10):
        np.random.seed(6000 + i)
        
        # Apply transformation and add noise
        T_true = random_rigid_transform()
        dst = apply_transform(src, T_true)
        noise = np.random.normal(scale=0.02, size=dst.shape)
        dst_noisy = dst + noise
        
        # Test with partial overlap
        mask = np.random.choice([True, False], size=(src.shape[0],), p=[0.8, 0.2])
        src_partial = src[mask]
        dst_partial = dst_noisy[mask]
        
        # Run einit
        T_recovered = ellipsoid_init_icp(src_partial, dst_partial)
        src_aligned = apply_transform(src, T_recovered)
        dst_clean = apply_transform(src, T_true)
        error = np.sqrt(np.mean(np.linalg.norm(src_aligned - dst_clean, axis=1)**2))
        errors.append(error)
    
    mean_error = np.mean(errors)
    print(f"Einit RMSE: {mean_error:.6f} ± {np.std(errors):.6f}")
    
    # Einit should achieve good alignment for complex shapes like bunny
    assert mean_error < 0.08, f"Einit RMSE {mean_error:.6f} should be < 0.08"
    print("✓ Einit provides good bunny alignment (RMSE < 0.08)")


@pytest.mark.skipif(not OPENCV_AVAILABLE, reason="OpenCV not available")
def test_opencv_compatibility():
    """Test that einit transformations are compatible with OpenCV functions."""
    print(f"\n=== OPENCV COMPATIBILITY TEST ===")
    
    # Generate simple test data
    np.random.seed(12345)
    src = np.random.randn(100, 3) * 2
    T_true = random_rigid_transform()
    dst = apply_transform(src, T_true)
    
    # Get einit result
    T_einit = ellipsoid_init_icp(src, dst)
    
    # Test OpenCV compatibility
    src_f32 = src.astype(np.float32)
    dst_f32 = dst.astype(np.float32)
    
    # Apply einit transform
    src_transformed = apply_transform(src, T_einit).astype(np.float32)
    
    # Use OpenCV estimateAffine3D
    retval, affine_matrix, inliers = cv2.estimateAffine3D(
        src_transformed, dst_f32,
        ransacThreshold=0.1,
        confidence=0.95
    )
    
    print(f"OpenCV estimateAffine3D success: {retval}")
    print(f"Inlier ratio: {np.sum(inliers)/len(inliers) if inliers is not None else 'N/A'}")
    
    # Should work with OpenCV
    assert retval, "OpenCV estimateAffine3D should succeed with einit result"
    if inliers is not None:
        assert np.sum(inliers) > 0.5 * len(inliers), "Should have >50% inliers"
    
    print("✓ Einit transformations are compatible with OpenCV")


def test_comprehensive_integration():
    """Run all integration tests."""
    if not OPENCV_AVAILABLE:
        print("OpenCV not available - install with: pip install opencv-python-headless")
        return
        
    print("\n" + "="*60)
    print("COMPREHENSIVE EINIT INTEGRATION TESTS")
    print("="*60)
    
    test_sphere_einit_quality()
    test_cube_einit_quality() 
    test_bunny_einit_quality()
    test_opencv_compatibility()
    
    print("\n" + "="*60)
    print("ALL INTEGRATION TESTS PASSED")
    print("Key findings:")
    print("- Einit provides good initialization quality")
    print("- Fully compatible with OpenCV transformation format")
    print("="*60)


if __name__ == "__main__":
    test_comprehensive_integration()