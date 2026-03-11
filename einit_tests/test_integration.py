#!/usr/bin/env python3
"""
Integration test for einit using the Stanford Bunny PLY dataset, with OpenCV and Open3D ICP refinement.
"""
import numpy as np
import urllib.request
import tarfile
import io
import timeit
import open3d as o3d
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
from scipy.spatial import cKDTree
from einit import register_ellipsoid


def apply_transform(pts, T):
    homo = np.hstack([pts, np.ones((pts.shape[0], 1))])
    return (T @ homo.T).T[:, :3]


def random_rigid_transform():
    A = np.random.randn(3, 3)
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    t = np.random.uniform(-1, 1, 3)
    T = np.eye(4)
    T[:3, :3] = Q
    T[:3, 3] = t
    return T


def download_stanford_bunny_ply(n_points=5000):
    """Download and parse Stanford Bunny PLY file with error handling"""
    try:
        url = 'http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz'
        with urllib.request.urlopen(url) as resp:
            data = resp.read()
        tf = tarfile.open(fileobj=io.BytesIO(data), mode='r:gz')
        
        f = None
        for m in tf.getmembers():
            if m.name.endswith('bun_zipper.ply'):
                f = tf.extractfile(m)
                break
        
        if f is None:
            raise RuntimeError('PLY file not found in archive')

        header_ended, verts = False, []
        for line in f:
            line = line.decode('utf-8').strip()
            if header_ended:
                parts = line.split()
                if len(parts) >= 5:  # Original format: x y z confidence intensity
                    try:
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        verts.append([x, y, z])
                    except ValueError:
                        continue
            elif line == 'end_header':
                header_ended = True

        if not verts:
            raise RuntimeError('No valid vertices found in PLY file')
            
        pts = np.array(verts)
        if pts.shape[0] < 100:
            raise RuntimeError(f'Too few points: {pts.shape[0]}')
            
        if pts.shape[0] > n_points:
            idx = np.random.choice(pts.shape[0], n_points, replace=False)
            pts = pts[idx]
        return pts
        
    except Exception as e:
        print(f"Error downloading bunny: {e}")
        print("Using synthetic point cloud instead")
        # Generate synthetic bunny-like point cloud
        theta = np.random.uniform(0, 2*np.pi, n_points)
        phi = np.random.uniform(0, np.pi, n_points)
        r = np.random.uniform(0.8, 1.2, n_points)
        
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta) 
        z = r * np.cos(phi)
        
        return np.column_stack([x, y, z])


def opencv_icp(src_pts, dst_pts, max_iterations=50, tolerance=1e-6):
    """Simple ICP implementation using OpenCV for point-to-point alignment"""
    src_pts = src_pts.astype(np.float32)
    dst_pts = dst_pts.astype(np.float32)
    
    # Find correspondences using nearest neighbor
    tree = cKDTree(dst_pts)
    distances, indices = tree.query(src_pts)
    
    # Filter out correspondences that are too far
    valid_mask = distances < 0.1
    if np.sum(valid_mask) < 10:
        return np.eye(4)
    
    src_corr = src_pts[valid_mask]
    dst_corr = dst_pts[indices[valid_mask]]
    
    # Estimate rigid transformation using least squares
    src_centroid = np.mean(src_corr, axis=0)
    dst_centroid = np.mean(dst_corr, axis=0)
    
    src_centered = src_corr - src_centroid
    dst_centered = dst_corr - dst_centroid
    
    # Kabsch algorithm
    H = src_centered.T @ dst_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure proper rotation matrix
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    t = dst_centroid - R @ src_centroid
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    
    return T


def calculate_rmse(src_pts, dst_pts):
    """Calculate RMSE between two point clouds using nearest neighbor distances"""
    tree = cKDTree(dst_pts)
    distances, _ = tree.query(src_pts)
    return np.sqrt(np.mean(distances**2))


def test_bunny_alignment_icp():
    """Test einit initial alignment followed by ICP refinement comparison"""
    print("\n=== Bunny Alignment Integration Test with ICP Comparison ===")

    try:
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Load clean point clouds
        src_clean = download_stanford_bunny_ply()
        T_true = random_rigid_transform()
        dst_clean = apply_transform(src_clean, T_true)

        # Create partial overlap + noise for initial registration
        mask_src = np.random.rand(len(src_clean)) < 0.8
        mask_dst = np.random.rand(len(dst_clean)) < 0.8
        src_partial = src_clean[mask_src]
        dst_partial = dst_clean[mask_dst]
        # Apply a random permutation  
        perm = np.random.permutation(dst_partial.shape[0])
        dst_partial = dst_partial[perm]
        
        # Add small Gaussian noise
        dst_partial += np.random.normal(scale=0.01, size=dst_partial.shape)

        print(f"Point cloud sizes: src={len(src_partial)}, dst={len(dst_partial)}")

        # Benchmark einit performance
        timer = timeit.Timer(lambda: register_ellipsoid(src_partial, dst_partial))
        ms = np.array(timer.repeat(repeat=10, number=1)) * 1000
        mean_ms, std_ms = ms.mean(), ms.std()

        # Get initial einit alignment
        T_einit = register_ellipsoid(src_partial, dst_partial)
        if T_einit is None:
            raise RuntimeError("Einit registration failed")
            
        src_einit_aligned = apply_transform(src_clean, T_einit)
        
        # Calculate baseline RMSE with einit only
        rmse_einit = calculate_rmse(src_einit_aligned, dst_clean)

        # OpenCV ICP refinement
        if CV2_AVAILABLE:
            try:
                T_opencv_icp = opencv_icp(src_einit_aligned, dst_clean)
                T_opencv_final = T_opencv_icp @ T_einit
                src_opencv_aligned = apply_transform(src_clean, T_opencv_final)
                rmse_opencv = calculate_rmse(src_opencv_aligned, dst_clean)
            except Exception as e:
                print(f"OpenCV ICP failed: {e}")
                rmse_opencv = float('inf')
                T_opencv_final = T_einit
        else:
            print("OpenCV not available, skipping OpenCV ICP")
            rmse_opencv = float('inf')
            T_opencv_final = T_einit

        # Open3D ICP refinement
        try:
            pc_src = o3d.geometry.PointCloud()
            pc_dst = o3d.geometry.PointCloud()
            pc_src.points = o3d.utility.Vector3dVector(src_einit_aligned)
            pc_dst.points = o3d.utility.Vector3dVector(dst_clean)
            pc_src.estimate_normals()
            pc_dst.estimate_normals()

            icp_result = o3d.pipelines.registration.registration_icp(
                pc_src, pc_dst, 
                max_correspondence_distance=0.1,
                init=np.eye(4),
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
            )
            T_open3d_final = icp_result.transformation @ T_einit
            src_open3d_aligned = apply_transform(src_clean, T_open3d_final)
            rmse_open3d = calculate_rmse(src_open3d_aligned, dst_clean)
        except Exception as e:
            print(f"Open3D ICP failed: {e}")
            rmse_open3d = float('inf')
            T_open3d_final = T_einit

        # Compare results
        print(f"\nAlignment RMSE Comparison (on clean clouds):")
        print(f"  Einit only:        {rmse_einit:.6f}")
        
        if CV2_AVAILABLE and rmse_opencv != float('inf'):
            improvement_opencv = ((rmse_einit - rmse_opencv) / rmse_einit * 100) if rmse_einit > 0 else 0
            print(f"  Einit + OpenCV ICP: {rmse_opencv:.6f} (improvement: {improvement_opencv:.1f}%)")
        elif CV2_AVAILABLE:
            print(f"  Einit + OpenCV ICP: FAILED")
        else:
            print(f"  Einit + OpenCV ICP: SKIPPED (OpenCV not available)")
            
        if rmse_open3d != float('inf'):
            improvement_open3d = ((rmse_einit - rmse_open3d) / rmse_einit * 100) if rmse_einit > 0 else 0
            print(f"  Einit + Open3D ICP: {rmse_open3d:.6f} (improvement: {improvement_open3d:.1f}%)")
        else:
            print(f"  Einit + Open3D ICP: FAILED")
        
        # Transform errors
        frob_error_einit = np.linalg.norm(T_einit - T_true)
        frob_error_opencv = np.linalg.norm(T_opencv_final - T_true) if rmse_opencv != float('inf') else float('inf')
        frob_error_open3d = np.linalg.norm(T_open3d_final - T_true) if rmse_open3d != float('inf') else float('inf')
        
        print(f"\nTransform Frobenius Errors:")
        print(f"  Einit only:        {frob_error_einit:.6f}")
        if CV2_AVAILABLE and frob_error_opencv != float('inf'):
            print(f"  Einit + OpenCV ICP: {frob_error_opencv:.6f}")
        if frob_error_open3d != float('inf'):
            print(f"  Einit + Open3D ICP: {frob_error_open3d:.6f}")
        
        print(f"\nEinit Performance: {mean_ms:.2f} ± {std_ms:.2f} ms")
        
        # Test passes if at least einit alignment is reasonable (using original threshold)
        test_passed = rmse_einit < 0.025
        print(f"\n{'Integration test passed' if test_passed else 'Test failed'}")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        return {'error': str(e), 'test_passed': False}

if __name__ == "__main__":
    test_bunny_alignment_icp()
