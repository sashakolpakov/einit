#!/usr/bin/env python3
"""
Visualize failure cases from the bunny test with KD-tree einit, permutations, and ICP refinement
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
from scipy.spatial import cKDTree

# Add the parent directory to path to import einit
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from einit import ellipsoid_init_icp
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tests'))
from test_einit import download_stanford_bunny, apply_transform, random_rigid_transform

# Try to import OpenCV for ICP
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


def opencv_icp_refinement(src, dst, T_init, max_iterations=10):
    """OpenCV-based ICP refinement using estimateAffine3D."""
    if not OPENCV_AVAILABLE:
        return T_init, [np.inf]
    
    T_current = T_init.copy()
    rmse_history = []
    
    kdtree = cKDTree(dst)
    
    for iteration in range(max_iterations):
        # Apply current transformation
        src_transformed = apply_transform(src, T_current)
        
        # Find correspondences using KD-tree
        distances, indices = kdtree.query(src_transformed)
        
        # Filter by distance to get good correspondences
        max_dist = np.percentile(distances, 80)  # Use best 80% of matches
        valid_mask = distances <= max_dist
        
        if np.sum(valid_mask) < 10:
            break
            
        # Create correspondence arrays for OpenCV
        src_corr = src_transformed[valid_mask].astype(np.float32)
        dst_corr = dst[indices[valid_mask]].astype(np.float32)
        
        # Compute RMSE
        rmse = np.sqrt(np.mean(distances[valid_mask]**2))
        rmse_history.append(rmse)
        
        # Use OpenCV estimateAffine3D for refinement
        retval, affine_matrix, inliers = cv2.estimateAffine3D(
            src_corr, dst_corr,
            ransacThreshold=max_dist * 0.5,
            confidence=0.99
        )
        
        if not retval or affine_matrix is None:
            break
            
        # Build incremental transformation
        R_increment = affine_matrix[:3, :3]
        t_increment = affine_matrix[:3, 3]
        
        # Check if transformation is reasonable
        if np.linalg.norm(t_increment) > 1.0 or np.linalg.norm(R_increment - np.eye(3)) > 0.5:
            break
            
        T_increment = np.eye(4)
        T_increment[:3, :3] = R_increment
        T_increment[:3, 3] = t_increment
        
        # Apply incremental transformation
        T_current = T_increment @ T_current
        
        # Check convergence
        if iteration > 0 and abs(rmse_history[-1] - rmse_history[-2]) < 1e-6:
            break
    
    return T_current, rmse_history

def visualize_bunny_failures(noise_std=0.02, overlap_fraction=0.8, n_points=3000, show_worst=3):
    """Visualize the worst failure cases from bunny test with KD-tree einit, permutations, and OpenCV refinement"""
    print(f"Analyzing bunny test failures with KD-tree einit, permutations, and OpenCV refinement...")
    print(f"Parameters: {n_points} points, noise_std={noise_std}, overlap={overlap_fraction*100:.0f}%")
    print(f"")
    print(f"A test FAILS if :")
    print(f"  - Transform error > 0.08 (||T_recovered - T_true||_F)")
    print(f"  - OR Clean RMSE > 0.08 (alignment error on full point clouds)")
    
    # Load bunny
    src = download_stanford_bunny(n_points=n_points)
    
    failures = []
    
    # Run test cases and collect failures
    for i in range(100):
        np.random.seed(1000 + i)
        
        T_true = random_rigid_transform()
        dst = apply_transform(src, T_true)
        noise = np.random.normal(scale=noise_std, size=dst.shape)
        dst_noisy = dst + noise
        
        # Create SPATIAL partial overlap (not random sampling)
        # Method: Create two overlapping bounding boxes to simulate real partial overlap
        
        # Get source bounding box
        src_min, src_max = np.min(src, axis=0), np.max(src, axis=0)
        src_range = src_max - src_min
        
        # Create offset overlap regions 
        # Source region: slightly shifted bounding box
        offset = src_range * (1 - overlap_fraction) * 0.5  # Shift to create overlap_fraction overlap
        src_box_min = src_min + offset * np.random.uniform(-1, 1, 3)  # Random direction
        src_box_max = src_box_min + src_range * overlap_fraction / 0.8  # Ensure good coverage
        
        # Destination region: overlapping box
        dst_box_min = src_box_min - offset * 0.3  # Overlap with source box
        dst_box_max = dst_box_min + src_range * overlap_fraction / 0.8
        
        # Select points within each bounding box
        src_mask = np.all((src >= src_box_min) & (src <= src_box_max), axis=1)
        dst_mask = np.all((dst_noisy >= dst_box_min) & (dst_noisy <= dst_box_max), axis=1)
        
        src_o = src[src_mask]
        dst_o = dst_noisy[dst_mask]
        
        # Ensure we have enough points
        if len(src_o) < 50 or len(dst_o) < 50:
            # Fallback: use random sampling if spatial method fails
            n_src = max(50, int(len(src) * overlap_fraction))
            n_dst = max(50, int(len(dst_noisy) * overlap_fraction))
            src_indices = np.random.choice(len(src), n_src, replace=False)
            dst_indices = np.random.choice(len(dst_noisy), n_dst, replace=False)
            src_o = src[src_indices]
            dst_o = dst_noisy[dst_indices]
        
        # APPLY RANDOM PERMUTATION to destination to test robustness
        perm_idx = np.random.permutation(len(dst_o))
        dst_o_permuted = dst_o[perm_idx]
        
        # Run einit with KD-tree (should handle permutations)
        T_recovered = ellipsoid_init_icp(src_o, dst_o_permuted)
        
        # Compute errors - use full clouds for clean evaluation
        transform_error = np.linalg.norm(T_recovered - T_true, ord='fro')
        
        # Clean RMSE: alignment quality on full clouds (this is what matters)
        src_full_aligned = apply_transform(src, T_recovered)
        dst_full_true = apply_transform(src, T_true)
        clean_rmse = np.sqrt(np.mean(np.linalg.norm(src_full_aligned - dst_full_true, axis=1)**2))
        
        # For reference, also compute overlap region RMSE 
        src_o_aligned = apply_transform(src_o, T_recovered)
        dst_o_true = apply_transform(src_o, T_true)
        kdtree_true = cKDTree(dst_o_true)
        distances, _ = kdtree_true.query(src_o_aligned)
        overlap_rmse = np.sqrt(np.mean(distances**2))
        
        # Store failure cases with OpenCV ICP refinement (use clean RMSE for failure detection)
        if transform_error > 0.08 or clean_rmse > 0.08:
            # Run OpenCV ICP refinement on the partial overlap data (with permutation)
            T_icp, rmse_history = opencv_icp_refinement(src_o, dst_o_permuted, T_recovered)
            
            # Compute post-OpenCV errors (clean RMSE on full clouds)
            src_full_icp_aligned = apply_transform(src, T_icp)
            clean_rmse_icp = np.sqrt(np.mean(np.linalg.norm(src_full_icp_aligned - dst_full_true, axis=1)**2))
            transform_error_icp = np.linalg.norm(T_icp - T_true, ord='fro')
            
            # For reference, also compute overlap region RMSE after ICP
            src_o_icp_aligned = apply_transform(src_o, T_icp)
            distances_icp, _ = kdtree_true.query(src_o_icp_aligned)
            overlap_rmse_icp = np.sqrt(np.mean(distances_icp**2))
            
            failures.append({
                'seed': 1000 + i,
                'transform_error': transform_error,
                'overlap_rmse': overlap_rmse,
                'clean_rmse': clean_rmse,
                'transform_error_icp': transform_error_icp,
                'overlap_rmse_icp': overlap_rmse_icp,
                'clean_rmse_icp': clean_rmse_icp,
                'T_true': T_true,
                'T_recovered': T_recovered,
                'T_icp': T_icp,
                'rmse_history': rmse_history,
                'src_o': src_o,
                'dst_o': dst_o_permuted,
                'src_full': src,
                'dst_full_true': dst_full_true,
                'src_o_aligned': src_o_aligned,
                'src_full_aligned': src_full_aligned,
                'src_o_icp_aligned': src_o_icp_aligned,
                'src_full_icp_aligned': src_full_icp_aligned,
                'permutation_applied': True
            })
    
    print(f"Found {len(failures)} failure cases out of 100 runs")
    
    if len(failures) == 0:
        print("No failures found!")
        return
    
    # Sort by worst error and show top failures
    failures.sort(key=lambda x: x['transform_error'], reverse=True)
    
    n_show = min(show_worst, len(failures))
    print(f"\nShowing {n_show} worst failures:")
    
    for i, failure in enumerate(failures[:n_show]):
        print(f"\nFailure {i+1}:")
        print(f"  Seed: {failure['seed']}")
        
        # Determine reason for failure
        transform_fails = failure['transform_error'] > 0.08
        clean_fails = failure['clean_rmse'] > 0.08
        
        if transform_fails and clean_fails:
            print(f"  FAILURE REASON: Both transform error AND clean RMSE exceed limits")
        elif transform_fails:
            print(f"  FAILURE REASON: Transform error exceeds limit")
        elif clean_fails:
            print(f"  FAILURE REASON: Clean RMSE exceeds limit")
        else:
            print(f"  ERROR: Should not be classified as failure")
        
        print(f"  Permutation applied: {failure['permutation_applied']}")
        print(f"  EINIT RESULTS:")
        print(f"    Transform error: {failure['transform_error']:.4f}")
        print(f"    Clean RMSE: {failure['clean_rmse']:.4f} (full cloud alignment)")
        print(f"    Overlap RMSE: {failure['overlap_rmse']:.4f} (partial region)")
        print(f"  OPENCV REFINEMENT RESULTS:")
        print(f"    Transform error: {failure['transform_error_icp']:.4f}")
        print(f"    Clean RMSE: {failure['clean_rmse_icp']:.4f} (full cloud alignment)")
        print(f"    Overlap RMSE: {failure['overlap_rmse_icp']:.4f} (partial region)")
        print(f"    OpenCV iterations: {len(failure['rmse_history'])}")
        
        # Calculate improvements
        transform_improvement = failure['transform_error'] / max(failure['transform_error_icp'], 1e-8)
        clean_improvement = failure['clean_rmse'] / max(failure['clean_rmse_icp'], 1e-8)
        overlap_improvement = failure['overlap_rmse'] / max(failure['overlap_rmse_icp'], 1e-8)
        
        print(f"  IMPROVEMENT:")
        print(f"    Transform error: {transform_improvement:.1f}x")
        print(f"    Clean RMSE: {clean_improvement:.1f}x")
        print(f"    Overlap RMSE: {overlap_improvement:.1f}x")
        
        # Check if OpenCV fixed the failure
        opencv_transform_ok = failure['transform_error_icp'] <= 0.08
        opencv_clean_ok = failure['clean_rmse_icp'] <= 0.08
        
        if opencv_transform_ok and opencv_clean_ok:
            print(f"    Status: OPENCV FIXED THE FAILURE")
        elif opencv_transform_ok or opencv_clean_ok:
            print(f"    Status: OpenCV partially improved")
        else:
            print(f"    Status: OpenCV did not improve")
        
        # Create visualization with 4 plots
        fig = plt.figure(figsize=(16, 12))
        
        # Plot 1: EINIT alignment
        ax1 = fig.add_subplot(221, projection='3d')
        ax1.scatter(failure['src_full_aligned'][:, 0], failure['src_full_aligned'][:, 1], failure['src_full_aligned'][:, 2], 
                   c='red', alpha=0.5, s=6, label='EINIT Aligned')
        ax1.scatter(failure['dst_full_true'][:, 0], failure['dst_full_true'][:, 1], failure['dst_full_true'][:, 2], 
                   c='blue', alpha=0.5, s=6, label='True Target')
        ax1.set_title(f'EINIT: RMSE={failure["clean_rmse"]:.4f}')
        ax1.legend()
        
        # Plot 2: OpenCV alignment
        ax2 = fig.add_subplot(222, projection='3d')
        ax2.scatter(failure['src_full_icp_aligned'][:, 0], failure['src_full_icp_aligned'][:, 1], failure['src_full_icp_aligned'][:, 2], 
                   c='green', alpha=0.5, s=6, label='OpenCV Aligned')
        ax2.scatter(failure['dst_full_true'][:, 0], failure['dst_full_true'][:, 1], failure['dst_full_true'][:, 2], 
                   c='blue', alpha=0.5, s=6, label='True Target')
        ax2.set_title(f'OpenCV: RMSE={failure["clean_rmse_icp"]:.4f}')
        ax2.legend()
        
        # Plot 3: Error distribution comparison
        ax3 = fig.add_subplot(223)
        errors_einit = np.linalg.norm(failure['src_full_aligned'] - failure['dst_full_true'], axis=1)
        errors_icp = np.linalg.norm(failure['src_full_icp_aligned'] - failure['dst_full_true'], axis=1)
        
        ax3.hist(errors_einit, bins=30, alpha=0.5, color='red', label=f'EINIT')
        ax3.hist(errors_icp, bins=30, alpha=0.5, color='green', label=f'OpenCV')
        ax3.set_xlabel('Point-wise Error')
        ax3.set_ylabel('Count')
        ax3.set_title('Error Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: OpenCV improvement convergence
        ax4 = fig.add_subplot(224)
        if failure['rmse_history']:
            ax4.plot(failure['rmse_history'], 'g-', marker='o', markersize=4)
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('RMSE')
            ax4.set_title(f'OpenCV Convergence ({len(failure["rmse_history"])} iter)')
            ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'Failure Case {i+1} (Seed {failure["seed"]})', fontsize=12)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Make window properly resizable and larger
        mng = plt.get_current_fig_manager()
        try:
            # Set window to be resizable and properly sized
            if hasattr(mng, 'window'):
                if hasattr(mng.window, 'wm_geometry'):
                    mng.window.wm_geometry("2000x1200+50+50")
                    mng.window.resizable(True, True)
                elif hasattr(mng, 'resize'):
                    mng.resize(2000, 1200)
            # For different backends
            elif hasattr(mng, 'resize'):
                mng.resize(2000, 1200)
        except Exception as e:
            print(f"Could not resize window: {e}")
        
        plt.show()
        
        # Print key transformation info
        print(f"  True transformation (det={np.linalg.det(failure['T_true'][:3,:3]):.3f}):")
        print(f"    {failure['T_true']}")
        print(f"  EINIT transformation (det={np.linalg.det(failure['T_recovered'][:3,:3]):.3f}):")
        print(f"    {failure['T_recovered']}")
        print(f"  OpenCV transformation (det={np.linalg.det(failure['T_icp'][:3,:3]):.3f}):")
        print(f"    {failure['T_icp']}")

    # Summary statistics
    print(f"\n" + "="*80)
    print(f"SUMMARY: EINIT WITH PERMUTATIONS + OPENCV REFINEMENT")
    print(f"="*80)
    
    fixed_count = 0
    partial_count = 0
    not_fixed_count = 0
    transform_improvements = []
    rmse_improvements = []
    
    for failure in failures:
        opencv_transform_ok = failure['transform_error_icp'] <= 0.08
        opencv_clean_ok = failure['clean_rmse_icp'] <= 0.08
        
        if opencv_transform_ok and opencv_clean_ok:
            fixed_count += 1
        elif opencv_transform_ok or opencv_clean_ok:
            partial_count += 1
        else:
            not_fixed_count += 1
            
        transform_improvements.append(failure['transform_error'] / max(failure['transform_error_icp'], 1e-8))
        rmse_improvements.append(failure['clean_rmse'] / max(failure['clean_rmse_icp'], 1e-8))
    
    print(f"Total failures analyzed: {len(failures)}")
    print(f"Completely fixed by OpenCV: {fixed_count} ({fixed_count/len(failures)*100:.1f}%)")
    print(f"Partially fixed by OpenCV: {partial_count} ({partial_count/len(failures)*100:.1f}%)")
    print(f"Not fixed by OpenCV: {not_fixed_count} ({not_fixed_count/len(failures)*100:.1f}%)")
    print(f"")
    print(f"Average transform error improvement: {np.mean(transform_improvements):.1f}x")
    print(f"Average clean RMSE improvement: {np.mean(rmse_improvements):.1f}x")
    print(f"")
    print(f"Note: All tests used random permutations of destination points")
    print(f"      to verify KD-tree robustness in einit")


if __name__ == "__main__":
    visualize_bunny_failures()