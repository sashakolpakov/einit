#!/usr/bin/env python3
"""
Real partial overlap test and visualization for the bunny case.
This demonstrates EINIT working correctly on genuine partial overlap scenarios,
unlike artificial bbox crops.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# Add the parent directory to path to import einit
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tests'))

from einit import ellipsoid_init_icp
from test_einit import download_stanford_bunny, apply_transform, random_rigid_transform

def create_real_partial_overlap(src, overlap_fraction=0.8, noise_std=0.02, visualize=False):
    """
    Create a REAL partial overlap scenario like broken vs restored statue.
    
    Method:
    1. Apply random transformation to get dst
    2. Add noise to dst
    3. Randomly remove points from BOTH src and dst to simulate partial overlap
    4. Ensure the remaining points still correspond geometrically
    """
    
    # Apply transformation 
    T_true = random_rigid_transform()
    dst_full = apply_transform(src, T_true)
    
    # Add noise
    noise = np.random.normal(scale=noise_std, size=dst_full.shape)
    dst_noisy = dst_full + noise
    
    # Create partial overlap by randomly removing points
    n_points = len(src)
    n_overlap = int(n_points * overlap_fraction)
    
    # Randomly select DIFFERENT points for source and destination (breaks correspondence)
    src_overlap_indices = np.random.choice(n_points, n_overlap, replace=False)
    dst_overlap_indices = np.random.choice(n_points, n_overlap, replace=False)
    
    # Add some non-overlapping points to make it realistic
    n_src_only = int(n_points * 0.1)  # 10% points only in source
    n_dst_only = int(n_points * 0.1)  # 10% points only in destination
    
    # Source partial cloud: just the randomly selected points
    src_indices = src_overlap_indices
    src_partial = src[src_indices]
    
    # Destination partial cloud: different randomly selected points  
    dst_indices = dst_overlap_indices
    dst_partial = dst_noisy[dst_indices]
    
    if visualize:
        # Show full clouds with selected/discarded points marked
        fig = plt.figure(figsize=(12, 5))
        
        # Plot 1: Source cloud with selected points marked
        ax1 = fig.add_subplot(121, projection='3d')
        # Get discarded indices
        src_discarded_indices = np.setdiff1d(np.arange(len(src)), src_indices)
        # Show discarded points in gray
        ax1.scatter(src[src_discarded_indices, 0], src[src_discarded_indices, 1], src[src_discarded_indices, 2], 
                   c='gray', alpha=0.5, s=15, label=f'Discarded ({len(src_discarded_indices)} pts)')
        # Show selected points in red
        ax1.scatter(src[src_indices, 0], src[src_indices, 1], src[src_indices, 2], 
                   c='red', alpha=1.0, s=30, label=f'Selected ({len(src_indices)} pts)')
        ax1.set_title('Source: Selected vs Discarded')
        ax1.legend()
        
        # Plot 2: Destination cloud with selected points marked
        ax2 = fig.add_subplot(122, projection='3d')
        # Get discarded indices
        dst_discarded_indices = np.setdiff1d(np.arange(len(dst_noisy)), dst_indices)
        # Show discarded points in gray
        ax2.scatter(dst_noisy[dst_discarded_indices, 0], dst_noisy[dst_discarded_indices, 1], dst_noisy[dst_discarded_indices, 2], 
                   c='gray', alpha=0.5, s=15, label=f'Discarded ({len(dst_discarded_indices)} pts)')
        # Show selected points in blue
        ax2.scatter(dst_noisy[dst_indices, 0], dst_noisy[dst_indices, 1], dst_noisy[dst_indices, 2], 
                   c='blue', alpha=1.0, s=30, label=f'Selected ({len(dst_indices)} pts)')
        ax2.set_title('Destination: Selected vs Discarded')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    return src_partial, dst_partial, T_true, src_overlap_indices

def test_and_visualize_real_overlap():
    """Test EINIT on real partial overlap with detailed visualization"""
    print("REAL PARTIAL OVERLAP TEST WITH VISUALIZATION")
    print("=" * 60)
    
    # Load bunny
    src = download_stanford_bunny(n_points=1000)
    
    # Test different overlap levels
    overlap_levels = [0.9, 0.8, 0.7, 0.6]
    
    for overlap_frac in overlap_levels:
        print(f"\n{'='*60}")
        print(f"TESTING {overlap_frac*100:.0f}% OVERLAP")
        print(f"{'='*60}")
        
        # Set seed for reproducible results
        np.random.seed(42)
        
        # Create real partial overlap without region visualization
        src_partial, dst_partial, T_true, overlap_indices = create_real_partial_overlap(
            src, overlap_fraction=overlap_frac, noise_std=0.02, visualize=False)
        
        print(f"Created partial overlap:")
        print(f"  Full cloud: {len(src)} points")
        print(f"  True overlap: {len(overlap_indices)} points ({len(overlap_indices)/len(src)*100:.1f}%)")
        print(f"  Source partial: {len(src_partial)} points")
        print(f"  Dest partial: {len(dst_partial)} points")
        
        # Test EINIT
        print(f"\nRunning EINIT...")
        T_recovered = ellipsoid_init_icp(src_partial, dst_partial, positive_only=True)
        
        # Evaluate results
        src_full_aligned = apply_transform(src, T_recovered)
        dst_full_true = apply_transform(src, T_true)
        
        full_rmse = np.sqrt(np.mean(np.linalg.norm(src_full_aligned - dst_full_true, axis=1)**2))
        transform_error = np.linalg.norm(T_recovered - T_true, ord='fro')
        
        # Also evaluate on just the overlap region
        overlap_src = src[overlap_indices]
        overlap_aligned = apply_transform(overlap_src, T_recovered)
        overlap_true = apply_transform(overlap_src, T_true)
        overlap_rmse = np.sqrt(np.mean(np.linalg.norm(overlap_aligned - overlap_true, axis=1)**2))
        
        success = full_rmse < 0.05
        
        print(f"\nResults:")
        print(f"  Full cloud RMSE: {full_rmse:.6f}")
        print(f"  Transform error: {transform_error:.6f}")
        print(f"  Overlap RMSE: {overlap_rmse:.6f}")
        print(f"  Transform error: {transform_error:.6f}")
        print(f"  Result: {'✓ PASS' if success else '✗ FAIL'}")
        
        # Create results visualization
        fig = plt.figure(figsize=(20, 5))
        
        # Plot 1: EINIT alignment result
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(src_full_aligned[:, 0], src_full_aligned[:, 1], src_full_aligned[:, 2], 
                   c='green', alpha=0.6, s=20, label='EINIT Aligned')
        ax1.scatter(dst_full_true[:, 0], dst_full_true[:, 1], dst_full_true[:, 2], 
                   c='blue', alpha=0.6, s=20, label='True Target')
        ax1.set_title(f'EINIT Result (RMSE={full_rmse:.4f})')
        ax1.legend()
        
        # Plot 2: Error visualization
        ax2 = fig.add_subplot(132, projection='3d')
        errors = np.linalg.norm(src_full_aligned - dst_full_true, axis=1)
        scatter = ax2.scatter(src_full_aligned[:, 0], src_full_aligned[:, 1], src_full_aligned[:, 2], 
                            c=errors, cmap='hot', alpha=0.7, s=20)
        ax2.set_title('Point-wise Errors (Red=High)')
        plt.colorbar(scatter, ax=ax2, shrink=0.8)
        
        # Plot 3: Overlap region alignment
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.scatter(overlap_aligned[:, 0], overlap_aligned[:, 1], overlap_aligned[:, 2], 
                   c='green', alpha=0.8, s=30, label='Overlap Aligned')
        ax3.scatter(overlap_true[:, 0], overlap_true[:, 1], overlap_true[:, 2], 
                   c='blue', alpha=0.8, s=30, label='Overlap True')
        ax3.set_title(f'Overlap Region (RMSE={overlap_rmse:.4f})')
        ax3.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Wait for user input before next test
        if overlap_frac > 0.6:
            input("Press Enter to continue to next overlap level...")

if __name__ == "__main__":
    # Run the tests and visualizations
    test_and_visualize_real_overlap()