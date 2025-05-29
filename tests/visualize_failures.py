#!/usr/bin/env python3
"""
Visualize failure cases from the bunny test
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# Add the parent directory to path to import einit
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from einit import ellipsoid_init_icp
from test_einit import download_stanford_bunny, apply_transform, random_rigid_transform

def visualize_bunny_failures(noise_std=0.02, overlap_fraction=0.8, n_points=3000, show_worst=5):
    """Visualize the worst failure cases from bunny test"""
    print(f"Analyzing bunny test failures...")
    print(f"Parameters: {n_points} points, noise_std={noise_std}, overlap={overlap_fraction*100:.0f}%")
    
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
        mask = np.random.choice([True, False], size=(src.shape[0],), p=[overlap_fraction, 1-overlap_fraction])
        src_o = src[mask]
        dst_o = dst_noisy[mask]

        min_points = min(len(src_o), len(dst_o))
        src_o = src_o[:min_points]
        dst_o = dst_o[:min_points]
        
        T_recovered = ellipsoid_init_icp(src_o, dst_o)
        
        # Compute errors
        transform_error = np.linalg.norm(T_recovered - T_true, ord='fro')
        
        src_clean_aligned = apply_transform(src, T_recovered)
        dst_clean = apply_transform(src, T_true)
        clean_rmse = np.sqrt(np.mean(np.linalg.norm(src_clean_aligned - dst_clean, axis=1)**2))
        
        # Store failure cases
        if transform_error > 0.08 or clean_rmse > 0.08:
            failures.append({
                'seed': 1000 + i,
                'transform_error': transform_error,
                'clean_rmse': clean_rmse,
                'T_true': T_true,
                'T_recovered': T_recovered,
                'src_o': src_o,
                'dst_o': dst_o,
                'src_full': src,
                'dst_clean': dst_clean,
                'src_aligned': src_clean_aligned
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
        print(f"  Transform error: {failure['transform_error']:.4f}")
        print(f"  Clean RMSE: {failure['clean_rmse']:.4f}")
        
        # Create visualization
        fig = plt.figure(figsize=(12, 6))
        
        # Plot 1: Algorithm result (full clouds)
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(failure['src_aligned'][:, 0], failure['src_aligned'][:, 1], failure['src_aligned'][:, 2], 
                   c='red', alpha=0.4, s=8, label='Aligned Source')
        ax1.scatter(failure['dst_clean'][:, 0], failure['dst_clean'][:, 1], failure['dst_clean'][:, 2], 
                   c='blue', alpha=0.4, s=8, label='True Target')
        ax1.set_title('Algorithm Result')
        ax1.legend()
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # Plot 2: Error distribution
        ax2 = fig.add_subplot(122)
        errors = np.linalg.norm(failure['src_aligned'] - failure['dst_clean'], axis=1)
        ax2.hist(errors, bins=50, alpha=0.7, color='orange')
        ax2.axvline(failure['clean_rmse'], color='red', linestyle='--', 
                   label=f'RMSE={failure["clean_rmse"]:.4f}')
        ax2.set_xlabel('Point-wise Error')
        ax2.set_ylabel('Count')
        ax2.set_title('Error Distribution')
        ax2.legend()
        
        plt.suptitle(f'Failure Case {i+1}: Transform Error = {failure["transform_error"]:.4f}', 
                    fontsize=16)
        plt.tight_layout()
        plt.show()
        
        # Print transformation matrices
        print(f"  True transformation:")
        print(f"    {failure['T_true']}")
        print(f"  Recovered transformation:")
        print(f"    {failure['T_recovered']}")

if __name__ == "__main__":
    visualize_bunny_failures()