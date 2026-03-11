#!/usr/bin/env python3
"""
Diagnostic visualization: why does the coloured-cube alignment fail,
and where exactly do features help or not?

Produces a 4-panel figure:
  1. Source cube (coloured faces) + destination cube (grey) — the problem
  2. Eigenvalue spectra: geometry-only vs feature-augmented covariance
  3. The 8 candidate rotations — spatial RMSE + feature distance for each
  4. Best alignment: geometry-only vs feature-augmented — overlaid

Run:  python einit_examples/cube_feature_matching.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D      # noqa: F401
from scipy.spatial import cKDTree
from einit import register_ellipsoid

# --------------------------------------------------------------------------
# Cube factory (same as test_features.py)
# --------------------------------------------------------------------------
def make_colored_cube(grid_n=15):
    base = np.array([0.5, 0.5, 0.5])
    dr, dg, db = 0.40, 0.25, 0.125
    face_colors = [
        base + np.array([-dr,   0,   0]),  # -X
        base + np.array([+dr,   0,   0]),  # +X
        base + np.array([  0, -dg,   0]),  # -Y
        base + np.array([  0, +dg,   0]),  # +Y
        base + np.array([  0,   0, -db]),  # -Z
        base + np.array([  0,   0, +db]),  # +Z
    ]
    face_labels = ['-X', '+X', '-Y', '+Y', '-Z', '+Z']
    grid = np.linspace(-1, 1, grid_n)
    face = np.array(np.meshgrid(grid, grid)).reshape(2, -1).T
    faces_xyz, faces_rgb, labels = [], [], []
    for i, (axis, val) in enumerate(
            [(0,-1),(0,1),(1,-1),(1,1),(2,-1),(2,1)]):
        f = np.insert(face, axis, val, axis=1)
        faces_xyz.append(f)
        faces_rgb.append(np.tile(face_colors[i], (len(f), 1)))
        labels.extend([face_labels[i]] * len(f))
    return np.vstack(faces_xyz), np.vstack(faces_rgb), labels

def apply_transform(pts, T):
    homo = np.hstack([pts, np.ones((len(pts), 1))])
    return (T @ homo.T).T[:, :3]

def random_rigid_transform():
    A = np.random.randn(3, 3)
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    t = np.random.uniform(-1, 1, 3)
    T = np.eye(4); T[:3,:3] = Q; T[:3,3] = t
    return T

# --------------------------------------------------------------------------
# Re-implement the inner loop to extract per-candidate diagnostics
# --------------------------------------------------------------------------
def diagnose_alignment(src_pts, dst_pts, src_feat=None, dst_feat=None,
                       feature_weight=0.0):
    """Return eigenvalues, per-candidate stats, and the best transform."""
    P = src_pts - src_pts.mean(0)
    Q = dst_pts - dst_pts.mean(0)

    use_feat = src_feat is not None and feature_weight > 0

    if use_feat:
        fstd = dst_feat.std(0)
        fstd[fstd < 1e-10] = 1.0
        gs = fstd.max()
        Fc_src = src_feat / gs
        Fc_dst = dst_feat / gs
        Exf_s = P.T @ Fc_src
        Exf_d = Q.T @ Fc_dst
        Exx_s = P.T @ P;  Eff_s = Exf_s @ Exf_s.T
        Exx_d = Q.T @ Q;  Eff_d = Exf_d @ Exf_d.T
        tr_s = np.trace(Exx_s) / max(np.trace(Eff_s), 1e-12)
        tr_d = np.trace(Exx_d) / max(np.trace(Eff_d), 1e-12)
        Ep = Exx_s + feature_weight * tr_s * Eff_s
        Eq = Exx_d + feature_weight * tr_d * Eff_d
        # KD-tree features
        sf_kdt = feature_weight * src_feat / fstd
        df_kdt = feature_weight * dst_feat / fstd
    else:
        Ep = P.T @ P
        Eq = Q.T @ Q

    Lp, Up = np.linalg.eigh(Ep)
    Lq, Uq = np.linalg.eigh(Eq)

    # Build KD-trees
    if use_feat:
        kdt_aug = cKDTree(np.hstack([Q, df_kdt]))
    kdt_spatial = cKDTree(Q)

    # Auto threshold
    ss = min(1000, len(Q))
    si = np.random.choice(len(Q), ss, replace=False)
    tt = cKDTree(Q[si])
    sd, _ = tt.query(Q[si], k=2)
    max_dist = 3.0 * np.median(sd[:, 1])

    sign_combos = [[1,1,1],[-1,1,1],[1,-1,1],[1,1,-1],
                   [-1,-1,1],[-1,1,-1],[1,-1,-1],[-1,-1,-1]]

    candidates = []
    for signs in sign_combos:
        D = np.diag(signs)
        U = Uq @ D @ Up.T
        Pt = P @ U.T

        # Spatial nearest-neighbour RMSE
        sp_dist, sp_idx = kdt_spatial.query(Pt)
        sp_valid = sp_dist <= max_dist
        sp_inliers = sp_valid.sum()
        sp_rmse = float(np.sqrt(np.mean(sp_dist[sp_valid]**2))) if sp_inliers > 0 else 999.

        # Feature-augmented nearest-neighbour
        if use_feat:
            _, aug_idx = kdt_aug.query(np.hstack([Pt, sf_kdt]))
            aug_sp_dist = np.linalg.norm(Pt - Q[aug_idx], axis=1)
            aug_valid = aug_sp_dist <= max_dist
            aug_inliers = aug_valid.sum()
            aug_rmse = float(np.sqrt(np.mean(aug_sp_dist[aug_valid]**2))) if aug_inliers > 0 else 999.
            # Feature mismatch for those correspondences
            if aug_inliers > 0:
                feat_dist = float(np.mean(np.linalg.norm(
                    src_feat[aug_valid] - dst_feat[aug_idx[aug_valid]], axis=1)))
            else:
                feat_dist = 999.
        else:
            aug_rmse = sp_rmse
            aug_inliers = sp_inliers
            feat_dist = 0.

        candidates.append(dict(
            signs=signs, U=U,
            sp_rmse=sp_rmse, sp_inliers=sp_inliers,
            aug_rmse=aug_rmse, aug_inliers=aug_inliers,
            feat_dist=feat_dist,
        ))

    return dict(
        Lp=Lp, Lq=Lq, Up=Up, Uq=Uq,
        candidates=candidates,
        max_dist=max_dist,
        centroid_src=src_pts.mean(0),
        centroid_dst=dst_pts.mean(0),
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    np.random.seed(42)
    src_xyz, src_rgb, labels = make_colored_cube(grid_n=15)
    T_true = random_rigid_transform()
    dst_xyz = apply_transform(src_xyz, T_true)
    dst_rgb = src_rgb.copy()
    dst_xyz += np.random.normal(0, 0.02, dst_xyz.shape)

    # Partial overlap
    rng = np.random.default_rng(7)
    n = len(src_xyz)
    si = rng.choice(n, int(0.8*n), replace=False)
    di = rng.choice(n, int(0.8*n), replace=False)
    src_p, dst_p = src_xyz[si], dst_xyz[di]
    src_c, dst_c = src_rgb[si], dst_rgb[di]

    fw = 1.0
    diag_geo  = diagnose_alignment(src_xyz, dst_xyz, feature_weight=0.0)
    diag_feat = diagnose_alignment(src_xyz, dst_xyz, src_rgb, dst_rgb, feature_weight=fw)

    # Also get the actual transforms (full cloud, no partial overlap)
    T_geo  = register_ellipsoid(src_xyz, dst_xyz)
    T_feat = register_ellipsoid(src_xyz, dst_xyz, src_features=src_rgb,
                                dst_features=dst_rgb, feature_weight=fw)

    # ======================================================================
    # FIGURE 1 — The problem: source (coloured) vs destination (grey)
    # ======================================================================
    fig1 = plt.figure(figsize=(14, 5))
    fig1.suptitle('Coloured Cube Alignment — The Problem', fontsize=14)

    ax1a = fig1.add_subplot(131, projection='3d')
    ax1a.scatter(*src_xyz.T, c=src_rgb, s=4, alpha=0.7)
    ax1a.set_title('Source (coloured faces)')

    ax1b = fig1.add_subplot(132, projection='3d')
    ax1b.scatter(*dst_xyz.T, c='0.6', s=4, alpha=0.5)
    ax1b.set_title('Destination (rotated)')

    ax1c = fig1.add_subplot(133, projection='3d')
    ax1c.scatter(*dst_xyz.T, c=dst_rgb, s=4, alpha=0.5, label='dst (true colour)')
    ax1c.scatter(*src_xyz.T, c=src_rgb, s=4, alpha=0.5, label='src (unaligned)')
    ax1c.set_title('Overlay before alignment')
    ax1c.legend(fontsize=8)

    fig1.tight_layout()
    fig1.savefig('einit_examples/cube_diag_1_problem.png', dpi=150)
    print('Saved cube_diag_1_problem.png')

    # ======================================================================
    # FIGURE 2 — Eigenvalue spectra
    # ======================================================================
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(10, 4))
    fig2.suptitle('Eigenvalue Spectra: Geometry-Only vs Feature-Augmented', fontsize=13)

    x = np.arange(3)
    w = 0.35
    ax2a.bar(x - w/2, diag_geo['Lp'],  w, label='Geometry only', color='steelblue')
    ax2a.bar(x + w/2, diag_feat['Lp'], w, label=f'+ Features (w={fw})', color='coral')
    ax2a.set_xticks(x); ax2a.set_xticklabels(['axis 0','axis 1','axis 2'])
    ax2a.set_title('Source covariance eigenvalues')
    ax2a.legend(fontsize=9)
    ax2a.set_ylabel('eigenvalue')

    # Show the ratio of largest/smallest eigenvalue
    ratio_geo  = diag_geo['Lp'][-1]  / max(diag_geo['Lp'][0],  1e-12)
    ratio_feat = diag_feat['Lp'][-1] / max(diag_feat['Lp'][0], 1e-12)
    ax2b.bar(['Geo only', f'+ Features'], [ratio_geo, ratio_feat],
             color=['steelblue', 'coral'])
    ax2b.set_title('Eigenvalue ratio  λ_max / λ_min')
    ax2b.set_ylabel('ratio')
    ax2b.axhline(1.0, color='grey', ls='--', lw=0.8, label='isotropic (degenerate)')
    ax2b.legend(fontsize=9)

    fig2.tight_layout()
    fig2.savefig('einit_examples/cube_diag_2_eigenvalues.png', dpi=150)
    print('Saved cube_diag_2_eigenvalues.png')

    # ======================================================================
    # FIGURE 3 — Per-candidate diagnostics (8 sign combos)
    # ======================================================================
    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
    fig3.suptitle('8 Sign-Combo Candidates: Geometry-Only vs Feature-Augmented', fontsize=13)

    idx = np.arange(8)
    sign_labels = [''.join('+' if s==1 else '-' for s in c['signs'])
                   for c in diag_geo['candidates']]

    # Spatial RMSE per candidate
    ax = axes3[0]
    geo_rmses = [c['sp_rmse'] for c in diag_geo['candidates']]
    feat_rmses = [c['aug_rmse'] for c in diag_feat['candidates']]
    ax.bar(idx - 0.2, geo_rmses,  0.35, label='Geo only (spatial KD)', color='steelblue')
    ax.bar(idx + 0.2, feat_rmses, 0.35, label='+ Features (augmented KD)', color='coral')
    ax.set_xticks(idx); ax.set_xticklabels(sign_labels, fontsize=7, rotation=45)
    ax.set_ylabel('Spatial RMSE (inliers)')
    ax.set_title('RMSE per candidate')
    ax.legend(fontsize=8)
    ax.set_ylim(0, min(max(geo_rmses + feat_rmses) * 1.3, 2.0))

    # Inlier count per candidate
    ax = axes3[1]
    geo_inl  = [c['sp_inliers']  for c in diag_geo['candidates']]
    feat_inl = [c['aug_inliers'] for c in diag_feat['candidates']]
    ax.bar(idx - 0.2, geo_inl,  0.35, label='Geo only', color='steelblue')
    ax.bar(idx + 0.2, feat_inl, 0.35, label='+ Features', color='coral')
    ax.set_xticks(idx); ax.set_xticklabels(sign_labels, fontsize=7, rotation=45)
    ax.set_ylabel('Inlier count')
    ax.set_title('Inliers per candidate')
    ax.legend(fontsize=8)

    # Feature distance per candidate (only meaningful for feature-augmented)
    ax = axes3[2]
    feat_dists = [c['feat_dist'] for c in diag_feat['candidates']]
    bars = ax.bar(idx, feat_dists, 0.5, color='coral')
    ax.set_xticks(idx); ax.set_xticklabels(sign_labels, fontsize=7, rotation=45)
    ax.set_ylabel('Mean feature distance')
    ax.set_title('Colour mismatch per candidate\n(lower = better face-colour match)')
    # Highlight the best one
    best_i = int(np.argmin(feat_dists))
    bars[best_i].set_edgecolor('black'); bars[best_i].set_linewidth(2)

    fig3.tight_layout()
    fig3.savefig('einit_examples/cube_diag_3_candidates.png', dpi=150)
    print('Saved cube_diag_3_candidates.png')

    # ======================================================================
    # FIGURE 4 — Alignment results
    # ======================================================================
    fig4 = plt.figure(figsize=(14, 5))
    fig4.suptitle('Alignment Results', fontsize=14)

    aligned_geo  = apply_transform(src_xyz, T_geo)
    aligned_feat = apply_transform(src_xyz, T_feat)

    # RMSE
    tree = cKDTree(dst_xyz)
    d_geo, _  = tree.query(aligned_geo)
    d_feat, _ = tree.query(aligned_feat)
    rmse_geo  = float(np.sqrt(np.mean(d_geo**2)))
    rmse_feat = float(np.sqrt(np.mean(d_feat**2)))

    ax4a = fig4.add_subplot(131, projection='3d')
    ax4a.scatter(*dst_xyz.T, c=dst_rgb, s=4, alpha=0.3, label='destination')
    ax4a.scatter(*aligned_geo.T, c='steelblue', s=4, alpha=0.5, label='geo aligned')
    ax4a.set_title(f'Geometry only\nRMSE = {rmse_geo:.4f}')
    ax4a.legend(fontsize=8)

    ax4b = fig4.add_subplot(132, projection='3d')
    ax4b.scatter(*dst_xyz.T, c=dst_rgb, s=4, alpha=0.3, label='destination')
    ax4b.scatter(*aligned_feat.T, c='coral', s=4, alpha=0.5, label='feat aligned')
    ax4b.set_title(f'Feature-augmented (w={fw})\nRMSE = {rmse_feat:.4f}')
    ax4b.legend(fontsize=8)

    # Ground truth overlay for reference
    aligned_true = apply_transform(src_xyz, T_true)
    ax4c = fig4.add_subplot(133, projection='3d')
    ax4c.scatter(*dst_xyz.T, c=dst_rgb, s=4, alpha=0.3, label='destination')
    ax4c.scatter(*aligned_true.T, c='green', s=4, alpha=0.5, label='ground truth')
    ax4c.set_title('Ground truth alignment')
    ax4c.legend(fontsize=8)

    fig4.tight_layout()
    fig4.savefig('einit_examples/cube_diag_4_results.png', dpi=150)
    print('Saved cube_diag_4_results.png')

    # ======================================================================
    # Print summary
    # ======================================================================
    print('\n' + '='*60)
    print('DIAGNOSTIC SUMMARY')
    print('='*60)
    print(f'Eigenvalue ratio (geo  only): {ratio_geo:.3f}  '
          f'(1.0 = isotropic = degenerate)')
    print(f'Eigenvalue ratio (+ features): {ratio_feat:.3f}')
    print(f'Eigenvalues (geo ): {diag_geo["Lp"]}')
    print(f'Eigenvalues (feat): {diag_feat["Lp"]}')
    print()
    print(f'max_correspondence_distance: {diag_geo["max_dist"]:.4f}')
    print()
    print('Per-candidate (geometry only):')
    for i, c in enumerate(diag_geo['candidates']):
        print(f'  {sign_labels[i]}  inliers={c["sp_inliers"]:4d}  '
              f'RMSE={c["sp_rmse"]:.4f}')
    print()
    print('Per-candidate (feature augmented):')
    for i, c in enumerate(diag_feat['candidates']):
        print(f'  {sign_labels[i]}  inliers={c["aug_inliers"]:4d}  '
              f'RMSE={c["aug_rmse"]:.4f}  feat_dist={c["feat_dist"]:.4f}')
    print()
    print(f'Final RMSE (geo  only): {rmse_geo:.4f}')
    print(f'Final RMSE (+ features): {rmse_feat:.4f}')
    print(f'Final RMSE (ground truth): 0.0000')

    plt.show()

if __name__ == '__main__':
    main()
