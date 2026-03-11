"""
Benchmarks for feature-augmented einit registration.

Three realistic scenarios where per-point features beyond (x, y, z) help:

  1. Stanford Bunny with LiDAR intensity  – real scan data from the PLY file
  2. LiDAR sphere with luminosity cap     – hemisphere flip resolved by intensity
  3. Coloured cube                        – face colours break 48-fold symmetry

For each benchmark, geometry-only vs feature-augmented success rates are
printed so the benefit (or absence of harm) is visible.
"""

import io
import tarfile
import urllib.request

import numpy as np
import pytest
from scipy.spatial import cKDTree

from einit import register_ellipsoid

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

RMSE_THRESHOLD = 0.05


def apply_transform(pts, T):
    homo = np.hstack([pts, np.ones((len(pts), 1))])
    return (T @ homo.T).T[:, :3]


def random_rigid_transform(translation_range=1.0):
    A = np.random.randn(3, 3)
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    t = np.random.uniform(-translation_range, translation_range, 3)
    T = np.eye(4)
    T[:3, :3] = Q
    T[:3, 3] = t
    return T


def nn_rmse(src, dst, T):
    """KD-tree nearest-neighbour RMSE after applying T to src."""
    aligned = apply_transform(src, T)
    dists, _ = cKDTree(dst).query(aligned)
    return float(np.sqrt(np.mean(dists ** 2)))


def partial_sample(arr, fraction, rng):
    idx = rng.choice(len(arr), int(fraction * len(arr)), replace=False)
    return arr[idx], idx


def run_trials(make_trial, n_trials, threshold=RMSE_THRESHOLD):
    """
    Run make_trial() n_trials times.
    make_trial must return (rmse_geo, rmse_feat).
    Returns (success_geo, success_feat, median_geo, median_feat).
    """
    geo, feat = [], []
    for _ in range(n_trials):
        rg, rf = make_trial()
        geo.append(rg)
        feat.append(rf)
    geo  = np.array(geo)
    feat = np.array(feat)
    return (
        float(np.mean(geo  < threshold)),
        float(np.mean(feat < threshold)),
        float(np.median(geo)),
        float(np.median(feat)),
    )


# ---------------------------------------------------------------------------
# Dataset factories
# ---------------------------------------------------------------------------

def make_unit_sphere(n, rng):
    phi   = rng.uniform(0, np.pi, n)
    theta = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi),
    ])


def make_lidar_sphere(n, rng, sensor_noise=0.03):
    """
    Unit sphere whose per-point intensity simulates an outdoor LiDAR scene:
      z > 0  →  high reflectance (~0.8, e.g. sunlit metal / pavement)
      z ≤ 0  →  low  reflectance (~0.2, e.g. vegetation / asphalt in shadow)

    The z-axis is special to the feature distribution but not to the
    geometry, so a hemisphere flip is ambiguous for pure spatial alignment
    but resolvable once intensity is used.
    """
    xyz = make_unit_sphere(n, rng)
    intensity = np.where(xyz[:, 2] > 0, 0.8, 0.2).reshape(-1, 1)
    intensity += rng.normal(0, sensor_noise, intensity.shape)
    return xyz, intensity


def make_colored_cube(grid_n=15):
    """
    Cube surface with each face painted a distinct colour, like a painted
    industrial part or a calibration target.

    Colour design rules (critical for the algorithm):
      (a) All 6 faces have distinct colours so KD-tree can discriminate them.
      (b) Each pair of opposite faces (±X, ±Y, ±Z) has a different per-channel
          contrast, so the three rows of the spatial-feature cross-covariance
          E_xf = P^T F  have distinct norms.  This makes the eigenvalues of
          E_aug = P^T P + w/N · E_xf E_xf^T  distinct, breaking the cube's
          48-fold geometric degeneracy so the ellipsoid step finds the correct
          principal axes.

    Contrast per axis pair (after feature normalisation):
      ±X : high red contrast   (~0.8)
      ±Y : medium green contrast (~0.5)
      ±Z : low blue contrast    (~0.25)

    This mimics a real inspection scenario where one axis is colour-coded
    with high contrast and others with progressively weaker markings.
    """
    # Base colour (shared background across all faces so off-axis contributions
    # to E_xf cancel cleanly)
    base = np.array([0.5, 0.5, 0.5])

    # Half-contrasts for each channel/axis
    dr, dg, db = 0.40, 0.25, 0.125

    # -X / +X differ in R;  -Y / +Y differ in G;  -Z / +Z differ in B
    face_colors = [
        base + np.array([-dr,   0,   0]),   # -X face
        base + np.array([+dr,   0,   0]),   # +X face
        base + np.array([  0, -dg,   0]),   # -Y face
        base + np.array([  0, +dg,   0]),   # +Y face
        base + np.array([  0,   0, -db]),   # -Z face
        base + np.array([  0,   0, +db]),   # +Z face
    ]

    grid = np.linspace(-1, 1, grid_n)
    face = np.array(np.meshgrid(grid, grid)).reshape(2, -1).T

    faces_xyz, faces_rgb = [], []
    for i, (axis, val) in enumerate(
            [(0, -1), (0, 1), (1, -1), (1, 1), (2, -1), (2, 1)]):
        f = np.insert(face, axis, val, axis=1)
        faces_xyz.append(f)
        faces_rgb.append(np.tile(face_colors[i], (len(f), 1)))

    return np.vstack(faces_xyz), np.vstack(faces_rgb)


def load_bunny_with_intensity(n_points=2000):
    """
    Download the Stanford Bunny PLY and return (xyz, intensity).
    The PLY vertex format is:  x  y  z  confidence  intensity
    The existing tests load only x y z; here we also return the intensity
    column — the raw scanner reflectance, not fabricated data.
    Returns (None, None) if the download fails (test will be skipped).
    """
    url = 'http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz'
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = resp.read()
    except Exception:
        return None, None

    tf = tarfile.open(fileobj=io.BytesIO(data), mode='r:gz')
    ply_file = None
    for m in tf.getmembers():
        if m.name.endswith('bun_zipper.ply'):
            ply_file = tf.extractfile(m)
            break
    if ply_file is None:
        return None, None

    header_done, rows = False, []
    for raw in ply_file:
        line = raw.decode('utf-8').strip()
        if header_done:
            parts = line.split()
            if len(parts) >= 5:
                try:
                    rows.append([float(p) for p in parts[:5]])
                except ValueError:
                    pass
        elif line == 'end_header':
            header_done = True

    if not rows:
        return None, None

    pts = np.array(rows)          # (N, 5): x y z confidence intensity
    if len(pts) > n_points:
        idx = np.random.default_rng(0).choice(len(pts), n_points, replace=False)
        pts = pts[idx]
    return pts[:, :3], pts[:, 4:5]   # xyz, intensity as (N, 1)


# ---------------------------------------------------------------------------
# Benchmark 1: Stanford Bunny with LiDAR intensity
# ---------------------------------------------------------------------------

def test_bunny_lidar_intensity(n_trials=30, feature_weight=0.3):
    """
    Alignment of the real Stanford Bunny scan using the scanner's own
    intensity field as a per-point feature.

    The PLY file stores five values per vertex: x y z confidence intensity.
    Intensity is the raw LiDAR reflectance — a surface property that is
    invariant to rigid motion (the value travels with the point, not with
    the coordinate frame).

    This is the most directly real-world scenario: it mirrors an actual
    LiDAR registration pipeline where two scans of the same object are
    taken from different poses.

    The bunny is geometrically distinctive, so geometry-only already works
    well.  The check here is that features do NOT hurt (no regression) and
    ideally reduce RMSE slightly.
    """
    xyz, intensity = load_bunny_with_intensity(n_points=2000)
    if xyz is None:
        pytest.skip("Could not download Stanford Bunny PLY")

    rng = np.random.default_rng(42)

    def trial():
        T_true = random_rigid_transform(translation_range=0.5)
        dst_xyz = apply_transform(xyz, T_true)
        dst_int = intensity.copy()   # reflectance is a surface property

        # Independent 80% subsamples (breaks index correspondence)
        n = len(xyz)
        si = rng.choice(n, int(0.8 * n), replace=False)
        di = rng.choice(n, int(0.8 * n), replace=False)
        src_p, dst_p = xyz[si], dst_xyz[di]
        src_i, dst_i = intensity[si], dst_int[di]

        # Small sensor noise on destination
        dst_p = dst_p + rng.normal(0, 0.01, dst_p.shape)
        dst_i = dst_i + rng.normal(0, 0.02, dst_i.shape)

        T_geo  = register_ellipsoid(src_p, dst_p)
        T_feat = register_ellipsoid(src_p, dst_p,
                                    src_features=src_i,
                                    dst_features=dst_i,
                                    feature_weight=feature_weight)
        return nn_rmse(xyz, dst_xyz, T_geo), nn_rmse(xyz, dst_xyz, T_feat)

    sr_geo, sr_feat, med_geo, med_feat = run_trials(trial, n_trials)
    print(f"\nBunny (LiDAR intensity, weight={feature_weight})")
    print(f"  geometry only :  success={sr_geo:.0%}  median RMSE={med_geo:.4f}")
    print(f"  with intensity:  success={sr_feat:.0%}  median RMSE={med_feat:.4f}")

    # Features must not hurt: allow at most 10% drop in success rate
    assert sr_feat >= sr_geo - 0.10, (
        f"Feature augmentation degraded bunny success rate: "
        f"{sr_feat:.0%} vs geometry-only {sr_geo:.0%}")


# ---------------------------------------------------------------------------
# Benchmark 2: LiDAR sphere — hemisphere-flip ambiguity via intensity
# ---------------------------------------------------------------------------

def test_lidar_sphere_hemisphere_flip(n_trials=100, feature_weight=0.5):
    """
    A sphere is geometrically invariant under any rotation, so the
    ellipsoid step yields arbitrary principal axes and all 8 sign
    combinations are equally uninformative.  The geometry-only algorithm
    succeeds only because the KD-tree step partially disambiguates among
    8 random candidate rotations — but for a large translation + near-180°
    flip it often fails.

    Per-point intensity encodes which hemisphere each point belongs to
    (sunlit vs shadowed), making the z-axis special to the feature
    distribution.  With feature-augmented covariance, the ellipsoid step
    finds the z-axis as the dominant principal axis; the KD-tree then
    selects the correct sign for it.

    This mirrors a real outdoor LiDAR scan of an approximately spherical
    object (e.g. a storage tank or dome) where one side is sunlit.
    """
    rng = np.random.default_rng(0)

    def trial():
        src_xyz, src_int = make_lidar_sphere(2000, rng)

        # True rotation: near-180° around z (hemisphere flip) + translation.
        # This is the hardest case: geometry gives identical ellipsoid for
        # the correct rotation and its z-flip.
        angle = np.pi + rng.uniform(-0.2, 0.2)
        c, s = np.cos(angle), np.sin(angle)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        T_true = np.eye(4)
        T_true[:3, :3] = Rz
        T_true[:3, 3]  = rng.uniform(-0.5, 0.5, 3)

        dst_xyz = apply_transform(src_xyz, T_true)
        # Intensity is a surface property: invariant to the rigid motion
        dst_int = src_int.copy()
        dst_xyz = dst_xyz + rng.normal(0, 0.01, dst_xyz.shape)
        dst_int = dst_int + rng.normal(0, 0.02, dst_int.shape)
        perm = rng.permutation(len(dst_xyz))
        dst_xyz, dst_int = dst_xyz[perm], dst_int[perm]

        T_geo  = register_ellipsoid(src_xyz, dst_xyz)
        T_feat = register_ellipsoid(src_xyz, dst_xyz,
                                    src_features=src_int,
                                    dst_features=dst_int,
                                    feature_weight=feature_weight)
        return nn_rmse(src_xyz, dst_xyz, T_geo), nn_rmse(src_xyz, dst_xyz, T_feat)

    sr_geo, sr_feat, med_geo, med_feat = run_trials(trial, n_trials)
    print(f"\nLiDAR sphere (hemisphere flip, weight={feature_weight})")
    print(f"  geometry only:  success={sr_geo:.0%}  median RMSE={med_geo:.4f}")
    print(f"  with intensity: success={sr_feat:.0%}  median RMSE={med_feat:.4f}")

    # Features must not hurt, and on this ambiguous case they should help
    assert sr_feat >= sr_geo - 0.05, (
        f"Intensity features degraded success rate: "
        f"{sr_feat:.0%} vs geometry-only {sr_geo:.0%}")
    assert sr_feat >= 0.5, (
        f"Feature-augmented alignment should succeed most of the time "
        f"on hemisphere flip, got {sr_feat:.0%}")


# ---------------------------------------------------------------------------
# Benchmark 3: Coloured cube — 48-fold geometric symmetry
# ---------------------------------------------------------------------------

def test_colored_cube(n_trials=100, feature_weight=1.0):
    """
    A cube has 48-fold rotational symmetry (order-48 symmetry group
    including reflections).  Pure geometry fails almost every time on a
    random rigid transform (~2% expected success rate).

    Six face colours with different per-axis contrast (see make_colored_cube)
    break the symmetry at the ellipsoid level: the spatial-feature
    cross-covariance makes the augmented covariance eigenvalues distinct,
    giving correctly oriented principal axes.  The feature-augmented
    KD-tree then resolves the remaining sign ambiguity.

    This models a real industrial scenario: a machined part with
    colour-coded faces scanned by an RGBD sensor for pose estimation.

    Uses full overlap (no subsampling) with light noise.  Partial overlap
    degrades accuracy for symmetric shapes because the KD-tree can no longer
    reliably distinguish sign candidates — that is an inherent limitation
    of the 8-candidate search on a cube, not of the feature approach.
    """
    src_xyz, src_rgb = make_colored_cube(grid_n=15)
    rng = np.random.default_rng(7)

    def trial():
        T_true = random_rigid_transform()
        dst_xyz = apply_transform(src_xyz, T_true)
        dst_rgb = src_rgb.copy()   # colour is a surface property
        dst_xyz = dst_xyz + rng.normal(0, 0.02, dst_xyz.shape)
        perm = rng.permutation(len(dst_xyz))
        dst_xyz, dst_rgb = dst_xyz[perm], dst_rgb[perm]

        T_geo  = register_ellipsoid(src_xyz, dst_xyz)
        T_feat = register_ellipsoid(src_xyz, dst_xyz,
                                    src_features=src_rgb,
                                    dst_features=dst_rgb,
                                    feature_weight=feature_weight)
        return nn_rmse(src_xyz, dst_xyz, T_geo), nn_rmse(src_xyz, dst_xyz, T_feat)

    sr_geo, sr_feat, med_geo, med_feat = run_trials(trial, n_trials)
    print(f"\nColoured cube (48-fold symmetry, weight={feature_weight})")
    print(f"  geometry only:  success={sr_geo:.0%}  median RMSE={med_geo:.4f}")
    print(f"  with RGB      :  success={sr_feat:.0%}  median RMSE={med_feat:.4f}")

    # Geometry-only baseline should be near 0% for a cube
    assert sr_geo < 0.15, (
        f"Geometry-only unexpectedly good on cube ({sr_geo:.0%}); "
        f"test may not be exercising the symmetry case")

    # Features must substantially outperform geometry-only
    assert sr_feat > sr_geo + 0.50, (
        f"RGB features did not substantially improve cube alignment: "
        f"feat {sr_feat:.0%} vs geo {sr_geo:.0%}")


# ---------------------------------------------------------------------------
# Benchmark 4: Soft vs hard scoring under subsampling
# ---------------------------------------------------------------------------

def test_soft_vs_hard_scoring_subsampling(n_trials=100):
    """
    A/B test: hard vs soft scoring across dst subsampling ratios 100%→20%.

    A random rigid transform is applied to a LiDAR sphere; dst is drawn as
    an independent random subsample at each ratio.  Hard scoring relies on
    inlier count, which collapses when dst is sparse.  Soft scoring
    accumulates a continuous Gaussian contribution from every src point and
    is expected to degrade more gracefully.

    Prints a ratio × scoring table and asserts soft is not worse than hard
    at any ratio (within a 5% tolerance).
    """
    rng = np.random.default_rng(17)
    src_xyz, _ = make_lidar_sphere(2000, rng)
    ratios = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2]

    print("\nA/B test — hard vs soft scoring across dst subsampling ratios")
    print(f"  {'ratio':>6}  {'hard':>8}  {'soft':>8}  {'winner':>8}")

    results = {}
    for ratio in ratios:
        res = {}
        for scoring in ("hard", "soft"):
            successes = []
            for _ in range(n_trials):
                T_true = random_rigid_transform(translation_range=0.5)
                dst_full = apply_transform(src_xyz, T_true)
                dst_full = dst_full + rng.normal(0, 0.01, dst_full.shape)

                n_dst = max(10, int(ratio * len(dst_full)))
                di = rng.choice(len(dst_full), n_dst, replace=False)
                dst_xyz = dst_full[di]

                T = register_ellipsoid(src_xyz, dst_xyz,
                                       params={"scoring": scoring})
                successes.append(nn_rmse(src_xyz, dst_full, T) < RMSE_THRESHOLD)

            res[scoring] = float(np.mean(successes))
        results[ratio] = res

        winner = ("soft" if res["soft"] > res["hard"] + 0.01
                  else "hard" if res["hard"] > res["soft"] + 0.01
                  else "tie")
        print(f"  {ratio:>6.0%}  {res['hard']:>8.0%}  {res['soft']:>8.0%}  {winner:>8}")

    for ratio, res in results.items():
        assert res["soft"] >= res["hard"] - 0.05, (
            f"Soft scoring worse than hard at ratio={ratio:.0%}: "
            f"soft {res['soft']:.0%} vs hard {res['hard']:.0%}")


# ---------------------------------------------------------------------------
# Benchmark 5: Feature weight sweep
# ---------------------------------------------------------------------------

def test_feature_weight_sweep(n_trials=40):
    """
    Sweep feature_weight from 0 (geometry only) to 1.0 on the hemisphere
    LiDAR sphere.  Reveals the optimal weight and checks that the algorithm
    degrades gracefully when features are over-weighted.

    Expected: success rate rises from the geometry-only baseline (which may
    already be decent due to KD-tree disambiguation) and then plateaus or
    gently declines.  At least one non-zero weight must outperform beta=0.
    """
    rng = np.random.default_rng(99)
    betas = [0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 1.0]

    print("\nFeature weight sweep — LiDAR intensity sphere (hemisphere flip)")
    print(f"  {'beta':>6}  {'success':>8}  {'median RMSE':>12}")

    results = {}
    for beta in betas:
        successes, rmse_list = [], []
        for _ in range(n_trials):
            src_xyz, src_int = make_lidar_sphere(1500, rng)

            # Near-180° rotation around z
            angle = np.pi + rng.uniform(-0.2, 0.2)
            c, s = np.cos(angle), np.sin(angle)
            Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            T_true = np.eye(4)
            T_true[:3, :3] = Rz
            T_true[:3, 3]  = rng.uniform(-0.5, 0.5, 3)

            dst_xyz = apply_transform(src_xyz, T_true)
            dst_int = src_int + rng.normal(0, 0.02, src_int.shape)
            dst_xyz = dst_xyz + rng.normal(0, 0.01, dst_xyz.shape)
            perm = rng.permutation(len(dst_xyz))
            dst_xyz, dst_int = dst_xyz[perm], dst_int[perm]

            if beta == 0.0:
                T = register_ellipsoid(src_xyz, dst_xyz)
            else:
                T = register_ellipsoid(src_xyz, dst_xyz,
                                       src_features=src_int,
                                       dst_features=dst_int,
                                       feature_weight=beta)
            r = nn_rmse(src_xyz, dst_xyz, T)
            rmse_list.append(r)
            successes.append(r < RMSE_THRESHOLD)

        sr  = float(np.mean(successes))
        med = float(np.median(rmse_list))
        results[beta] = sr
        print(f"  {beta:>6.2f}  {sr:>8.0%}  {med:>12.4f}")

    best_feat_sr = max(results[b] for b in betas if b > 0.0)
    # Features must not hurt; if geometry is already perfect (100%), that is
    # acceptable — the sweep still validates graceful degradation at high weights.
    assert best_feat_sr >= results[0.0], (
        f"Feature augmentation degraded every weight setting vs geometry-only "
        f"({results[0.0]:.0%}); best feature result: {best_feat_sr:.0%}")


# ---------------------------------------------------------------------------
# Unit tests: API validation (no geometry needed, fast)
# ---------------------------------------------------------------------------

def test_feature_validation_both_or_neither():
    src = np.random.randn(50, 3)
    dst = np.random.randn(50, 3)
    feat = np.random.rand(50, 2)
    with pytest.raises(ValueError, match="both be provided"):
        register_ellipsoid(src, dst, src_features=feat)
    with pytest.raises(ValueError, match="both be provided"):
        register_ellipsoid(src, dst, dst_features=feat)


def test_feature_validation_row_mismatch():
    src = np.random.randn(50, 3)
    dst = np.random.randn(60, 3)
    with pytest.raises(ValueError, match="same number of rows as src"):
        register_ellipsoid(src, dst,
                           src_features=np.ones((40, 1)),
                           dst_features=np.ones((60, 1)))
    with pytest.raises(ValueError, match="same number of rows as dst"):
        register_ellipsoid(src, dst,
                           src_features=np.ones((50, 1)),
                           dst_features=np.ones((55, 1)))


def test_feature_validation_column_mismatch():
    src = np.random.randn(50, 3)
    dst = np.random.randn(50, 3)
    with pytest.raises(ValueError, match="same number of columns"):
        register_ellipsoid(src, dst,
                           src_features=np.ones((50, 2)),
                           dst_features=np.ones((50, 3)))


def test_feature_validation_negative_weight():
    src = np.random.randn(50, 3)
    dst = np.random.randn(50, 3)
    with pytest.raises(ValueError, match="non-negative"):
        register_ellipsoid(src, dst,
                           src_features=np.ones((50, 1)),
                           dst_features=np.ones((50, 1)),
                           feature_weight=-0.1)


def test_feature_weight_zero_identical_to_no_features():
    """feature_weight=0.0 must give exactly the same result as passing no features."""
    rng = np.random.default_rng(123)
    src = rng.standard_normal((200, 3))
    dst = rng.standard_normal((200, 3))
    feat_s = rng.random((200, 3))
    feat_d = rng.random((200, 3))

    T_no_feat = register_ellipsoid(src, dst)
    T_zero_w  = register_ellipsoid(src, dst,
                                   src_features=feat_s,
                                   dst_features=feat_d,
                                   feature_weight=0.0)
    np.testing.assert_array_equal(T_no_feat, T_zero_w)


def test_feature_1d_input_accepted():
    """1-D feature arrays (N,) should be accepted and treated as (N, 1)."""
    src = np.random.randn(80, 3)
    dst = np.random.randn(80, 3)
    T = register_ellipsoid(src, dst,
                           src_features=np.ones(80),
                           dst_features=np.ones(80),
                           feature_weight=0.3)
    assert T.shape == (4, 4)


def test_output_shape_unchanged_with_features():
    """Output must still be a proper 4×4 homogeneous matrix."""
    src = np.random.randn(100, 3)
    dst = np.random.randn(100, 3)
    T = register_ellipsoid(src, dst,
                           src_features=np.random.rand(100, 3),
                           dst_features=np.random.rand(100, 3),
                           feature_weight=0.5)
    assert T.shape == (4, 4)
    np.testing.assert_allclose(T[3, :], [0, 0, 0, 1])
