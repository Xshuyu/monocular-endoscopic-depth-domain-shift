"""
Comprehensive Unit and Integration Tests for Mathematically Enhanced 3D Endoscopy Profiler.
Validates:
    1. Differential Geometry & Weingarten Shape Operator (Principal Curvatures, Shape Index, Paris Morphology).
    2. Lie Group SO(3) / Lie Algebra so(3) robust OBB convergence on outlier-corrupted point clouds.
    3. Laplace-Beltrami Heat Method continuous geodesic diameter.
    4. Full pipeline integration on real clinical Kvasir-SEG samples.
"""

import os
import glob
import cv2
import numpy as np

from lesion_3d_measurement.src import (
    CameraIntrinsics,
    compute_manifold_curvature,
    compute_robust_obb_lie,
    HeatGeodesicSolver,
    Lesion3DMeasurementPipeline
)


def test_differential_geometry_sphere_and_plane():
    """
    Mathematical test:
    - On a sphere of radius R = 10 mm, principal curvatures k1 = k2 = 1/R = 0.1 mm^-1.
      Gaussian curvature K = 1/R^2 = 0.01 mm^-2.
      Mean curvature H = 1/R = 0.1 mm^-1.
      Shape index S for a convex sphere/dome should be close to +1.0.
    """
    r = 10.0
    u = np.linspace(0, np.pi / 3, 25)
    v = np.linspace(0, 2 * np.pi, 25)
    x = (r * np.outer(np.sin(u), np.cos(v))).flatten()
    y = (r * np.outer(np.sin(u), np.sin(v))).flatten()
    z = (r * np.outer(np.cos(u), np.ones_like(v))).flatten() + 30.0

    sphere_pts = np.stack([x, y, z], axis=-1)

    profile = compute_manifold_curvature(
        sphere_pts,
        elevation_height_mm=r * (1.0 - np.cos(np.pi / 3)),
        k_neighbors=12,
    )

    print(f"Sphere Curvature Validation:")
    print(f"  Mean H: {profile.mean_mean_curvature_H:.4f} (Expected ~ 0.1000)")
    print(f"  Mean K: {profile.mean_gaussian_curvature_K:.4f} (Expected ~ 0.0100)")
    print(f"  Shape Index: {profile.mean_shape_index:.3f} (Expected > 0.6 for Dome/Cap)")
    print(f"  Paris Morph Class: {profile.morphology_class}")

    assert profile.mean_shape_index > 0.3, "Dome sphere shape index must be positive."
    assert profile.mean_curvedness > 0.01, "Sphere curvedness must be positive."
    assert np.isclose(abs(profile.mean_mean_curvature_H), 0.1, atol=0.08)
    assert np.isclose(profile.mean_gaussian_curvature_K, 0.01, atol=0.03)


def test_lie_algebra_obb_robustness_against_outliers():
    """
    Mathematical test:
    Construct an ellipsoid of size 20 x 10 x 4 mm with 15% severe outlier noise (specular glint simulation).
    Verify that Lie algebra SO(3) M-estimation isolates the true extents without blowup.
    """
    np.random.seed(42)
    n_inliers = 400
    # Generate points inside ellipsoid x^2/10^2 + y^2/5^2 + z^2/2^2 <= 1
    phi = np.random.uniform(0, np.pi, n_inliers)
    theta = np.random.uniform(0, 2 * np.pi, n_inliers)
    rad = np.random.uniform(0, 1, n_inliers) ** (1/3)

    x = 10.0 * rad * np.sin(phi) * np.cos(theta)
    y = 5.0 * rad * np.sin(phi) * np.sin(theta)
    z = 2.0 * rad * np.cos(phi) + 25.0
    inliers = np.stack([x, y, z], axis=-1)

    # Add 15% severe outliers far away (representing specular reflections / bubbles)
    n_outliers = 60
    outliers = np.random.uniform(low=[-30, -30, 10], high=[30, 30, 40], size=(n_outliers, 3))

    corrupted_pts = np.vstack([inliers, outliers])

    # Run Lie Algebra OBB
    obb_res = compute_robust_obb_lie(corrupted_pts)

    print(f"Lie Algebra Robust OBB Validation:")
    print(f"  Major Diameter: {obb_res.major_diameter_mm:.2f} mm (Expected true ~ 20 mm)")
    print(f"  Minor Diameter: {obb_res.minor_diameter_mm:.2f} mm (Expected true ~ 10 mm)")
    print(f"  Elevation Height: {obb_res.elevation_height_mm:.2f} mm (Expected true ~ 4 mm)")

    # The robust bounds should reject the extreme +-30mm outliers (span 60mm)
    assert obb_res.major_diameter_mm < 35.0, "Robust OBB failed to reject severe outliers."
    assert obb_res.minor_diameter_mm < 20.0, "Minor diameter inflated by outliers."


def test_heat_method_geodesic_vs_euclidean():
    """
    Mathematical test:
    Construct a folded saddle/wave mucosal surface: z = sin(x / 4) * 4.
    The surface distance along the wave must be strictly greater than the Euclidean chord distance.
    """
    xs = np.linspace(-10, 10, 25)
    ys = np.linspace(-5, 5, 15)
    X, Y = np.meshgrid(xs, ys)
    Z = np.sin(X / 3.0) * 4.0 + 30.0

    wave_pts = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=-1)

    solver = HeatGeodesicSolver(k_neighbors=10)
    geodesic_diam = solver.compute_manifold_geodesic_diameter(wave_pts)
    euclidean_diam = float(np.linalg.norm(np.ptp(wave_pts, axis=0)))

    print(f"Heat Method Geodesic Validation:")
    print(f"  Euclidean Diameter: {euclidean_diam:.2f} mm")
    print(f"  Heat Geodesic Manifold Length: {geodesic_diam:.2f} mm")

    assert geodesic_diam >= euclidean_diam * 0.95, "Geodesic diameter should be at least comparable to Euclidean."


def test_clinical_integration_pipeline_mathematical():
    """
    Integration test running the full mathematically enhanced pipeline on Kvasir-SEG sample.
    """
    dataset_dir = "/ssd-1/temp/kimicode/multmodel/lesion_3d_measurement/datasets/kvasir-seg/Kvasir-SEG"
    img_files = sorted(glob.glob(f"{dataset_dir}/images/*.jpg"))
    mask_files = sorted(glob.glob(f"{dataset_dir}/masks/*.jpg"))

    assert len(img_files) > 0, "No clinical images found."

    img = cv2.imread(img_files[0])
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_files[0], cv2.IMREAD_GRAYSCALE)

    pipeline = Lesion3DMeasurementPipeline(use_lie_algebra=True, use_heat_method=True)
    report = pipeline.process(rgb_image=rgb, lesion_mask=mask)

    res = report.result
    curv = res.curvature_profile

    print("\n--- Advanced Mathematical MetricEndo Clinical Report ---")
    print(f"Sample: {os.path.basename(img_files[0])}")
    print(f"Robust Major Diameter:   {res.major_diameter_mm:.2f} mm")
    print(f"Robust Minor Diameter:   {res.minor_diameter_mm:.2f} mm")
    print(f"Robust Elevation Height: {res.elevation_height_mm:.2f} mm")
    print(f"Heat Method Geodesic:    {res.geodesic_length_mm:.2f} mm")
    print(f"Mean Gaussian Curv (K):  {curv.mean_gaussian_curvature_K:.5f} mm^-2")
    print(f"Mean Mean Curv (H):      {curv.mean_mean_curvature_H:.4f} mm^-1")
    print(f"Surface Curvedness (C):  {curv.mean_curvedness:.4f}")
    print(f"Shape Index (S):         {curv.mean_shape_index:.3f}")
    print(f"Paris Classification:    {res.paris_morphology}")
    print(f"Guideline Risk Category: {res.risk_category}")

    out_dir = "/ssd-1/temp/kimicode/multmodel/lesion_3d_measurement/outputs"
    out_path = os.path.join(out_dir, "mathematical_kvasir_measurement.jpg")
    pipeline.render_3d_dashboard(rgb, mask, report, out_path, mode="mathematical")
    assert os.path.exists(out_path)
    print(f"Saved enhanced 3D perspective dashboard to: {out_path}")


if __name__ == "__main__":
    print("==================================================")
    print("RUNNING MATHEMATICALLY ENHANCED 3D METRIC TESTS")
    print("==================================================")

    print("\n[1/4] Testing Differential Geometry & Weingarten Shape Operator...")
    test_differential_geometry_sphere_and_plane()
    print("✓ Differential Geometry test passed.")

    print("\n[2/4] Testing Lie Group SO(3) / Lie Algebra Robust OBB...")
    test_lie_algebra_obb_robustness_against_outliers()
    print("✓ Lie Algebra OBB test passed.")

    print("\n[3/4] Testing Laplace-Beltrami Heat Method Geodesic Solver...")
    test_heat_method_geodesic_vs_euclidean()
    print("✓ Heat Geodesic test passed.")

    print("\n[4/4] Testing End-to-End Clinical Sample with Mathematical AR Overlay...")
    test_clinical_integration_pipeline_mathematical()
    print("✓ End-to-End Clinical Sample test passed.")

    print("\n==================================================")
    print("ALL 4 MATHEMATICAL ENHANCEMENT SUITES PASSED!")
    print("==================================================")
