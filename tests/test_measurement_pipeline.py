"""
Unit and Integration Tests for 3D Lesion Measurement Pipeline.
"""

import os
import glob
import cv2
import numpy as np

from lesion_3d_measurement.src import (
    CameraIntrinsics,
    backproject_depth_to_pointcloud,
    project_pointcloud_to_image,
    MetricScaleCalibrator,
    ToolDimensionPrior,
    estimate_lesion_size_3d,
    GeometricEndoscopyDepthEstimator,
    Lesion3DMeasurementPipeline
)


def test_camera_projection_roundtrip():
    """Verify that 3D back-projection and 2D re-projection are mathematical inverses."""
    w, h = 640, 480
    intrinsics = CameraIntrinsics.from_fov(w, h, fov_deg=140.0)

    # Synthetic flat depth plane at Z = 30.0 mm
    depth = np.full((h, w), 30.0, dtype=np.float32)

    # Synthetic circular mask in center
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (320, 240), 40, 255, -1)

    pts_3d, coords_2d = backproject_depth_to_pointcloud(depth, intrinsics, mask=mask)
    assert len(pts_3d) > 0

    reprojected_2d, valid = project_pointcloud_to_image(pts_3d, intrinsics)
    assert np.all(valid)

    # Difference between original (u,v) and reprojected (u,v) should be virtually 0
    diff = np.max(np.abs(coords_2d - reprojected_2d))
    assert diff < 1e-3, f"Max re-projection error: {diff}"


def test_metric_scale_calibration_with_tool():
    """Verify tool-anchored metric scale factor recovery."""
    w, h = 640, 480
    intrinsics = CameraIntrinsics.from_fov(w, h, fov_deg=140.0)
    calibrator = MetricScaleCalibrator()

    # Create a synthetic relative depth map
    rel_depth = np.ones((h, w), dtype=np.float32) * 0.5

    # Simulate an endoscopic tool shaft: vertical rectangle of width = 24 pixels
    tool_mask = np.zeros((h, w), dtype=np.uint8)
    tool_mask[200:400, 300:324] = 255  # width = 24 px

    prior = ToolDimensionPrior(sheath_diameter_mm=2.4)
    scale, tool_depth = calibrator.calibrate_with_tool(rel_depth, tool_mask, intrinsics, prior)

    # Physical check: Z = f * D / w
    f = intrinsics.fx
    expected_z = (f * 2.4) / 24.0
    expected_scale = expected_z / 0.5

    assert np.isclose(tool_depth, expected_z, rtol=0.1)
    assert np.isclose(scale, expected_scale, rtol=0.1)


def test_synthetic_sphere_size_estimation():
    """Verify 3D size estimation on a known geometric shape (hemisphere of radius R=5mm -> diameter=10mm)."""
    r = 5.0  # mm
    u = np.linspace(0, np.pi / 2, 30)
    v = np.linspace(0, 2 * np.pi, 30)
    x = (r * np.outer(np.sin(u), np.cos(v))).flatten()
    y = (r * np.outer(np.sin(u), np.sin(v))).flatten()
    z = (r * np.outer(np.cos(u), np.ones_like(v))).flatten() + 25.0  # placed at depth 25mm

    pts_3d = np.stack([x, y, z], axis=-1)
    res = estimate_lesion_size_3d(pts_3d)

    # Major and minor diameters should be close to 2 * R = 10 mm
    assert np.isclose(res.major_diameter_mm, 10.0, atol=1.5), f"Major: {res.major_diameter_mm}"
    assert np.isclose(res.minor_diameter_mm, 10.0, atol=1.5), f"Minor: {res.minor_diameter_mm}"
    assert np.isclose(res.elevation_height_mm, 5.0, atol=1.5), f"Height: {res.elevation_height_mm}"
    assert res.geodesic_length_mm >= res.major_diameter_mm


def test_real_kvasir_sample_pipeline():
    """End-to-end integration test running on an actual Kvasir-SEG clinical sample."""
    dataset_dir = "/ssd-1/temp/kimicode/multmodel/lesion_3d_measurement/datasets/kvasir-seg/Kvasir-SEG"
    img_files = sorted(glob.glob(f"{dataset_dir}/images/*.jpg"))
    mask_files = sorted(glob.glob(f"{dataset_dir}/masks/*.jpg"))

    assert len(img_files) > 0 and len(mask_files) > 0, "Kvasir-SEG images not found."

    # Pick first clinical sample
    img_path = img_files[0]
    mask_path = mask_files[0]

    bgr = cv2.imread(img_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    pipeline = Lesion3DMeasurementPipeline(fov_deg=140.0)
    report = pipeline.process(rgb_image=rgb, lesion_mask=mask)

    assert report.result.major_diameter_mm > 0.0
    assert report.result.minor_diameter_mm > 0.0
    assert report.result.elevation_height_mm > 0.0
    assert report.result.geodesic_length_mm >= report.result.major_diameter_mm
    assert report.annotated_image.shape[0] == rgb.shape[0]
    assert report.annotated_image.shape[1] >= rgb.shape[1]

    # Save output visualization
    out_dir = "/ssd-1/temp/kimicode/multmodel/lesion_3d_measurement/outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sample_kvasir_measurement.jpg")
    pipeline.render_3d_dashboard(rgb, mask, report, out_path, mode="clinical")
    assert os.path.exists(out_path)

    print("\n--- Clinical Lesion 3D Measurement Report ---")
    print(f"Sample: {os.path.basename(img_path)}")
    print(f"Major Diameter (L): {report.result.major_diameter_mm:.2f} mm")
    print(f"Minor Diameter (W): {report.result.minor_diameter_mm:.2f} mm")
    print(f"Elevation Height (H): {report.result.elevation_height_mm:.2f} mm")
    print(f"Geodesic Path Length: {report.result.geodesic_length_mm:.2f} mm")
    print(f"Surface Area: {report.result.surface_area_mm2:.2f} mm²")
    print(f"Estimated Volume: {report.result.volume_mm3:.2f} mm³")
    print(f"Clinical Risk Stratification: {report.result.risk_category}")
    print(f"Annotated visualization saved to: {out_path}")


if __name__ == "__main__":
    print("Running test_camera_projection_roundtrip...")
    test_camera_projection_roundtrip()
    print("✓ test_camera_projection_roundtrip passed.")

    print("\nRunning test_metric_scale_calibration_with_tool...")
    test_metric_scale_calibration_with_tool()
    print("✓ test_metric_scale_calibration_with_tool passed.")

    print("\nRunning test_synthetic_sphere_size_estimation...")
    test_synthetic_sphere_size_estimation()
    print("✓ test_synthetic_sphere_size_estimation passed.")

    print("\nRunning test_real_kvasir_sample_pipeline...")
    test_real_kvasir_sample_pipeline()
    print("✓ test_real_kvasir_sample_pipeline passed.")

    print("\n==========================================")
    print("ALL TESTS COMPLETED AND PASSED SUCCESSFULLY!")
    print("==========================================")
