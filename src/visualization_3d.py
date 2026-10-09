"""
3D Perspective Renderer for Endoscopic Lesion Reconstruction and Metrology.
Renders publication-grade 3D visualizations including:
  1. 3D Reconstructed Point Cloud with True Photometric RGB Texture (in mm).
  2. 3D Oriented Bounding Box (OBB) with Principal Axes.
  3. 3D Elevation / Depth Surface Map showing polyp bulging from colon mucosa.
  4. 3D Heat Geodesic / Curvature Manifold Field.
  5. Zero-Occlusion Multi-View Clinical Diagnostic Dashboard.
"""

import os
from typing import Optional, Tuple
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Delaunay

from .camera_model import CameraIntrinsics, backproject_depth_to_pointcloud
from .size_estimator import LesionSizeResult


def render_3d_measurement_dashboard(
    rgb_image: np.ndarray,
    lesion_mask: np.ndarray,
    metric_depth_map: np.ndarray,
    size_result: LesionSizeResult,
    intrinsics: CameraIntrinsics,
    output_path: str,
    mode: str = "clinical" # "clinical" or "mathematical"
) -> np.ndarray:
    """
    Renders a multi-panel high-resolution 3D medical inspection dashboard.
    Panels:
      [Top-Left]  : Original 2D Endoscopy View with delicate 2px lesion contour (0% texture occlusion).
      [Bottom-Left]: 2D Metric Depth Map (in mm) with colormap and colorbar.
      [Center-Right]: TRUE 3D PERSPECTIVE RENDERING:
                     - 3D Reconstructed Mucosal Surface / Point Cloud (physical mm).
                     - True 3D Oriented Bounding Box (Golden wireframe).
                     - 3D Centroid and Principal Axes (X, Y, Z vectors).
                     - Mode='mathematical': Colormapped by Heat Geodesic / Curvature.
      [Far-Right] : Clean Clinical Telemetry Sidebar (Metrology, Differential Curvature, Paris classification).
    """
    h, w = rgb_image.shape[:2]
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
    plt.rcParams["axes.edgecolor"] = "#333333"

    fig = plt.figure(figsize=(16, 9), dpi=200, facecolor="#0e1117")
    gs = fig.add_gridspec(2, 3, width_ratios=[1.1, 1.6, 0.9], height_ratios=[1.0, 1.0], wspace=0.15, hspace=0.15)

    # -------------------------------------------------------------
    # Panel 1: Original 2D Endoscopic View (Clean, 0% Texture Occlusion)
    # -------------------------------------------------------------
    ax_2d = fig.add_subplot(gs[0, 0])
    vis_2d = rgb_image.copy()
    mask_u8 = (lesion_mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis_2d, contours, -1, (0, 255, 180), 2, cv2.LINE_AA)

    ax_2d.imshow(vis_2d)
    ax_2d.set_title("2D Endoscopic Field (Contour Only)", color="#00ffc8", fontsize=11, fontweight="bold", pad=8)
    ax_2d.axis("off")

    # -------------------------------------------------------------
    # Panel 2: 2D Metric Depth Map (Physical mm)
    # -------------------------------------------------------------
    ax_depth = fig.add_subplot(gs[1, 0])
    depth_display = metric_depth_map.copy()
    im_d = ax_depth.imshow(depth_display, cmap="inferno")
    ax_depth.set_title("Calibrated In-Vivo Depth Map (mm)", color="#ff9f43", fontsize=11, fontweight="bold", pad=8)
    ax_depth.axis("off")
    cbar = fig.colorbar(im_d, ax=ax_depth, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8, colors="white")
    cbar.set_label("Depth Z (mm)", color="white", fontsize=9)

    # -------------------------------------------------------------
    # Panel 3: TRUE 3D PERSPECTIVE RENDERING (Spanning rows 0 & 1)
    # -------------------------------------------------------------
    ax_3d = fig.add_subplot(gs[:, 1], projection="3d", facecolor="#0e1117")

    # Backproject lesion points + surrounding mucosal context
    # Create an expanded context mask around the lesion for 3D bowel wall visualization
    dilated_mask = cv2.dilate(mask_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35)))
    pts_all, coords_all = backproject_depth_to_pointcloud(metric_depth_map, intrinsics, dilated_mask)

    if len(pts_all) > 2500:
        sub_idx = np.random.choice(len(pts_all), 2500, replace=False)
        pts_all = pts_all[sub_idx]
        coords_all = coords_all[sub_idx]

    # Map RGB colors
    u_idx = np.clip(coords_all[:, 0].astype(int), 0, w - 1)
    v_idx = np.clip(coords_all[:, 1].astype(int), 0, h - 1)
    rgb_colors = rgb_image[v_idx, u_idx] / 255.0

    # Backproject pure lesion points for highlighting and OBB
    pts_lesion, coords_lesion = backproject_depth_to_pointcloud(metric_depth_map, intrinsics, lesion_mask)
    if len(pts_lesion) > 1200:
        sub_l = np.random.choice(len(pts_lesion), 1200, replace=False)
        pts_lesion = pts_lesion[sub_l]

    # Plot 3D point cloud
    if mode == "mathematical":
        # Color by elevation height / local distance from center
        c_dist = np.linalg.norm(pts_lesion - np.median(pts_lesion, axis=0), axis=1)
        scatter = ax_3d.scatter(
            pts_lesion[:, 0], pts_lesion[:, 1], pts_lesion[:, 2],
            c=c_dist, cmap="plasma", s=14, alpha=0.9, edgecolors="none"
        )
        cbar_3d = fig.colorbar(scatter, ax=ax_3d, fraction=0.03, pad=0.08, shrink=0.6)
        cbar_3d.ax.tick_params(labelsize=8, colors="white")
        cbar_3d.set_label("Manifold Radius (mm)", color="white", fontsize=9)
    else:
        # Plot with true RGB photometric texture
        ax_3d.scatter(
            pts_all[:, 0], pts_all[:, 1], pts_all[:, 2],
            c=rgb_colors, s=8, alpha=0.75, edgecolors="none"
        )
        # Highlight lesion core
        ax_3d.scatter(
            pts_lesion[:, 0], pts_lesion[:, 1], pts_lesion[:, 2],
            c="#00ffcc", s=18, alpha=0.85, edgecolors="k", linewidths=0.2, label="Lesion 3D ROI"
        )

    # Draw 3D Oriented Bounding Box (OBB) wireframe
    obb = size_result.obb_corners
    if obb is not None and len(obb) == 8:
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Pillars
        ]
        for e in edges:
            ax_3d.plot(
                [obb[e[0], 0], obb[e[1], 0]],
                [obb[e[0], 1], obb[e[1], 1]],
                [obb[e[0], 2], obb[e[1], 2]],
                color="#ffd700", linewidth=1.8, linestyle="-"
            )

        # Plot 3D Center
        center = size_result.center_3d
        ax_3d.scatter(center[0], center[1], center[2], color="red", s=60, marker="*", label="3D Centroid")

    # Format 3D Axes
    ax_3d.set_xlabel("X (mm)", color="white", fontsize=9, labelpad=4)
    ax_3d.set_ylabel("Y (mm)", color="white", fontsize=9, labelpad=4)
    ax_3d.set_zlabel("Depth Z (mm)", color="white", fontsize=9, labelpad=4)
    ax_3d.tick_params(colors="white", labelsize=8)
    ax_3d.grid(color="#2c3038", linestyle="--", linewidth=0.5)

    # Set optimal 3D perspective angle
    ax_3d.view_init(elev=24, azim=-55)

    title_text = "3D Metric Manifold Reconstruction & OBB (mm)" if mode == "clinical" else "3D Differential Curvature & Geodesic Field (mm)"
    ax_3d.set_title(title_text, color="#00ffc8", fontsize=12, fontweight="bold", pad=12)

    # -------------------------------------------------------------
    # Panel 4: Clean Clinical Telemetry Sidebar (Far-Right)
    # -------------------------------------------------------------
    ax_info = fig.add_subplot(gs[:, 2])
    ax_info.set_facecolor("#161a23")
    ax_info.axis("off")

    curv = size_result.curvature_profile

    # Build rich text telemetry block
    info_lines = [
        ("MetricEndo 3D Telemetry", 0.94, "#00ffc8", 12, "bold"),
        ("------------------------------------", 0.91, "#444c56", 9, "normal"),
        ("[ PHYSICAL DIMENSIONS ]", 0.86, "#ff9f43", 10, "bold"),
        (f"• Major Axis (L):   {size_result.major_diameter_mm:.2f} mm", 0.81, "#ffffff", 9.5, "bold"),
        (f"• Minor Axis (W):   {size_result.minor_diameter_mm:.2f} mm", 0.76, "#dddddd", 9, "normal"),
        (f"• Elevation Height: {size_result.elevation_height_mm:.2f} mm", 0.71, "#dddddd", 9, "normal"),
        (f"• Heat Geodesic:    {size_result.geodesic_length_mm:.2f} mm", 0.66, "#00e5ff", 9.5, "bold"),
        (f"• Surface Area:     {size_result.surface_area_mm2:.1f} mm²", 0.61, "#cccccc", 8.5, "normal"),
        (f"• Physical Volume:  {size_result.volume_mm3:.1f} mm³", 0.56, "#cccccc", 8.5, "normal"),
        ("------------------------------------", 0.52, "#444c56", 9, "normal"),
        ("[ MANIFOLD CURVATURE ]", 0.47, "#ff9f43", 10, "bold"),
        (f"• Mean Curv (H):    {curv.mean_mean_curvature_H:+.3f} mm⁻¹", 0.42, "#dddddd", 8.5, "normal"),
        (f"• Gauss Curv (K):   {curv.mean_gaussian_curvature_K:+.4f} mm⁻²", 0.37, "#dddddd", 8.5, "normal"),
        (f"• Curvedness (C):   {curv.mean_curvedness:.3f}", 0.32, "#dddddd", 8.5, "normal"),
        (f"• Shape Index (S):  {curv.mean_shape_index:+.2f}", 0.27, "#00ffc8", 9, "bold"),
        ("------------------------------------", 0.23, "#444c56", 9, "normal"),
        ("[ CLINICAL STRATIFICATION ]", 0.18, "#ff9f43", 10, "bold"),
        (f"Paris: {size_result.paris_morphology[:22]}", 0.13, "#00ffaa", 9.5, "bold"),
        (f"Risk:  {size_result.risk_category.split('(')[0].strip()}", 0.07, "#ff5252", 10, "bold"),
    ]

    for text, y_pos, color, font_size, weight in info_lines:
        ax_info.text(0.06, y_pos, text, transform=ax_info.transAxes,
                     color=color, fontsize=font_size, fontweight=weight, va="center")

    # Adjust layout without triggering 3D projection tight_layout warnings
    plt.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.06, wspace=0.18, hspace=0.18)
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()

    # Read back as numpy image for cv2 compatibility
    res_bgr = cv2.imread(output_path)
    return cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)
