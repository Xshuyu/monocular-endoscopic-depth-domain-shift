"""
Camera Intrinsic and Back-Projection Module for Endoscopic Imaging.
Handles projection between 2D pixel space and 3D physical coordinate space.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class CameraIntrinsics:
    """
    Endoscopic camera intrinsic parameters.
    Default parameters correspond to standard wide-angle colonoscopes (~140° FOV).
    """
    fx: float  # Focal length x (pixels)
    fy: float  # Focal length y (pixels)
    cx: float  # Principal point x (pixels)
    cy: float  # Principal point y (pixels)
    width: int
    height: int

    @classmethod
    def from_fov(cls, width: int, height: int, fov_deg: float = 140.0) -> "CameraIntrinsics":
        """
        Derives intrinsic parameters assuming symmetric center and given Field of View.
        """
        fov_rad = np.radians(fov_deg)
        # Endoscopic wide-angle approximation
        f = (width / 2.0) / np.tan(fov_rad / 2.0)
        return cls(
            fx=float(f),
            fy=float(f),
            cx=float(width / 2.0),
            cy=float(height / 2.0),
            width=width,
            height=height
        )

    def to_matrix(self) -> np.ndarray:
        """Returns 3x3 intrinsic matrix K."""
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)


def backproject_depth_to_pointcloud(
    depth_map: np.ndarray,
    intrinsics: CameraIntrinsics,
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Back-project a 2D depth map (in mm) to a 3D point cloud (X, Y, Z in mm).

    Args:
        depth_map: 2D array of metric depth in millimeters, shape (H, W).
        intrinsics: CameraIntrinsics object.
        mask: Optional binary mask (H, W), non-zero pixels indicate points to back-project.

    Returns:
        points_3d: (N, 3) array of 3D coordinates [X, Y, Z] in mm.
        valid_coords: (N, 2) array of corresponding (u, v) pixel coordinates.
    """
    h, w = depth_map.shape[:2]
    u_grid, v_grid = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    if mask is not None:
        valid_mask = (mask > 0) & (depth_map > 0) & np.isfinite(depth_map)
    else:
        valid_mask = (depth_map > 0) & np.isfinite(depth_map)

    u = u_grid[valid_mask]
    v = v_grid[valid_mask]
    z = depth_map[valid_mask]

    # Standard pinhole back-projection:
    # X = (u - cx) * Z / fx
    # Y = (v - cy) * Z / fy
    # Z = Z
    x = (u - intrinsics.cx) * z / intrinsics.fx
    y = (v - intrinsics.cy) * z / intrinsics.fy

    points_3d = np.stack([x, y, z], axis=-1)
    coords_2d = np.stack([u, v], axis=-1)

    return points_3d, coords_2d


def project_pointcloud_to_image(
    points_3d: np.ndarray,
    intrinsics: CameraIntrinsics
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D points [X, Y, Z] back to 2D image plane [u, v].

    Args:
        points_3d: (N, 3) coordinates in camera frame (Z > 0).
        intrinsics: CameraIntrinsics object.

    Returns:
        coords_2d: (N, 2) pixel coordinates.
        valid_mask: (N,) boolean mask indicating points in front of camera.
    """
    x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
    valid_mask = z > 1e-4

    u = np.zeros_like(x)
    v = np.zeros_like(y)

    u[valid_mask] = (x[valid_mask] * intrinsics.fx / z[valid_mask]) + intrinsics.cx
    v[valid_mask] = (y[valid_mask] * intrinsics.fy / z[valid_mask]) + intrinsics.cy

    coords_2d = np.stack([u, v], axis=-1)
    return coords_2d, valid_mask
