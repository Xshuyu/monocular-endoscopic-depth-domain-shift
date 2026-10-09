"""
Differential Geometry Module for Endoscopic Mucosal Manifolds.
Computes First and Second Fundamental Forms, Weingarten Matrix (Shape Operator),
Principal Curvatures (k1, k2), Gaussian Curvature (K), Mean Curvature (H),
Curvedness (C), and Shape Index (S) to quantify lesion Paris morphology.
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any
import numpy as np
from scipy.spatial import cKDTree


@dataclass
class ManifoldCurvatureProfile:
    """
    Quantifies the intrinsic and extrinsic differential geometry of the lesion.
    """
    mean_gaussian_curvature_K: float   # Intrinsic curvature (mm^-2)
    mean_mean_curvature_H: float       # Extrinsic curvature (mm^-1)
    mean_principal_k1: float           # Max principal curvature
    mean_principal_k2: float           # Min principal curvature
    mean_curvedness: float             # Curvature intensity metric
    mean_shape_index: float            # Local surface topology [-1.0, 1.0]
    morphology_class: str              # Clinical morphological subtype (Paris classification mapping)


def fit_local_monge_patch(
    neighbors: np.ndarray,
    center: np.ndarray,
    normal: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fits a local osculating quadratic Monge patch:
        z = 0.5 * (a * x^2 + 2 * b * x * y + c * y^2) + d * x + e * y
    around the local Darboux tangent frame (u, v, n).

    Returns:
        (a, b, c): Coefficients of the second fundamental form II.
    """
    # Construct orthonormal Darboux basis: (t1, t2, n)
    n = normal / (np.linalg.norm(normal) + 1e-8)

    # Use the local tangent covariance for a stable Darboux frame.  An
    # arbitrary global reference can be nearly aligned with a tangent
    # direction and makes the quadratic fit ill-conditioned.
    tangent = neighbors - center
    tangent_cov = tangent.T @ tangent / max(len(tangent), 1)
    eigvals, eigvecs = np.linalg.eigh(tangent_cov)
    t1 = eigvecs[:, -1]
    t1 -= n * np.dot(t1, n)
    t1 /= (np.linalg.norm(t1) + 1e-8)
    t2 = np.cross(n, t1)
    t2 /= (np.linalg.norm(t2) + 1e-8)

    R = np.stack([t1, t2, n], axis=0)  # Shape (3, 3)

    # Transform neighbors to local coordinates centered at 'center'
    rel_pts = neighbors - center
    local_pts = np.dot(rel_pts, R.T)  # (K, 3) -> (x, y, z)

    # Normalize local coordinates before least squares. Without this, the
    # quadratic and linear columns have different physical scales and small
    # point perturbations can produce ill-conditioned curvature estimates.
    x = local_pts[:, 0]
    y = local_pts[:, 1]
    z = local_pts[:, 2]
    scale = max(float(np.median(np.linalg.norm(rel_pts, axis=1))), 1e-6)
    x_n, y_n, z_n = x / scale, y / scale, z / scale

    # Design matrix for quadratic surface: z = 0.5*a*x^2 + b*x*y + 0.5*c*y^2 + d*x + e*y
    # Using least squares
    A = np.stack([0.5 * x_n**2, x_n * y_n, 0.5 * y_n**2, x_n, y_n], axis=1)
    try:
        distances = np.linalg.norm(rel_pts, axis=1)
        bandwidth = max(float(np.median(distances)), 1e-6)
        weights = np.exp(-0.5 * (distances / bandwidth) ** 2)
        Aw = A * np.sqrt(weights)[:, None]
        zw = z * np.sqrt(weights)
        coeffs, _, _, _ = np.linalg.lstsq(Aw, zw, rcond=None)
        # z_n = 0.5*a_n*x_n^2 + ...; physical curvature coefficients are
        # coefficient/scale after converting all coordinates back to mm.
        a, b, c = coeffs[0] / scale, coeffs[1] / scale, coeffs[2] / scale
    except Exception:
        a, b, c = 0.0, 0.0, 0.0

    return a, b, c


def classify_paris_morphology(shape_index: float, curvedness: float, elevation_height_mm: float) -> str:
    """
    Maps continuous differential invariants (Shape Index S and Curvedness C)
    into the internationally recognized clinical Paris classification:
    - 0-Ip: Pedunculated (有蒂型隆起)
    - 0-Is: Sessile (广基隆起型)
    - 0-IIa: Slightly elevated (浅表微隆起型)
    - 0-IIb: Completely flat (平坦型)
    - 0-IIc: Slightly depressed / Ulcerated (浅凹陷型/溃疡型，恶性浸润风险最高)
    """
    # Shape index interpretation:
    # S in (-1.0, -0.625]: Cup / Pit / Depressed
    # S in (-0.625, -0.125]: Rut / Valley
    # S in (-0.125, +0.125]: Saddle / Flat ridge
    # S in (+0.125, +0.625]: Ridge / Cylinder
    # S in (+0.625, +1.0]: Dome / Cap / Sphere
    if shape_index < -0.2:
        return "Paris 0-IIc (Depressed/Ulcerated, High Invasive Risk)"
    elif curvedness < 0.05 and elevation_height_mm < 1.2:
        return "Paris 0-IIb (Flat Mucosa)"
    elif elevation_height_mm < 2.5:
        return "Paris 0-IIa (Slightly Elevated, Superficial)"
    elif shape_index > 0.6 and elevation_height_mm > 7.0:
        return "Paris 0-Ip (Pedunculated Polyp with Stalk)"
    else:
        return "Paris 0-Is (Sessile Polyp, Elevated Base)"


def compute_manifold_curvature(
    points_3d: np.ndarray,
    elevation_height_mm: float,
    k_neighbors: int = 12,
    normal_orientation: str = "camera",
    random_seed: int = 0,
    normal_reference_point: np.ndarray = None,
) -> ManifoldCurvatureProfile:
    """
    Evaluates extrinsic and intrinsic differential invariants across the lesion surface.

    Args:
        points_3d: (N, 3) point cloud coordinates in millimeters.
        elevation_height_mm: Physical elevation height of the lesion.
        k_neighbors: Number of neighboring points for local quadric patch estimation.

    Returns:
        ManifoldCurvatureProfile containing Riemannian and Euclidean differential invariants.
    """
    n_pts = len(points_3d)
    if n_pts < k_neighbors:
        return ManifoldCurvatureProfile(
            mean_gaussian_curvature_K=0.0,
            mean_mean_curvature_H=0.0,
            mean_principal_k1=0.0,
            mean_principal_k2=0.0,
            mean_curvedness=0.0,
            mean_shape_index=0.0,
            morphology_class="Unclassified (Insufficient Points)"
        )

    tree = cKDTree(points_3d)
    dists, indices = tree.query(points_3d, k=k_neighbors)

    # 1. Estimate local normals via PCA covariance of neighborhoods
    k1_list = []
    k2_list = []
    K_list = []
    H_list = []
    curvedness_list = []
    shape_index_list = []

    # Sample a representative subset for computational efficiency if point cloud is large
    sample_size = min(n_pts, 300)
    rng = np.random.default_rng(random_seed)
    sample_indices = np.sort(rng.choice(n_pts, sample_size, replace=False))

    for idx in sample_indices:
        neighbor_pts = points_3d[indices[idx]]
        center = points_3d[idx]

        # Local normal from covariance matrix
        cov = np.cov(neighbor_pts - center, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        normal = eigenvectors[:, 0]  # Normal corresponds to smallest eigenvalue

        # Orient normals consistently. For visible camera-facing patches,
        # enforce the camera direction. For analytic surfaces, callers may
        # request outward orientation; this avoids flipping the lower half of
        # a full sphere toward +Z.
        if normal_orientation == "camera":
            if normal[2] > 0:
                normal = -normal
        elif normal_orientation == "outward":
            ref_point = np.zeros(3) if normal_reference_point is None else np.asarray(normal_reference_point, dtype=float)
            if np.dot(normal, center - ref_point) < 0:
                normal = -normal
        elif normal_orientation == "none":
            pass
        else:
            raise ValueError("normal_orientation must be 'camera', 'outward', or 'none'")

        # 2. Fit Monge Patch to obtain Second Fundamental Form: II = [a, b; b, c]
        a, b, c = fit_local_monge_patch(neighbor_pts, center, normal)

        # Weingarten matrix in orthonormal frame: W = II (since I is identity at Darboux tangent plane)
        # Eigenvalues of [a, b; b, c] are the principal curvatures k1 and k2
        trace = a + c
        det = a * c - b**2
        discriminant = max(0.0, trace**2 - 4.0 * det)
        sqrt_disc = np.sqrt(discriminant)

        k1 = 0.5 * (trace + sqrt_disc)
        k2 = 0.5 * (trace - sqrt_disc)

        # Gaussian curvature K = k1 * k2, Mean curvature H = 0.5 * (k1 + k2)
        K = k1 * k2
        H = 0.5 * (k1 + k2)

        # Curvedness C = sqrt((k1^2 + k2^2) / 2)
        C = np.sqrt(max(1e-8, 0.5 * (k1**2 + k2**2)))

        # Shape Index S = (2 / pi) * arctan((k1 + k2) / (k1 - k2)), where k1 >= k2
        denom = k1 - k2
        if abs(denom) < 1e-6:
            S = 1.0 if (k1 + k2) > 0 else -1.0
        else:
            S = (2.0 / np.pi) * np.arctan((k1 + k2) / denom)

        k1_list.append(k1)
        k2_list.append(k2)
        K_list.append(K)
        H_list.append(H)
        curvedness_list.append(C)
        shape_index_list.append(S)

    mean_K = float(np.mean(K_list))
    mean_H = float(np.mean(H_list))
    mean_k1 = float(np.mean(k1_list))
    mean_k2 = float(np.mean(k2_list))
    mean_C = float(np.mean(curvedness_list))
    mean_S = float(np.mean(shape_index_list))

    morph_class = classify_paris_morphology(mean_S, mean_C, elevation_height_mm)

    return ManifoldCurvatureProfile(
        mean_gaussian_curvature_K=mean_K,
        mean_mean_curvature_H=mean_H,
        mean_principal_k1=mean_k1,
        mean_principal_k2=mean_k2,
        mean_curvedness=mean_C,
        mean_shape_index=mean_S,
        morphology_class=morph_class
    )
