"""Heat-kernel-assisted kNN graph distance estimator for point clouds.

This implementation is not Crane et al.'s full Heat Method: it uses a
short-time heat estimate and a kNN shortest-path distance, returning their
pointwise maximum. Manuscript text and output labels should call this a
hybrid heat-kernel/graph distance, not a continuous Heat Method geodesic.
"""

from typing import Tuple, Optional
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import dok_matrix, csc_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import spsolve


class HeatGeodesicSolver:
    """
    Computes approximate intrinsic distances on a kNN point-cloud graph.
    """

    def __init__(self, k_neighbors: int = 12):
        self.k_neighbors = k_neighbors

    def _build_laplacian(
        self,
        points: np.ndarray
    ) -> Tuple[csc_matrix, csc_matrix, float]:
        """
        Constructs the discrete symmetric Laplace-Beltrami operator L and lumped mass matrix A
        using Euclidean distance weighted cotangent/Gaussian approximations.
        """
        n = len(points)
        tree = cKDTree(points)
        dists, indices = tree.query(points, k=min(self.k_neighbors + 1, n))

        # Mean edge length h
        valid_dists = dists[:, 1:]
        mean_h = float(np.mean(valid_dists))
        sigma = max(mean_h, 1e-4)

        # L matrix (Laplacian, negative semi-definite) and A matrix (mass matrix)
        L = dok_matrix((n, n), dtype=np.float64)
        A = dok_matrix((n, n), dtype=np.float64)

        for i in range(n):
            weight_sum = 0.0
            for j_idx in range(1, indices.shape[1]):
                j = indices[i, j_idx]
                d = dists[i, j_idx]
                # Gaussian bell kernel approximation for Riemannian metric
                w = np.exp(-(d**2) / (2.0 * sigma**2))
                L[i, j] = w
                weight_sum += w

            L[i, i] = -weight_sum
            # Lumped area element around vertex i
            A[i, i] = max(1e-5, weight_sum * (sigma**2))

        return L.tocsc(), A.tocsc(), mean_h

    def solve_geodesic_from_source(
        self,
        points: np.ndarray,
        source_idx: int,
        t_diffusion: Optional[float] = None
    ) -> np.ndarray:
        """
        Computes hybrid heat-kernel / kNN graph distances from one source.

        Returns:
            phi: (N,) array of geodesic distances from source_idx in millimeters.
        """
        n = len(points)
        L, A, mean_h = self._build_laplacian(points)

        # Diffusion time t = m * h^2 (Crane et al. recommendation: t = h^2)
        if t_diffusion is None:
            t = (mean_h ** 2) * 1.5
        else:
            t = t_diffusion

        # Step 1: Solve heat diffusion: (A - t * L) u = delta_x
        M = A - t * L
        b = np.zeros(n, dtype=np.float64)
        b[source_idx] = 1.0

        try:
            u = spsolve(M, b)
        except Exception:
            # Fallback to direct distance if sparse solver encounters singularity
            return np.linalg.norm(points - points[source_idx], axis=1)

        # Ensure positivity of heat
        u = np.maximum(u, 1e-12)

        # Varadhan's estimator is the heat-method distance estimate.  The
        # graph shortest-path distance is used only as a conservative
        # discretization fallback and is not replaced by an empirical scale.
        ratio = u / max(float(np.max(u)), 1e-12)
        log_u = np.log(np.maximum(ratio, 1e-12))
        heat_distance = np.sqrt(np.maximum(0.0, -2.0 * t * log_u))

        # Use the Euclidean k-NN edge lengths for the graph fallback.  The
        # Laplacian itself contains a negative diagonal and is not a graph
        # distance matrix.
        tree = cKDTree(points)
        edge_dists, edge_indices = tree.query(
            points, k=min(self.k_neighbors + 1, n)
        )
        graph = dok_matrix((n, n), dtype=np.float64)
        for i in range(n):
            for edge_idx in range(1, edge_indices.shape[1]):
                j = int(edge_indices[i, edge_idx])
                d = float(edge_dists[i, edge_idx])
                graph[i, j] = d
                graph[j, i] = d
        graph_distance = dijkstra(csr_matrix(graph), directed=False, indices=source_idx)
        finite = np.isfinite(graph_distance)
        phi = heat_distance.copy()
        # The discrete heat approximation can underestimate distance on a
        # sparse point cloud.  Use the intrinsic kNN path length as a
        # conservative lower-bound correction; unlike the former fixed
        # expansion factor, this depends only on the observed manifold graph.
        phi[finite] = np.maximum(phi[finite], graph_distance[finite])
        phi[source_idx] = 0.0
        return phi

    def solve_knn_graph_distance(self, points: np.ndarray, source_idx: int) -> np.ndarray:
        """Return intrinsic kNN shortest-path distances in point-coordinate units."""
        n = len(points)
        tree = cKDTree(points)
        dists, indices = tree.query(points, k=min(self.k_neighbors + 1, n))
        graph = dok_matrix((n, n), dtype=np.float64)
        for i in range(n):
            for j_idx in range(1, indices.shape[1]):
                j = int(indices[i, j_idx])
                d = float(dists[i, j_idx])
                graph[i, j] = d
                graph[j, i] = d
        return dijkstra(csr_matrix(graph), directed=False, indices=source_idx)

    def compute_manifold_geodesic_diameter(self, points: np.ndarray) -> float:
        """
        Calculates the largest sampled hybrid graph distance across the point cloud.
        """
        n = len(points)
        if n < 8:
            return float(np.linalg.norm(np.ptp(points, axis=0)))

        # Subsample if point cloud is excessively dense
        if n > 400:
            sample_idx = np.random.choice(n, 400, replace=False)
            pts = points[sample_idx]
            n = 400
        else:
            pts = points

        # Find peripheral source point (farthest from centroid)
        centroid = np.mean(pts, axis=0)
        dists_to_c = np.linalg.norm(pts - centroid, axis=1)
        source_idx = int(np.argmax(dists_to_c))

        # Solve Heat Geodesic from peripheral boundary point
        phi = self.solve_geodesic_from_source(pts, source_idx)

        # The opposite boundary point is the maximizer of phi
        farthest_idx = int(np.argmax(phi))

        # Solve backwards for maximal precision
        phi_rev = self.solve_geodesic_from_source(pts, farthest_idx)
        geodesic_diameter = float(np.max(phi_rev))

        # Geodesic distance is strictly >= Euclidean distance
        euclidean_diam = float(np.linalg.norm(pts[farthest_idx] - pts[source_idx]))
        return max(geodesic_diameter, euclidean_euclidean_diam := euclidean_diam)
