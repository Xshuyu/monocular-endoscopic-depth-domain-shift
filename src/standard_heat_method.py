"""Standard cotangent-mesh Heat Method (Crane et al.) for triangle meshes.

This implementation is deliberately separate from the legacy point-cloud
hybrid estimator. It performs heat diffusion, normalized face-gradient,
weak-form divergence assembly and a Poisson solve with a Dirichlet source.
"""

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve
from scipy.sparse.csgraph import dijkstra


def _mesh_operators(vertices: np.ndarray, faces: np.ndarray):
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=int)
    n = len(v)
    stiffness_rows, stiffness_cols, stiffness_data = [], [], []
    mass = np.zeros(n, dtype=float)
    face_areas = np.zeros(len(f), dtype=float)
    face_grads = np.zeros((len(f), 3, 3), dtype=float)

    for fi, tri in enumerate(f):
        ids = tri
        p = v[ids]
        cross = np.cross(p[1] - p[0], p[2] - p[0])
        twice_area = np.linalg.norm(cross)
        if twice_area <= 1e-14:
            raise ValueError(f"Degenerate triangle at face {fi}")
        area = 0.5 * twice_area
        normal = cross / twice_area
        face_areas[fi] = area
        mass[ids] += area / 3.0
        # Gradients of barycentric basis functions, each orthogonal to the
        # opposite edge and with magnitude edge_length/(2 area).
        face_grads[fi, 0] = np.cross(normal, p[2] - p[1]) / twice_area
        face_grads[fi, 1] = np.cross(normal, p[0] - p[2]) / twice_area
        face_grads[fi, 2] = np.cross(normal, p[1] - p[0]) / twice_area
        for local_opposite in range(3):
            a = (local_opposite + 1) % 3
            b = (local_opposite + 2) % 3
            ea = p[a] - p[local_opposite]
            eb = p[b] - p[local_opposite]
            cot = float(np.dot(ea, eb) / max(np.linalg.norm(np.cross(ea, eb)), 1e-14))
            weight = 0.5 * cot
            ia, ib = ids[a], ids[b]
            stiffness_rows.extend((ia, ib, ia, ib))
            stiffness_cols.extend((ia, ib, ib, ia))
            stiffness_data.extend((weight, weight, -weight, -weight))
    K = coo_matrix((stiffness_data, (stiffness_rows, stiffness_cols)), shape=(n, n)).tocsr()
    M = diags(np.maximum(mass, 1e-12), format="csr")
    return K, M, mass, face_areas, face_grads


def heat_method_distance(vertices: np.ndarray, faces: np.ndarray, source_idx: int, time_scale: float = 1.0):
    """Compute approximate geodesic distances from one mesh vertex.

    Args:
        vertices: (V, 3) coordinates in any consistent length unit.
        faces: (F, 3) triangle indices.
        source_idx: source vertex index.
        time_scale: multiplier for t=h^2, where h is mean mesh-edge length.

    Returns:
        Nonnegative distance vector of shape (V,).
    """
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=int)
    if v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("Expected vertices (V,3) and triangular faces (F,3)")
    if not (0 <= int(source_idx) < len(v)):
        raise ValueError("source_idx is outside the vertex range")
    K, M, mass, areas, grads = _mesh_operators(v, f)
    edges = np.vstack((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]))
    h = np.mean(np.linalg.norm(v[edges[:, 0]] - v[edges[:, 1]], axis=1))
    t = float(time_scale) * h * h
    delta = np.zeros(len(v), dtype=float)
    delta[source_idx] = 1.0
    # FEM heat equation uses M u0 on the right-hand side; u0 is the nodal
    # delta initial condition.
    rhs = M @ delta
    u = spsolve(M + t * K, rhs)

    grad_u = np.einsum("fij,fi->fj", grads, u[f])
    grad_norm = np.linalg.norm(grad_u, axis=1)
    X = -grad_u / np.maximum(grad_norm[:, None], 1e-14)

    # Weak-form divergence load: b_i = integral <grad(phi_i), X> dA.
    div_load = np.zeros(len(v), dtype=float)
    for local in range(3):
        np.add.at(div_load, f[:, local], areas * np.einsum("fi,fi->f", grads[:, local, :], X))

    # Solve the singular Poisson system with one fixed Dirichlet source.
    keep = np.ones(len(v), dtype=bool)
    keep[source_idx] = False
    phi = np.zeros(len(v), dtype=float)
    phi[keep] = spsolve(K[keep][:, keep], div_load[keep])
    phi -= phi[source_idx]
    # Sign follows the weak-form convention above; orient globally once from
    # the source-neighbour shell, never by per-mesh shortest-path fitting.
    nbr = np.unique(f[np.any(f == source_idx, axis=1)])
    nbr = nbr[nbr != source_idx]
    if len(nbr) and np.median(phi[nbr]) < 0:
        phi = -phi
    return phi


def triangulate_parametric_grid(vertices: np.ndarray, rows: int, cols: int, wrap_cols: bool = False):
    """Triangulate a row-major structured surface grid."""
    faces = []
    col_cells = cols if wrap_cols else cols - 1
    for i in range(rows - 1):
        for j in range(col_cells):
            j1 = (j + 1) % cols
            a = i * cols + j
            b = i * cols + j1
            c = (i + 1) * cols + j
            d = (i + 1) * cols + j1
            faces.append((a, b, d))
            faces.append((a, d, c))
    return np.asarray(faces, dtype=int)
