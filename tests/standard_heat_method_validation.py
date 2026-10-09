"""Convergence checks for the standard triangular-mesh Heat Method."""

import csv
import json
from pathlib import Path
import platform
import sys

import numpy as np
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from lesion_3d_measurement.src.standard_heat_method import heat_method_distance, triangulate_parametric_grid

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "heat_method_validation"


def make_plane(n):
    axis = np.linspace(-10.0, 10.0, n)
    x, y = np.meshgrid(axis, axis)
    v = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
    f = triangulate_parametric_grid(v, n, n)
    source = (n // 2) * n + n // 2
    return v, f, source


def make_sphere_cap(n_lat, n_lon, radius=10.0, cap_angle=np.pi / 3):
    theta = np.linspace(0.02, cap_angle, n_lat)
    phi = np.linspace(0.0, 2 * np.pi, n_lon, endpoint=False)
    t, p = np.meshgrid(theta, phi, indexing="ij")
    v = np.column_stack([
        (radius * np.sin(t) * np.cos(p)).ravel(),
        (radius * np.sin(t) * np.sin(p)).ravel(),
        (radius * (1 - np.cos(t))).ravel(),
    ])
    f = triangulate_parametric_grid(v, n_lat, n_lon, wrap_cols=True)
    # Source at cap pole; deterministic tie chooses first ring vertex.
    source = 0
    return v, f, source


def validate_case(name, mesh_builder, sizes, time_scale=1.0, interior_sources=True):
    rows = []
    for size in sizes:
        v, f, base_source = mesh_builder(size)
        source_indices = [base_source]
        if name == "plane" and interior_sources and size >= 21:
            source_indices.extend([(size // 2) * size + size // 3, (size // 2) * size + (2 * size // 3)])
        for source in sorted(set(source_indices)):
            dist = heat_method_distance(v, f, source, time_scale=time_scale)
            euclid = np.linalg.norm(v - v[source], axis=1)
            # Graph shortest path is a discrete comparator.
            tree = cKDTree(v)
            d, idx = tree.query(v, k=min(7, len(v)))
            rr, cc, ww = [], [], []
            for i in range(len(v)):
                for j in range(1, idx.shape[1]):
                    a, b = i, int(idx[i, j])
                    w = float(d[i, j])
                    rr.extend((a, b)); cc.extend((b, a)); ww.extend((w, w))
            from scipy.sparse import csr_matrix
            graph = csr_matrix((ww, (rr, cc)), shape=(len(v), len(v)))
            graph_dist = dijkstra(graph, directed=False, indices=source)
            if name == "plane":
                # Euclidean distance is the exact planar geodesic.
                truth = euclid
                mae = float(np.mean(np.abs(dist - truth)))
                max_err = float(np.max(np.abs(dist - truth)))
            else:
                # Analytic spherical geodesic distance from the source vertex.
                radius = 10.0
                # Parameterization z=R(1-cos(theta)) has sphere center z=R.
                center = np.array([0.0, 0.0, radius])
                unit = (v - center) / radius
                cosine = np.clip(unit @ unit[source], -1.0, 1.0)
                truth = radius * np.arccos(cosine)
                mae = float(np.mean(np.abs(dist - truth)))
                max_err = float(np.max(np.abs(dist - truth)))
            row = {
                "case": name, "resolution": size, "vertices": len(v), "faces": len(f),
                "source_idx": int(source),
                "heat_mae_mm": mae, "heat_max_abs_error_mm": max_err,
                "mean_graph_minus_heat_mm": float(np.mean(graph_dist - dist)),
                "min_heat_minus_euclidean_mm": float(np.min(dist - euclid)),
                "minimum_distance_mm": float(np.min(dist)),
                "euclidean_violation_fraction": float(np.mean(dist + 1e-8 < euclid)),
            }
            if name == "sphere_cap":
                row["analytic_geodesic_max_mm"] = float(np.max(truth))
            rows.append(row)
    return rows


def time_scale_validation():
    candidates = (.25, .5, 1.0, 2.0, 4.0)
    validation_scores = []
    for tscale in candidates:
        val = validate_case("plane", make_plane, [21], time_scale=tscale, interior_sources=False)
        val += validate_case("sphere_cap", lambda n: make_sphere_cap(n, 2 * n), [17], time_scale=tscale, interior_sources=False)
        validation_scores.append({
            "time_scale": tscale,
            "validation_mean_mae_mm": float(np.mean([r["heat_mae_mm"] for r in val])),
            "validation_max_error_mm": float(max(r["heat_max_abs_error_mm"] for r in val)),
            "validation_chord_violation": float(np.mean([r["euclidean_violation_fraction"] for r in val])),
        })
    selected = min(validation_scores, key=lambda r: r["validation_mean_mae_mm"])["time_scale"]
    hold = validate_case("plane", make_plane, [41, 61], time_scale=selected, interior_sources=True)
    hold += validate_case("sphere_cap", lambda n: make_sphere_cap(n, 2 * n), [33, 49], time_scale=selected, interior_sources=False)
    return {
        "candidate_scores": validation_scores,
        "selected_time_scale_on_validation": selected,
        "holdout_results": hold,
        "holdout_mean_mae_mm": float(np.mean([r["heat_mae_mm"] for r in hold])),
        "holdout_max_abs_error_mm": float(max(r["heat_max_abs_error_mm"] for r in hold)),
        "holdout_mean_chord_violation_fraction": float(np.mean([r["euclidean_violation_fraction"] for r in hold])),
    }


def main():
    results = []
    results += validate_case("plane", make_plane, [11, 21, 41, 61])
    results += validate_case("sphere_cap", lambda n: make_sphere_cap(n, 2 * n), [9, 17, 33, 49])
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "heat_method_convergence.csv").open("w", newline="") as f:
        fields = list(dict.fromkeys(key for row in results for key in row))
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(results)
    grouped = {}
    for case_name in ("plane", "sphere_cap"):
        case_rows = [r for r in results if r["case"] == case_name]
        by_res = {}
        for resolution in sorted({r["resolution"] for r in case_rows}):
            sub = [r for r in case_rows if r["resolution"] == resolution]
            by_res[str(resolution)] = {
                "mean_mae_mm": float(np.mean([r["heat_mae_mm"] for r in sub])),
                "max_error_mm": float(np.max([r["heat_max_abs_error_mm"] for r in sub])),
                "mean_euclidean_violation_fraction": float(np.mean([r["euclidean_violation_fraction"] for r in sub])),
                "n_sources": len(sub),
            }
        grouped[case_name] = by_res
    payload = {
        "method": "standard cotangent-mesh Heat Method",
        "acceptance": {
            "criterion": "decreasing mesh-refinement error and vanishing geodesic < chord violation",
            "passed": False,
            "reason": "interior-source plane max error and Euclidean-bound violations do not show stable convergence across tested meshes",
        },
        "grouped_convergence": grouped,
        "time_scale_validation": time_scale_validation(),
        "results": results,
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__},
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
