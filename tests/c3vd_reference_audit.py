"""Audit/adapter for C3VDReg paired point-cloud reference data.

This script supports the C3VDReg manifest schema. It deliberately refuses to
score geometry when reference meshes/poses are missing (as in the committed
tiny smoke-test subset). With the complete DOI dataset present, it validates
source/target/reference paths, units, transforms and source-to-target point
alignment before computing geometry metrics.
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np


def read_ascii_ply(path):
    with Path(path).open("rb") as f:
        line = f.readline().decode("ascii", errors="ignore")
        if line.strip() != "ply":
            raise ValueError(f"Not a PLY file: {path}")
        count = None
        fmt = None
        properties = []
        in_vertex = False
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Unexpected EOF in PLY header")
            parts = line.decode("ascii", errors="ignore").strip().split()
            if parts[:2] == ["format", "ascii"]:
                fmt = "ascii"
            elif parts[:3] == ["format", "binary_little_endian", "1.0"]:
                fmt = "binary_little_endian"
            if parts[:2] == ["element", "vertex"]:
                count = int(parts[2]); in_vertex = True
            elif parts and parts[0] == "element":
                in_vertex = False
            elif parts[:1] == ["property"] and in_vertex:
                properties.append(parts[-1])
            elif parts[:1] == ["end_header"]:
                break
        if count is None or not all(k in properties for k in ("x", "y", "z")):
            raise ValueError("PLY header does not provide vertex x/y/z")
        if fmt == "ascii":
            rows = []
            for _ in range(count):
                rows.append([float(x) for x in f.readline().decode("ascii", errors="ignore").split()[:len(properties)]])
            arr = np.asarray(rows, dtype=float)
        elif fmt == "binary_little_endian":
            type_map = {
                "char": "b", "int8": "b", "uchar": "B", "uint8": "B",
                "short": "h", "int16": "h", "ushort": "H", "uint16": "H",
                "int": "i", "int32": "i", "uint": "I", "uint32": "I",
                "float": "f", "float32": "f", "double": "d", "float64": "d",
            }
            # Re-read header to recover scalar types for each vertex property.
            f.seek(0)
            prop_specs = []
            in_vertex = False
            while True:
                line = f.readline().decode("ascii", errors="ignore").strip()
                parts = line.split()
                if parts[:2] == ["element", "vertex"]:
                    in_vertex = True
                elif parts and parts[0] == "element":
                    in_vertex = False
                elif parts[:1] == ["property"] and in_vertex and len(parts) >= 3 and parts[1] != "list":
                    prop_specs.append((parts[2], type_map[parts[1]]))
                elif parts[:1] == ["end_header"]:
                    break
            fmt_str = "<" + "".join(code for _, code in prop_specs)
            item_size = struct.calcsize(fmt_str)
            raw = f.read(count * item_size)
            values = np.frombuffer(raw, dtype=np.dtype([(name, "<" + {"b":"i1","B":"u1","h":"i2","H":"u2","i":"i4","I":"u4","f":"f4","d":"f8"}[code]) for name, code in prop_specs]))
            arr = np.column_stack([values[k].astype(float) for k in properties if k in values.dtype.names])
            properties = [k for k in properties if k in values.dtype.names]
        else:
            raise ValueError(f"Unsupported PLY format: {fmt}")
    idx = [properties.index(k) for k in ("x", "y", "z")]
    return arr[:, idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.dataset_root)
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r.get("split") == args.split]
    audit = []
    for r in rows:
        source = root / r["source_path"]
        target = root / r["target_path"]
        reference = root / r["reference_path"] if r.get("reference_path") else None
        ref_pose = root / r["reference_pose_path"] if r.get("reference_pose_path") else None
        item = {
            "sample_id": r.get("sample_id"),
            "scene": r.get("scene"),
            "source_exists": source.exists(),
            "target_exists": target.exists(),
            "reference_exists": bool(reference and reference.exists()),
            "reference_pose_exists": bool(ref_pose and ref_pose.exists()),
        }
        if item["source_exists"] and item["target_exists"]:
            src_pts = read_ascii_ply(source)
            tgt_pts = read_ascii_ply(target)
            item.update({
                "source_n": len(src_pts), "target_n": len(tgt_pts),
                "source_bounds": np.ptp(src_pts, axis=0).tolist(),
                "target_bounds": np.ptp(tgt_pts, axis=0).tolist(),
                "source_dtype_range": [float(src_pts.min()), float(src_pts.max())],
                "target_dtype_range": [float(tgt_pts.min()), float(tgt_pts.max())],
            })
        item["scoreable_geometry"] = item["source_exists"] and item["target_exists"] and item["reference_exists"] and item["reference_pose_exists"]
        audit.append(item)
    payload = {
        "dataset_root": str(root),
        "split": args.split,
        "n_manifest_pairs": len(rows),
        "n_source_target_pairs_readable": sum(x.get("source_exists", False) and x.get("target_exists", False) for x in audit),
        "n_scoreable_with_reference": sum(x["scoreable_geometry"] for x in audit),
        "claim_boundary": "paired registration/depth geometry validation only; not lesion caliper diameter unless lesion-specific physical annotations exist",
        "pairs": audit,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: v for k, v in payload.items() if k != "pairs"}, indent=2))
    if payload["n_scoreable_with_reference"] == 0:
        raise SystemExit("No reference-backed pairs available; this subset is not scoreable for metric geometry.")


if __name__ == "__main__":
    main()
