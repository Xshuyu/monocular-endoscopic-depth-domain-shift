# External dataset validation plan and current status

## Dataset options and what they validate

### C3VD / C3VDReg (recommended next external reference)

The C3VD paper describes paired colonoscopy RGB/depth, camera pose, surface/mesh and coverage references derived from high-fidelity colon models. C3VDReg derives local-to-local point-cloud pairs from C3VD depth reprojection and CT-mesh raycasting, with source-to-target pose transforms and millimetre translation metrics.

This supports external validation of:

- depth-map back-projection and coordinate/unit consistency;
- point-cloud geometry and OBB stability across real colonoscopic views;
- registration/pose metrics if using the C3VDReg official protocol.

It does **not** provide lesion-specific caliper diameters for every frame, so it cannot directly validate clinical polyp-size MAE or Paris classification.

Official references:

- C3VD paper: Bobrow et al., *Medical Image Analysis* 88 (2023), 102956; arXiv:2206.08903.
- C3VD dataset landing page: https://durr.jhu.edu/C3VD/ (HTTP access from this environment returned 403).
- C3VDReg dataset DOI from project README: https://doi.org/10.5522/04/30640043.
- C3VDReg code: https://github.com/linzhe001/C3VDReg.

### EndoNeRF (secondary, stereo/depth workflow)

The public EndoNeRF repository provides sample stereo endoscopy sequences organized with depth maps, tool masks and camera poses. Its paper/repository explicitly notes geometric ground-truth limitations and focuses evaluation on photometric reconstruction. It may test data-loader compatibility, relative-depth and stereo workflow, but should not be treated as independent metric truth for lesion diameter.

- Repository: https://github.com/med-air/EndoNeRF
- Paper: Wang et al., MICCAI 2022, arXiv:2206.15255.

## Local access status

- C3VDReg repository was cloned to `/tmp/opencode/C3VDReg` at commit `de18d37cc0a00875ffb73326d1a067a60389425f`.
- The included `c3vdreg_tiny_8192` subset has only six smoke-test pairs. Its manifest references CT meshes and poses that are not included in the tiny subset.
- Local audit of the two tiny test pairs: both source and target PLYs parse, but zero pairs have the reference mesh and pose locally; the audit script correctly refuses metric scoring.
- Full C3VDReg download could not be obtained from this environment: outbound data-host access is blocked/returns 403. No full external benchmark score is claimed.
- Project-local Kvasir-SEG has 1000 paired image/mask IDs but no metric size truth.

## Reproducible adapter

Use `tests/c3vd_reference_audit.py` after downloading the full dataset under its official terms:

```bash
PYTHONPATH=/path/to/project \
python tests/c3vd_reference_audit.py \
  --dataset-root /path/to/C3VDReg/data \
  --manifest /path/to/C3VDReg/manifests/c3vd_raycasting_v1.jsonl \
  --split test \
  --output outputs/c3vd_reference_audit_test.json
```

The adapter verifies point-cloud file access and reference/pose availability first. It intentionally refuses to score a subset without independent reference geometry. The full-data follow-up should use the official held-out C3VDReg test scenes and report registration/depth geometry metrics separately from lesion diameter.

## Needed for true lesion-size validation

To validate absolute lesion diameter, a separate reference source is required:

1. A calibrated physical phantom with known lesion dimensions, varied shape, pose, material and wet/specular conditions; or
2. A prospective/retrospective clinical cohort with independent post-resection caliper/pathology dimensions matched to the imaged lesion, under appropriate ethics and data-use permissions.

C3VD depth/mesh truth and lesion caliper truth are different endpoints; one cannot substitute for the other.
