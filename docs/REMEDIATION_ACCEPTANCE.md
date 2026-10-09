# Algorithm remediation acceptance protocol

This protocol is frozen before the remediation reruns. It separates endpoint
definitions and prevents selecting a correction after inspecting test results.

## OBB

- Test analytically sampled ellipsoids with randomized SO(3) rotation.
- Report full-support and 1--99% support as distinct estimands.
- Compare PCA, MAD-filtered PCA and the SO(3) quantile-volume estimator on the
  same points, with the same radial filter and the same endpoint quantiles.
- Require corner-edge / reported-axis consistency < 1e-6 mm and rotation
  invariance < 0.01 mm.
- Report clean bias separately from contamination robustness. Do not claim an
  SO(3) contribution unless its paired benefit remains after Holm correction.
- Any adaptive rejection gate must report clean coverage, abstention rate by
  contamination level, and validation/test seed separation; accuracy without
  coverage is insufficient.

## Scale cue

- State the pinhole, instrument-axis and projected-width assumptions.
- Test 20/30/45 mm, width quantization, mask erosion/dilation and instrument
  axis tilt; compare against known camera-to-tool distance.
- Calibrate a confidence or rejection threshold on a separate validation set.
- Never call prior-scaled depth metric ground truth. Low-confidence scale must
  be reported as unavailable or as an interval, not silently clamped.

## Curvature

- Use a visible spherical cap, plane and saddle with analytical references.
- Make point sampling deterministic and test 0, 0.005 and 0.01 mm normal noise.
- Predeclare allowable H and K error; if Gaussian curvature fails under the
  stated noise regime, remove it from clinical morphology claims.

## Surface distance

- A kNN shortest path is labelled graph geodesic, not the standard Heat Method.
- Standard Heat Method claims require triangular mesh cotangent Laplacian,
  mass matrix, face-gradient normalization and Poisson integration, plus
  analytical mesh convergence tests.
- Do not infer ESD margin from unlabelled Kvasir-SEG outputs.

## Statistics and provenance

- Save per-case CSV, deterministic bootstrap CI, paired differences and Holm-
  adjusted p-values.
- Record generator/source hashes, seeds, dependencies and inclusion counts.
- Kvasir-SEG is descriptive only unless independent metric size truth is added.
