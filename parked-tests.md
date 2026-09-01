# Parked Semantic Feature Tests

These are deferred experiments for improving FG-CLIP semantic distillation. Test each change independently against the current committed frontier before combining changes.

## Common Validation

Keep poses, TSDF surface, selected frames, hypotheses, prompts, and evaluation labels fixed. Record:

- semantic top-1 and field coverage;
- per-class, small-object, and boundary accuracy where practical;
- Stage 12 wall time and peak VRAM;
- instance metrics, which should remain unchanged;
- semantic PCA PLY quality for visual inspection.

Treat small metric changes as noise until reproduced.

## Experiments

1. **Hypothesis-union accumulation.** Allocate and update temporary per-point features only for surface points belonging to at least one finalized hypothesis. Preserve the original point indexing when pooling and writing artifacts. Expected gain is modest because current hypothesis coverage is already high.

2. **PCA visualization optimization.** Replace the full SVD over up to `100k x 768` reconstructed descriptors with a validated deterministic covariance, randomized, or incremental top-3 PCA method. This affects visualization time only, not stored descriptors or semantic metrics.

3. **FG-CLIP 2 migration study.** Independently evaluate `fg-clip2-large` and `fg-clip2-so400m`. Treat this as a model and embedding-space migration, not a config swap: validate dense-feature APIs, text embeddings, prompts, dimensions, model revision, accuracy, speed, and VRAM from scratch.

## Rejected Unless Requirements Change

- Do not store 768-D features inside every TSDF voxel; it greatly increases volume memory and changes the audited surface-feature semantics.
- Do not add PCA-128 storage while only compact hypothesis descriptors are persisted; current storage savings are negligible and truncated PCA does not preserve cosine similarity exactly.
- Do not spatially smooth across hypotheses; it risks blending instance boundaries. Reconsider only with an explicit edge- or atom-constrained formulation and a demonstrated noise problem.
- Do not use GT poses in production metrics. A GT-pose path may exist only as a separately labeled oracle diagnostic.
