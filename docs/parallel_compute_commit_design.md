# RT Pipeline Parallel Compute + Ordered Commit Design

## Execution model

The refactor introduces a **two-stage engine**:

1. `compute_stage(scan)`
   - Runs in worker pool.
   - Performs DICOM conversion only.
   - Produces `ResultEnvelope` with the converted raw NIfTI and timing metadata.

2. `commit_stage(scan)`
   - Runs under a single ordered commit cursor.
   - Applies all stateful operations in deterministic scan order: RTPSpy motion correction, FD/DVARS histories, fieldmap unwarp, nuisance regression, space transforms, status logging, and score publication.

## Ordered commit invariants

- A central reorder buffer stores `ResultEnvelope` entries by `scan`.
- `next_scan_to_commit` is the only commit cursor.
- Runtime monotonicity check: every commit must satisfy `scan > last_committed_scan`.
- Failed/missing scans are committed deterministically as failed envelopes so the cursor always advances.

## Failure / retry behavior

- Compute stage retries each scan up to `max_retries`.
- If compute still fails, a failed envelope is inserted and committed in order.
- If the commit cursor stalls and newer scans are buffered, timeout (`commit_wait_timeout_s`) injects a deterministic failure envelope for the stalled scan.

## Rollback switch

- `pipeline_engine=legacy` keeps the prior `_process_scan` path.
- New behavior is enabled by `pipeline_engine=parallel_ordered` (default).

## Diagram

```text
incoming DICOM
    |
    v
 priority queue -> worker pool -> compute_stage(scan) -> ResultEnvelope
                                (DICOM -> raw NIfTI)
                                       |
                                       v
                                 reorder_buffer[scan]
                                       |
                                       v
                           commit cursor next_scan_to_commit
                                       |
                                       v
                                 commit_stage(scan)
                                       |
                    MC -> fieldmap unwarp -> regression/transform/score
```
