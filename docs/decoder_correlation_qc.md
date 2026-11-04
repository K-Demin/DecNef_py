# Decoder-ROI spatial correlation during a run

The live pipeline compares each aligned BOLD image with one fixed EPI reference:

`r_t = Pearson(reference[fixed_QC_ROI], BOLD_t[fixed_QC_ROI])`

This is a correlation **across voxels** at each TR. It measures spatial pattern
stability, independently of the decoder score. It is calculated after unwarping,
motion correction and final spatial transforms, **before smoothing, cGLM and
voxel normalization**, including startup volumes. Decoder weights do not enter
the correlation; their effective feature mask supplies the initial ROI.

## Settings

Use the same settings JSON as the run launcher:

```json
{
  "enable_decoder_correlation_qc": true,
  "decoder_qc_reference": "/study/decoder_training_mean_bold.nii.gz",
  "decoder_qc_roi_mask": null
}
```

| Setting | Default | Meaning |
| --- | --- | --- |
| `enable_decoder_correlation_qc` | `true` | Calculate and record spatial QC for voxel-decoder runs. |
| `decoder_qc_reference` | `null` | Preferred: mean EPI from decoder construction, already in the final analysis space and exact output grid. With `null`, the prepared corrected session EPI is mapped once into that grid as a labelled **session-reference fallback**. |
| `decoder_qc_roi_mask` | `null` | Optional feature ROI NIfTI on the same final grid, intersected with the effective decoder mask for QC only. Supply a genuine feature mask when a decoder file's nonzero background does not represent its trained ROI. |

The reference and optional ROI must match the final image's shape and physical
affine. A 4D image containing exactly one volume is accepted. The script does not
estimate a new registration for these supplied files or silently change their
coordinate systems. T1/MNI references must match the decoder/output grid; an
EPI-space reference must match the stable prepared EPI grid. Anatomical T1 images
and weight maps are not suitable substitutes for a mean BOLD reference.

The fallback tests stability against session preparation. It cannot establish
agreement with the decoder-training acquisition. Updating an anatomical
registration does not repair a mismatched supplied training reference.

## Fixed ROI and missing coverage

Once per run, the QC ROI is defined as the effective decoder mask intersected
with the prepared reference's geometric coverage and mapped brain mask, then
with the optional QC ROI. The saved metadata records the full decoder count,
fixed QC count and excluded decoder fraction. These exclusions affect QC only;
they do not remove or reweight decoder features.

The same fixed voxels are compared at every TR. A later missing voxel produces
`incomplete_coverage` and a NaN correlation, alongside the available fraction.
The script does not replace missing samples with zeros or calculate a
potentially misleading correlation on a different ROI at each TR. Current
spatial support is used here, independently of cumulative cGLM eligibility.
The existing `coverage_rt.csv` still reports support for the full decoder.

Constant patterns, nonfinite values, empty ROIs and insufficient voxel counts
also produce NaN plus an explicit status. An intensity of zero alone does not
define missing acquisition coverage. Correlations cannot prove anatomical
accuracy, and a high value within a small covered ROI does not validate the
uncovered decoder features.

## Outputs

Within the usual `data/sub-ID/day/func/run/` directory:

| File | Contents |
| --- | --- |
| `decoder_qc/fixed_roi.nii` | The exact fixed QC ROI for visual inspection. |
| `decoder_qc/session_reference.nii` | Mapped reference when using the fallback. |
| `decoder_qc_reference.json` | Reference identity, coordinates, ROI counts, stage and missing-coverage policy. Also copied into `session_metadata.json`. |
| `decoder_qc_request.json` | Content identities for the input reference, fixed masks, grid, transforms and interpolation; prevents mixing different references when restarting a run. |
| `decoder_qc_rt.csv` | `volume_idx`, `reference_correlation`, `reference_coverage_fraction`, `correlation_status`; every spatially processed TR, including warmup. |
| `regression_status_rt.csv` | The same three QC columns beside FD, DVARS and readiness. |
| `qc_scores_motion.png` | Existing score/motion plot plus correlation and coverage for all recorded TRs. Short runs can produce a QC-only panel before scores are ready. |
| `run_qc.xlsx` | Written at completion. `Volumes` joins available score/QC CSVs by explicit `volume_idx`; `Summary` records counts and QC reference metadata. |

Workbook writing and plotting occur after processing, not once per TR. The
reference is mapped and centered once; each TR requires only ROI extraction and
a Pearson calculation. The behavioral launchers export after the processing
child exits. The direct pipeline also attempts workbook export on orderly stop.
An abrupt process or machine termination can leave only the CSVs; they remain
the primary per-TR record. Existing CSVs without the new columns remain readable
by the report helpers. Processing refuses to append to the older status-column
schema; use fresh run folders for new acquisitions. A restart with different
QC reference/ROI inputs is rejected before overwriting generated QC files.

PCA workflows without a voxel decoder do not infer an ROI from their geometry
template; status is `no_voxel_decoder`. Disabling QC records `disabled`. This
feature does not change scores, nuisance regression, motion/DVARS censoring,
normalization, acquisition masks or the registration approval gate. No automatic
correlation threshold is applied.

The decoder-training reference comparison follows the spatial-pattern check in
[Cortese et al. (2021), The DecNef collection](https://www.nature.com/articles/s41597-021-00845-7#Sec12).
The published check was trial-based. Calculating it per TR, using a
session-reference fallback and reporting missing coverage explicitly are
implementation choices here, rather than claims of equivalence to that study.
