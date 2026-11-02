# Brain selection and the processing mask

Incoming BOLD is unwarped and motion-corrected on the session EPI grid, then
processed in this order:

**Spatial mapping → optional masked smoothing → cumulative-GLM denoising →
voxel normalization.**

The response data, denoising mask and normalization statistics all use the final
analysis grid: EPI, T1 or MNI, as selected by `analysis_space`. For EPI analysis
with a session-to-reference bridge, this is the stable reference EPI grid.

## Anatomical selection and temporal processing

Preparation supplies the session EPI brain mask, normally SynthStrip's
`epi_mask_mean.nii` from the unwarped, motion-corrected reference mean. Review
that mask against the corrected EPI and aligned anatomy: changing the processing
order does not recover cortex omitted by the anatomical mask.

At run initialization, the anatomical brain mask is mapped onto the final grid
with nearest-neighbour interpolation. The temporal processing mask is that
mapped brain mask intersected with the selected analysis mask. Consequently,
`analysis_mask: "whole_brain"`, `"cortical_gm"` and `"custom"` control the response
voxels processed by denoising and normalization as well as smoothing. A zero
FWHM skips Gaussian filtering; a cortex/custom selection still applies.

Spatial mapping receives acquired motion-corrected BOLD before response
masking or temporal processing. There is no response-processing interpolation
margin: spatial interpolation has already happened by the time denoising starts.
The native cubic-margin helper is retained for historical callers and separate
native nuisance support; it no longer defines the response GLM mask.

GS, WM and ventricular nuisance signals are sampled separately from the
**native, unsmoothed, motion-corrected** volume using their native masks and
available acquisition support. Selecting cortex as the response region therefore
does not discard the WM/ventricular signals needed by the nuisance model. Motion
estimation, FD, DVARS, registration masks and fitted transforms retain their
native inputs and existing definitions.

## Missing samples

Per-volume acquisition support is mapped to the final grid once, using Linear
interpolation and a threshold of `1 - 1e-5` for both Linear and Lanczos signal
interpolation. It is then intersected with the anatomical/analysis selections.
This availability rule does not require every Lanczos neighbour to be acquired.
It is not a guarantee that every contributor exists at the acquisition boundary.

Masked smoothing excludes unavailable samples from both its numerator and its
normalizing denominator, and leaves those output samples unavailable. cGLM's
complete-history support and normalization's valid-statistics checks can further
reduce usable support. Neither smoothing nor a broader mask recovers missing
slices. Temporal support is already on the final grid and is not spatially
warped a second time. `coverage_rt.csv` records the corresponding final-grid
coverage; native MC support remains a separate input to motion/DVARS processing.

## Saved files and warmup

Each run saves these files under `func/<run>/processing_mask/`:

| File | Meaning |
| --- | --- |
| `brain_epi.nii` | Anatomical brain selection on the session EPI grid |
| `brain_final.nii` | Anatomical brain selection mapped onto the final analysis grid |
| `processing_final.nii` | Final-grid response selection used by denoising and normalization |
| `policy.json` | Policy/version, processing order, nuisance source, interpolation, source identity and voxel counts |

`mc/` retains native motion-corrected BOLD. `t1/`, `mni/` or bridged `epi_ref/`
contain spatially mapped BOLD in original intensity units. Optional `smooth/`
or `masked/` outputs are the next, pre-denoising stage. `reg/` contains the
denoising output; `norm/` is the final normalized feature source for decoding.
All stages from spatial mapping onward use the final analysis affine.

Spatial and smoothed images remain visible during calibration. Unready `norm/`
volumes are zero-filled, their usable `support/` masks are empty and the decoder
score remains unavailable. Check `glm_ready`, `normalization_ready` and
`reg_ready` in `regression_status_rt.csv`; visible anatomy alone does not imply
analysis readiness. Normalization reference images in `norm/` also use the
final analysis grid.

### Decoder contribution volumes

`save_decoder_contributions: true` (default) saves the linear decoder's voxel
terms to `func/<run>/decoder_contributions/vol_XXXXX_decoder_contributions.nii`.
Each float32 voxel is the final `norm/` value multiplied by its effective decoder
weight, using ROI-TXT weights when supplied. The image keeps the input affine and
grid; voxels outside the decoder mask are zero. No further masking is applied.

Summing the image and adding the scalar `bias` in `decoder_contributions/decoder.json`
reconstructs `scores.csv`'s `score_raw`, within floating-point rounding. The bias
is not distributed across voxels. These are raw decision-value contributions,
before scalar score normalization or probability/percentile feedback.

Only ready volumes with finite scores produce maps; warmup and scoring-disabled
runs do not. This covers the standard volumetric linear decoder and the active
Olga direct scorer, not the separate PCA scoring pathway. Set the option to
`false` to skip the additional files. Their calculation and uncompressed NIfTI
write are included in the existing `SCORE` stage timing.

## Processing provenance and reuse

The response policy is `final_analysis_grid`, version 1. Metadata records the
order `spatial_transform`, `smoothing`, `denoising`, `normalization` and nuisance
source `native_unsmoothed_mc`. The former `source_grid_cubic_halo` policy remains
recognizable as historical provenance and is not treated as equivalent.

**Use a fresh run output directory and rebuild RS score/PCA calibration assets
after this order change.** Scalar score references and PCA assets compare
normalization, coverage and processing-mask provenance. Missing or incompatible
modern provenance is rejected. Source mask paths, hashes and native nuisance
padding widths are acquisition-specific and need not match across RS and NF.
Compatible unchanged anatomical registrations can still be reused.

Lanczos remains opt-in through `gpu_resampler_interpolation: "lanczos"`; the
default is `"linear"`. Lanczos requires the CUDA/Triton backend. Temporal work
now scales with the selected final-grid response voxels, so a finer T1/MNI grid
can cost more than native EPI processing.

The earlier 2026-11-02 Linux/RTX 4090 margin comparison used the historical
temporal-before-spatial order. Its three- versus four-voxel margin result does
not validate the current order. The runtime tests separately exercise current
final-grid processing with actual smoothing, GLM and normalization; scanner
replay, registration quality and decoder recalibration remain distinct checks.
