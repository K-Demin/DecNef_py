# Brain selection and the processing mask

The pipeline keeps the intended brain region separate from the region processed
before spatial interpolation. Denoising only inside a tight brain mask would
replace neighbouring measured signals with zeros before Lanczos or Linear
interpolation can use them.

## Automatic construction

Preparation supplies the session EPI brain mask, normally SynthStrip's
`epi_mask_mean.nii` from the unwarped, motion-corrected reference mean. Review
that anatomical mask against the corrected EPI and aligned anatomy: adding a
processing margin does not correct a brain mask that excludes cortex.

At run initialization, the pipeline makes a separate processing mask on the same
EPI grid. The margin follows the selected spatial interpolation automatically;
there is no additional JSON setting.

| Spatial operation after temporal processing | Processing margin |
| --- | --- |
| Lanczos-3 | Three source voxels in each direction along each axis |
| Linear | One source voxel in each direction along each axis |
| Native EPI output with no spatial mapping | None |

These are **cubic** margins, clipped to the source array. For 1.5 mm EPI voxels,
the Lanczos margin is 4.5 mm along each axis. An EPI-to-reference-EPI bridge is a
spatial mapping and receives a margin too. The kernel-based construction covers
the interpolation neighbours of output points whose nearest-neighbour source
centre is inside the anatomical mask. It does not assert that every selected
voxel was acquired or has a valid temporal estimate.

Both cumulative-GLM denoising and subsequent voxel normalization use the same
processing mask. Intermediate `reg/` and `norm/` files can therefore contain
processed values just outside the anatomical brain mask. Those values supply
interpolation neighbours; they do not enlarge the final analysis region.

The original anatomical mask continues to define whole-brain selection, global
signal sampling and brain-based QC. WM/ventricle nuisance masks, configured
DVARS masks, and cortical/custom analysis selections are unchanged. This does
not rerun SynthStrip or change registration masks or fitted transforms.

## Final output and missing samples

The anatomical brain mask is mapped to the final grid with nearest-neighbour
interpolation. Per-volume acquisition and usable-processing support are mapped
separately with **Linear** interpolation and thresholded at `1 - 1e-5`, for both
Linear and Lanczos signal interpolation. Final usable support intersects the
mapped brain mask. It no longer requires every Lanczos neighbour to lie inside
the anatomical brain mask.

The processing margin does not recover missing slices, voxels rejected by the
GLM's complete-history policy, or voxels without valid normalization statistics.
Lanczos can still be affected by unavailable neighbours near those boundaries;
the availability rule is not a guarantee that every Lanczos contributor exists.
The established per-volume support rules and `coverage_rt.csv` remain relevant.

## Saved files and restarting

Each run saves these files under `func/<run>/processing_mask/`:

| File | Meaning |
| --- | --- |
| `brain_epi.nii` | Original anatomical selection on the session EPI grid |
| `processing_epi.nii` | Expanded region used by denoising and normalization |
| `brain_final.nii` | Anatomical selection mapped onto the final output grid |
| `policy.json` | Policy/version, interpolation, margin, source identity and voxel counts |

Processing policy is also recorded in run metadata and
`preprocessing_order.json`. **Use a fresh run output directory:** the new margin
changes processed values, including potentially the default Linear path, and
old temporal output/calibration must not be mixed with the new run. Compatible
unchanged anatomical registrations can still be reused.

Scalar RS score references and PCA assets/references record and check the
normalization, coverage policy and mask policy/version/interpolation. PCA also
records whether its source was raw MC or normalized data. Missing or conflicting
modern provenance requires rebuilding/reprocessing the reference; it is not
silently accepted because voxel grids happen to match. Acquisition-specific
mask paths, hashes and padding widths are not required to match across RS/NF.

Lanczos remains opt-in through `gpu_resampler_interpolation: "lanczos"`; the
default stays `"linear"`. Lanczos requires the CUDA/Triton backend. The margin
is constructed once, while processing additional voxels increases temporal
processing work.

Kernel-footprint and runtime tests cover the padding construction, actual GLM
and normalization, final masks, missing slices, original-volume outputs and QC.

On 2026-11-02, an isolated Linux/RTX 4090 control replay used 48 saved Olga MC
volumes (61–108), native-unit CUDA GLM, fixed post-GLM z-scoring and GPU Lanczos.
Three- and four-voxel margins produced identical final support across 17 ready
volumes; their maximum final-value difference was 0.00007424 in normalized
units. All finite-value, final-brain and mapped-coverage checks passed. The
anatomical mask contained 375,245 voxels; the three-voxel processing mask
contained 478,804 (27.6% more), without enlarging final anatomical selection.

This was a controlled restart using the same original-brain GS regressor and
polynomial order 1, not the original study's complete temporal model or an NF
score replay. It used historical PA-reference MC data with one fixed transform
from the earlier AP-reference comparison, so it does not validate registration
accuracy. A fresh end-to-end scanner replay, decoder calibration and complete
runtime measurement remain separate checks. Reproducible script and results:
`output/processing_mask_integration_20261102/validate_native.py` and
`output/processing_mask_integration_20261102/native_results/results.json`.
