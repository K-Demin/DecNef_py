# Unwarp before motion correction

The pipeline order is `unwarp_then_mc_v1`.

1. Select the run's calibration using the existing `--ap-block` and `--pa-block` options. With neither supplied, use the common fieldmap directory.
2. Average each raw AP/PA series independently. Do not motion-correct either series, register them to each other, or register them to the session reference.
3. Estimate PyHySCO's field from those native means.
4. Apply that field to each incoming raw BOLD volume, using the configured AP or PA polarity.
5. Motion-correct the unwarped volume to the fixed, corrected session reference (`func/trans/epi_unwarped_mean.nii[.gz]`).
6. Compute motion/FD and DVARS, regress nuisance signals, and use the existing EPI-to-T1/MNI transforms and decoder.

For a new session, offline preparation also unwarps before MC. Its first corrected BOLD volume supplies the initial MC target; the resulting aligned mean becomes the fixed online reference. Subsequent runs retain that reference. Changing the calibration pair does not rebuild anatomical segmentation, nuisance masks, or spatial transforms.

## Files and migration

Native calibration products live inside the selected pair folder under `native_unwarp_v1/`, with a provenance manifest. Old `pyhysco_epi-*`, AP/PA MC products, and legacy fields are never used as native fields. Raw `AP.nii[.gz]` and `PA.nii[.gz]` must be available to rebuild calibration.

Existing corrected session references and their matching masks/transforms can be reused for online processing. The former distorted `rt_ref_epi.nii` is no longer the online motion target. New preparation also writes corrected data under legacy reference filenames for compatibility.

Use a fresh run output directory for replaying old processed runs. The pipeline rejects mixing old MC-first outputs, changed calibration, or a changed reference with a run's new products. To regenerate offline reference assets, prepare in a fresh day directory; do not overwrite a previous reference while retaining its transforms.

Output names remain compatible with existing consumers:

| Product | Contents |
| --- | --- |
| `raw/vol_*.nii` | Raw distorted BOLD |
| `unwarped/vol_*_uw_native.nii` | Unwarped BOLD before MC, when intermediate saving is enabled (also written by file fallback) |
| `mc/vol_*_mc.nii` | Unwarped and motion-corrected BOLD |
| `reg/vol_*_reg.nii` | Downstream denoised/normalized volume |

Fully corrected volumes are written once, in `mc/`. The `unwarped` PCA/stream option is a compatibility alias for this product; historical runs can still be read from their old `unwarped/vol_*_mc_uw.nii` paths. No duplicate is written for new runs. FD describes registration of corrected BOLD; DVARS uses corrected and aligned BOLD with the corrected reference mask.

## Geometry and supported correction

This path uses PyHySCO. The old ANTs AP-to-PA fallback is rejected explicitly. The AP/PA and BOLD acquisition grids must match, and the fixed reference must have the same grid prescription. Head motion within that grid is handled by MC after unwarping.

PyHySCO's voxel phase-encoding axis is separate from AP/PA polarity. The axis comes from `PhaseEncodingDirection` in calibration JSON; AP is the first PyHySCO input (+1), PA the second (-1). New calibration conversion retains JSON. Without calibration metadata, the historical voxel-axis-2 convention is used with a warning. Known mismatches in PE direction or readout time are rejected; no readout-time displacement scaling is performed.

The preloaded PyHySCO applier remains initialized once per run. Both its fast path and the file-based fallback now precede RTPSpy MC. Processing remains ordered for motion, regression, and scoring. This change does not claim lower latency or a different number of interpolation steps; verify timing and image alignment with a representative recorded run before live use.
