# Preparing partial-coverage EPI

Partial-brain registration is an explicit preparation option. Existing commands
continue to use `full_brain`; the default registration recipe is unchanged.
Use `partial_epi` when the reference EPI covers a slab rather than the full
brain. It estimates Translation and Rigid transforms, with an optional SyN
refinement. It does not reconstruct cortex outside the acquired field of view.

## Settings

| Setting | Default | Choices and effect |
| --- | --- | --- |
| `registration_mode` | `"full_brain"` | `"full_brain"` or `"partial_epi"`; selects the EPI-to-T1 preparation recipe. |
| `registration_initialization` | `"auto"` | `"auto"`, `"identity"` or `"center_of_mass"`. Auto uses center of mass for full-brain EPI and header-based identity for partial EPI. |
| `partial_epi_syn` | `false` | Boolean; enables SyN refinement after the partial-mode linear stages. `true` requires `registration_mode="partial_epi"`. |

Identity initialization relies on the image headers already placing EPI and T1
reasonably near one another. It avoids shifting a partial slab to match the
center of mass of the entire T1. If headers do not provide a suitable starting
point, inspect the images and choose initialization explicitly. These settings
do not establish that registration succeeded; inspect the resulting alignment.

## Registration and transform directions

In `partial_epi` mode, ANTs fixes the corrected session EPI reference and moves
`anat/T1_brain.nii` (the SynthStrip brain extracted from the N4-corrected T1).
The metric is mutual information, restricted by the EPI brain mask. Translation
uses a single resolution; Rigid uses shrink factors `2x2x1`. Optional SyN uses
`2x1`. The schedules follow the [ANTs-linked partial-slab example](https://github.com/ntustison/PartialSlabEpiT1ImageRegistration).
Our `auto` initialization uses identity for partial EPI, rather than the example's
center of mass, to retain a reasonable alignment already provided by the headers.
Translation and rotation are still optimized after identity initialization.

The files in `func/trans` retain the same meaning for both registration modes:

| File | Meaning and consumers |
| --- | --- |
| `epi2t1_Composite.h5` | EPI → T1. BOLD resampling and EPI-mask mapping. Copied from the native **inverse** in partial mode. |
| `epi2t1_InverseComposite.h5` | T1 → EPI. WM, ventricle, cortical and custom T1 masks. Copied from the native **forward** in partial mode. |
| `epi2t1_Warped.nii` | Corrected EPI reference on the T1 grid; inspect against `anat/T1_N4.nii`. |
| `epi2t1_InverseWarped.nii` | T1 on the EPI grid; inspect against `epi_unwarped_mean.nii`. |

Native partial-mode ANTs outputs are also retained under `t1_to_partial_epi_*`.
`registration.json` records their relationship to the canonical filenames.
For MNI BOLD output, `antsApplyTransforms` receives
`-t warp_T1_to_MNI_synth.nii -t epi2t1_Composite.h5`: EPI → T1, then T1 → MNI.
The `-r` reference selects the output grid. Tissue masks retain nearest-neighbor
interpolation. CPU and GPU paths use the same semantic transform chain.

## Direct preparation

`run_preproc.py` accepts the three equivalent CLI flags. Example identifiers
and paths below must be replaced with the actual scanner series:

```bash
python run_preproc.py \
  --sub 99988 --day partial-session \
  --base-data /SSD2/DecNef_py/data \
  --incoming-root /path/to/session_dicoms \
  --struct-block 1 --ap-block 2 --pa-block 3 --epi-block 4 \
  --registration-mode partial_epi
```

This uses `auto` initialization, which resolves to identity in partial mode,
and leaves SyN disabled. Add `--registration-initialization center_of_mass` to
request that initialization explicitly, or `--partial-epi-syn` to enable SyN.
`--no-partial-epi-syn` explicitly disables the refinement.

The direct preparation command does not load `rt_settings.json`. It records
its selected settings in the generated session `config.json`.

## Automatic preparation from the resting-state launcher

Add these fields to the runtime JSON passed to `rs_realtime_parallel.py`:

```json
{
  "registration_mode": "partial_epi",
  "registration_initialization": "auto",
  "partial_epi_syn": false
}
```

These fields can be part of a complete existing participant settings file.
The launcher loads that file, applies any explicitly supplied registration
CLI overrides, checks cache compatibility, and forwards the effective values
to `run_preproc.py`. An omitted registration flag does not override the JSON.
For example:

```bash
python rs_realtime_parallel.py \
  --sub 99988 --day partial-session --run 4 \
  --base-data /SSD2/DecNef_py/data \
  --incoming-root /path/to/session_dicoms \
  --struct-block 1 --ap-block 2 --pa-block 3 \
  --settings-file /path/to/participant_settings.json
```

## Existing prepared sessions

The registration cache records the selected recipe and its inputs and outputs.
An incompatible request is refused before scan staging, costly preparation, or
rewriting `config.json`. The RS launcher also checks before starting its display
or waiting for reference scans. It does not silently reuse the other mode's
transforms or delete the previous preparation.

New fits are published only after both transform files and diagnostic image
grids are checked. `registration_pending.json` marks an interrupted preparation;
it must not be treated as a finished registration. GPU grids from older code
are rebuilt once because they lack transform provenance. Subsequent reuse checks
both image geometries and the contents of the transform files.

Unchanged legacy full-brain preparations remain usable. New runs record spatial
transforms and enabled nuisance-mask inputs; an old run directory cannot be
reused with a newly versioned registration. The per-volume coverage update also
requires a fresh run output directory for runs processed before that update;
this alone does not require rebuilding an unchanged registration.

When changing an existing preparation to another mode or incompatible settings,
use a fresh preparation/session directory. Rebuild the corrected reference,
registration and derived masks together, then point the intended runs at that
prepared session. Do not mix its transforms with an old reference or manually
remove the cache record to bypass the check.

This is separate from `truncate_t1_to_epi_fov`, which crops a PCA output reference
after an EPI-to-T1 transform already exists. Cropping does not fix registration
and does not restore missing decoder features. Check the transformed EPI against
the acquired portion of T1, the WM/ventricle mask overlap, and decoder coverage
before using the prepared data for feedback.

## Coverage after motion correction

The anatomical analysis mask describes the selected brain tissue. It is separate
from each volume's geometric field of view. Motion can move an acquisition edge
inside that fixed mask, especially with partial coverage.

Each realtime volume now reapplies its already estimated motion to an all-ones
support image using AFNI linear interpolation, without another registration.
Only fully supported samples are retained (`support >= 1 - 1e-5`). Support is
mapped through the same EPI → T1/MNI transform chain and intersected with the
analysis mask for smoothing. The Gaussian numerator and denominator use this
intersection, and unavailable output voxels stay zero. Valid zero and negative
residuals are retained. When coverage includes the full analysis mask, the cached
smoothing denominator is reused.

Coverage images are saved under the run's `coverage/` directory as
`vol_00001_epi.nii` and, for transformed output, `vol_00001_t1.nii` or
`vol_00001_mni.nii`. They contain binary validity, not BOLD intensities. The
`mc_fov_v1` policy is recorded in run provenance to prevent mixing old and new
processing. Coverage work is included in the existing MC and TRANS speed timings.
The CPU transform path performs an additional mask warp per volume; the GPU path
uses its existing sampling grid.

This support represents the corrected input array's geometric extent through MC
and final resampling. It is not per-volume brain segmentation or detection of
signal dropout; it does not detect PyHySCO border extrapolation or certify every
tap of the wider heptic BOLD interpolation kernel. It also does not implement
missing-observation fitting in the cumulative GLM: missing samples can still
affect calibration means, nuisance estimates and earlier fitted coefficients.
Unsupported current residuals are zeroed before final spatial interpolation.
Decoder weights are unchanged; unavailable features are not recovered.

## Online volume viewer

With the existing volume streamer enabled, the first row continues to show
`volume_stream_kind`. The second row shows the same volume's T1-stage image,
after spatial resampling and coverage masking, before smoothing. It is explicitly
unavailable for runs that do not produce a T1 intermediate (EPI or combined MNI
processing); displaying QC does not add a T1 registration or resampling step.
Each row uses its own geometry, canonical orientation and display contrast.
