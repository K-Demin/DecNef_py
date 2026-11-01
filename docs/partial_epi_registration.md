# Preparing EPI-to-anatomy registration

Preparation uses one T1-to-EPI registration recipe for both full-brain and
partial-coverage acquisitions. ANTs fixes the corrected session EPI reference
and moves the anatomical T1. Header-based identity initialization is followed
by Translation and Rigid optimization, with optional SyN refinement.

For new full-brain preparations, this changes the previous recipe: there is no
Affine stage, and SyN is now disabled by default. Existing prepared transforms
remain usable by realtime consumers. Registration does not reconstruct cortex
outside the acquired field of view.

## Settings

| Setting | Default | Choices and effect |
| --- | --- | --- |
| `registration_initialization` | `"identity"` | `"identity"` uses the image headers as the starting alignment. `"center_of_mass"` explicitly aligns intensity centroids first. `"auto"` is a compatibility alias for `"identity"`. |
| `registration_syn` | `false` | Boolean; `true` adds SyN nonlinear refinement after Translation and Rigid. |

Identity initialization relies on the image headers already placing EPI and T1
reasonably near one another. It avoids shifting a partial slab to match the
center of mass of the entire T1. If headers do not provide a suitable starting
point, inspect the images and choose initialization explicitly. These settings
do not establish that registration succeeded; inspect the resulting alignment.

All three preparation entry points load a profile only when given
`--settings-file /path/to/participant_settings.json`. Without this argument they
use built-in defaults; they do not automatically discover `rt_settings.json`.
These fields can be included in a complete participant profile:

```json
{
  "registration_initialization": "identity",
  "registration_syn": false
}
```

Explicit CLI options override the corresponding profile values. Omitted options
preserve the profile values. Use `--registration-initialization center_of_mass`,
`--registration-syn`, or `--no-registration-syn` as needed. The effective
registration settings are saved in the prepared session's `config.json`.

There is no active `registration_mode` setting or `--registration-mode` flag.
For old JSON profiles, known `registration_mode` values (`"full_brain"` and
`"partial_epi"`) are accepted with a warning and ignored. The old JSON key
`partial_epi_syn` maps to `registration_syn` with a warning; supplying conflicting
values for both keys is rejected. The old `--partial-epi-syn` CLI flag is replaced
by `--registration-syn`.

## Registration and transform directions

For either acquisition coverage, ANTs fixes the corrected session EPI reference
and moves `anat/T1_brain.nii` (the SynthStrip brain extracted from the
N4-corrected T1).
The metric is mutual information, restricted by the EPI brain mask. Translation
uses a single resolution; Rigid uses shrink factors `2x2x1`. Optional SyN uses
`2x1`. The schedules follow the [ANTs-linked partial-slab example](https://github.com/ntustison/PartialSlabEpiT1ImageRegistration).
Our default initialization uses identity, rather than the example's
center of mass, to retain a reasonable alignment already provided by the headers.
Translation and rotation are still optimized after identity initialization.

The files in `func/trans` retain their existing semantic directions:

| File | Meaning and consumers |
| --- | --- |
| `epi2t1_Composite.h5` | EPI → T1. BOLD resampling and EPI-mask mapping. Copied from the native **inverse** transform. |
| `epi2t1_InverseComposite.h5` | T1 → EPI. WM, ventricle, cortical and custom T1 masks. Copied from the native **forward** transform. |
| `epi2t1_Warped.nii` | Corrected EPI reference on the T1 grid; inspect against `anat/T1_N4.nii`. |
| `epi2t1_InverseWarped.nii` | T1 on the EPI grid; inspect against `epi_unwarped_mean.nii`. |

New native ANTs outputs are retained under `t1_to_epi_*`.
`registration.json` version 2 records their relationship to the canonical filenames.
For MNI BOLD output, `antsApplyTransforms` receives
`-t warp_T1_to_MNI_synth.nii -t epi2t1_Composite.h5`: EPI → T1, then T1 → MNI.
The `-r` reference selects the output grid. Tissue masks retain nearest-neighbor
interpolation. CPU and GPU paths use the same semantic transform chain.

## Preparation commands

The same settings-file loading and registration CLI overrides apply to all three
entry points:

| Command | Role |
| --- | --- |
| `run_preproc.py` | Prepare the session reference, transforms and masks directly. |
| `rs_realtime_parallel.py` | Prepare if needed, then run resting-state acquisition and calibration. |
| `rs_pca_score_all_rois.py` | Prepare if needed, then process resting-state data and score existing ROI PCA models. |

Replace example identifiers, paths and scanner series numbers with the actual
values. Direct preparation:

```bash
python run_preproc.py \
  --sub 99988 --day session-10 \
  --base-data /SSD2/DecNef_py/data \
  --incoming-root /path/to/session_dicoms \
  --struct-block 1 --ap-block 2 --pa-block 3 --epi-block 4 \
  --settings-file /path/to/participant_settings.json
```

Automatic preparation from the resting-state launcher:

```bash
python rs_realtime_parallel.py \
  --sub 99988 --day session-10 --run 4 \
  --base-data /SSD2/DecNef_py/data \
  --incoming-root /path/to/session_dicoms \
  --struct-block 1 --ap-block 2 --pa-block 3 \
  --settings-file /path/to/participant_settings.json
```

Resting-state scoring with existing ROI PCA models:

```bash
python rs_pca_score_all_rois.py \
  --sub 99988 --day session-10 --run 4 \
  --base-data /SSD2/DecNef_py/data \
  --incoming-root /path/to/session_dicoms \
  --struct-block 1 --ap-block 2 --pa-block 3 \
  --pca-root /path/to/existing_pca_models \
  --settings-file /path/to/participant_settings.json
```

All three commands check registration-cache compatibility before scan staging or
costly preparation. Both automatic launchers forward the resolved choices to
`run_preproc.py`. Neurofeedback, PsychoPy, robot and BrainPong launchers consume
the prepared maps; they need no separate partial-coverage registration flags.

## Existing prepared sessions

The registration cache records the selected recipe and its inputs and outputs.
An incompatible request is refused before scan staging, costly preparation, or
rewriting `config.json`. The automatic launchers also check before waiting for
reference scans; the RS launcher checks before starting its display. The check
does not delete or overwrite an incompatible preparation.

New registrations use `registration.json` version 2. A version 1 partial-EPI
preparation can be reused when its resolved initialization, SyN choice, input
identities and canonical output identities match. Its existing files and marker
are retained without renaming or refitting. Version 1 full-brain registrations
and unversioned transform caches cannot certify the new recipe and are rejected
by preparation; use a fresh preparation/session directory for a new fit.

New fits are published only after both transform files and diagnostic image
grids are checked. `registration_pending.json` marks an interrupted preparation;
it must not be treated as a finished registration. GPU grids from older code
are rebuilt once because they lack transform provenance. Subsequent reuse checks
both image geometries and the contents of the transform files.

Existing realtime consumers can still use unchanged legacy full-brain
preparations; the new preparation checks do not refit them automatically.
New runs record spatial transforms and enabled nuisance-mask inputs; an old run
directory cannot be reused with a newly versioned registration. The per-volume coverage update also
requires a fresh run output directory for runs processed before that update;
this alone does not require rebuilding an unchanged registration.

When changing an existing preparation to the new recipe or incompatible settings,
use a fresh preparation/session directory. Rebuild the corrected reference,
registration and derived masks together, then point the intended runs at that
prepared session. Do not mix its transforms with an old reference or manually
remove the cache record to bypass the check.

This is separate from `truncate_t1_to_epi_fov`, which crops a PCA output reference
after an EPI-to-T1 transform already exists. Cropping does not fix registration
and does not restore missing decoder features. Check the transformed EPI against
the acquired portion of T1, the WM/ventricle mask overlap, and decoder coverage
before using the prepared data for feedback.

Command construction and cache handling are covered by automated tests. Native
ANTs anatomical alignment has not been validated in this Windows development
environment; inspect registration QC on representative full-brain and partial
acquisitions before relying on the new recipe.

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
