# Reusable offline preparation and session alignment

The offline bundle contains a corrected EPI reference, its EPI↔T1 transforms,
T1 anatomy and tissue masks, the T1→MNI transform, and the named MNI template.
It is copied into a new directory with an `assets.json` manifest. Existing
bundles are not overwritten. It contains no fieldmap for a future acquisition,
resting-state score distribution, or fitted temporal-regression state.
An optional `epi_registration_reference` image supplies an enhanced fitting
reference on the same grid as `epi_reference`; it does not change the bundle's
coordinate system or original reference image.

At the scanner, a short current-session reference is aligned to the offline EPI
reference. Online volumes are unwarped with their current acquisition's fieldmap,
motion-corrected to the session reference, denoised, and mapped through the session
EPI→offline EPI→T1→MNI chain as required by the selected analysis space. The
session's nuisance masks are mapped onto its own EPI grid. The anatomical
registration is therefore reusable without requiring identical slab prescriptions.

## Option 1: import fMRIPrep outputs

Use matching assets from one subject and anatomical reference. The importer
accepts explicit paths; it does not choose among several runs or spaces.

For fMRIPrep 23.2.1 (the naming present in the supplied data), the useful filename patterns are:

| CLI input | Typical derivative path beneath `sub-<id>/` |
| --- | --- |
| `--epi-reference` | `func/sub-<id>_<run-specifiers>_desc-coreg_boldref.nii.gz` |
| `--epi-mask` | Binary brain mask on that native EPI reference grid; supply an existing matching mask or prepare it using the inverse EPI-to-T1 affine and nearest-neighbor interpolation. |
| `--epi-to-t1` | `func/sub-<id>_<run-specifiers>_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt` |
| `--t1` | `anat/sub-<id>_desc-preproc_T1w.nii.gz` |
| `--t1-mask` | `anat/sub-<id>_desc-brain_mask.nii.gz` |
| `--wm-probseg` | `anat/sub-<id>_label-WM_probseg.nii.gz` |
| `--aseg` | `func/sub-<id>_<run-specifiers>_space-T1w_desc-aseg_dseg.nii.gz` |
| `--t1-to-mni` | `anat/sub-<id>_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5` |
| `--mni-template` | The T1-weighted template image for that exact named MNI space, supplied from the template installation. |

Include `ses-*` folders/entities where applicable. `space-T1w_desc-aparcaseg_dseg`
is also suitable for `--aseg`. These segmentations use FreeSurfer labels; the
three-tissue anatomical `dseg` is not an aseg substitute. See the
[versioned fMRIPrep output documentation](https://fmriprep.org/en/23.2.1/outputs.html).

The table lists the nine image/transform inputs required by the current importer.
It also requires `--subject`, `--mni-space`, and a new `--output` directory.
An image stored as `(X, Y, Z, 1)` is accepted as a single volume and written into
the bundle as `(X, Y, Z)`, preserving voxel values and spatial forms. Actual
multivolume references are rejected; the importer does not select or average
time points. The original input files remain intact.
The current bundle schema requires the MNI inputs even when online
`analysis_space` is `t1`; those transforms are then unused for the final BOLD
output. The decoder NIfTI and any decoder bias file are selected separately by
the realtime launcher and are not part of this spatial bundle.

The person preparing the bundle selects a matching native EPI reference and
coregistration transform, the corresponding T1 and tissue images, and a matching
MNI template/transform. The importer checks file structure and grids; it does
not infer the anatomical origin of arbitrary input files.

Newer fMRIPrep output naming uses `space-run_boldref.nii.gz` with
`from-run_to-T1w_mode-image_desc-coreg_xfm.txt`. Session/subject template boldrefs
instead require their matching `from-session_to-T1w` or `from-subject_to-T1w`
transform. Match the transform's source space to the chosen reference; see the
[current reference-space definitions](https://fmriprep.org/en/latest/outputs.html).

`--epi-reference` must be a **distortion-corrected reference in its EPI reference
space**, not a volume already resampled into T1 or MNI. Confirm susceptibility
correction in the fMRIPrep report; the importer cannot infer it from intensities.
Supply its matching binary `--epi-mask` on exactly the same shape and affine.
Do not substitute an HMC-only reference merely because its filename contains
`boldref`.

The fMRIPrep importer records the supplied, already enhanced reference as the
registration reference too. It does not run N4 again on that image. Older bundles
with recorded fMRIPrep-import provenance are likewise recognized during session
preparation, even if they do not yet contain the optional registration-reference
entry. The filenames do not establish this provenance by themselves.

`--aseg` and `--wm-probseg` must already share the T1's physical coordinate space.
Different voxel sizes are allowed: the importer resamples labels with nearest
neighbor interpolation and probabilities linearly onto the T1 grid. Raw
`sourcedata/freesurfer/.../mri/aseg.mgz` may require a separate fsnative→T1w
transform; changing its header or resampling its grid alone is insufficient.

Example with explicit paths:

```bash
conda activate rt_pipe
python prepare_offline.py import-fmriprep \
  --subject 99988 \
  --epi-reference /derivatives/sub-99988/func/sub-99988_task-rest_desc-coreg_boldref.nii.gz \
  --epi-mask /inputs/matching_native_epi_brain_mask.nii.gz \
  --epi-to-t1 /derivatives/sub-99988/func/sub-99988_task-rest_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt \
  --t1 /derivatives/sub-99988/anat/sub-99988_desc-preproc_T1w.nii.gz \
  --t1-mask /derivatives/sub-99988/anat/sub-99988_desc-brain_mask.nii.gz \
  --wm-probseg /derivatives/sub-99988/anat/sub-99988_label-WM_probseg.nii.gz \
  --aseg /derivatives/sub-99988/func/sub-99988_task-rest_space-T1w_desc-aseg_dseg.nii.gz \
  --t1-to-mni /derivatives/sub-99988/anat/sub-99988_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5 \
  --mni-template /templates/MNI152NLin2009cAsym_T1w.nii.gz \
  --mni-space MNI152NLin2009cAsym \
  --output /offline/sub-99988/reference-v1
```

The importer requires `antsApplyTransforms` when converting ITK affine text into
canonical HDF5 transforms. It checks that the text contains one invertible 3D
ITK affine, then obtains its inverse with `[affine.txt,1]` on the EPI reference
grid. FSL matrices and a series of HMC matrices are not accepted as substitutes.
For an existing `--epi-to-t1 transform.h5`, also provide the matching inverse
with `--t1-to-epi inverse.h5`; nonlinear composites are not inverted automatically.

WM is thresholded at probability ≥0.95. Ventricles use labels 4, 43, 5, 44, 14 and
15. The cortical mask uses labels 3/42 and 1000–2999 when present. Derived masks
are intersected with the T1 brain mask; an empty WM/ventricle mask stops import.
Input paths and SHA256 identities are recorded as provenance.

The T1→MNI transform, supplied template and decoder must refer to the **same
named template space**. `MNI152NLin2009cAsym` and `MNI152NLin6Asym` are not
interchangeable. The final resampling still uses the decoder's voxel grid;
recording `--mni-space` does not convert an incorrectly selected template.

## Option 2: export an existing prepared DecNef day

```bash
python prepare_offline.py export-prepared \
  --subject 99988 \
  --prepared-day /data/sub-99988/offline-day \
  --mni-template /templates/MNI152NLin2009cAsym_T1w.nii.gz \
  --mni-space MNI152NLin2009cAsym \
  --output /offline/sub-99988/reference-v1
```

This copies `func/trans/epi_unwarped_mean.nii`, `epi_mask_mean.nii`, canonical
`epi2t1_Composite.h5` and `epi2t1_InverseComposite.h5`, the N4 T1 and masks, and
the corresponding T1→MNI transform. It handles `.nii.gz` images too. Anatomy
comes from `func/trans/offline_anat` for imported preparations, otherwise from
the subject's `anat` folder. WM/ventricle masks are resampled onto the T1 grid;
a cortical mask is included when an existing mask or FastSurfer parcellation
is available. Explicit template arguments also support preparations created
with `--no-save-config`. Interrupted registrations cannot be exported.
If preparation produced an N4 registration-reference copy, export includes it
as `epi_registration_reference` with the original reference retained separately.
Older native preparations without that copy remain usable: when N4 is enabled,
session preparation enhances a local copy of their anchor before fitting the
EPI bridge, leaving the original preparation/bundle intact.

## Preparing and starting a new session

Use the same profile in preparation and the intended realtime launcher:

```json
{
  "offline_assets": "/offline/sub-99988/reference-v1/assets.json",
  "registration_initialization": "identity",
  "registration_direction": "epi_to_t1",
  "registration_syn": true,
  "epi_reference_n4": true,
  "registration_qc_policy": null
}
```

`registration_direction` selects the native EPI↔T1 fit when creating anatomical
registration locally. Both directions use Rigid→Affine→SyN by default. Imported
anatomical transforms retain their recorded meanings; selecting a direction does
not refit them. Session EPI↔reference EPI alignment uses its separate rigid recipe.

`epi_reference_n4` defaults to `true` and must be a JSON boolean. It applies N4
to a separate current-session registration reference after unwarping, MC averaging
and SynthStrip masking. The original mean remains the MC target and incoming
BOLD intensities are unchanged. For the bridge anchor, a supplied enhanced
reference is reused; otherwise a local N4 copy is created, except for recognized
fMRIPrep imports whose reference is already enhanced. No N4 runs per TR.
Use `false`, or the explicit CLI override `--no-epi-reference-n4`, to fit the
original references instead. `--epi-reference-n4` enables it explicitly.

Preparation saves `epi_registration_reference.nii` with its `.json` provenance
and, for an enhanced bridge anchor, `reference_epi_registration.nii` in the
transform folder. Earlier isolated Linux measurements took about 4.1 seconds
per N4 reference; an old unenhanced anchor can require another correction.
Those measurements are not a full new-session runtime benchmark. Changing
the setting requires a fresh preparation/session directory and new QC approval.

Direct preparation can consume the bundle:

```bash
python run_preproc.py --sub 99988 --day session-10 \
  --base-data /data --incoming-root /scanner/session-10 \
  --ap-block 2 --pa-block 3 --epi-block 4 \
  --settings-file /profiles/sub-99988.json \
  --offline-assets /offline/sub-99988/reference-v1/assets.json
```

To initialize the current session before neurofeedback:

```bash
python prepare_session.py --sub 99988 --day session-10 --reference-run 4 \
  --base-data /data --incoming-root /scanner/session-10 \
  --ap-block 2 --pa-block 3 --settings-file /profiles/sub-99988.json
```

Reference scans 11–30 form the current-session reference. Use a separate reference
acquisition before the timed NF task when visual QC is required. If preparation
uses the first NF run itself, startup waits for preparation/QC; the processor then
handles existing volumes using its normal skip/warmup settings. It does not
automatically start at scan 31. Session assets are written to `func/session_trans`;
later runs reuse the accepted alignment.
The session fieldmap pair must match the current acquisition. The bundle does
not replace that pair.

Inspect the generated `registration_qc_*.png/json` reports. With no QC policy,
manual review is required:

```bash
python check_registration.py --trans /data/sub-99988/session-10/func/session_trans \
  --approve --reviewer YOUR_NAME
```

Canonical warped images and QC use original intensities through the estimated
maps. N4 fitting copies are saved separately, so correcting intensity shading
alone cannot be mistaken for improved alignment in those comparisons.

A supplied `--registration-qc-policy policy.json` applies the configured QC
criteria; it is not a universal accuracy guarantee. Import/export validates
files, transform declarations and image grids, but does not establish that an
anatomical fit converged correctly. Review alignment and decoder coverage on
representative data before using the bundle for feedback. Native ANTs transform
composition was checked numerically, and selected real-data alignment and GPU
application comparisons have passed. A complete imported bundle through the
first live NF run has not yet been validated on the affected acquisitions.

Spatial preparation can use a short reference acquisition without a full
resting-state run. Whether feedback can omit resting-state **score calibration**
depends on its scoring mode and available calibration statistics. Offline spatial
assets do not supply those statistics or eliminate online normalization/GLM warmup.

Runtime normalization defaults to `voxel_norm_mode: "zscore"`,
`voxel_norm_reference: "fixed"`, and `voxel_norm_ref_volumes: 30`. After the
native-unit cumulative GLM becomes ready, the next 30 emitted residuals establish
a fixed voxel-wise mean/sample SD; scoring begins with the subsequent normalized
volume. This occurs in the session EPI grid before spatial transforms and
smoothing. A short spatial reference acquisition does not replace this runtime
calibration. `mean` is also available and divides residuals by a reference mean
of raw MC intensities; `running` updates statistics from prior observations.
Keep the runtime settings and trained decoder/PCA feature units consistent.
