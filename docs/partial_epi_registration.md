# Registration, offline reuse and session alignment

Offline assets are optional. With no prior participant data, run `run_preproc.py`
(or the RS launcher) on the current T1/AP/PA/reference BOLD. With existing data,
import fMRIPrep or export a prepared DecNef day using `prepare_offline.py`.
See [the exact input files and commands](offline_preparation.md).

Both routes then use a current-session EPI-to-EPI alignment. A separate resting
state is not required for spatial preparation. Current-run denoising warmup and
feedback calibration are separate requirements.

## Anatomical recipe

| Setting | Default | Meaning |
| --- | --- | --- |
| registration_direction | epi_to_t1 | epi_to_t1 or t1_to_epi |
| registration_initialization | identity | Header coordinates; center_of_mass explicit; auto aliases identity |
| registration_syn | true | SyN after Rigid/Affine; false retains Rigid/Affine |
| epi_reference_n4 | true | N4-correct a separate EPI reference for registration fitting |
| offline_assets | null | Path to an imported/exported assets.json |
| registration_qc_policy | null | Study-validated numeric thresholds; absent means visual review |

Both estimation directions use the same restored stages:

| Stage | Iteration limits | Shrink factors | Smoothing (vox) |
| --- | --- | --- | --- |
| Rigid[0.1] | 1000x500x250x100 | 8x4x2x1 | 3x2x1x0 |
| Affine[0.1] | 1000x500x250x100 | 8x4x2x1 | 3x2x1x0 |
| SyN[0.1,1,0] | 100x70x50x20 | 6x4x2x1 | 3x2x1x0 |

Each stage uses MI, 32 bins, Regular sampling 0.2, tolerance 1e-7, window 15.
Affine/SyN restore June/September stage schedules, with the subsequently requested
Rigid stage first. These limits do not guarantee convergence. Reversing fixed
and moving changes physical pyramid resolution when voxel sizes differ.

Masks follow the corresponding images in either direction. Identity assumes
reasonable header overlap; a partial slab's centroid is not a full brain's
centroid, so choose centre-of-mass initialization deliberately.

All launchers accept the shared registration CLI flags and JSON profile fields.
Explicit CLI values override profiles. A profile containing
`"registration_syn": false` still disables SyN and must be updated if unwanted.
`--epi-reference-n4` and `--no-epi-reference-n4` explicitly override the JSON
`epi_reference_n4` boolean; omission preserves the profile.

```bash
python run_preproc.py --sub 001 --day 1 --base-data /data \
  --incoming-root /scanner/export --struct-block 1 \
  --ap-block 2 --pa-block 3 --epi-block 4 \
  --registration-direction t1_to_epi --registration-syn
```

`epi2t1_Composite.h5` always maps EPI images into T1; InverseComposite maps T1
into EPI. Warped is EPI on the T1 grid; InverseWarped is T1 on the EPI grid.
Native outputs are retained without changing their headers. Canonical warped
images and registration QC use the original image intensities through the fitted
transforms, even when N4-enhanced images were used to estimate them.
`registration.json` identifies the recipe and reference-enhancement settings; older short-recipe caches
are not silently reinterpreted as the restored nonlinear fit.

The reference preparation order is unwarping → motion correction → observed-sample
mean → SynthStrip brain mask → N4 registration copy → spatial fitting. With
`epi_reference_n4: true`, `epi_registration_reference.nii` and its `.json` sidecar
are saved in the transform folder. The original `epi_unwarped_mean.nii` remains
the motion-correction target. N4 does not run on each incoming volume, change the
existing brain mask, or replace the intensities entering temporal processing.

For the EPI bridge, the current-session fitting reference receives N4. An offline
bundle's optional `epi_registration_reference` is reused as the anchor; legacy
bundles identified as fMRIPrep imports reuse their already enhanced reference.
An older raw prepared anchor instead receives N4 in a local copy named
`reference_epi_registration.nii`. Input bundles are not modified. Setting
`epi_reference_n4: false` uses the original references for fitting.

Earlier isolated Linux experiments measured approximately 4.1 seconds for N4
per reference; an older raw anchor may require a second reference correction.
This is preparation cost, not per-TR work, and is not a new end-to-end runtime
benchmark. N4 improved some difficult fits but did not rescue every case; assess
the transformed original images and anatomy, not increased N4-image correlation
alone. This tested after-MC N4 placement is not an exact recreation of fMRIPrep's
reference workflow.

Changing this setting requires fresh anatomical/session preparation and renewed
QC. Existing transforms are not silently reused as if N4 had been enabled when
they were fitted; keep the previous preparation intact for comparison.

## Session preparation and gate

Scans 11-30 of a current reference acquisition are unwarped, motion corrected,
and averaged over acquired observations. Prefer a short reference block before
NF when visual review is required. The NF launcher can prepare from its initial
volumes if no session reference exists, but a required review stops startup;
do not begin the timed task before preparation/QC is finished.

```bash
python prepare_session.py --sub 001 --day 1 --base-data /data \
  --incoming-root /scanner/export --reference-run 4 \
  --ap-block 2 --pa-block 3 --settings-file study_settings.json
```

For offline reuse, add `--offline-assets /data/sub-001/offline/assets.json`
or put the same path in the settings file used by NF.

The bridge fixes current EPI and moves the prepared EPI, using their masks.
It is a separate rigid MI fit: 500x250x50 iterations, shrink 2x2x1, smoothing
2x1x0. The inverse supplies current EPI -> prepared EPI. Very small slabs can
have insufficient information; a current whole-brain EPI intermediate may help.

Files in `day/func/session_trans/`:

- `epi_unwarped_mean.nii`, `epi_mask_mean.nii`: current MC reference/grid.
- `epi_registration_reference.nii` and `.json`: N4 fitting copy and provenance when enabled.
- `reference_epi.nii`, `reference_epi_mask.nii`: stable prepared EPI grid.
- `reference_epi_registration.nii`: selected enhanced bridge anchor when N4 is enabled.
- `session_to_reference_*.h5`: EPI bridge and inverse.
- `epi2t1_*.h5`: composed current-EPI/T1 mappings.
- `offline_anat/`: exact anatomical images and masks.
- `registration_qc_*.json/.png`: EPI/EPI, EPI/T1 and T1/template diagnostics.
- `session_alignment.json`: frozen sources, settings, outputs and QC set.

Review overlays/reports, then record explicit review:

```bash
python check_registration.py --trans /data/sub-001/1/func/session_trans \
  --approve --reviewer YourName
```

The gate is checked before every NF run; unchanged approvals are reused.
Later runs retain the current-session MC target. Changed slab geometry requires
new preparation. Run-specific fieldmaps remain native to their acquisitions.

In EPI analysis mode, NF output is mapped to the stable prepared EPI grid and
saved in `epi_ref/`. EPI decoders must match that grid. T1/MNI output uses the
composed mapping directly. This keeps native-grid MC/denoising separate from
the decoder's coordinate system.

See [QC metrics and threshold rules](registration_qc.md). Pearson correlation is
for EPI/EPI or T1/template; EPI/T1 uses cross-contrast metrics and overlays.
June decoder-score reproducibility r=0.83 is not an alignment threshold.

## Coverage and normalization

Raw MC images can contain extracranial MRI signal. Final analysis outputs are
explicitly zeroed outside valid brain support. Negative and exactly-zero
denoised values inside that support remain valid observations. Intermediate
`reg/` and `norm/` images can retain a processing margin around the anatomical
brain mask to supply spatial-interpolation neighbours; see
[processing_masks.md](processing_masks.md).

Warmup retains raw units and reg_ready=false within the processing mask.
The cumulative GLM fits and emits residuals in native intensity units.
Voxel normalization follows it in the session EPI grid, before spatial
transforms and smoothing. The default is `voxel_norm_mode: "zscore"`,
`voxel_norm_reference: "fixed"`, and `voxel_norm_ref_volumes: 30`: collect
30 emitted regression-ready residuals, freeze their voxel-wise mean and sample
SD, and normalize from the next volume. This calibration is additional to GLM
startup. `running` uses prior observations and updates after the current volume.
The alternative `mean` is `100 * residual / raw_MC_reference_mean`; it never
divides by the residual mean. Mean mode calibrates raw MC observations during
GLM startup. Both modes exclude missing observations and do not clip values.
`reg/` contains native-unit output; `norm/` contains normalized EPI output and
is zero-filled until normalization is ready. Scores remain NaN until GLM and
normalization are ready; the status CSV records both flags separately.
Spatial interpolation/smoothing can change voxel variance afterward.

The cumulative GLM uses a conservative complete-history policy: a voxel that
loses acquisition coverage remains unavailable for that run. This avoids fitting
missing values as measured zeros, but can reduce coverage at moving slab edges.
Feature indexing and decoder weights stay fixed; unavailable entries are zero.
`coverage_rt.csv` reports acquired/usable brain and decoder fractions.
Without regression, calibrated voxels may reappear when coverage returns.
Empty enabled tissue masks make that timepoint unavailable for fitting.
DVARS uses the intersection of adjacent volumes' acquired support.

`coverage/` is geometric FOV, `support/` is actual analysis support, and
`support_original/` is written only when the optional original stream differs.
Smoothing normalizes within actual support. Equal support mappings are reused;
distinct masks add resampling work, which must be timed on the scanner.

Use fresh run output directories for this coverage/normalization version.
Existing outputs are preserved; do not mix them into the new temporal history.
