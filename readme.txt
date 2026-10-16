Realtime fMRI Preprocessing Pipeline
====================================

Overview
--------

This package prepares subject/day data for realtime fMRI (DecNef-style) runs.
It runs:

- N4 bias correction on T1
- FastSurfer segmentation (offline, GPU)
- Cortical / GM mask generation (without CSF)
- Skull-stripping of T1 and EPI (SynthStrip)
- T1 -> MNI registration via SynthMorph
- PyHySCO distortion correction from native AP/PA means (no AP/PA motion correction)
- Raw BOLD unwarping, then motion correction to the fixed corrected session reference
- T1 -> corrected EPI registration via ANTs; use its inverse for EPI -> T1
- EPI -> MNI resampling through EPI -> T1 and the SynthMorph T1 -> MNI transform
- Optional composed transforms for online use

All heavy transforms for realtime are precomputed offline.

Processing order, output compatibility, and migration: docs/unwarp_before_mc.md

EPI-to-anatomy registration (full-brain and partial coverage)
----------------------------------------------------------

Preparation uses one recipe: fixed corrected EPI, moving T1, header-based
identity initialization, then Translation and Rigid optimization. The default
settings are "registration_initialization": "identity" and
"registration_syn": false. center_of_mass is an explicit initialization option;
auto is a compatibility alias for identity. Optional SyN adds nonlinear
refinement. New full-brain preparation therefore changes the old recipe: there
is no Affine stage, and SyN is disabled by default.

run_preproc.py, rs_realtime_parallel.py and rs_pca_score_all_rois.py all accept
--settings-file /path/to/participant_settings.json. Without it they use built-in
defaults; rt_settings.json is not loaded automatically. Explicit
--registration-initialization, --registration-syn or --no-registration-syn
options override the profile; omitted options preserve it. Effective choices
are recorded in the prepared session's config.json. Neurofeedback, PsychoPy,
robot and BrainPong reuse these prepared maps without separate partial flags.

The canonical epi2t1_Composite.h5 still maps EPI -> T1, and
epi2t1_InverseComposite.h5 still maps T1 -> EPI. Native ANTs outputs from new
fits are retained as t1_to_epi_*. There is no active registration_mode flag.
Old JSON registration_mode values full_brain/partial_epi warn and are ignored;
partial_epi_syn warns and aliases registration_syn. Conflicting aliases fail.

All three preparation commands check cache compatibility before staging scans
or rewriting config.json. New fits record registration.json version 2.
Equivalent version 1 partial-EPI caches can be reused after input/output and
settings checks. Old full-brain or unversioned caches require a fresh preparation
directory for the new recipe; unchanged prepared maps remain usable by existing
realtime consumers. Rebuild the reference and derived transforms together when
changing preparation. This is separate from truncate_t1_to_epi_fov, which only
changes a PCA output grid. Inspect alignment and mask/decoder coverage for both
full-brain and partial acquisitions; native anatomical registration has not been
validated in the Windows development environment.
Commands, settings and QC guidance: docs/partial_epi_registration.md

Final analysis masking and spatial smoothing
--------------------------------------------

Configure these fields in rt_settings.json or your participant settings JSON:
  "analysis_mask": "whole_brain",   # whole_brain, cortical_gm, or custom
  "analysis_mask_file": null,       # custom binary NIfTI path
  "analysis_mask_space": "final",   # custom coordinates: final, epi, t1, mni
  "smoothing_fwhm_mm": 0.0          # FWHM in mm; 0 means no Gaussian filtering
(The explanatory # comments above are not valid JSON.)

The defaults (whole_brain + 0 mm) retain the existing complete bypass, with no
new final-stage files. cortical_gm/custom apply masking even at 0 mm. Positive
FWHM applies normalized Gaussian smoothing within the selected analysis mask.
Use the same analysis mask and smoothing recipe for training and online scoring.

Example: cortex only without smoothing:
  "analysis_mask": "cortical_gm", "smoothing_fwhm_mm": 0.0
Example: cortex only with a 4 mm Gaussian:
  "analysis_mask": "cortical_gm", "smoothing_fwhm_mm": 4.0
Example: custom mask already on the final grid:
  "analysis_mask": "custom", "analysis_mask_file": "/path/to/mask.nii.gz",
  "analysis_mask_space": "final", "smoothing_fwhm_mm": 4.0
Use absolute custom paths; relative paths are resolved from the process directory.

Order: unwarp -> MC -> regression/normalization -> EPI/T1/MNI output ->
       final analysis masking / masked smoothing -> scoring.
The decoder ROI can be a smaller subset of the analysis/smoothing mask.

whole_brain uses the corrected session EPI brain mask. cortical_gm selects the
bilateral cortical DKT parcels from anat/fastsurfer/<subject>/mri/
aparc.DKTatlas+aseg.deep.mgz; WM, CSF, subcortical GM, cerebellum, and unknown
labels are excluded. It reuses existing FastSurfer segmentation, with no new
segmentation run. Custom masks must be finite binary 0/1 volumes.

Masks are mapped once per run into the exact final BOLD grid, with nearest-neighbor
interpolation. T1->EPI uses the inverse EPI-to-T1 composite, T1->MNI uses the
anatomical warp, and EPI->T1/MNI uses the BOLD transform chain. Same-space masks
may be resampled between grids. A custom "final" mask must already match exactly.
MNI->EPI/T1 mask conversion is not supported; supply a mask in final coordinates.
Cortex/custom masks are intersected with transformed EPI brain coverage. Empty,
nonfinite, nonbinary custom, or incompatible masks fail before volume processing.

Each volume also carries geometric FOV support from its estimated MC transform
through the same final spatial transform. This is independent of voxel intensity.
The effective mask is the fixed analysis mask intersected with current support.
Missing current samples are zeroed before spatial interpolation and final output.
This does not add a missing-observation policy to cumulative nuisance regression.

Positive-FWHM mathematics uses normalized convolution:
Gaussian(effective_mask * volume) / Gaussian(effective_mask), zero outside;
float64, per-axis mm-to-voxel conversion, zero padding, truncate=4, and original-value
fallback where the denominator is below 1e-6. Buffers are reused; the denominator
is recomputed when coverage removes analysis voxels and cached otherwise.
At 0 mm, cortex/custom mode copies in-mask values unchanged and zeros everything
else, with no Gaussian calls. This remains volumetric, not surface-geodesic smoothing.

Outputs under func/<run>/:
  smooth/vol_XXXXX_smooth.nii       positive FWHM, masked and smoothed
  masked/vol_XXXXX_masked.nii       cortex/custom masking with 0 mm
  corresponding *_orig.nii         optional original-score comparison stream
Each active folder includes mask.nii (the fixed analysis mask). Per-volume support
is saved separately in coverage/vol_XXXXX_epi.nii and, when applicable, *_t1.nii
or *_mni.nii. New coverage handling requires fresh run outputs, but an unchanged
prepared registration can be reused. Final volumes
are published atomically for PCA readers. Stream "score_input" to see final output;
"mc" and "unwarped" continue to show the earlier corrected volumes.
The online viewer adds a second row for the same volume's T1 stage before
smoothing. EPI/MNI runs without a T1 intermediate show that row as unavailable.

PCA final reg/t1/mni readers follow the selected final stage from run metadata.
Preparation accepts --pca-input smooth or masked; auto follows the final stage,
as does reg/t1 when it matches the final analysis space. Only the selected main
volume suffix is merged, excluding masks and *_orig files. Earlier intermediate
streams remain explicitly selectable. Existing PCA models are not retrained.
Mask source/reference/transform identities and FWHM are recorded. Use a fresh run
output directory when changing them. SMOOTH/MASK timing logs include volume writes.

Label references:
https://deep-mi.org/FastSurfer/stable/overview/OUTPUT_FILES.html
https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT


Environments
------------

We assume two conda environments:

1) rt_pipe
   - Python 3.9
   - ANTs (antsRegistration, antsApplyTransforms, ComposeMultiTransform)
   - FSL (fslmaths, flirt, fslcpgeom)
   - FreeSurfer (mri_binarize, mri_synthstrip, mri_synthmorph)
   - Nilearn (optional, for QC plots)

2) fastsurfer
   - FastSurfer installation (run_fastsurfer.sh)
   - CUDA-visible if using GPU

Typical usage:

    conda activate rt_pipe
    python run_preproc.py --sub 00085 --day 4 --settings-file ./rt_settings.json
    python rt_pipeline.py --sub 00085 --day 2 --run 11 --incoming-root /home/sin/DecNef_pain_Dec23/realtime/incoming/pain7T/20251105.20251105_00085.Kostya  --base-data /SSD2/DecNef_py/data
    python -m fmri_rt_preproc.prep_surface_rois --root /SSD2/DecNef_py/data --subj 00085 --day 2_copy

To stage raw DICOMs for the structural scan and AP/PA fieldmaps before preprocessing a transfer run, provide the incoming folder and block/run numbers (the structural block is optional if anat already contains T1*.nii* or DICOM files). When staging EPIs, pass the block/run for the transfer scan; the script will keep scans 11-30 (dropping the first 10) and place the converted NIfTIs into func/trans:

    python run_preproc.py --sub 00085 --day 4 \
      --incoming-root /path/to/incoming/dicoms \
      --ap-block 7 --pa-block 8 --struct-block 3 --epi-block 11 \
      --settings-file ./rt_settings.json

For full fastsurfer preproc + ROI masks (you can run it before run_preproc):
    python -m fmri_rt_preproc.prep_surface_rois \
      --root /SSD2/DecNef_py/data \
      --subj 00085 \
      --day 2_copy

Example of behavioral experiment:
usage: rt_psychopy_parallel.py [-h] --sub SUB --day DAY --run RUN
                               [--incoming-root INCOMING_ROOT]
                               [--base-data BASE_DATA]
                               [--max-points MAX_POINTS]
                               [--decoder-template DECODER_TEMPLATE]
rt_psychopy_parallel.py: error: the following arguments are required: --sub, --day, --run




Data Organization
-----------------

For each subject/day, data should be organized like this:

    <project_root>/
      sub-0001/
        anat/
            T1.nii.gz                 # raw structural
        day-01/
          fmap/
            AP.nii.gz                 # 8x AP volumes (currently "down")
            PA.nii.gz                 # 8x PA volumes (currently "up")
          func/
            run-01/
              epi.nii.gz              # raw EPI for this run
            run-02/
              epi.nii.gz
          config.json
          logs/
            preproc.log               # optional

The script will create additional files (T1_N4, masks, warps, etc.) inside
anat/, fmap/, and func/run-XX/.


config.json
-----------

Each subject/day has a small JSON config that tells the pipeline where things are:

Example:

    {
      "subject_id": "0001",
      "day_id": "01",
      "root": "/project_root/sub-0001/day-01",
      "phase_encoding": {
        "ap_label": "down",
        "pa_label": "up"
      },
      "templates": {
        "mni_t1": "/path/to/freesurfer/average/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c.nii.gz"
      },
      "runs": [
        {"id": "run-01", "epi_file": "func/run-01/epi.nii.gz"},
        {"id": "run-02", "epi_file": "func/run-02/epi.nii.gz"}
      ]
    }

Notes:

- "root" is the subject/day folder.
- "mni_t1" should point to the standard MNI152 09c template shipped with FreeSurfer.
- "runs" lists all runs you want to pre-process.


Running the pipeline
--------------------

1) Make sure T1.nii.gz, AP.nii.gz, PA.nii.gz, and all epi.nii.gz files
   exist in the correct folders.

2) Check that ANTs, FSL, FreeSurfer, and Nilearn are available in rt_pipe:

       conda activate rt_pipe
       which antsRegistration
       which fslmaths
       which mri_binarize
       python -c "import nilearn"

3) Run the preprocessing:

       conda activate rt_pipe
       python run_preproc.py --sub 0001 --day 01 --base-data /project_root --settings-file /path/to/participant_settings.json

4) The script will:

   - Create anat/T1_N4.nii.gz (N4 bias correction).
   - Run FastSurfer in the fastsurfer env and create aparc+aseg.mgz etc.
   - Create:
        - anat/brainmask_noCSF_filled.nii.gz
        - anat/T1_brain.nii.gz
        - anat/T1_mask_skull.nii.gz
        - anat/T1_combined_mask.nii.gz
   - Run SynthMorph:
        - anat/warp_T1_to_MNI_synth.nii.gz
        - anat/T1_warped_to_MNI_synth.nii.gz
   - Average raw AP/PA independently, without motion correction.
     Raw AP.nii[.gz] and PA.nii[.gz] stay in the selected fmap/pair-* folder
     (or fmap/ when no explicit pair is selected). Derived products are in
     that folder's native_unwarp_v1/ subdirectory:
        - AP_mean.nii and PA_mean.nii
        - pyhysco_native-EstFieldMap.nii
        - calibration.json
   - For each run:
        - func/run-XX/epi_first.nii
        - func/run-XX/epi_mc.nii
        - func/run-XX/motion.1D
        - func/run-XX/epi_unwarped.nii
        - func/run-XX/epi_brain.nii.gz
        - func/run-XX/epi_mask.nii.gz
        - func/run-XX/epi_unwarped_mean.nii.gz
        - func/run-XX/epi2t1_* transforms (Warped, InverseWarped, Composite.h5)
        - func/run-XX/epi_in_MNI.nii.gz
        - optional: func/run-XX/qc_epi_in_MNI.png (if QC plotting is enabled)
   - For realtime reuse, create day-level references in func/trans:
        - rt_ref_epi.nii and rt_ref_epi_mask.nii (legacy aliases; corrected data in new preparations)
        - epi_unwarped_mean.nii and epi_mask_mean.nii (unwarped analysis grid)


Motion Correction
-----------------

The preprocessing script now performs RTPSpy motion correction itself.

For each func/run-XX, the script writes func/run-XX/epi_mc.nii and motion.1D.
The first run unwarps raw BOLD, motion-corrects it to its first corrected volume,
and establishes func/trans/epi_unwarped_mean.nii as the fixed corrected session
reference. EPI-to-T1 registration and nuisance masks use this corrected reference.
AP and PA are averaged without motion correction and remain in acquisition geometry.

For realtime, rt_pipeline.py first unwarps each incoming raw volume using the
selected pair's native_unwarp_v1/pyhysco_native-EstFieldMap.nii, then motion-corrects
it to func/trans/epi_unwarped_mean.nii[.gz].
Regression, voxel normalization, EPI-to-T1/MNI transforms, and decoder scoring
all operate on the unwarped analysis stream.


Realtime / DecNef Usage
-----------------------

The offline preprocessing prepares the anatomy, fieldmaps, references, masks,
and transforms needed by rt_pipeline.py. During realtime:

- compute_stage converts each incoming DICOM to a raw NIfTI.
- commit_stage runs stateful processing in scan order:
    - fieldmap unwarp using the selected pair's native_unwarp_v1/pyhysco_native-EstFieldMap.nii
    - RTPSpy motion correction to epi_unwarped_mean.nii[.gz]
    - FD/DVARS censor bookkeeping
    - nuisance regression and voxel normalization on the unwarped stream
    - optional EPI->T1 or EPI->T1->MNI transform
    - decoder scoring and score publication


QC
--

If Nilearn is installed, you can optionally generate visual QC:

- Overlay epi_in_MNI.nii.gz on MNI template.
- Save as PNG in func/run-XX/qc_epi_in_MNI.png.

This is purely for visual inspection (alignment of EPI and MNI / decoder).
It is not used in realtime.


Contact / Notes
---------------

- If ANTs fails with a Python version error, make sure rt_pipe is Python 3.9,
  not 3.10 (ANTs does not support 3.10 in your current build).
- If FastSurfer fails, check CUDA visibility and that you can run
  "run_fastsurfer.sh" inside the fastsurfer environment.
- The AP/PA naming in your current folders is flipped ("AP" == down, "PA" == up),
  but the pipeline just treats them as AP.nii.gz and PA.nii.gz, so future
  renaming will not break the logic.

Measure realtime speed
----------------------

Append --test-speed to your usual rs_realtime_parallel.py command. It runs the
same pipeline and records timings; stop normally with Ctrl+C after acquisition.
The run folder (sub-<subject>/<day>/func/<run>/) receives:

  speed_test.csv   per-volume measurements, flushed as volumes finish
  speed_test.png   16:9 report generated after stopping, outside the TR loop

The plot is one per-stage duration chart. Labels follow the script: DICOM
conversion, FMAP unwarping, MC plus FD, REG denoising (including any physiology
wait), TRANS spatial resampling, SMOOTH/MASK, SCORE and output I/O. The internal
QC timing is labeled DVARS: signal-change and censor checks (FD and DVARS flags).
Disabled stages with no measured work are omitted.

The same chart includes incoming-volume overhead:
  WAIT       idle time until the next incoming volume is detected
  FILE       creation-to-close callback interval, when both events are observed
  WATCHDOG   detection to conversion-worker start, including worker queueing
  ORDER      conversion completion to ordered processing start
These input intervals can overlap and should not be added together. FILE includes
event-delivery delay and is not a pure transfer-time measurement. It is
blank/not observed if the filesystem supplies no close event, the creation event
was missed, or the close arrives after the completed CSV row was flushed.

Summaries use ready volumes when available, preferring fresh live conversions.
Backlog and reused-NIfTI measurements are separate cohorts, never pooled into
the selected summary. The subtitle identifies the cohort and each row gives n.
Whiskers are approximate 95% moving-block-bootstrap confidence intervals
of each stage's median, using 2,000 resamples and blocks of ceil(n^(1/3)) adjacent
measurements. A fixed seed makes regeneration repeatable; fewer than 8 observations
or nonconsecutive volume indices are labeled CI unavailable. Missing observations
are not treated as adjacent TRs. These are uncertainty intervals for the within-run
median, not a range containing 95% of individual TR times. The bootstrap assumes
approximate stationarity and short-range dependence. Per-volume timings in the
CSV allow inspection of drift; the intervals do not predict a future run.
Method: https://arch.readthedocs.io/en/stable/bootstrap/timeseries-bootstraps.html

Existing files are marked backlog, reused raw NIfTIs are marked cached, and
failed volumes have their own status. Backlog detection intervals do not measure
scanner cadence. Input timing begins when the watcher sees a file, not at the
scanner trigger; file-transfer time before detection is therefore not measured.
Processing includes file I/O and repeated conversion/processing attempts;
end-to-end latency additionally includes queue delays and retry backoff. An
interrupted volume may be absent, but completed rows remain in the CSV.
Profiling is off by default. Matplotlib is only imported to make the report.
Initialization/anatomy preparation and external display rendering are excluded.
Each profiled invocation replaces this run's previous speed_test.csv/.png.

To regenerate the report (set --tr to the run's configured TR, in seconds):

    python rt_speed.py /path/to/run/speed_test.csv --tr 1.0

Global runtime settings (new)
-----------------------------

You can now keep shared realtime parameters in one JSON file and reuse it across:
- rt_pipeline.py
- rt_psychopy_parallel.py
- rs_realtime_parallel.py

A ready-to-edit template is included at the repo root:

    rt_settings.json

Use it like this:

    python rt_pipeline.py ... --settings-file ./rt_settings.json

Commenting convention in JSON
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

JSON does not support native comments, so the template uses `_comment*` keys
(e.g., `_comment_timing`, `_comment_biopac`) to explain each settings block.
Those keys are ignored by the loader and are safe to keep in place.

What each settings block controls
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Timing/model (`TR`, `analysis_space`, `mot_reg`, `max_poly_order`, `enable_motion_regression`, `voxel_norm_ref_volumes`):
  controls temporal assumptions and nuisance model complexity.
  - `analysis_space = "mni"` (default): apply EPI→T1→MNI normalization before scoring.
  - `analysis_space = "epi"`: skip normalization and score the cleaned MC volume in native EPI space.
  - `voxel_norm_ref_volumes` (default `1`): shared normalization-window setting used in both modes. With regression ON, it sets RTPSpy `wait_num = voxel_norm_ref_volumes - 1`, so scaling mean is built from that many initial volumes; with regression OFF, it averages the first `voxel_norm_ref_volumes` volumes for the same voxel-wise percent-signal scaling (`Y / Y_mean * 100`).
- Tissue regressors (`use_gs`, `use_wm`, `use_vent`):
  toggles global/WM/ventricle nuisance regressors.
- Censoring (`fd_thr`, `dvars_thr_robust_z`, `censor_plus1`, etc.):
  controls motion/outlier censor regressors.
- BIOPAC (`biopac_*`):
  defaults for receiving/using physio regressors and handshake behavior.
- Runtime (`max_workers`, `max_retries`):
  parallelism and retry policy for realtime processing.

If you use `analysis_space = "epi"` with scoring enabled, pass an EPI-space decoder via
`--decoder-template` so decoder dimensions/space match the processed volume.

Per-volume output folders (under `func/<run_id>/`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- `raw/`: raw incoming NIfTI volumes before RT motion correction.
- `mc/`: unwarped, motion-corrected volumes in the fixed corrected EPI reference space; saved once and shared by PCA, streaming, and original-score consumers.
- `unwarped/`: optional `_uw_native` intermediate volumes after unwarping, before motion correction.
- `reg/`: nuisance-cleaned (and voxel-normalized) volumes in native EPI space; this is the source volume that is later warped when `analysis_space` is `t1` or `mni`.
- `t1/` (only when `analysis_space = "t1"`):
  - `vol_XXXXX_t1.nii`: the `reg/` volume warped to T1/decoder space (used for denoised scoring).
  - `vol_XXXXX_t1_orig.nii`: the fully corrected `mc/` volume warped to T1/decoder space (used as the non-denoised comparison score).
- `mni/` (only when `analysis_space = "mni"`): equivalent pair (`*_mni.nii` and `*_mni_orig.nii`) in MNI/decoder space.

CLI flags still work and can override values for a single run.

Closed-loop T2 (SPM + AFNI) pipeline
-----------------------------------

If you only point to folders, use:

    python -m fmri_rt_preproc.t2_spm_afni_closed_loop \
      --subject-root /path/to/sub-00085 \
      --day day-02 \
      --spm-dir /opt/spm12 \
      --mni-template /path/to/MNI152_T1_1mm.nii.gz \
      --segmentation /path/to/segmentation.nii.gz \
      --dg-labels 17 53

What it does automatically:
- Searches `sub-XXXX/anat` for `T2*.nii*`; if missing, converts DICOMs in that folder via `dcm2niix`.
- Runs SPM segmentation on T2 (GM/WM/CSF + bias-corrected output).
- Runs AFNI cleanup (`3dUnifize`, `3dSkullStrip`) and optional MNI alignment (`3dAllineate`).
- Optionally extracts a DG mask from a segmentation volume using provided labels.
- Writes `day-XX/t2_pipeline/pipeline_summary.json` with all outputs.

Note on DG labels:
- `--dg-labels 17 53` are a practical fallback (whole hippocampus proxy in aparc+aseg style maps).
- For true dentate gyrus extraction, pass the atlas-specific DG labels from your own segmentation output.
