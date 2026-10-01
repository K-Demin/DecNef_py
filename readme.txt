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
- EPI(mean) -> T1 -> MNI registration via ANTs
- Optional composed transforms for online use

All heavy transforms for realtime are precomputed offline.

Processing order, output compatibility, and migration: docs/unwarp_before_mc.md

Final spatial smoothing
-----------------------

Set "smoothing_fwhm_mm" in rt_settings.json or your participant settings JSON.
The default is 0.0: no smoothing, no mask preparation, and no smooth/ outputs.
For example, 4.0 enables a 4 mm FWHM Gaussian in the final analysis space.
Use the same smoothing recipe for decoder training and online scoring.

Order: unwarp -> MC -> regression/normalization -> EPI/T1/MNI output -> smoothing -> scoring.
This is spatial smoothing of each volume independently; no temporal smoothing.

The mask is the corrected session whole-brain EPI mask, not a grey-matter mask
or the decoder's nonzero weights. EPI output uses it directly. T1/MNI output
uses that mask transformed once per run onto the actual final reference grid,
with the same transform chain as BOLD and nearest-neighbor interpolation.
No additional segmentation is run. The prepared binary mask is saved in smooth/mask.nii.

The algorithm matches smooth_masked in the supplied volume_smooth.py:
Gaussian(mask * volume) / Gaussian(mask) inside the mask, zero outside;
float64, per-axis FWHM-to-sigma conversion, zero padding, truncate=4,
and original-value fallback where the denominator is below 1e-6.
The denominator and buffers are cached once. This whole-brain version permits
mixing between tissue types inside the mask, as requested.

Enabled outputs (under func/<run>/):
  smooth/vol_XXXXX_smooth.nii       final denoised/normalized scoring input
  smooth/vol_XXXXX_smooth_orig.nii  optional original-score comparison input
Earlier-step files remain unchanged. Files are published atomically for PCA readers.
The volume streamer's "score_input" option shows the final smoothed stream;
"mc" and "unwarped" still show the unsmoothed corrected volumes.

PCA readers using the final reg/t1/mni stream follow smoothing when enabled in
the run metadata. PCA preparation supports --pca-input smooth; auto mode selects
it for enabled runs, as does reg/t1 when that mode matches the final analysis space.
Preparation merges only *_smooth.nii (not the mask or *_smooth_orig.nii).
Choosing an earlier intermediate PCA stream explicitly still uses that stream.
Existing PCA models are not retrained automatically.

FWHM, mask/reference identities, and transforms are recorded for reproducibility.
Use a fresh run output directory when changing smoothing settings. SMOOTH timing
logs include per-volume filtering and output writes; validate total latency by replay.


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
    python run_preproc.py --sub 00085 --day 4
    python rt_pipeline.py --sub 00085 --day 2 --run 11 --incoming-root /home/sin/DecNef_pain_Dec23/realtime/incoming/pain7T/20251105.20251105_00085.Kostya  --base-data /SSD2/DecNef_py/data
    python -m fmri_rt_preproc.prep_surface_rois --root /SSD2/DecNef_py/data --subj 00085 --day 2_copy

To stage raw DICOMs for the structural scan and AP/PA fieldmaps before preprocessing a transfer run, provide the incoming folder and block/run numbers (the structural block is optional if anat already contains T1*.nii* or DICOM files). When staging EPIs, pass the block/run for the transfer scan; the script will keep scans 11-30 (dropping the first 10) and place the converted NIfTIs into func/trans:

    python run_preproc.py --sub 00085 --day 4 \
      --incoming-root /path/to/incoming/dicoms \
      --ap-block 7 --pa-block 8 --struct-block 3 --epi-block 11

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
       python run_preproc.py /path/to/sub-0001/day-01/config.json

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
