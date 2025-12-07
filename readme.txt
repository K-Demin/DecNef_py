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
- AP/PA-based distortion correction via ANTs (AP->PA warp)
- EPI unwarping + motion correction (via AFNI/RTPSpy, external)
- EPI(mean) -> T1 -> MNI registration via ANTs
- Optional composed transforms for online use

All heavy transforms for realtime are precomputed offline.


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
    python rt_pipeline.py   --sub 00085 --day 2 --run 11 --block 11 --incoming-root /home/sin/DecNef_pain_Dec23/realtime/incoming/pain7T/20251105.20251105_00085.Kostya  --base-data /SSD2/DecNef_py/data

For full fastsurfer preproc + ROI masks (you can run it before run_preproc):
    python -m fmri_rt_preproc.prep_surface_rois \
      --root /SSD2/DecNef_py/data \
      --subj 00085 \
      --day 2_copy



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
   - Compute AP_mean.nii.gz and PA_mean.nii.gz in fmap/
   - Compute AP2PA_Warped.nii.gz and AP2PA_InverseWarped.nii.gz (AP->PA warp)
   - For each run:
        - func/run-XX/epi_unwarped.nii.gz
        - func/run-XX/epi_brain.nii.gz
        - func/run-XX/epi_mask.nii.gz
        - func/run-XX/epi_unwarped_mean.nii.gz
        - func/run-XX/epi2t1_* transforms (Warped, InverseWarped, Composite.h5)
        - func/run-XX/epi_in_MNI.nii.gz
        - optional: func/run-XX/qc_epi_in_MNI.png (if QC plotting is enabled)


Motion Correction
-----------------

The pipeline expects motion-corrected EPI to be provided by RTPSpy / AFNI's
RTP_VOLREG.

For each func/run-XX:

- The preprocessing script currently assumes a file:
      func/run-XX/epi_mc.nii.gz
  produced externally.

- It then computes:
      func/run-XX/epi_unwarped_mean.nii.gz
  from epi_mc.nii.gz via fslmaths -Tmean.

Integration with RTP_VOLREG can be added later as a small wrapper around
AFNI / RTPSpy commands.


Realtime / DecNef Usage
-----------------------

The idea is:

- Precompute all static transforms and masks using this pipeline.
- A separate realtime worker (rt_worker.py) will:
    - Load the epi->T1 composite transforms and T1->MNI warp.
    - Load AP->PA warp if needed.
    - For each new volume:
        - Apply the transforms using antsApplyTransforms.
        - Extract ROI / decoder signal in MNI space.

For now, all realtime-specific code is out of scope; this pipeline prepares
the anatomy, fieldmaps, and transforms so that realtime can run with minimal
overhead.


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
