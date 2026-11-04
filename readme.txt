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
- Direction-selectable EPI/T1 registration via ANTs, or imported offline mappings
- EPI -> MNI resampling through EPI -> T1 and the selected T1 -> MNI transform
- Optional composed transforms for online use

Spatial mappings are estimated during preparation, then reused for realtime resampling.

Processing order, output compatibility, and migration: docs/unwarp_before_mc.md

EPI-to-anatomy registration and offline reuse
--------------------------------------------

Both registration_direction choices (epi_to_t1, t1_to_epi) now use Rigid,
Affine and the restored historical SyN schedule. Defaults: direction=epi_to_t1,
registration_initialization=center_of_mass, registration_threads=1 (ITK),
registration_bridge_initialization=identity, registration_syn=true. Explicit settings
profiles still override defaults, including an existing registration_syn=false.

epi_reference_n4=true (default) adds a separate N4-corrected EPI reference for
registration: unwarp -> MC -> observed-sample mean -> SynthStrip mask -> N4 copy
-> anatomical or EPI-to-EPI fit. The original mean remains the MC target;
incoming volumes and temporal processing do not receive N4. Configure the JSON
boolean, or override it with --epi-reference-n4 / --no-epi-reference-n4.

The transform folder saves epi_registration_reference.nii and its .json sidecar.
EPI bridges reuse an optional enhanced offline epi_registration_reference,
recognize already enhanced fMRIPrep imports, or create an N4 copy of an older
raw anchor as reference_epi_registration.nii. Original bundles remain intact.
Canonical warped outputs and QC show the original image intensities through
the fitted maps, so intensity correction alone is not reported as improved alignment.
Earlier Linux reference experiments measured about 4.1 s per N4 correction;
an unenhanced old anchor can require a second correction. There is no per-TR N4
cost. These are isolated preparation timings, not a new full-session benchmark.
Use fresh preparation/session directories and renew registration QC when this
setting changes. N4 can help fitting but does not guarantee convergence.

prepare_offline.py imports an explicit matching fMRIPrep file set or exports a
prepared DecNef day. Offline input is optional: run_preproc.py can compute all
anatomy and reference mappings during a first visit. The exact input filenames,
folder layout and commands are in docs/offline_preparation.md.

prepare_session.py builds the current EPI-to-prepared-EPI bridge from the first
reference acquisition. It saves func/session_trans and produces registration QC
reports/overlays. Review these before the timed NF task. NF, probability NF,
PsychoPy, robot and BrainPong use this shared gate and reuse the approved session
alignment. RS/preparation processing does not require NF approval.

Canonical epi2t1_Composite.h5 always means EPI->T1; InverseComposite means T1->EPI.
The fixed/moving choice does not change those meanings. Imported MNI transforms
keep their actual file format and exact template identity. Both CPU and GPU
consumers use the selected session mappings.

New registrations have manifest version 3; the short, linear-only recipe is
not silently reused as a nonlinear registration. Old data are preserved.
Changed coverage/normalization also requires fresh per-run output directories.
Denoising warmup uses the processing mask, and missing observations are
excluded from calibration. The cumulative GLM
uses explicitly reported complete-history voxel support; see the coverage
tradeoff and runtime implications in docs/partial_epi_registration.md.

Commands/settings: docs/partial_epi_registration.md
Offline files/fMRIPrep: docs/offline_preparation.md
Alignment metrics/review: docs/registration_qc.md
Live decoder-ROI correlation and Excel export: docs/decoder_correlation_qc.md

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

Order: unwarp -> MC -> native-unit cumulative GLM -> voxel normalization -> EPI/T1/MNI output ->
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
The effective mask is the fixed analysis mask intersected with usable signal support.
Missing current samples are zeroed before spatial interpolation and final output.
The cumulative GLM uses complete-history support: once a voxel loses coverage, it
stays excluded for the rest of that run. See the coverage policy above.

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
is saved separately: coverage/ holds geometric FOV and support/ holds usable
analysis samples, in EPI and the selected final space. New coverage handling
requires fresh run outputs, but a compatible unchanged
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
    - cumulative nuisance GLM in native intensity units on the corrected EPI stream
    - voxel normalization (fixed-reference z-score by default)
    - optional EPI->T1 or EPI->T1->MNI transform
    - selected analysis mask and optional spatial smoothing
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

- Timing/model (`TR`, `analysis_space`, `mot_reg`, `max_poly_order`, `enable_motion_regression`):
  controls temporal assumptions and nuisance model complexity.
  - `analysis_space = "mni"` (default): apply EPI→T1→MNI spatial transforms before scoring.
  - `analysis_space = "t1"`: apply EPI→T1 before scoring.
  - `analysis_space = "epi"`: score in the fixed session EPI grid; voxel normalization still applies.
- Voxel normalization (`voxel_norm_mode`, `voxel_norm_reference`, `voxel_norm_ref_volumes`):
  - `voxel_norm_mode = "zscore"` (default), or `"mean"`.
  - `voxel_norm_reference = "fixed"` (default), or `"running"`.
  - `voxel_norm_ref_volumes = 30` (default): N reference samples; N must be at least 2 for zscore, or 1 for mean. With regression enabled, N also sets the GLM startup minimum. See the timing and formulas below.
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

Voxel normalization: units, calibration and readiness
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The cumulative GLM fits the measured corrected EPI intensities without first
percent-scaling them. Let y_v(t) be raw MC intensity and r_v(t) the emitted GLM
residual for voxel v. Normalization follows the GLM in the session EPI grid,
before T1/MNI interpolation, analysis masking and smoothing:

  zscore: z_v(t) = (r_v(t) - mean_reference(r_v)) / sample_SD_reference(r_v)
  mean:   q_v(t) = 100 * r_v(t) / mean_reference(y_v)

Mean mode divides by the raw BOLD reference mean, never by a near-zero residual
mean. It expresses residuals as a percentage of reference BOLD intensity; it does
not add 100 back. Neither mode clips the resulting values or bounds them to a
fixed range. Sample SD uses N-1 in its denominator. Invalid or unavailable
observations do not become measured zero-intensity calibration samples.

With `fixed`, zscore collects the first N emitted regression-ready residuals,
freezes their per-voxel mean/SD, then normalizes starting with the NEXT volume.
Thus N=30 means GLM startup PLUS 30 ready residuals before usable normalized
feedback. It does not mean that all startup completes in 30 TRs. GLM
design/readiness or missing support can delay feedback further.
`skip_first_trs` shifts the presentation timeline; those acquired volumes still
enter processing. `--baseline-trs` controls presentation timing independently;
it does not change the JSON normalization window. Allow the task to wait for
normalization readiness even when the planned presentation baseline has ended.

With `running`, zscore also waits for at least N ready residuals, but continues
updating the reference. Each current volume uses statistics from prior volumes;
it is added only afterward. The reference therefore changes over time, unlike
the fixed mode. Running is cumulative, not a sliding window.

Mean mode calibrates on the first N raw MC observations during GLM startup.
Fixed freezes that raw mean; running updates it after each current volume, so
the current denominator still uses prior observations. With regression disabled,
the corrected MC data replace r_v(t) and z-score calibration starts immediately.

The z-score reference is voxel-wise and temporal, not the spatial mean of each
EPI. Subsequent interpolation and smoothing combine voxels: final decoder-grid
voxels are not guaranteed to retain a sample SD of exactly 1. Calibration also
does not guarantee that future volumes have the same distribution.

Decoder scoring then uses the resulting features: decision = dot(weights, x) +
intercept. The probability launcher must not apply another voxel z-score. A
sigmoid or reference-run normalization of that scalar decision is a different
operation. PCA's saved training mean/SD remain part of its model contract; old
PCA models and decoder weights must be checked against the new upstream units
and processing order, and retrained/recalibrated when those contracts differ.
Fixed early-run z-scoring is not identical to offline whole-run standardization.

For future runs intended for PCA preparation in T1, MNI or a bridged reference
EPI grid, add `--enable-original-score` to the usual `rs_realtime_parallel.py`
or `rs_pca_score_all_rois.py` command. It is opt-in and defaults to false.
This preserves a same-grid original-intensity `*_orig.nii` comparison stream
for tSNR (temporal mean / temporal SD) voxel ranking; z-scored residuals cannot
provide meaningful tSNR. Native session-EPI PCA uses the existing `mc/` stream.
PCA preparation reports an error if required original-intensity volumes are
missing, rather than substituting normalized data. The additional comparison
stream adds per-TR spatial resampling, any enabled smoothing, and output I/O;
measure the resulting latency before using it during a study. PCA fitting still
uses normalized, readiness-filtered data and retains its own trained mean/SD.

These settings change output units and warmup. Use fresh per-run output folders;
normalization choices and stage are recorded in provenance so cached outputs
from a different normalization policy cannot silently mix into a run.
`regression_status_rt.csv` distinguishes `glm_ready`, `normalization_ready` and
`normalization_reference_count`. Its existing `reg_ready` field now means overall
scoring readiness for compatibility with consumers; scores remain NaN until
both stages are ready. `norm/` images are zero-filled during calibration, rather
than exposing raw-intensity warmup as normalized data. Reference mean, sample
SD/count images and JSON metadata are saved under `norm/` at calibration and
shutdown (including final running statistics), not rewritten every TR.

GPU interpolation (final TRANS stage)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set these in the study settings JSON to use the optional sharper interpolator:
    "use_gpu_resampler": true,
    "gpu_resampler_device": "cuda",
    "gpu_resampler_interpolation": "lanczos"

The default remains "linear". Lanczos requires CUDA and Triton; an explicitly
selected Lanczos run stops during initialization if the backend is unavailable.
This option applies EPI-to-reference-EPI, EPI-to-T1, or EPI-to-MNI transforms.
It does not change registration fitting, PyHySCO unwarping, or heptic motion
correction. EPI output with no spatial transform remains a passthrough.

The Lanczos kernel matches ANTs LanczosWindowedSinc (radius 3, zero exterior,
unnormalized weights). Coordinates and weights are cached; kernels are warmed
during initialization. Legacy Linear grids are rebuilt for Lanczos to retain
the exact source coordinates at image edges. Switching interpolation requires
a fresh run output directory, as enforced by preprocessing_order.json.

Before spatial interpolation, denoising and normalization use an automatically
expanded processing mask: a cubic margin of three EPI voxels per axis for
Lanczos, one for Linear, or none for native EPI output without spatial mapping.
The original anatomical and nuisance masks remain separate. No extra JSON
setting is required. This processing change also requires fresh run outputs.

Final brain selection uses nearest-neighbor mask mapping. Missing samples are
tracked independently of intensities with Linear availability interpolation
and threshold 1-1e-5, for either signal kernel. Final support intersects the
mapped anatomical mask; it no longer requires the whole Lanczos neighbourhood
to be inside that brain mask. Signed residuals and legitimate zeros remain valid.

Missing slices or unavailable temporal estimates are not restored by padding,
and Lanczos values beside such gaps may still be affected. Inspect
coverage_rt.csv (especially usable_decoder_voxels) and alignment QC. Missing
decoder features remain zero; the coverage flag alone does not block scoring.
Use the same processing/interpolation policy when calibrating a decoder baseline.
See docs/processing_masks.md for saved masks, provenance and validation status;
native session replay of the newly integrated processing margin is pending.

Per-volume output folders (under `func/<run_id>/`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- `raw/`: raw incoming NIfTI volumes before RT motion correction.
- `mc/`: unwarped, motion-corrected volumes in the fixed corrected EPI reference space; saved once and shared by PCA, streaming, and original-score consumers.
- `unwarped/`: optional `_uw_native` intermediate volumes after unwarping, before motion correction.
- `reg/`: cumulative-GLM output in native EPI intensity units, including the processing margin when spatial mapping follows (warmup is not yet a fitted residual).
- `norm/`: post-GLM voxel-normalized volumes in session EPI space, with the same processing margin; this is the source later spatially mapped and restricted to final brain/analysis support.
- `t1/` (only when `analysis_space = "t1"`):
  - `vol_XXXXX_t1.nii`: the `norm/` volume warped to T1/decoder space (used for denoised scoring).
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
