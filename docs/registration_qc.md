# Registration QC before neurofeedback

Registration produces aligned images; QC checks those results. A numerical
similarity is not proof of anatomical alignment. The first NF preparation in
each session generates reports for the current session reference and its
transform chain. The NF launcher checks the expected report set and current
assets before continuing. Review approval is reused only while its inputs and
QC policy remain unchanged.

## Reports and review

The transform folder contains `registration_qc_epi_t1.json`,
`registration_qc_t1_mni.json`, and, for the session-to-reference bridge,
`registration_qc_epi_epi.json`. Each report has a matching PNG showing fixed
and aligned images plus a red/cyan overlay in three orthogonal voxel planes.
These planes can be oblique. The images are scaled independently for display;
the quantitative metrics use their original values.

Inspect corresponding anatomy, ventricles, cortical boundaries, acquired slab
limits, and missing signal. The montage samples three planes; inspect the
NIfTIs in a viewer when additional locations are needed.

```bash
python check_registration.py --trans /path/to/session/func/session_trans
```

Without validated automatic criteria, this prints `review_required` and exits
with status 1. After inspecting the images, explicitly record the review:

```bash
python check_registration.py --trans /path/to/session/func/session_trans \
  --approve --reviewer "Operator name" --note "Checked anatomy and slab coverage"
```

This creates `.approval.json` sidecars bound to the report IDs. It does not
perform a visual review automatically. Reports that failed data or configured
threshold checks cannot be waived by this command. Correct the cause and
regenerate QC. Repeat the original NF command after approval; the launcher
checks that the session assets and complete expected report set still match.

The optional `--weights decoder.nii.gz --min-frac 0.90` preserves the old
decoder signal-presence guard. It counts nonzero decoder voxels with EPI
intensity above 10% of the median positive EPI intensity. This is descriptive,
not geometric coverage or registration accuracy, and cannot approve a run.
When `--weights` is supplied, an empty decoder or fraction below `--min-frac`
still exits with status 1 and prevents approval, preserving the earlier guard.
Omit `--weights` to check only the registration reports.

## Metrics

All supplied images and masks must already share the fixed image's shape and
physical affine. QC does not register, resample, or relabel image headers.
The supplied evaluation mask defines the region. Optional aligned-brain and
geometric-support masks further restrict paired samples. Similarity excludes
nonfinite values and zero fill in either image; shared background zeros cannot
inflate the result. Nonfinite image values in the evaluated region still cause
a hard failure rather than being silently ignored.

| Metric | Meaning and limits |
| --- | --- |
| `pearson_r` | Spatial, mean-centered Pearson correlation across valid paired voxels for `epi_epi` and `t1_t1`. Invariant to positive linear intensity scaling. Not calculated for `epi_t1`, whose tissue contrasts differ. |
| `nmi` | Histogram normalized mutual information: `(H(fixed)+H(aligned))/H(joint)`, using 32 bins by default. Normally between 1 and 2. Histogram definition, mask and contrast affect the value; it is not the ANTs optimization cost. |
| `edge_correlation` | Pearson correlation of gradient magnitudes inside a one-voxel-eroded valid overlap. Gradients account for voxel spacing. Erosion excludes zero-fill/mask boundaries. This custom descriptive metric is not AFNI LPC or boundary-based registration. |
| `signal_overlap_fraction` | Finite, nonzero paired samples divided by the supplied evaluation-mask voxels. It combines signal presence and supplied support, not pure geometric coverage. |
| `geometric_coverage_fraction` | Portion of the evaluation mask covered by a separately supplied transformed acquisition-support mask. Unavailable without that mask. |
| `mask_dice` | Dice overlap between separately supplied evaluation and aligned masks. Unavailable without an aligned mask. A partial acquisition naturally has less overlap with a whole-brain mask. |

For T1-to-MNI QC, the current preparation uses the transformed subject T1 brain
mask as **evaluation support**. It does not supply that same mask a second time
as an independent template segmentation, so Dice is unavailable. Its overlap
fraction describes paired signal inside this support; it does not measure the
fraction of the whole MNI template covered. Metadata records these mask roles.

Empty masks, fewer than three valid paired samples, constant images within the
overlap, and nonfinite evaluated image values fail QC. Wrong dimensions,
geometry, malformed masks, or unreadable inputs stop report creation.

## Automatic thresholds

There are no universal numerical cutoffs built into this QC module. In
particular, a decoder-score reproducibility correlation is a different
statistic and must not become a registration-correlation threshold.

`registration_qc_policy` names a JSON file. It may contain lower-bound settings
under `epi_epi`, `epi_t1` and `t1_mni`, plus `thresholds_validated` (default false).
Each section can use:

| Setting | Available range |
| --- | --- |
| `pearson_r_min` | -1 to 1; same-contrast comparisons only |
| `nmi_min` | 1 to 2 |
| `edge_correlation_min` | -1 to 1 |
| `signal_overlap_min` | 0 to 1 |
| `geometric_coverage_min` | 0 to 1; requires a supplied support mask |
| `mask_dice_min` | 0 to 1; requires a separate aligned mask |

An unmet threshold, including an unavailable requested metric, blocks the
report. Automatic pass additionally requires `thresholds_validated: true`, a
similarity criterion (`pearson_r_min` for EPI/EPI or T1/T1; `nmi_min` or
`edge_correlation_min` for EPI/T1), and `signal_overlap_min`. Otherwise visual
review remains necessary. Setting the validation flag records the lab's
decision; the software does not establish scientific validation. Validate
criteria on reviewed good/bad registrations matching the acquisition,
processing, evaluation mask and histogram settings.

JSON reports contain SHA256 identities for every image, mask and supplied
transform. The report ID also binds metrics, comparison type, algorithm
parameters and threshold policy. A change invalidates previous approval.
Report content is deterministic; review PNG paths are not part of this ID.

## Implementation and sources

`fmri_rt_preproc.registration_qc.build_registration_qc()` generates a report.
`load_registration_qc_report()` validates its identity and current inputs.
`approve_registration_qc()` records explicit human review.
`assert_qc_gate()` checks an explicit list of required reports; the session
launcher is responsible for supplying and verifying the complete session list.

The metric distinction follows the [FSL FLIRT documentation](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/flirt/faq.html),
which separates same-modality correlation from intermodal similarity measures.
[AFNI's alignment documentation](https://afni.nimh.nih.gov/pub/dist/doc/program_help/align_epi_anat.py.html)
explains why its EPI/T1 localized Pearson cost uses opposite tissue contrast;
that cost is not a positive global-correlation acceptance threshold.
[fMRIPrep documents visual QA reports](https://fmriprep.org/en/stable/outputs.html)
as a means to inspect processing quality. These sources motivate the QC
distinctions, not a universally validated automatic gate or the custom edge
metric implemented here.
