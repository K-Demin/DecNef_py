# October 16 registration regression audit

This records the source-code changes that motivated the registration repair.
Subsequent acquisition checks are separate from this audit. The historical June
decoder-score correlation (r = 0.83) was independently reproduced, but recovery
of that result with the new pipeline has not been established.

## Changes that can be established from Git

| Revision | Relevant behavior |
| --- | --- |
| `1615d02` (before October 16) | Historical anatomical Affine/SyN route; no new per-volume coverage policy. |
| `71381f0` (October 16, 11:24 +09:00) | Added partial mode and geometric coverage. Default full-brain registration retained Rigid → Affine → SyN; partial mode used T1→EPI Translation → Rigid with optional short SyN. |
| `3956b2b` (October 16, 14:41 +09:00) | Made the partial recipe universal: T1→EPI Translation → Rigid, SyN disabled by default, no Affine stage, identity/header initialization. Changed preparation/configuration/cache acceptance. |

The second commit does **not** change `rt_pipeline.py`, `rtp_regress.py`,
`gpu_ants_like_resampler.py`, `decoder_score.py`, or the NF scoring launchers.
It can nevertheless change downstream volumes and scores because it changes the
transforms estimated during preparation. Merely running updated NF code against
an existing preparation does not establish which recipe generated those transforms.
Inspect `registration.json`, its source identities, and the run's provenance.

The old full-brain `partial_epi_syn: false` setting meant "do not add partial-mode
SyN"; full-brain SyN was still enabled. The second commit's legacy-key translation
turned this into `registration_syn: false`. The repair preserves the former
full-brain semantics while retaining explicitly selected modern settings.

## Why one study working does not yet isolate the cause

The launcher paths use shared preprocessing, but their inputs and selected
settings are not necessarily equal. Check the actual command and saved settings,
not only the repository's current profile.

- At the audited October 16 revision, `rt_nf_events_prob_parallel.py` substituted a scorer that z-scored each voxel
  against prior regression-ready volumes of the current run, adds the trained
  intercept, and applies sigmoid to a trial-level decision value.
- The standard scorer uses a weighted sum of the incoming features. Optional
  reference-run score normalization occurs at the score level.
- Per-voxel temporal z-scoring can cancel a fixed positive gain and fixed offset
  at each voxel, apart from the SD floor. It does not repair spatial misalignment,
  time-varying missing coverage, or a wrong voxel-to-weight correspondence.
- The checked-in `olga_settings.json` itself describes a logistic decoder and
  requests AP, GS/WM/ventricular regression and T1 output. Establish each run's
  effective settings from its saved command and provenance.

The subsequent normalization change is separate from the October 16 registration
audit: the runtime now defaults to fixed-reference post-GLM voxel z-scoring in
EPI space, before spatial transforms/smoothing, with no second probability-scorer
voxel z-score. Its calibration starts after GLM readiness. Match this policy
explicitly when replaying historical runs; do not attribute its effects to the
registration recipe.

An additional, older preparation mismatch was found: the RS launchers did not
forward `--settings-file` to their preparation subprocess, and the single-run
configuration used a PA default. Live processing could use AP from a study
profile while its prepared reference used PA. This predates the second October
16 commit. Numerical replay of supplied AP-acquisition references confirmed that
the PA path had actually been used; this is one identified error, not an
explanation for every registration failure. The repair passes the profile through
and copies its phase-encoding/method settings into preparation; regression tests
cover AP propagation.

## Validation design

Preserve the supplied files. Write each test variant to a separate output
directory; do not update or overwrite the acquisition's cached transforms.

1. Establish the working/failing commands, settings, source NIfTI geometry,
   fieldmap polarity, actual transform recipe, and whether caches were reused.
2. Hold the corrected EPI reference, T1, masks, fieldmap, decoder and temporal
   processing fixed. Compare the historical anatomical fit, the second-October-16
   fit, and the restored recipe in each requested direction. Inspect overlays,
   tissue/mask alignment and decoder coverage, not only a scalar similarity.
3. Hold transforms fixed. Compare legacy/new coverage and normalization on the
   same volume sequence. Examine valid support, first-regressed-volume units,
   tissue regressors and voxel time series before considering final feedback.
4. Compare decoder features and raw decision values before score normalization,
   percentiles or sigmoid. Recompute the reported reproducibility measure using
   the same runs, timepoints and exclusion rules as the June comparison.
5. Repeat the critical checks on a known-working full-brain/logistic acquisition.
   Synthetic tests establish code behavior; these acquisition comparisons are
   needed to establish that the methods preserve the intended study behavior.

The proposed offline-import/session-bridge route is a separate functional change.
It should not be used as the sole comparison against the old pipeline when
identifying the October regression: that would change multiple spatial steps
at once.
