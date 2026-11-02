"""Scheduling hints only; actual pipeline readiness remains authoritative."""


def normalization_startup_hint(settings):
    """Earliest possible scored volume, not a promise of processing completion.

    Volume indices are one-based pipeline input indices. Presentation-only
    skipped TRs are not added. The first successful GLM fit seeds z-scoring
    from its stored residual history; no second calibration window is added.
    Scoring starts on the next input volume, using prior observations only.
    Additional GLM/design, coverage, and processing delays can make actual
    readiness later than this lower bound.
    """
    count = int(settings.voxel_norm_ref_volumes)
    mode = str(settings.voxel_norm_mode)
    if count < 1 or mode not in {"zscore", "mean"}:
        raise ValueError("Expected a positive JSON normalization window and zscore or mean mode")
    first_glm = count if settings.enable_motion_regression else 1
    first_score = count + 1
    return {
        "voxel_norm_mode": mode,
        "voxel_norm_reference": str(settings.voxel_norm_reference),
        "voxel_norm_ref_volumes": count,
        "glm_first_ready_lower_bound": first_glm,
        "first_score_volume_lower_bound": first_score,
        "normalization_space": "motion-corrected EPI, after cumulative GLM and before spatial transforms",
        "estimate_kind": "lower bound; GLM design, coverage and processing can delay readiness",
    }


def first_trial_window_volumes(*, baseline_trs, skip_first_trs, iti_trs, cue_trs, scans_trs, score_delay):
    """Match the NF schedule's first score window, expressed as input volumes."""
    cue_start = baseline_trs + iti_trs + 1
    scans_end = baseline_trs + iti_trs + cue_trs + scans_trs
    score_start = min(cue_start + max(0, score_delay), scans_end)
    return skip_first_trs + score_start, skip_first_trs + scans_end
