#!/usr/bin/env python
import time
import csv
import logging
import argparse
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Any
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, CancelledError
from typing import Optional

import nibabel as nib
import numpy as np
import torch

from fmri_rt_preproc.native_fieldmap import (
    PIPELINE_ORDER, native_paths, load_calibration, find_source,
    validate_bold, require_same_grid, source_identity,
)

from motion_fd import fd_from_rtpspy_delta
from fmri_rt_preproc.volume_smooth import prepare_smoother, smooth_file
from fmri_rt_preproc.analysis_mask import (
    final_output_kind, mask_provenance, session_epi_bridge, epi_output_reference,
    prepare_analysis_mask,
)
from fmri_rt_preproc.processing_mask import (
    make_processing_mask, processing_margin_vox,
    FINAL_PROCESSING_MASK_POLICY, FINAL_PROCESSING_MASK_POLICY_VERSION,
    FINAL_NUISANCE_SOURCE, FINAL_PROCESSING_ORDER,
    canonical_final_processing_mask,
)

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from fmri_rt_preproc.RTPSpy_tools.rtp_volreg import RtpVolreg
from fmri_rt_preproc.RTPSpy_tools.rtp_regress import RtpRegress
from fmri_rt_preproc.pyhysco_apply import apply_pyhysco_fieldmap, PreloadedPyHyscoApplier
from fmri_rt_preproc.utils import run  # your existing run() wrapper

from decoder_score import DecoderScorer
from fmri_rt_preproc.decoder_qc import DecoderSpatialCorrelation, write_run_qc_workbook
from biopac_rt.biopac_receiver import (
    BiopacReceiverConfig,
    BiopacRetroTSReceiver,
    BiopacRetroTSFileBuffer,
)
from rt_global_settings import load_regressor_settings
from volume_streamer import VolumeStreamerConfig, VolumeStreamerHandle

# ---------- Logging setup ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
log = logging.getLogger("rt_pipeline")

# Silence noisy sub-loggers (keep only warnings+)
logging.getLogger("fmri_rt_preproc").setLevel(logging.WARNING)
logging.getLogger("RtpVolreg").setLevel(logging.WARNING)
logging.getLogger("watchdog.observers.inotify_buffer").setLevel(logging.WARNING)

# ---------- Regressor config ----------
REGRESSOR_SETTINGS = load_regressor_settings()
COVERAGE_POLICY = "separate_brain_processing_observed_support_v3"
TEMPORAL_PROCESSING_ORDER = "spatial_glm_smooth_norm_v2"

from fmri_rt_preproc.offline_assets import t1_to_mni_transform
from fmri_rt_preproc.voxel_normalization import VoxelNormalizer
SUPPORT_THRESHOLD = 1.0 - 1e-5


def normalization_spec():
    spec = {"version": 3, "stage": "after_spatial_smoothing_glm",
            "space": str(REGRESSOR_SETTINGS.analysis_space).lower(), "glm_input_units": "original_intensity",
            "mode": getattr(REGRESSOR_SETTINGS, "voxel_norm_mode", "zscore"),
            "reference": getattr(REGRESSOR_SETTINGS, "voxel_norm_reference", "fixed"),
            "ref_volumes": int(getattr(REGRESSOR_SETTINGS, "voxel_norm_ref_volumes", 30)),
            "current_volume_in_reference": False, "sample_sd_ddof": 1,
            "mean_denominator": "pre_glm_transformed_smoothed_bold", "clipping": None}
    if spec["mode"] == "zscore" and getattr(REGRESSOR_SETTINGS, "enable_motion_regression", True):
        # References calibrated from the old second window are not equivalent.
        spec.update(zscore_startup="first_glm_fit_residual_history")
    return spec


def processing_mask_spec(cfg):
    """Record final-grid temporal selection and the unchanged native nuisance source."""
    space = str(REGRESSOR_SETTINGS.analysis_space).lower()
    spatial = space in {"t1", "mni"} or session_epi_bridge(cfg) is not None
    interpolation = str(getattr(REGRESSOR_SETTINGS, "gpu_resampler_interpolation", "linear")).lower()
    path = getattr(cfg, "rt_unwarped_analysis_ref_mask", None)
    return {"policy": FINAL_PROCESSING_MASK_POLICY, "version": FINAL_PROCESSING_MASK_POLICY_VERSION,
            "margin_vox": 0, "space": space,
            "nuisance_source": FINAL_NUISANCE_SOURCE,
            "processing_order": list(FINAL_PROCESSING_ORDER),
            "nuisance_margin_vox": processing_margin_vox(interpolation, spatial),
            "interpolation": interpolation, "spatial_resampling": spatial,
            "brain_mask": source_identity(path) if path is not None and path.exists() else None,
            "final_brain_mask_interpolation": "NearestNeighbor",
            "availability_interpolation": "Linear"}


def prepare_processing_masks(cfg, analysis_mask=None):
    """Keep native QC anatomy separate from final-grid temporal selection."""
    path = cfg.rt_unwarped_analysis_ref_mask
    if not path.exists():
        log.warning("[MASK] Brain mask missing at %s; using available acquisition support.", path)
        return None, analysis_mask, None, None
    source = nib.load(str(path))
    require_same_grid(source, nib.load(str(cfg.rt_unwarped_analysis_ref_epi)))
    brain = np.asarray(source.dataobj).astype(bool)
    folder = cfg.rt_work_dir / "processing_mask"
    folder.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(brain.astype(np.uint8), source.affine), folder / "brain_epi.nii")
    space = str(REGRESSOR_SETTINGS.analysis_space).lower()
    final_brain, final_affine = prepare_analysis_mask(
        cfg, "whole_brain", space, resolve_analysis_reference(cfg, space), folder, run)
    processing = final_brain.copy()
    if analysis_mask is not None:
        if analysis_mask.shape != processing.shape:
            raise ValueError("Analysis mask and final brain mask grids differ.")
        processing &= analysis_mask
    process_path = folder / "processing_final.nii"
    nib.save(nib.Nifti1Image(processing.astype(np.uint8), final_affine), process_path)
    nib.save(nib.Nifti1Image(final_brain.astype(np.uint8), final_affine), folder / "brain_final.nii")
    spec = processing_mask_spec(cfg)
    spec.update(brain_voxels=int(final_brain.sum()), processing_voxels=int(processing.sum()))
    (folder / "policy.json").write_text(json.dumps(spec, indent=2) + "\n")
    log.info("[MASK] final %s grid: brain=%d; temporal processing=%d voxels",
             space, final_brain.sum(), processing.sum())
    return brain, processing, final_brain, process_path


def native_nuisance_means(handler, data, support):
    """Sample the existing native, unsmoothed nuisance ROIs independently of responses."""
    masks = getattr(handler, "native_nuisance_masks", None)
    if masks is None:
        return None
    if getattr(handler, "_infer_native_nuisance_mask", False):
        # Preserve RTPSpy's first-volume nonzero mask fallback on its original
        # source grid, independently of the new final-grid response mask.
        first_mask = data != 0
        for label in masks:
            masks[label] &= first_mask
        handler._infer_native_nuisance_mask = False
    means = {}
    for label, mask in masks.items():
        values = data[mask & support]
        means[label] = float(values.mean()) if values.size and np.isfinite(values).all() else float("nan")
    return means


def save_normalization_reference(cfg, normalizer, affine, volume_idx):
    """Persist calibration/final sufficient statistics, never every-TR images."""
    if normalizer.mean is None:
        return
    folder = cfg.rt_work_dir / "norm"
    folder.mkdir(parents=True, exist_ok=True)
    for name, values in (("mean", normalizer.mean), ("std", normalizer.std),
                         ("count", normalizer.count)):
        if values is not None:
            nib.save(nib.Nifti1Image(np.asarray(values, dtype=np.float32), affine),
                     str(folder / f"reference_{name}.nii"))
    payload = {**normalization_spec(), "reference_count": normalizer.reference_count,
               "statistics_saved_after_volume": int(volume_idx), "ready": normalizer.ready,
               "sample_source": "pre_glm_transformed_smoothed_bold" if normalizer.mode == "mean" else "regression_ready_output"}
    (folder / "reference.json").write_text(json.dumps(payload, indent=2) + "\n")


@dataclass
class ResultEnvelope:
    scan: int
    volume_idx: int
    dicom_path: Path
    volume_timestamp: float
    success: bool
    error: Optional[str] = None
    traceback_text: Optional[str] = None
    attempts: int = 0
    raw_nii: Optional[Path] = None
    mc_nii: Optional[Path] = None
    score_input_nii: Optional[Path] = None
    score_input_orig_nii: Optional[Path] = None
    mc_data: Optional[np.ndarray] = None
    affine: Optional[np.ndarray] = None
    motion_vec: Optional[np.ndarray] = None
    meta: dict[str, Any] = field(default_factory=dict)

def log_step(step: str, vol: int, extra: str = "", start_t=None, speed=None):
    """Compact colored/clean log."""
    v = f"{vol:05d}"
    if start_t is not None:
        dt = time.perf_counter() - start_t
        if speed is not None:
            speed.stage(vol, step, dt)
        log.info(f"[{step:<5}] vol {v}  {extra}  ({dt*1000:.1f} ms)")
    else:
        log.info(f"[{step:<5}] vol {v}  {extra}")




class ScoreEventTracker:
    """Infer per-volume event/stage labels from session_metadata.json."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self._mode = None
        self._skip_first_trs = 0
        self._baseline_trs = 0
        self._n_trials = 0
        self._stage_defs: list[tuple[str, int]] = []
        self._load_metadata()

    def _load_metadata(self) -> None:
        metadata_path = self.run_dir / "session_metadata.json"
        if not metadata_path.exists():
            return
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except (OSError, json.JSONDecodeError):
            return

        if isinstance(metadata.get("fixation_display"), dict):
            fix = metadata["fixation_display"]
            self._mode = "rest"
            self._skip_first_trs = int(fix.get("skip_first_trs", 0) or 0)
            self._baseline_trs = int(fix.get("baseline_trs", 0) or 0)

        psychopy = metadata.get("psychopy")
        if not isinstance(psychopy, dict):
            return
        script = psychopy.get("script")
        if script != "rt_nf_events_parallel.py":
            return

        self._mode = "nf_events"
        self._skip_first_trs = int(psychopy.get("skip_first_trs", 0) or 0)
        self._baseline_trs = int(psychopy.get("baseline_trs", 0) or 0)
        self._n_trials = int(psychopy.get("n_trials", 0) or 0)
        self._stage_defs = [
            ("iti", int(psychopy.get("iti_trs", 0) or 0)),
            ("cue", int(psychopy.get("cue_trs", 0) or 0)),
            ("scans", int(psychopy.get("scans_trs", 0) or 0)),
            ("delay", int(psychopy.get("delay_trs", 0) or 0)),
            ("feedback", int(psychopy.get("feedback_trs", 0) or 0)),
        ]

    def for_volume(self, volume_idx: int) -> Optional[str]:
        if self._mode is None:
            return None
        if volume_idx <= self._skip_first_trs:
            return "background"

        exp_tr = volume_idx - self._skip_first_trs

        if self._mode == "rest":
            if exp_tr <= self._baseline_trs:
                return "background"
            return "rest"

        if exp_tr <= self._baseline_trs:
            return "background"

        if self._mode != "nf_events":
            return None

        stage_cycle: list[str] = []
        for stage, dur in self._stage_defs:
            stage_cycle.extend([stage] * max(0, dur))
        if not stage_cycle:
            return None

        tr_after_baseline = exp_tr - self._baseline_trs
        cycle_len = len(stage_cycle)
        trial_idx = (tr_after_baseline - 1) // cycle_len
        if trial_idx >= self._n_trials:
            return "post_task"
        stage_pos = (tr_after_baseline - 1) % cycle_len
        return stage_cycle[stage_pos]


def append_score(
    csv_path: Path,
    volume_idx: int,
    raw_score: float,
    original_score: Optional[float] = None,
    reg_ready: Optional[bool] = None,
    timestamp: Optional[float] = None,
    event_type: Optional[str] = None,
) -> float:
    if timestamp is None:
        timestamp = time.time()
    exists = csv_path.exists()

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["volume_idx", "timestamp", "score_raw", "score_original", "reg_ready", "event_type"])
        writer.writerow([volume_idx, timestamp, raw_score, original_score, int(reg_ready) if reg_ready is not None else "", event_type or ""])
    return timestamp


def append_score_z(
    csv_path: Path,
    volume_idx: int,
    timestamp: float,
    raw_score: float,
    z_score: float,
    ref_stats: dict,
    reg_ready: Optional[bool] = None,
) -> None:
    exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(
                [
                    "volume_idx",
                    "timestamp",
                    "score_raw",
                    "score_z",
                    "ref_run",
                    "ref_mean",
                    "ref_std",
                    "ref_n",
                    "ref_used_reg_ready",
                    "reg_ready",
                ]
            )
        writer.writerow(
            [
                volume_idx,
                timestamp,
                raw_score,
                z_score,
                ref_stats["run"],
                ref_stats["mean"],
                ref_stats["std"],
                ref_stats["n"],
                int(ref_stats.get("used_reg_ready", False)),
                int(reg_ready) if reg_ready is not None else "",
            ]
        )

def append_motion(motion_path: Path, motion_vec: np.ndarray):
    """
    Append a single 6-parameter motion vector to a text file (AFNI-style 1D).
    """
    motion_path.parent.mkdir(parents=True, exist_ok=True)
    with open(motion_path, "a") as f:
        f.write(" ".join(f"{x:.6f}" for x in motion_vec) + "\n")

def append_fd(fd_path: Path, volume_idx: int, fd_value: float):
    """
    Append a single FD value to a CSV file: volume_idx, fd.
    """
    fd_path.parent.mkdir(parents=True, exist_ok=True)
    exists = fd_path.exists()
    with open(fd_path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["volume_idx", "fd"])
        w.writerow([volume_idx, fd_value])

class ProcSrc:
    """Holds the current 3D volume for RTPSpy mask regressors (GS/WM/Vent)."""
    def __init__(self):
        self.proc_data = None


class RTPStyleVoxelNormalizer:
    """
    Historical scaling helper retained for compatibility with old callers.

    The live pipeline uses VoxelNormalizer after GLM instead. This legacy helper
    includes the current sample and clips values at 200, as the old path did.
    """

    def __init__(self, ref_volumes: int = 1, brain_mask: Optional[np.ndarray] = None):
        self.ref_volumes = max(1, int(ref_volumes))
        self._sum: Optional[np.ndarray] = None
        self._observations: Optional[np.ndarray] = None
        self._count = 0
        self._y_mean: Optional[np.ndarray] = None
        self._y_mean_mask: Optional[np.ndarray] = None
        self._brain_mask = brain_mask.astype(bool, copy=False) if brain_mask is not None else None
        self.output_support: Optional[np.ndarray] = None

    def apply(self, vol_data: np.ndarray, support_mask=None) -> np.ndarray:
        data = np.asarray(vol_data, dtype=np.float32)
        if support_mask is None:
            observed = np.ones(data.shape, dtype=bool)
        else:
            observed = np.asarray(support_mask)
            if observed.shape != data.shape or observed.dtype.kind != "b":
                raise ValueError("Normalization support must be boolean and match the volume grid.")
            observed = observed.copy()
        if self._brain_mask is not None:
            if self._brain_mask.shape != data.shape:
                raise ValueError("Normalization brain mask and volume grids differ.")
            observed &= self._brain_mask
        if not np.isfinite(data[observed]).all():
            raise ValueError("Nonfinite acquired values inside the normalization mask.")

        if self._count < self.ref_volumes:
            if self._sum is None:
                self._sum = np.zeros_like(data, dtype=np.float64)
                self._observations = np.zeros(data.shape, dtype=np.int32)
            self._sum[observed] += data[observed]
            self._observations[observed] += 1
            self._count += 1
            # An unavailable acquisition is not a measured zero. Freeze each
            # reference mean after this window using only its observed samples.
            self._y_mean = np.divide(self._sum, self._observations,
                                     out=np.zeros_like(self._sum),
                                     where=self._observations > 0).astype(np.float32)
            self._y_mean_mask = np.abs(self._y_mean) > 1e-6
            if self._brain_mask is not None:
                self._y_mean_mask &= self._brain_mask
            if self._count >= self.ref_volumes:
                self._sum = None
                self._observations = None

        out = np.zeros_like(data, dtype=np.float32)
        mask = self._y_mean_mask & observed
        self.output_support = mask
        out[mask] = data[mask] / self._y_mean[mask] * 100.0
        out[out > 200.0] = 200.0
        return out

class MotionRegressor:
    def __init__(
        self,
        volreg: RtpVolreg,
        reg_mask: Optional[Path] = None,
        gs_mask: Optional[Path] = None,
        wm_mask: Optional[Path] = None,
        vent_mask: Optional[Path] = None,
        mot_reg: str = "mot6",
        max_poly_order: float = np.inf,
        TR: float = 1,
        max_scan_length: int = 1000,
        norm_ref_volumes: int = 1,
        enable_fd_censor_reg: bool = False,
        enable_dvars_censor_reg: bool = False,
        phys_reg: str = "None",
        rtp_physio: Optional[object] = None,
        normalization_mode: Optional[str] = None,
        normalization_reference: str = "fixed",
        startup_residual_transform=None,
    ):
        wait_num = max(0, int(norm_ref_volumes) - 1)
        kwargs = dict(
            mot_reg=mot_reg,
            volreg=volreg,
            mask_file=str(reg_mask) if reg_mask is not None else 0,
            TR=TR,
            wait_num=wait_num,
            max_poly_order=max_poly_order,
            save_proc=False,
            online_saving=False,
            reg_retro_proc=False,
            max_scan_length=max_scan_length,
            spike_reg_num=200,
            phys_reg=phys_reg,
            rtp_physio=rtp_physio,
            scale_before_glm=False,
            startup_ref_volumes=(int(norm_ref_volumes) if normalization_mode == "zscore" else None),
            startup_ref_reference=normalization_reference,
            startup_residual_transform=startup_residual_transform,
        )
        self._regress = RtpRegress(**kwargs)

        if gs_mask is not None:
            self._regress.set_param("GS_reg", True)
            self._regress.set_param("GS_mask", str(gs_mask))
        if wm_mask is not None:
            self._regress.set_param("WM_reg", True)
            self._regress.set_param("WM_mask", str(wm_mask))
        if vent_mask is not None:
            self._regress.set_param("Vent_reg", True)
            self._regress.set_param("Vent_mask", str(vent_mask))

        self._ready = False

    def apply(
        self,
        mc_img: nib.Nifti1Image,
        volume_idx: int,
        fd_censor: int = 0,
        dvars_censor: int = 0,
        support_mask=None,
        nuisance_means=None,
    ) -> tuple[np.ndarray, bool]:
        """
        Returns (cleaned_vol, regressed_bool). Always returns a tuple.
        """

        self._regress.last_proc_ready = False
        self._regress.output_support = np.zeros(mc_img.shape[:3], dtype=bool)

        # Ensure regressor init/ready state
        if not self._ready:
            try:
                self._ready = bool(self._regress.ready_proc())
            except Exception as exc:
                log.error(f"[REG] Failed to prepare regressor: {exc}")
                return np.zeros(mc_img.shape[:3], dtype=np.float32), False

        if not self._ready:
            return np.zeros(mc_img.shape[:3], dtype=np.float32), False

        # Do regression
        try:
            # RTPSpy modifies mc_img in-place
            self._regress.do_proc(
                mc_img,
                vol_idx=volume_idx - 1,
                fd_censor=fd_censor,
                dvars_censor=dvars_censor,
                support_mask=support_mask,
                nuisance_means=nuisance_means,
            )

            regressed = bool(getattr(self._regress, "last_proc_ready", False))

            cleaned = np.asarray(mc_img.dataobj, dtype=np.float32)
            return cleaned, regressed

        except Exception as exc:
            log.error(f"[REG] Motion regression failed at vol {volume_idx:05d}: {exc}")
            self._regress.output_support.fill(False)
            return np.zeros(mc_img.shape[:3], dtype=np.float32), False

    @property
    def output_support(self):
        """Brain voxels with observations throughout this run's fitted history."""
        return self._regress.output_support

    @property
    def brain_mask(self):
        return self._regress.maskV

    def take_startup_reference(self):
        """Consume the first fit's residual moments without replaying volumes."""
        moments = self._regress.startup_residual_moments
        self._regress.startup_residual_moments = None
        return moments

    def get_regressors(self, volume_idx: int) -> tuple[Optional[list[str]], Optional[np.ndarray]]:
        des_mtx = getattr(self._regress, "desMtx", None)
        if des_mtx is None:
            return None, None

        idx = volume_idx - 1
        try:
            row = des_mtx[idx]
        except Exception:
            return None, None

        if hasattr(row, "detach"):
            row = row.detach().cpu().numpy()

        row = np.asarray(row, dtype=float)

        reg_names = list(getattr(self._regress, "reg_names", []))
        if len(reg_names) < row.shape[0]:
            extra = row.shape[0] - len(reg_names)
            reg_names.extend([f"poly_{i:02d}" for i in range(extra)])

        return reg_names, row





# ---------- Simple config for this RT session ----------

@dataclass
class RTSessionConfig:
    subject: str
    day: str
    run: str
    incoming_root: Path
    base_data: Path
    decoder_template: Optional[Path] = None
    decoder_roi_txt: Optional[Path] = None
    reference_score_run: Optional[str] = None
    reference_score_stats: Optional[dict] = None
    enable_scoring: bool = True
    enable_original_score: bool = False
    test_speed: bool = False
    # Reference runs establish the current-session spatial assets. NF only
    # consumes those assets, whether fitted online or anchored to offline data.
    preparation_only: bool = False
    # Process one fixed existing-DICOM snapshot and return without a watchdog.
    recorded_only: bool = False

    # NEW
    ap_block: Optional[int] = None
    pa_block: Optional[int] = None

    @property
    def fmap_dir(self) -> Path:
        if self.ap_block is None and self.pa_block is None:
            return self.day_root / "fmap"

        if self.ap_block is None or self.pa_block is None:
            raise ValueError("Set both --ap-block and --pa-block, or neither.")

        return (
            self.day_root
            / "fmap"
            / f"pair-ap{int(self.ap_block):03d}_pa{int(self.pa_block):03d}"
        )

    @property
    def subject_root(self) -> Path:
        return self.base_data / f"sub-{self.subject}"

    @property
    def day_root(self) -> Path:
        return self.subject_root / self.day

    @property
    def trans_dir(self) -> Path:
        return self.day_root / "func" / "trans"

    @property
    def rt_wm_mask(self) -> Path:
        return self.trans_dir / "rt_WM_mask.nii"

    @property
    def rt_vent_mask(self) -> Path:
        return self.trans_dir / "rt_Vent_mask.nii"

    @property
    def rt_work_dir(self) -> Path:
        """
        Where we put per-volume NIfTIs, logs, etc.

        Runs are stored under func/XXX, where XXX corresponds to the middle
        element of the DICOM name (historically called "block").
        """
        d = self.day_root / "func" / self.run
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def incoming_dir(self) -> Path:
        # The full incoming folder, you gave an example root:
        return self.incoming_root

    @property
    def rt_raw_dir(self) -> Path:
        d = self.rt_work_dir / "raw"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def rt_mc_dir(self) -> Path:
        d = self.rt_work_dir / "mc"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def rt_reg_dir(self) -> Path:
        d = self.rt_work_dir / "reg"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def rt_mni_dir(self) -> Path:
        d = self.rt_work_dir / "mni"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def rt_unwarp_dir(self) -> Path:
        d = self.rt_work_dir / "unwarped"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def rt_distorted_motion_ref_epi(self) -> Path:
        """
        Legacy filename retained for external callers; not the online MC target.
        Older sessions contain distorted data; new preparations contain corrected data.
        """
        return self.trans_dir / "rt_ref_epi.nii"

    @property
    def rt_distorted_motion_ref_mask(self) -> Path:
        """
        Legacy reference-mask filename; not the online DVARS mask.
        """
        return self.trans_dir / "rt_ref_epi_mask.nii"

    @property
    def rt_unwarped_analysis_ref_epi(self) -> Path:
        """
        Unwarped EPI reference used for EPI->T1/MNI transform grids.
        """
        return _prefer_uncompressed_nifti(self.trans_dir / "epi_unwarped_mean.nii")

    @property
    def rt_unwarped_analysis_ref_mask(self) -> Path:
        """
        Brain mask in the unwarped EPI reference grid.
        """
        return _prefer_uncompressed_nifti(self.trans_dir / "epi_mask_mean.nii")

    @property
    def rt_motion_ref_epi(self) -> Path:
        """Corrected reference used for motion correction after unwarping."""
        return self.rt_unwarped_analysis_ref_epi

    @property
    def rt_motion_ref_mask(self) -> Path:
        """Corrected reference mask used after unwarping and motion correction."""
        return self.rt_unwarped_analysis_ref_mask

    @property
    def rt_ref_epi(self) -> Path:
        """Deprecated alias for the unwarped analysis reference."""
        return self.rt_unwarped_analysis_ref_epi

    @property
    def rt_ref_mask(self) -> Path:
        """Deprecated alias for the unwarped analysis-reference mask."""
        return self.rt_unwarped_analysis_ref_mask


def resolve_decoder_template(cfg: RTSessionConfig) -> Path:
    return cfg.decoder_template or (
        Path(cfg.base_data).parent
        / "decoders"
        / "rweights_NSF_grouppred_cvpcrTMP_nonzeros.nii"
    )


def resolve_analysis_reference(cfg, space):
    if space != "epi":
        return resolve_decoder_template(cfg)
    reference = epi_output_reference(cfg)
    decoder = getattr(cfg, "decoder_template", None)
    if decoder is not None:
        try:
            require_same_grid(nib.load(str(decoder)), nib.load(str(reference)))
        except ValueError as exc:
            raise ValueError("EPI decoder/template must match the stable EPI output grid; "
                             "use a decoder prepared on reference_epi.nii.") from exc
    return reference


def prepare_decoder_correlation_qc(cfg, handler):
    """Freeze the spatial QC reference/ROI once; never use normalized features.

    A supplied training mean tests agreement with decoder construction. Otherwise
    the corrected session reference provides a labelled within-session check.
    Only the QC ROI is restricted to the reference's observable brain; decoder
    weights and the analysis mask remain untouched.
    """
    enabled = bool(getattr(REGRESSOR_SETTINGS, "enable_decoder_correlation_qc", True))
    scorer = getattr(handler, "scorer", None)
    if not enabled or scorer is None:
        return None
    space = str(REGRESSOR_SETTINGS.analysis_space).lower()
    folder = cfg.rt_work_dir / "decoder_qc"
    folder.mkdir(parents=True, exist_ok=True)
    analysis_path = resolve_analysis_reference(cfg, space)
    analysis_image = nib.load(str(analysis_path))
    source = cfg.rt_unwarped_analysis_ref_epi
    source_image = nib.load(str(source))
    bridge = session_epi_bridge(cfg) if space == "epi" else None
    spatial = space != "epi" or bridge is not None
    brain = getattr(handler, "final_brain_mask", None)
    explicit_reference = getattr(REGRESSOR_SETTINGS, "decoder_qc_reference", None)
    roi_path = getattr(REGRESSOR_SETTINGS, "decoder_qc_roi_mask", None)
    decoder_identity = source_identity(resolve_decoder_template(cfg))
    roi_txt = getattr(cfg, "decoder_roi_txt", None)
    # Bind inputs before overwriting any generated reference/support metadata.
    # Generated fallback mtimes are deliberately excluded: the same request may
    # be restarted, but existing rows must never change their reference or ROI.
    import hashlib
    from fmri_rt_preproc.offline_assets import file_identity
    def mask_identity(values):
        if values is None:
            return None
        array = np.ascontiguousarray(np.asarray(values, dtype=bool))
        return {"shape": list(array.shape), "sha256": hashlib.sha256(array.tobytes()).hexdigest()}
    transforms = ([bridge] if bridge is not None else
                  [cfg.trans_dir / "epi2t1_Composite.h5"] if spatial else [])
    if space == "mni":
        transforms.insert(0, t1_to_mni_transform(cfg))
    request = {"version": 2, "analysis_space": space,
               "analysis_grid": {"shape": list(analysis_image.shape[:3]), "affine": analysis_image.affine.tolist()},
               "session_reference": file_identity(source),
               "explicit_reference": file_identity(Path(explicit_reference).expanduser()) if explicit_reference else None,
               "decoder": decoder_identity,
               "decoder_roi_txt": source_identity(roi_txt) if roi_txt is not None and roi_txt.is_file() else None,
               "decoder_mask": mask_identity(scorer.mask), "final_brain_mask": mask_identity(brain),
               "explicit_roi": file_identity(Path(roi_path).expanduser()) if roi_path else None,
               "transforms": [file_identity(transform) for transform in transforms],
               "backend": "gpu_resampler" if spatial and handler.gpu_resampler is not None else "ants" if spatial else "identity",
               "interpolation": str(getattr(REGRESSOR_SETTINGS, "gpu_resampler_interpolation", "linear"))
               if spatial and handler.gpu_resampler is not None else "linear"}
    binding_path = cfg.rt_work_dir / "decoder_qc_request.json"
    csv_path = cfg.rt_work_dir / "decoder_qc_rt.csv"
    if csv_path.exists() and csv_path.stat().st_size:
        try:
            previous = json.loads(binding_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = None
        if previous != request:
            raise ValueError("Existing decoder QC rows use a different QC version or incompatible inputs. "
                             "Use a fresh run directory to preserve the existing files.")
    binding_temporary = binding_path.with_suffix(".json.tmp")
    binding_temporary.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")
    binding_temporary.replace(binding_path)
    # The same support mapping used for incoming volumes defines a fixed ROI.
    # This is independent of the first TR's motion and of cGLM history.
    reference_support = final_volume_support(
        cfg, handler, np.ones(source_image.shape[:3], dtype=bool),
        source_image.affine, 0, space, folder_name="decoder_qc/reference_support")
    if brain is not None:
        reference_support &= brain
    reference_path = Path(explicit_reference).expanduser().resolve() if explicit_reference else folder / "session_reference.nii"
    if not explicit_reference:
        if not spatial:
            require_same_grid(source_image, analysis_image)
            nib.save(nib.Nifti1Image(np.asarray(source_image.dataobj, dtype=np.float32),
                                    analysis_image.affine), str(reference_path))
        elif handler.gpu_resampler is not None:
            handler.gpu_resampler.resample_nifti_to_nifti(source, reference_path)
        else:
            transforms = [bridge] if bridge is not None else [cfg.trans_dir / "epi2t1_Composite.h5"]
            if space == "mni":
                transforms.insert(0, t1_to_mni_transform(cfg))
            command = ["antsApplyTransforms", "-d", "3", "-i", str(source),
                       "-r", str(analysis_path), "-o", str(reference_path),
                       "-n", "Linear", "--float", "1"]
            for transform in transforms:
                command.extend(["-t", str(transform)])
            run(command)
    qc = DecoderSpatialCorrelation(
        reference_path, scorer.mask, analysis_image=analysis_image,
        fixed_roi_mask=reference_support,
        decoder_weights=getattr(scorer, "w", None),
        roi_image=Path(roi_path).expanduser().resolve() if roi_path else None,
        reference_kind="decoder_training_mean" if explicit_reference else "prepared_session_reference",
        reference_label="Decoder-training mean EPI" if explicit_reference else "Prepared session EPI reference (fallback)")
    metadata = dict(qc.metadata)
    metadata.update(stage="after_spatial_before_smoothing_glm_normalization",
                    analysis_space=space, reference=source_identity(reference_path),
                    decoder=decoder_identity,
                    roi_file=source_identity(Path(roi_path).expanduser().resolve()) if roi_path else None,
                    session_reference=source_identity(source),
                    reporting_only=True, missing_coverage_policy="pearson_on_available_fixed_roi_voxels",
                    weight_loss_basis="squared_weights_of_full_decoder_mask_before_temporal_processing")
    nib.save(nib.Nifti1Image(qc.roi.astype(np.uint8), analysis_image.affine),
             str(folder / "fixed_roi.nii"))
    metadata_path = cfg.rt_work_dir / "decoder_qc_reference.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    session_path = cfg.rt_work_dir / "session_metadata.json"
    if session_path.exists():
        session = json.loads(session_path.read_text(encoding="utf-8"))
        session["decoder_correlation_qc"] = metadata
        session_path.write_text(json.dumps(session, indent=2) + "\n", encoding="utf-8")
    log.info("[QC] Spatial decoder-ROI correlation: %s; fixed ROI %s voxels",
             metadata["reference_kind"], int(qc.roi.sum()))
    return qc


def record_decoder_correlation(cfg, handler, volume_idx, image_path, support):
    """Record raw spatial agreement for every TR, including GLM warmup."""
    qc = getattr(handler, "decoder_qc", None)
    if qc is None:
        return {"reference_correlation": float("nan"),
                "reference_coverage_fraction": float("nan"),
                "correlation_status": "disabled" if not getattr(
                    REGRESSOR_SETTINGS, "enable_decoder_correlation_qc", True) else "no_voxel_decoder",
                "decoder_weight_loss_fraction": float("nan")}
    image = image_path if hasattr(image_path, "dataobj") else nib.load(str(image_path))
    result = qc.evaluate(image, coverage=support).as_dict()
    path = cfg.rt_work_dir / "decoder_qc_rt.csv"
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["volume_idx", *result])
        if not exists:
            writer.writeheader()
        writer.writerow({"volume_idx": volume_idx, **result})
    return result


def maybe_init_gpu_resampler(cfg: RTSessionConfig):
    interpolation = str(getattr(REGRESSOR_SETTINGS, "gpu_resampler_interpolation", "linear")).lower()
    if interpolation not in {"linear", "lanczos"}:
        raise ValueError("gpu_resampler_interpolation must be 'linear' or 'lanczos'.")
    if not bool(getattr(REGRESSOR_SETTINGS, "use_gpu_resampler", False)):
        if interpolation == "lanczos":
            raise ValueError("Lanczos requires use_gpu_resampler=true.")
        return None

    analysis_space = str(REGRESSOR_SETTINGS.analysis_space).lower()
    bridge = session_epi_bridge(cfg) if analysis_space == "epi" else None
    if analysis_space not in {"t1", "mni"} and bridge is None:
        log.info("[GPU] Disabled: analysis_space=%s does not require warp.", analysis_space)
        return None

    from gpu_ants_like_resampler import GpuAntsLikeResampler, precompute_sampling_grid

    if bridge is not None:
        fixed_ref = resolve_analysis_reference(cfg, "epi")
        transforms = [bridge]
        out_dir = cfg.trans_dir / "gpu_grids" / "epi_to_reference"
    elif analysis_space == "t1":
        fixed_ref = resolve_decoder_template(cfg)
        transforms = [cfg.trans_dir / "epi2t1_Composite.h5"]
        out_dir = cfg.trans_dir / "gpu_grids" / "epi_to_t1"
    else:
        fixed_ref = resolve_decoder_template(cfg)
        transforms = [
            t1_to_mni_transform(cfg),
            cfg.trans_dir / "epi2t1_Composite.h5",
        ]
        out_dir = cfg.trans_dir / "gpu_grids" / "epi_to_mni"

    moving_ref = cfg.rt_unwarped_analysis_ref_epi
    missing = [p for p in [fixed_ref, *transforms, moving_ref] if not p.exists()]
    if missing:
        if interpolation == "lanczos":
            raise FileNotFoundError(f"Cannot initialize Lanczos resampling; missing inputs: {missing}")
        log.warning("[GPU] Disabled: missing inputs for GPU resampler: %s", missing)
        return None

    # The builder validates both geometries and the ordered transform contents;
    # grids from older versions without that provenance are rebuilt once.
    grid_path = precompute_sampling_grid(
        moving_ref=moving_ref,
        fixed_ref=fixed_ref,
        transforms=transforms,
        out_dir=out_dir,
        interpolation=interpolation,
    )

    device = str(getattr(REGRESSOR_SETTINGS, "gpu_resampler_device", "cuda"))
    resampler = GpuAntsLikeResampler(grid_path=grid_path, device=device, mode="bilinear",
                                    interpolation=interpolation)
    log.info("[GPU] Resampler ready (%s, device=%s, interpolation=%s).",
             grid_path, resampler.device, interpolation)
    return resampler


def _build_pyhysco_applier(
    cfg: RTSessionConfig,
    prototype_vol_path: Path,
) -> Optional[PreloadedPyHyscoApplier]:
    method = str(getattr(REGRESSOR_SETTINGS, "fieldmap_method", "pyhysco")).lower()
    if method != "pyhysco":
        return None
    if not bool(getattr(REGRESSOR_SETTINGS, "use_preloaded_pyhysco", True)):
        return None

    fmap_dir = cfg.fmap_dir
    pyhysco_field = _preferred_pyhysco_fieldmap(fmap_dir)
    if not pyhysco_field.exists():
        log.warning("[FMAP] Preloaded PyHySCO disabled: missing fieldmap at %s", pyhysco_field)
        return None
    if not prototype_vol_path.exists():
        log.warning("[FMAP] Preloaded PyHySCO disabled: missing prototype volume at %s", prototype_vol_path)
        return None

    epi_pe = str(getattr(REGRESSOR_SETTINGS, "epi_phase_encoding", "PA")).upper()
    polarity = 1 if epi_pe == "AP" else -1
    phase_encoding_direction = load_calibration(fmap_dir)["phase_encoding_axis"]
    device = str(getattr(REGRESSOR_SETTINGS, "pyhysco_device", "cuda"))
    backend = str(getattr(REGRESSOR_SETTINGS, "pyhysco_backend", "grid_sample")).lower()
    applier = PreloadedPyHyscoApplier(
        prototype_vol_path=prototype_vol_path,
        fieldmap_path=pyhysco_field,
        phase_encoding_direction=phase_encoding_direction,
        polarity=polarity,
        device=device,
        dtype=torch.float32,
        interpolation_backend=backend,
    )
    log.info(
        "[FMAP] Preloaded PyHySCO applier ready (device=%s, backend=%s, prototype=%s).",
        applier.device,
        backend,
        prototype_vol_path,
    )
    return applier


def _resolve_pyhysco_fieldmap(fmap_dir: Path) -> Path:
    return native_paths(fmap_dir)[1]


def maybe_init_pyhysco_applier(cfg: RTSessionConfig) -> Optional[PreloadedPyHyscoApplier]:
    # Prototype is calibration-local, independent of the session motion target.
    return _build_pyhysco_applier(cfg, native_paths(cfg.fmap_dir)[0] / "AP_mean.nii")


def _prefer_uncompressed_nifti(path_nii: Path) -> Path:
    """Prefer .nii, but transparently fall back to .nii.gz when needed."""
    if path_nii.exists():
        return path_nii
    gz = path_nii.with_suffix(path_nii.suffix + ".gz")
    if gz.exists():
        return gz
    return path_nii


def resolve_decoder_roi_txt(cfg: RTSessionConfig) -> Optional[Path]:
    if cfg.decoder_roi_txt is None:
        return None
    if not cfg.decoder_roi_txt.exists():
        log.warning("[SCORE] Decoder ROI txt not found at %s; falling back to decoder NIfTI mask.", cfg.decoder_roi_txt)
        return None
    return cfg.decoder_roi_txt


def load_reference_score_stats(cfg: RTSessionConfig, run_id: Optional[str]) -> Optional[dict]:
    if not run_id:
        return None
    run_dir = cfg.day_root / "func" / str(run_id)
    scores_path = run_dir / "scores.csv"
    if not scores_path.exists():
        log.warning("[SCORE] Reference scores.csv not found at %s", scores_path)
        return None

    metadata_path = run_dir / "session_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid reference session metadata in {metadata_path}")
    # Older behavioral references contain scores and optional readiness logs,
    # without a preprocessing contract. Do not invent one for those recordings.
    legacy_reference = not any(key in metadata for key in (
        "normalization", "coverage_policy", "processing_mask", "temporal_processing_order",
    ))
    recorded_normalization = metadata.get("normalization")
    current_normalization = normalization_spec()
    if legacy_reference:
        log.warning(
            "[SCORE] Legacy score reference in %s has no preprocessing metadata; "
            "using its recorded scores and legacy sample selection.", run_dir,
        )
    elif recorded_normalization != current_normalization:
        # A score reference is an explicit analysis choice. Preserve its
        # recorded units for review without holding up the live run.
        log.warning(
            "[SCORE] Reference voxel normalization differs or is unrecorded in %s; "
            "using the selected score reference. Recorded=%s; current=%s",
            run_dir, recorded_normalization, current_normalization,
        )
    # RS and NF can use different source masks/grids and different amounts of
    # padding (for example native RS EPI versus a bridged NF EPI). Compare the
    # processing method, not those acquisition-specific identities. A changed
    # method can change decoder-score statistics even with identical voxel units.
    expected_processing = {
        "policy": FINAL_PROCESSING_MASK_POLICY,
        "version": FINAL_PROCESSING_MASK_POLICY_VERSION,
        "nuisance_source": FINAL_NUISANCE_SOURCE,
        "processing_order": list(FINAL_PROCESSING_ORDER),
        "interpolation": str(getattr(REGRESSOR_SETTINGS, "gpu_resampler_interpolation", "linear")).lower(),
    }
    recorded_processing = metadata.get("processing_mask")
    if isinstance(recorded_processing, dict):
        # Zero-FWHM references have no Gaussian operation to reorder. Preserve
        # them only when both recorded and current settings explicitly agree.
        legacy_zero_width = (metadata.get("smoothing_fwhm_mm")
                             if getattr(REGRESSOR_SETTINGS, "smoothing_fwhm_mm", 0) == 0 else None)
        try:
            recorded_processing = canonical_final_processing_mask(recorded_processing, legacy_zero_width)
        except ValueError:
            recorded_processing = None
    if not legacy_reference and (metadata.get("coverage_policy") != COVERAGE_POLICY
            or not isinstance(recorded_processing, dict)
            or any(recorded_processing.get(key) != value
                   for key, value in expected_processing.items())):
        raise ValueError(
            f"Reference scores in {run_dir} use a different or unrecorded spatial masking/interpolation policy. "
            "Collect or reprocess a fresh RS reference with the current preprocessing settings; "
            "the existing reference scores cannot be reused."
        )

    reg_ready_map = None
    reg_status_path = run_dir / "regression_status_rt.csv"
    if reg_status_path.exists():
        reg_ready_map = {}
        with open(reg_status_path, newline="") as f:
            reader = csv.DictReader(f)
            if "volume_idx" in (reader.fieldnames or []) and "reg_ready" in (reader.fieldnames or []):
                for row in reader:
                    try:
                        vol = int(row["volume_idx"])
                        reg_ready_map[vol] = bool(int(row["reg_ready"]))
                    except (TypeError, ValueError):
                        continue
            else:
                log.warning("[SCORE] reg_ready column missing in %s", reg_status_path)
                reg_ready_map = None

    scores = []
    scores_all = []
    skipped_first_reg_ready = False
    with open(scores_path, newline="") as f:
        reader = csv.DictReader(f)
        has_score_reg_ready = bool(reader.fieldnames and "reg_ready" in reader.fieldnames)
        for row in reader:
            if row.get("reference_selection_role") == "first_ready_transient":
                # Legacy gameplay exports explicitly retain this row for the
                # old loader to drop. Modern exports contain gameplay rows only.
                skipped_first_reg_ready = True
                continue
            try:
                vol = int(row["volume_idx"])
                raw = float(row["score_raw"])
            except (TypeError, ValueError, KeyError):
                continue
            if not np.isfinite(raw):
                continue
            scores_all.append(raw)
            score_reg_ready = None
            if has_score_reg_ready:
                try:
                    score_reg_ready = bool(int(row["reg_ready"]))
                except (TypeError, ValueError, KeyError):
                    score_reg_ready = None
            ready = reg_ready_map.get(vol, False) if reg_ready_map is not None else score_reg_ready
            if ready is None:
                ready = legacy_reference
            if ready:
                scores.append(raw)

    used_reg_ready = (reg_ready_map is not None) or has_score_reg_ready
    if legacy_reference:
        # Preserve the historical loader's first-ready transient exclusion.
        # An explicitly marked transient was already removed above.
        if scores and not skipped_first_reg_ready:
            scores = scores[1:]
            skipped_first_reg_ready = True
        if used_reg_ready and not scores:
            log.warning(
                "[SCORE] No usable ready legacy reference scores in %s; "
                "falling back to its finite recorded scores.", scores_path,
            )
            scores = scores_all
            used_reg_ready = False
    # Modern normalization gates the first valid score after calibration, so
    # its first ready value is retained unless explicitly marked as transient.
    if used_reg_ready and not scores:
        log.warning(
            "[SCORE] No normalization-ready reference scores found in %s.",
            scores_path,
        )

    if len(scores) < 2:
        log.warning("[SCORE] Not enough reference scores in %s to compute z-score.", scores_path)
        return None

    values = np.asarray(scores, dtype=float)
    mean = float(np.nanmean(values))
    std = float(np.nanstd(values))
    if not np.isfinite(std) or std <= 0:
        log.warning("[SCORE] Reference score std invalid (%s) in %s.", std, scores_path)
        return None

    return {
        "run": str(run_id),
        "mean": mean,
        "std": std,
        "n": int(len(scores)),
        "used_reg_ready": used_reg_ready,
        "skipped_first_reg_ready": skipped_first_reg_ready,
    }

def _prefer_uncompressed_nifti(path_nii: Path) -> Path:
    if path_nii.exists():
        return path_nii
    gz = path_nii.with_suffix(path_nii.suffix + ".gz")
    if gz.exists():
        return gz
    return path_nii


def _preferred_pyhysco_fieldmap(fmap_dir: Path) -> Path:
    return native_paths(fmap_dir)[1]


def _fieldmap_ready(cfg: RTSessionConfig) -> bool:
    if str(REGRESSOR_SETTINGS.fieldmap_method).lower() != "pyhysco":
        raise ValueError("unwarp -> MC supports PyHySCO only; select fieldmap_method=pyhysco.")
    try:
        load_calibration(cfg.fmap_dir)
        return True
    except (OSError, ValueError):
        return False


def ensure_fieldmap_only(cfg: RTSessionConfig) -> None:
    """
    Ensure the selected AP/PA pair has fieldmap outputs.

    This is RT-side prerequisite checking.
    It does not run full preprocessing.
    It does not require structural block.
    It does not require a separate EPI block.
    It averages raw AP/PA without motion correction or a session reference.
    """

    if (cfg.ap_block is None) != (cfg.pa_block is None):
        raise ValueError("Set both --ap-block and --pa-block, or neither.")
    if _fieldmap_ready(cfg):
        log.info("[FMAP] Native fieldmap ready in %s", cfg.fmap_dir)
        return

    log.warning(
        "[FMAP] Fieldmap missing for AP/PA pair ap=%s pa=%s in %s; building fieldmap only.",
        cfg.ap_block,
        cfg.pa_block,
        cfg.fmap_dir,
    )

    # Reuse the DICOM staging/conversion helper from run_preproc.
    # It copies only the requested AP/PA DICOM blocks and converts them to fixed names.
    from run_preproc import _stage_convert_to_target
    from fmri_rt_preproc.pipeline import FMRIRealtimePreprocessor
    from types import SimpleNamespace

    cfg.fmap_dir.mkdir(parents=True, exist_ok=True)

    ap_path = cfg.fmap_dir / "AP.nii.gz"
    pa_path = cfg.fmap_dir / "PA.nii.gz"

    if cfg.ap_block is not None:
        _stage_convert_to_target(cfg.incoming_dir, int(cfg.ap_block), ap_path)
        _stage_convert_to_target(cfg.incoming_dir, int(cfg.pa_block), pa_path)
    ap_path = find_source(cfg.fmap_dir, "AP")
    pa_path = find_source(cfg.fmap_dir, "PA")

    # Minimal config object for FMRIRealtimePreprocessor._prepare_fieldmap().
    fieldmap_cfg = SimpleNamespace(
        subject_id=cfg.subject,
        day_id=cfg.day,
        root=cfg.day_root,
        subject_root=cfg.subject_root,
        ap_file=ap_path,
        pa_file=pa_path,
        fieldmap_dir=cfg.fmap_dir,
        fieldmap_method=str(REGRESSOR_SETTINGS.fieldmap_method).lower(),
        epi_phase_encoding=str(REGRESSOR_SETTINGS.epi_phase_encoding).upper(),
        decoder_template=Path("unused_decoder_template.nii"),
    )

    pipe = FMRIRealtimePreprocessor(fieldmap_cfg)
    pipe._prepare_fieldmap()

    if not _fieldmap_ready(cfg):
        raise FileNotFoundError(
            f"[FMAP] Fieldmap-only build finished, but required outputs are still missing in {cfg.fmap_dir}"
        )

    log.info("[FMAP] Fieldmap-only build complete: %s", cfg.fmap_dir)

def write_session_metadata(cfg: RTSessionConfig, decoder_template: Path) -> None:
    metadata_path = cfg.rt_work_dir / "session_metadata.json"
    payload = {}
    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            payload = {}
    payload.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "subject": cfg.subject,
            "day": cfg.day,
            "run": cfg.run,
            "incoming_root": str(cfg.incoming_root),
            "base_data": str(cfg.base_data),
            "decoder_template": str(decoder_template),
            "decoder_roi_txt": str(cfg.decoder_roi_txt) if cfg.decoder_roi_txt else None,
            "reference_score_run": cfg.reference_score_run,
            "reference_score_stats": cfg.reference_score_stats,
            "enable_original_score": cfg.enable_original_score,
            "preprocessing_order": PIPELINE_ORDER,
            "temporal_processing_order": TEMPORAL_PROCESSING_ORDER,
            "final_output": {"folder": "norm", "suffix": "norm",
                             "space": str(REGRESSOR_SETTINGS.analysis_space).lower()},
            "decoder_contributions": {
                "enabled": bool(cfg.enable_scoring and getattr(REGRESSOR_SETTINGS, "save_decoder_contributions", False)),
                "folder": "decoder_contributions", "input_folder": "norm",
                "score": "score_raw", "bias_included": False,
                "ready_finite_scores_only": True,
            },
            "coverage_space": str(REGRESSOR_SETTINGS.analysis_space).lower(),
            "coverage_policy": COVERAGE_POLICY,
            "processing_mask": processing_mask_spec(cfg),
            "normalization": normalization_spec(),
            "epi_reference_mapped": (str(REGRESSOR_SETTINGS.analysis_space).lower() == "epi"
                                     and session_epi_bridge(cfg) is not None),
            "smoothing_fwhm_mm": float(getattr(REGRESSOR_SETTINGS, "smoothing_fwhm_mm", 0)),
            "analysis_mask": getattr(REGRESSOR_SETTINGS, "analysis_mask", "whole_brain"),
            "analysis_mask_file": getattr(REGRESSOR_SETTINGS, "analysis_mask_file", None),
            "analysis_mask_space": getattr(REGRESSOR_SETTINGS, "analysis_mask_space", "final"),
            "fieldmap_dir": str(cfg.fmap_dir),
            "motion_reference": str(cfg.rt_motion_ref_epi),
            "tr": REGRESSOR_SETTINGS.TR,
            "regression": {
                "enable_motion_regression": REGRESSOR_SETTINGS.enable_motion_regression,
                "mot_reg": REGRESSOR_SETTINGS.mot_reg,
                "max_poly_order": REGRESSOR_SETTINGS.max_poly_order,
                "use_gs": REGRESSOR_SETTINGS.use_gs,
                "use_wm": REGRESSOR_SETTINGS.use_wm,
                "use_vent": REGRESSOR_SETTINGS.use_vent,
                "enable_fd_censor_reg": REGRESSOR_SETTINGS.enable_fd_censor_reg,
                "fd_thr": REGRESSOR_SETTINGS.fd_thr,
                "enable_dvars_censor_reg": REGRESSOR_SETTINGS.enable_dvars_censor_reg,
                "dvars_thr_robust_z": REGRESSOR_SETTINGS.dvars_thr_robust_z,
                "censor_plus1": REGRESSOR_SETTINGS.censor_plus1,
                "dvars_warmup": REGRESSOR_SETTINGS.dvars_warmup,
                "dvars_mask_source": REGRESSOR_SETTINGS.dvars_mask_source,
                "analysis_space": REGRESSOR_SETTINGS.analysis_space,
                "voxel_norm_ref_volumes": REGRESSOR_SETTINGS.voxel_norm_ref_volumes,
                "voxel_norm_mode": getattr(REGRESSOR_SETTINGS, "voxel_norm_mode", "zscore"),
                "voxel_norm_reference": getattr(REGRESSOR_SETTINGS, "voxel_norm_reference", "fixed"),
                "rt_max_scan_length": REGRESSOR_SETTINGS.rt_max_scan_length,
                "pipeline_engine": REGRESSOR_SETTINGS.pipeline_engine,
                "commit_wait_timeout_s": REGRESSOR_SETTINGS.commit_wait_timeout_s,
            },
            "biopac": {
                "enabled": REGRESSOR_SETTINGS.enable_biopac_physio,
                "phys_reg": REGRESSOR_SETTINGS.biopac_phys_reg,
                "mode": REGRESSOR_SETTINGS.biopac_mode,
                "file": str(REGRESSOR_SETTINGS.biopac_file) if REGRESSOR_SETTINGS.biopac_file else None,
                "host": REGRESSOR_SETTINGS.biopac_host,
                "port": REGRESSOR_SETTINGS.biopac_port,
                "timeout": REGRESSOR_SETTINGS.biopac_timeout,
                "handshake": REGRESSOR_SETTINGS.biopac_handshake,
                "start_online_only": REGRESSOR_SETTINGS.biopac_start_online_only,
                "poll_interval": REGRESSOR_SETTINGS.biopac_poll_interval,
            },
        }
    )
    temporary = metadata_path.with_name(".session_metadata.json")
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    temporary.replace(metadata_path)


# ---------- Filename parsing ----------

def parse_dicom_name(name: str):
    """
    Parse a Siemens-like DICOM filename: 001_000004_000003.dcm
    Returns (series_id, run_id, scan).

    001_000004_000003
     ^    ^       ^
     |    |       +-- scan (volume index)
     |    +---------- "block" in your description
     +--------------- constant "001"
    """
    stem = Path(name).stem
    parts = stem.split("_")
    if len(parts) != 3:
        return None
    series_str, block_str, scan_str = parts
    try:
        series_id = int(series_str)
        run_id = int(block_str)
        scan = int(scan_str)
    except ValueError:
        return None
    return series_id, run_id, scan


# ---------- Watchdog event handler ----------

class DICOMHandler(FileSystemEventHandler):
    def __init__(
        self,
        cfg: RTSessionConfig,
        score_queue: Optional[object] = None,
        start_biopac: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        self.speed = None
        if getattr(cfg, "test_speed", False):
            from rt_speed import SpeedRecorder
            self.speed = SpeedRecorder(cfg.rt_work_dir / "speed_test.csv")
        self.current_run = int(cfg.run)
        self.next_volume_idx = 1
        self.score_queue = score_queue
        self.biopac_receiver = None
        self._biopac_timeout = None
        self._pending = queue.PriorityQueue()
        self._pending_scans: set[int] = set()
        self._processed_scans: set[int] = set()
        self._inflight_scans: set[int] = set()
        self._lock = threading.Lock()
        self._order_cv = threading.Condition(self._lock)
        self._result_buffer: dict[int, ResultEnvelope] = {}
        self._scan_first_seen: dict[int, float] = {}
        self._next_scan_to_commit: Optional[int] = None
        self._next_scan_to_process: Optional[int] = None
        self._last_committed_scan: int = 0
        self._timed_out_scans: set[int] = set()
        self._engine_mode = str(getattr(REGRESSOR_SETTINGS, "pipeline_engine", "parallel_ordered")).lower()
        self._commit_wait_timeout_s = float(getattr(REGRESSOR_SETTINGS, "commit_wait_timeout_s", 1.0))
        max_workers = int(REGRESSOR_SETTINGS.max_workers)
        if max_workers > 10:
            log.warning(
                "[WATCHDOG] max_workers=%d is high for realtime offline mode; capping to 10.",
                max_workers,
            )
            max_workers = 10
        if max_workers < 1:
            log.warning("[WATCHDOG] max_workers=%d is invalid; using 1.", max_workers)
            max_workers = 1
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        log.info(
            "[ENGINE] mode=%s workers=%d commit_wait_timeout_s=%.2f",
            self._engine_mode,
            max_workers,
            self._commit_wait_timeout_s,
        )
        self._online_mode = False
        self._biopac_started = start_biopac
        self._biopac_run_started = False
        self._biopac_timelag_sum = 0.0
        self._biopac_timelag_count = 0
        self._biopac_timelag_path = self.cfg.rt_work_dir / "biopac_timelag.csv"
        self.reference_score_stats = cfg.reference_score_stats
        self.score_event_tracker = ScoreEventTracker(cfg.rt_work_dir)
        if self.reference_score_stats is not None:
            log.info(
                "[SCORE] Using reference run %s (mean=%.4f, std=%.4f, n=%s, reg_ready=%s)",
                self.reference_score_stats["run"],
                self.reference_score_stats["mean"],
                self.reference_score_stats["std"],
                self.reference_score_stats["n"],
                self.reference_score_stats.get("used_reg_ready", False),
            )
        self.gpu_resampler = maybe_init_gpu_resampler(cfg)
        self.pyhysco_applier = maybe_init_pyhysco_applier(cfg)
        self.smoother = prepare_smoother(
            cfg, getattr(REGRESSOR_SETTINGS, "smoothing_fwhm_mm", 0),
            str(REGRESSOR_SETTINGS.analysis_space).lower(),
            resolve_analysis_reference(cfg, str(REGRESSOR_SETTINGS.analysis_space).lower()), run,
            mask_type=getattr(REGRESSOR_SETTINGS, "analysis_mask", "whole_brain"),
            custom_file=getattr(REGRESSOR_SETTINGS, "analysis_mask_file", None),
            custom_space=getattr(REGRESSOR_SETTINGS, "analysis_mask_space", "final"),
        )
        (self.anatomical_brain_mask, self.processing_mask, self.final_brain_mask,
         self.processing_mask_path) = prepare_processing_masks(
             cfg, self.smoother.mask if self.smoother is not None else None)
        self._pyhysco_unwarped_save_notice_emitted = False
        self.volume_streamer = VolumeStreamerHandle(
            VolumeStreamerConfig(
                enabled=bool(getattr(REGRESSOR_SETTINGS, "enable_volume_streamer", False)),
                kind=str(getattr(REGRESSOR_SETTINGS, "volume_stream_kind", "unwarped")),
                every_n=int(getattr(REGRESSOR_SETTINGS, "volume_stream_every_n", 1)),
                max_queue=int(getattr(REGRESSOR_SETTINGS, "volume_stream_max_queue", 2)),
                window_title=str(
                    getattr(
                        REGRESSOR_SETTINGS,
                        "volume_stream_window_title",
                        "RT volume QC",
                    )
                ),
                window_x=int(getattr(REGRESSOR_SETTINGS, "volume_stream_window_x", 50)),
                window_y=int(getattr(REGRESSOR_SETTINGS, "volume_stream_window_y", 50)),
            ),
            logger=log,
        )
        self.volume_streamer.start()
        log.info(
            "[FMAP] Effective settings: use_preloaded_pyhysco=%s save_intermediate_unwarped=%s "
            "pyhysco_backend=%s pyhysco_device=%s",
            bool(getattr(REGRESSOR_SETTINGS, "use_preloaded_pyhysco", True)),
            bool(getattr(REGRESSOR_SETTINGS, "save_intermediate_unwarped", True)),
            str(getattr(REGRESSOR_SETTINGS, "pyhysco_backend", "grid_sample")),
            str(getattr(REGRESSOR_SETTINGS, "pyhysco_device", "cuda")),
        )

        # --- RTPSpy Volreg ---
        self.volreg = RtpVolreg(
            regmode='heptic',
            max_scan_length=REGRESSOR_SETTINGS.rt_max_scan_length,
        )
        self.volreg.ignore_init = 0
        self.volreg.save_proc = False

        # --- Motion reference: corrected session mean, fixed across runs ---
        self.ref_set = False
        ref_epi = self.cfg.rt_motion_ref_epi
        if not ref_epi.exists():
            raise FileNotFoundError(
                f"RT reference EPI not found at {ref_epi}. "
                f"Run the offline preprocessing pipeline first so "
                f"epi_unwarped_mean.nii is created in {self.cfg.trans_dir}."
            )
        self.volreg.set_ref_vol(str(ref_epi))
        self.ref_set = True

        # --- Motion / FD state ---
        self.motion_file = self.cfg.rt_work_dir / "motion_rt.1D"
        self.fd_file = self.cfg.rt_work_dir / "fd_rt.csv"
        self.prev_motion = None              # previous 6-vector
        self.brain_radius_mm = 50.0          # standard radius for FD
        self.pre_trial_scans = max(0, int(REGRESSOR_SETTINGS.skip_first_trs))

        # --- NEW: DVARS state ---
        self.prev_mc_for_dvars = None
        self.dvars_hist = []  # store prior DVARS to compute robust stats
        self.censor_next_fd = False
        self.censor_next_dvars = False
        self.last_dvars_val = float("nan")
        self.last_dvars_z = float("nan")

        # --- DVARS uses corrected, motion-aligned volumes and mask ---
        self.dvars_mask = None
        if REGRESSOR_SETTINGS.enable_dvars_censor_reg:
            mpath = cfg.rt_motion_ref_mask
            if not mpath.exists():
                log.warning(f"[DVARS] Mask missing at {mpath}; DVARS censor disabled.")
            else:
                mimg = nib.load(str(mpath))
                mdat = np.asanyarray(mimg.dataobj)
                self.dvars_mask = (mdat > 0.5)
                log.info(f"[DVARS] Using mask for DVARS: {mpath}")

        # --- Nuisance masks (made offline by pipeline.py) ---
        def resolve_mask(label: str, path: Path, enabled: bool) -> Optional[Path]:
            if not enabled:
                log.info(f"[REG] {label} regressor disabled by config.")
                return None
            if not path.exists():
                log.warning(f"[REG] {label} mask missing at {path}; skipping {label} regressor.")
                return None
            return path

        gs = resolve_mask("GS", cfg.rt_unwarped_analysis_ref_mask, REGRESSOR_SETTINGS.use_gs)
        wm = resolve_mask("WM", cfg.rt_wm_mask, REGRESSOR_SETTINGS.use_wm)
        vent = resolve_mask("Vent", cfg.rt_vent_mask, REGRESSOR_SETTINGS.use_vent)

        used = {k: v for k, v in {"GS": gs, "WM": wm, "Vent": vent}.items() if v is not None}
        self.native_nuisance_masks = {}
        self._infer_native_nuisance_mask = self.anatomical_brain_mask is None
        if REGRESSOR_SETTINGS.enable_motion_regression:
            native_reference = nib.load(str(cfg.rt_motion_ref_epi))
            native_processing = self.anatomical_brain_mask
            if native_processing is not None:
                mask_spec = processing_mask_spec(cfg)
                native_processing = make_processing_mask(
                    native_processing, mask_spec["interpolation"], mask_spec["spatial_resampling"])
            for label, path in used.items():
                image = nib.load(str(path))
                require_same_grid(image, native_reference)
                mask = np.asarray(image.dataobj).astype(bool)
                if native_processing is not None:
                    mask &= native_processing
                self.native_nuisance_masks[label] = mask
        if used:
            log.info("[REG] Using nuisance masks:\n" + "\n".join([f"  {k}={v}" for k, v in used.items()]))
        else:
            log.info("[REG] No nuisance masks enabled; running motion-only regression.")

        phys_reg = "None"
        rtp_physio = None
        if REGRESSOR_SETTINGS.enable_biopac_physio:
            if not REGRESSOR_SETTINGS.enable_motion_regression:
                log.warning("[BIOPAC] Motion regression disabled; skipping physio regressors.")
            else:
                expected_regressors = {
                    "RICOR8": 8,
                    "RVT5": 5,
                    "RVT+RICOR13": 13,
                }.get(REGRESSOR_SETTINGS.biopac_phys_reg, 8)
                biopac_cfg = BiopacReceiverConfig(
                    host=REGRESSOR_SETTINGS.biopac_host,
                    port=REGRESSOR_SETTINGS.biopac_port,
                    timeout=REGRESSOR_SETTINGS.biopac_timeout,
                    expected_regressors=expected_regressors,
                    handshake_tr=REGRESSOR_SETTINGS.TR if REGRESSOR_SETTINGS.biopac_handshake else None,
                    subject=cfg.subject,
                    day=cfg.day,
                    run=cfg.run,
                    output_path=self.cfg.rt_work_dir / "biopac_regressors_rx.csv",
                )
                if REGRESSOR_SETTINGS.biopac_mode == "file":
                    if REGRESSOR_SETTINGS.biopac_file is None:
                        raise ValueError("[BIOPAC] biopac_mode=file requires biopac_file to be set.")
                    rtp_physio = BiopacRetroTSFileBuffer(
                        REGRESSOR_SETTINGS.biopac_file,
                        timeout=biopac_cfg.timeout,
                        expected_regressors=expected_regressors,
                        poll_interval=REGRESSOR_SETTINGS.biopac_poll_interval,
                    )
                    phys_reg = REGRESSOR_SETTINGS.biopac_phys_reg
                    log.info(
                        "[BIOPAC] Using file-backed physio regressors (%s) from %s",
                        phys_reg,
                        REGRESSOR_SETTINGS.biopac_file,
                    )
                else:
                    self.biopac_receiver = BiopacRetroTSReceiver(biopac_cfg)
                    self._biopac_timeout = biopac_cfg.timeout
                    if start_biopac:
                        self.biopac_receiver.start()
                    else:
                        biopac_cfg.timeout = 0.0
                        log.info("[BIOPAC] Deferring receiver start until online mode.")
                    phys_reg = REGRESSOR_SETTINGS.biopac_phys_reg
                    rtp_physio = self.biopac_receiver
                    log.info(
                        "[BIOPAC] Enabled physio regressors (%s) on %s:%s",
                        phys_reg,
                        biopac_cfg.host,
                        biopac_cfg.port,
                    )

        if REGRESSOR_SETTINGS.enable_motion_regression:
            reg_mask = self.processing_mask_path
            if reg_mask is None:
                log.warning("[REG] Regression mask missing at %s; using non-zero voxels from first volume.", cfg.rt_unwarped_analysis_ref_mask)
            self.motion_regressor = MotionRegressor(
                self.volreg,
                reg_mask=reg_mask,
                gs_mask=gs,
                wm_mask=wm,
                vent_mask=vent,
                mot_reg=REGRESSOR_SETTINGS.mot_reg,
                max_poly_order=REGRESSOR_SETTINGS.max_poly_order,
                TR=REGRESSOR_SETTINGS.TR,
                max_scan_length=REGRESSOR_SETTINGS.rt_max_scan_length,
                norm_ref_volumes=REGRESSOR_SETTINGS.voxel_norm_ref_volumes,
                normalization_mode=REGRESSOR_SETTINGS.voxel_norm_mode,
                normalization_reference=REGRESSOR_SETTINGS.voxel_norm_reference,
                startup_residual_transform=(self.smoother.apply
                    if self.smoother is not None and self.smoother.output_kind == "smooth" else None),
                enable_fd_censor_reg=REGRESSOR_SETTINGS.enable_fd_censor_reg,
                enable_dvars_censor_reg=REGRESSOR_SETTINGS.enable_dvars_censor_reg,
                phys_reg=phys_reg,
                rtp_physio=rtp_physio,
            )
        else:
            log.info("[REG] Motion regression disabled by config.")
            self.motion_regressor = None

        norm_mask = self.processing_mask
        self.voxel_normalizer = VoxelNormalizer(
            mode=REGRESSOR_SETTINGS.voxel_norm_mode,
            reference=REGRESSOR_SETTINGS.voxel_norm_reference,
            ref_volumes=REGRESSOR_SETTINGS.voxel_norm_ref_volumes,
            brain_mask=norm_mask,
        )

        # --- Source container for GS/WM/Vent regressors (RTPSpy expects mask_src_proc.proc_data) ---
        self.proc_src = ProcSrc()
        if self.motion_regressor is not None:
            self.motion_regressor._regress.mask_src_proc = self.proc_src

        # --- Decoder / scorer ---
        if cfg.enable_scoring:
            decoder_path = resolve_decoder_template(cfg)
            roi_txt = resolve_decoder_roi_txt(cfg)
            if roi_txt is None:
                log.info("[SCORE] Decoder ROI txt not provided; using decoder NIfTI nonzero mask.")
            else:
                log.info("[SCORE] Using decoder ROI txt: %s", roi_txt)

            self.scorer = DecoderScorer(
                decoder_path,
                roi_txt=roi_txt,
                # Keep decoder baseline aligned with the shared normalization
                # window used by the RT pipeline.
                n_baseline=max(1, int(REGRESSOR_SETTINGS.voxel_norm_ref_volumes)),
            )
        else:
            log.info("[SCORE] Scoring disabled; skipping decoder initialization.")
            self.scorer = None
        self.decoder_qc = prepare_decoder_correlation_qc(cfg, self)

    def stop(self):
        if hasattr(self, "normalization_last_volume"):
            save_normalization_reference(self.cfg, self.voxel_normalizer,
                                         nib.load(str(resolve_analysis_reference(
                                             self.cfg, str(REGRESSOR_SETTINGS.analysis_space).lower()))).affine,
                                         self.normalization_last_volume)
        if self.biopac_receiver is not None:
            if self._biopac_run_started:
                self.biopac_receiver.send_run_end()
            self.biopac_receiver.stop()
        self._executor.shutdown(wait=False, cancel_futures=True)
        self.volume_streamer.stop()
        # Let the existing stop path return as before, but do not read a final
        # workbook snapshot while an active callback can still append CSV rows.
        # A non-daemon completion thread also covers direct rt_pipeline.py use;
        # behavioral parents export again after their processing child exits.
        def export_after_workers():
            self._executor.shutdown(wait=True)
            try:
                write_run_qc_workbook(self.cfg.rt_work_dir)
            except Exception:
                log.exception("[QC] Workbook export failed; per-volume CSVs remain available.")
        self._qc_export_thread = threading.Thread(
            target=export_after_workers, name="run-qc-export", daemon=False)
        self._qc_export_thread.start()

    def start_biopac(self):
        if self.biopac_receiver is None:
            return
        if self._biopac_timeout is not None:
            self.biopac_receiver.config.timeout = self._biopac_timeout
        self.biopac_receiver.start()
        self._biopac_started = True
        if self._online_mode and not self._biopac_run_started:
            self.biopac_receiver.send_run_start()
            self._biopac_run_started = True

    def enable_online_mode(self) -> None:
        self._online_mode = True

    def on_created(self, event):
        if event.is_directory:
            return

        path = Path(event.src_path)
        # `on_created` can fire before the writer closes/flushed the file.
        # We still enqueue as a fallback for filesystems that don't emit
        # close events, but the preferred path is `on_closed` below.
        log.info(f"[WATCHDOG] File created: {path}")
        self.enqueue_path(path, file_event="created")

    def on_closed(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        log.info(f"[WATCHDOG] File closed: {path}")
        self.enqueue_path(path, file_event="closed")

    def enqueue_path(self, path: Path, backlog: bool = False, file_event: Optional[str] = None) -> None:
        parsed = parse_dicom_name(path.name)
        if parsed is None:
            log.debug(f"[WATCHDOG] Ignoring non-matching file: {path.name}")
            return

        _, run_id, scan = parsed
        if run_id != self.current_run:
            log.debug(f"[WATCHDOG] Ignoring run {run_id}, expecting {self.current_run}")
            return
        if self.speed is not None:
            self.speed.arrive(scan, backlog=backlog)
            if file_event is not None:
                self.speed.file_event(scan, file_event)
        if scan in self._processed_scans or scan in self._pending_scans:
            return
        if self._online_mode and not self._biopac_run_started:
            if not self._biopac_started:
                self.start_biopac()
            elif self.biopac_receiver is not None:
                self.biopac_receiver.send_run_start()
                self._biopac_run_started = True
        self._pending.put((scan, path))
        self._pending_scans.add(scan)

    def next_pending(self, timeout: float = 0.1) -> Optional[tuple[int, Path]]:
        try:
            scan, path = self._pending.get(timeout=timeout)
        except queue.Empty:
            return None
        self._pending_scans.discard(scan)
        return scan, path

    def mark_processed(self, scan: int) -> None:
        self._processed_scans.add(scan)

    def submit_pending(self) -> bool:
        pending = self.next_pending(timeout=0.0)
        if pending is None:
            return False
        scan, path = pending
        with self._lock:
            if scan in self._processed_scans or scan in self._inflight_scans:
                return True
            volume_idx = scan
            self._inflight_scans.add(scan)
            self._scan_first_seen.setdefault(scan, time.time())
            if self._next_scan_to_commit is None:
                self._next_scan_to_commit = scan
        if self._engine_mode == "legacy":
            fut = self._executor.submit(self._process_scan, path, scan, volume_idx)
            fut.add_done_callback(lambda f, s=scan, v=volume_idx: self._on_scan_future_done(f, s, v))
        else:
            fut = self._executor.submit(self._compute_scan, path, scan, volume_idx)
            fut.add_done_callback(self._on_compute_future_done)
        return True

    def _on_scan_future_done(self, future, scan: int, volume_idx: int) -> None:
        try:
            exc = future.exception()
        except CancelledError:
            return
        if exc is None:
            return
        if self.speed is not None:
            self.speed.finish(volume_idx, False)

        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        log.error(
            "[WATCHDOG] Unhandled exception for scan %s (vol %05d): %s\n%s",
            scan,
            volume_idx,
            exc,
            tb,
        )

        with self._lock:
            if scan in self._inflight_scans:
                self._inflight_scans.discard(scan)
                self.mark_processed(scan)
                self._advance_expected_scan_locked()

    def _advance_expected_scan_locked(self) -> None:
        all_candidates = self._inflight_scans.union(self._pending_scans).union(self._result_buffer.keys())
        candidates = sorted(s for s in all_candidates if s > self._last_committed_scan)
        self._next_scan_to_commit = candidates[0] if candidates else None
        self._next_scan_to_process = self._next_scan_to_commit
        self._order_cv.notify_all()

    def _on_compute_future_done(self, future) -> None:
        try:
            env: ResultEnvelope = future.result()
        except Exception as exc:
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            log.error("[ENGINE] compute future failed: %s\n%s", exc, tb)
            return
        with self._lock:
            if env.scan <= self._last_committed_scan or env.scan in self._timed_out_scans:
                log.warning("[ENGINE] dropping late envelope scan=%s (last_committed=%s timed_out=%s)", env.scan, self._last_committed_scan, env.scan in self._timed_out_scans)
                return
            self._result_buffer[env.scan] = env
            self._order_cv.notify_all()
        self._drain_commit_ready()

    def _drain_commit_ready(self) -> None:
        while True:
            with self._lock:
                if self._next_scan_to_commit is None:
                    self._advance_expected_scan_locked()
                    if self._next_scan_to_commit is None:
                        return
                expected_scan = self._next_scan_to_commit
                env = self._result_buffer.pop(expected_scan, None)
                if env is None:
                    has_newer_ready = any(s > expected_scan for s in self._result_buffer)
                    is_missing = (expected_scan not in self._inflight_scans) and (expected_scan not in self._pending_scans)
                    if is_missing and has_newer_ready:
                        first_seen = self._scan_first_seen.get(expected_scan, time.time())
                        if time.time() - first_seen > self._commit_wait_timeout_s:
                            env = ResultEnvelope(
                                scan=expected_scan,
                                volume_idx=expected_scan,
                                dicom_path=Path("missing"),
                                volume_timestamp=time.time(),
                                success=False,
                                error=f"Commit timeout waiting for scan {expected_scan}",
                            )
                            self._timed_out_scans.add(expected_scan)
                            self._pending_scans.discard(expected_scan)
                    if env is None:
                        return
            self._commit_scan(env)

    def _process_scan(self, path: Path, scan: int, volume_idx: int) -> None:
        # Legacy path (kept for backward compatibility).
        log.info(f"[WATCHDOG] Preparing volume idx {volume_idx} (run={self.current_run}, scan={scan})")

        prepared_raw: Optional[Path] = None
        volume_timestamp: Optional[float] = None
        for attempt in range(1, REGRESSOR_SETTINGS.max_retries + 1):
            prepared_raw, volume_timestamp = prepare_volume_input(self.cfg, path, volume_idx, self.speed)
            if prepared_raw is not None:
                break
            log.warning(
                "[WATCHDOG] Retry scan %s prep (attempt %s/%s)",
                scan,
                attempt,
                REGRESSOR_SETTINGS.max_retries,
            )
            time.sleep(0.2)

        ok = False
        if prepared_raw is not None and volume_timestamp is not None:
            with self._order_cv:
                while self._next_scan_to_process is not None and scan != self._next_scan_to_process:
                    self._order_cv.wait(timeout=0.1)
            log.info(f"[WATCHDOG] Processing volume idx {volume_idx} (run={self.current_run}, scan={scan})")
            # Stateful stages have already advanced even if a later operation
            # fails. Replaying here would count this sample twice in the GLM
            # and normalization reference, and violate prior-sample scoring.
            started = self.speed.process_start(volume_idx) if self.speed else None
            try:
                ok = process_volume(
                    self.cfg,
                    self,
                    path,
                    volume_idx,
                    raw_nii=prepared_raw,
                    volume_timestamp=volume_timestamp,
                )
            except Exception:
                log.exception("[WATCHDOG] Stateful processing failed for scan %s; not retrying this sample.", scan)
            finally:
                if self.speed is not None:
                    self.speed.process_end(volume_idx, started)

        if not ok:
            log.error("[WATCHDOG] Scan %s failed; advancing without repeating stateful processing.", scan)
        if self.speed is not None:
            self.speed.finish(volume_idx, ok)
        with self._lock:
            self._inflight_scans.discard(scan)
            self.mark_processed(scan)
            self._advance_expected_scan_locked()

    def _compute_scan(self, path: Path, scan: int, volume_idx: int) -> ResultEnvelope:
        err_txt = None
        tb_txt = None
        for attempt in range(1, REGRESSOR_SETTINGS.max_retries + 1):
            try:
                env = compute_stage(self.cfg, path, scan=scan, volume_idx=volume_idx, attempt=attempt, speed=self.speed)
                env.attempts = attempt
                return env
            except Exception as exc:
                err_txt = str(exc)
                tb_txt = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                log.warning("[ENGINE] compute retry scan=%s attempt=%s/%s err=%s", scan, attempt, REGRESSOR_SETTINGS.max_retries, exc)
                time.sleep(0.2)
        return ResultEnvelope(
            scan=scan,
            volume_idx=volume_idx,
            dicom_path=path,
            volume_timestamp=time.time(),
            success=False,
            error=err_txt or "unknown compute failure",
            traceback_text=tb_txt,
            attempts=REGRESSOR_SETTINGS.max_retries,
        )

    def _commit_scan(self, env: ResultEnvelope) -> None:
        speed = getattr(self, "speed", None)
        started = speed.process_start(env.volume_idx) if speed else None
        ok = False
        try:
            ok = commit_stage(self.cfg, self, env)
        except Exception as exc:
            log.error("[ENGINE] commit failure scan=%s vol=%05d err=%s", env.scan, env.volume_idx, exc)
        finally:
            if speed is not None:
                speed.process_end(env.volume_idx, started)
                speed.finish(env.volume_idx, ok)
            with self._lock:
                if env.scan <= self._last_committed_scan:
                    log.warning("[COMMIT] Ignoring already-committed/late scan=%s (last=%s)", env.scan, self._last_committed_scan)
                else:
                    self._last_committed_scan = env.scan
                self._inflight_scans.discard(env.scan)
                self.mark_processed(env.scan)
                self._scan_first_seen.pop(env.scan, None)
                self._advance_expected_scan_locked()

    def process_file(self, path: Path):
        parsed = parse_dicom_name(path.name)
        if parsed is None:
            log.debug(f"[WATCHDOG] Ignoring non-matching file: {path.name}")
            return

        _, run_id, scan = parsed
        if run_id != self.current_run:
            log.debug(f"[WATCHDOG] Ignoring run {run_id}, expecting {self.current_run}")
            return

        if scan in self._processed_scans:
            return
        self.enqueue_path(path)


# ---------- Core processing hook (DICOM -> NIfTI -> MC -> FMAP) ----------

def final_volume_support(cfg, handler, support, epi_affine, volume_idx, space,
                         folder_name="coverage", mapped_support=None):
    """Carry this volume's acquisition or analysis support through the transform.

    Availability uses Linear interpolation for both signal kernels. The fixed
    anatomical mask is mapped independently with nearest-neighbour. Never
    shrink brain selection by requiring the entire Lanczos stencil inside it.
    """
    folder = cfg.rt_work_dir / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    epi_path = folder / f"vol_{volume_idx:05d}_epi.nii"
    nib.save(nib.Nifti1Image(support.astype(np.uint8), epi_affine), epi_path)
    bridge = session_epi_bridge(cfg) if space == "epi" else None
    brain = getattr(handler, "final_brain_mask", None) if folder_name != "coverage" else None
    if space == "epi" and bridge is None:
        valid = support.copy() if brain is None else support & brain
        nib.save(nib.Nifti1Image(valid.astype(np.uint8), epi_affine), epi_path)
        return valid
    final_kind = "epi_ref" if bridge is not None else space
    final_path = folder / f"vol_{volume_idx:05d}_{final_kind}.nii"
    reference = resolve_analysis_reference(cfg, space)
    ref = nib.load(str(reference))
    if mapped_support is not None:
        # Identical EPI support has the same final support; retain separate
        # auditable files without repeating an ANTs/GPU resampling operation.
        mapped = mapped_support
    elif handler.gpu_resampler is not None:
        mapped = handler.gpu_resampler.resample_support(support.astype(np.float32))
    else:
        transforms = [bridge] if bridge is not None else [cfg.trans_dir / "epi2t1_Composite.h5"]
        if space == "mni":
            transforms.insert(0, t1_to_mni_transform(cfg))
        cmd = ["antsApplyTransforms", "-d", "3", "-i", str(epi_path),
               "-r", str(reference), "-o", str(final_path), "-n", "Linear", "--float", "1"]
        for transform in transforms:
            cmd.extend(["-t", str(transform)])
        run(cmd)
        mapped_img = nib.load(str(final_path), mmap=False)
        require_same_grid(mapped_img, ref)
        mapped = np.asarray(mapped_img.dataobj).copy()
    if mapped.shape != ref.shape[:3] or not np.isfinite(mapped).all():
        raise ValueError("Invalid per-volume coverage on the final image grid.")
    valid = mapped >= SUPPORT_THRESHOLD
    if brain is not None:
        if brain.shape != valid.shape:
            raise ValueError("Anatomical brain mask and final support grids differ.")
        valid &= brain
    nib.save(nib.Nifti1Image(valid.astype(np.uint8), ref.affine), final_path)
    return valid


def record_volume_support(cfg, handler, volume_idx, brain_mask, mc_support,
                          cleaned_support, final_support):
    """Report support without shortening/reordering a trained decoder's features.

    GLM support is conservative: once a brain voxel lacks an observation it is
    unavailable for the remaining run. A no-regression reference mean instead
    uses observed samples, and a calibrated voxel can recover on a later TR.
    """
    brain_count = int(np.count_nonzero(brain_mask))
    acquired = int(np.count_nonzero(brain_mask & mc_support))
    usable = int(np.count_nonzero(brain_mask & cleaned_support))
    final_valid = final_support
    smoother = getattr(handler, "smoother", None)
    if smoother is not None:
        final_valid = final_valid & smoother.mask
    decoder_mask = getattr(getattr(handler, "scorer", None), "mask", None)
    decoder_count = usable_decoder = None
    fraction = None
    if decoder_mask is not None:
        decoder_mask = np.asarray(decoder_mask, dtype=bool)
        if decoder_mask.shape != final_valid.shape:
            raise ValueError("Decoder mask and final support grids differ.")
        decoder_count = int(np.count_nonzero(decoder_mask))
        usable_decoder = int(np.count_nonzero(decoder_mask & final_valid))
        fraction = usable_decoder / decoder_count if decoder_count else 0.0
    row = dict(volume_idx=volume_idx, brain_voxels=brain_count,
               acquired_brain_voxels=acquired, usable_brain_voxels=usable,
               usable_brain_fraction=usable / brain_count if brain_count else 0.0,
               decoder_voxels=decoder_count, usable_decoder_voxels=usable_decoder,
               decoder_coverage_fraction=fraction)
    handler.current_support_qc = row
    handler.decoder_support_complete = (usable_decoder == decoder_count
                                        if decoder_count is not None else None)
    path = cfg.rt_work_dir / "coverage_rt.csv"
    exists = path.exists()
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    counts = (acquired, usable, usable_decoder)
    if counts != getattr(handler, "_last_support_counts", None):
        log.info("[SUPPORT] vol=%05d acquired brain=%d/%d; usable brain=%d/%d; decoder=%s/%s",
                 volume_idx, acquired, brain_count, usable, brain_count,
                 usable_decoder, decoder_count)
        if decoder_count is not None and usable_decoder < decoder_count:
            log.warning("[SUPPORT] Decoder coverage %.2f%%; unavailable features are zero-filled, "
                        "not removed or reweighted. See coverage_rt.csv and support/.", 100. * fraction)
        handler._last_support_counts = counts


def zero_outside_support(path, support):
    """Remove unsupported resampled values; do not infer validity from signal."""
    if support.all():
        return
    img = nib.load(str(path), mmap=False)
    data = np.asarray(img.dataobj).copy()
    if data.shape != support.shape:
        raise ValueError("BOLD and per-volume coverage shapes differ.")
    data[~support] = 0
    nib.save(nib.Nifti1Image(data, img.affine), str(path))

def prepare_volume_input(cfg: RTSessionConfig, dicom_path: Path, volume_idx: int, speed=None) -> tuple[Optional[Path], Optional[float]]:
    if speed is None:
        return _prepare_volume_input(cfg, dicom_path, volume_idx)
    cached = (cfg.rt_raw_dir / f"vol_{volume_idx:05d}.nii").exists()
    started = speed.conversion_start(volume_idx, cached=cached)
    try:
        return _prepare_volume_input(cfg, dicom_path, volume_idx)
    finally:
        speed.conversion_end(volume_idx, started)


def _prepare_volume_input(cfg: RTSessionConfig, dicom_path: Path, volume_idx: int) -> tuple[Optional[Path], Optional[float]]:
    """
    Convert DICOM to raw NIfTI.
    This stage is safe to run ahead of ordered/stateful realtime steps.
    """
    volume_timestamp = time.time()

    # ---------- 1) DICOM -> raw NIfTI ----------
    t0 = time.perf_counter()
    raw_dir = cfg.rt_raw_dir
    raw_nii = raw_dir / f"vol_{volume_idx:05d}.nii"

    if not raw_nii.exists():
        for attempt in range(3):
            run([
                "dcm2niix",
                "-z", "n",  # no gzip
                "-s", "y",
                "-a", "y",
                "-b", "y",
                "-v", "0",
                "-f", f"vol_{volume_idx:05d}",
                "-o", str(raw_dir),
                str(dicom_path),
            ])

            produced = sorted(raw_dir.glob(f"vol_{volume_idx:05d}*.nii*"))
            if produced:
                raw_nii = produced[0]
                break
            log.warning("[DICOM] vol %05d conversion retry %d/3", volume_idx, attempt + 1)
            time.sleep(0.2)
        else:
            log.error(f"[DICOM] vol {volume_idx:05d} FAILED (no output)")
            return None, None

    log_step("DICOM", volume_idx, start_t=t0)

    return raw_nii, volume_timestamp


def run_worker_local_volreg(ref_epi: Path, raw_nii: Path, mc_nii: Path, volume_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    volreg = RtpVolreg(regmode='heptic')
    volreg.ignore_init = 0
    volreg.save_proc = False
    volreg.set_ref_vol(str(ref_epi))
    img = nib.load(str(raw_nii))
    data = np.asanyarray(img.dataobj).astype(np.float32)
    tmp_img = nib.Nifti1Image(data, img.affine, img.header.copy())
    tmp_img.set_filename(str(mc_nii))
    volreg.do_proc(tmp_img, vol_idx=volume_idx - 1)
    mc_data = np.asanyarray(tmp_img.dataobj).astype(np.float32)
    nib.save(nib.Nifti1Image(mc_data, img.affine), str(mc_nii))
    try:
        motion_vec = np.asarray(volreg._motion[volume_idx - 1]).astype(float)
    except Exception:
        motion_vec = np.zeros(6, dtype=float)
    return mc_data, img.affine, motion_vec


def compute_stage(cfg: RTSessionConfig, dicom_path: Path, scan: int, volume_idx: int, attempt: int = 1, speed=None) -> ResultEnvelope:
    raw_nii, volume_timestamp = prepare_volume_input(cfg, dicom_path, volume_idx, speed)
    if raw_nii is None or volume_timestamp is None:
        raise RuntimeError("prepare_volume_input failed")
    return ResultEnvelope(
        scan=scan,
        volume_idx=volume_idx,
        dicom_path=dicom_path,
        volume_timestamp=volume_timestamp,
        success=True,
        attempts=attempt,
        raw_nii=raw_nii,
    )


def commit_stage(cfg: RTSessionConfig, handler: "DICOMHandler", env: ResultEnvelope) -> bool:
    if not env.success:
        log.error("[COMMIT] scan=%s failed in compute: %s", env.scan, env.error)
        return False
    if env.raw_nii is None:
        log.error("[COMMIT] scan=%s missing raw input.", env.scan)
        return False
    return process_volume(
        cfg,
        handler,
        env.dicom_path,
        env.volume_idx,
        raw_nii=env.raw_nii,
        volume_timestamp=env.volume_timestamp,
    )


def process_volume(
    cfg: RTSessionConfig,
    handler: "DICOMHandler",
    dicom_path: Path,
    volume_idx: int,
    raw_nii: Optional[Path] = None,
    volume_timestamp: Optional[float] = None,
):
    """
    For each incoming DICOM:
      1) DICOM -> raw NIfTI (single volume)
      2) Fieldmap unwarp of raw volume
      3) Motion correction of unwarped volume -> mc NIfTI
      4) Native-unit denoising -> reg NIfTI; voxel normalization -> norm NIfTI
      5) Spatial mapping, coverage masking and optional masked smoothing
      6) Decode only when both GLM and normalization are ready
    """
    speed = getattr(handler, "speed", None)
    if raw_nii is None:
        raw_nii, prepared_timestamp = prepare_volume_input(cfg, dicom_path, volume_idx)
        if raw_nii is None or prepared_timestamp is None:
            return False
        volume_timestamp = prepared_timestamp
    elif volume_timestamp is None:
        volume_timestamp = time.time()

    # Unwarp raw BOLD in calibration geometry before estimating motion.
    uw_t0 = time.perf_counter()
    raw_img = nib.load(str(raw_nii))
    validate_bold(raw_img, cfg.fmap_dir, str(REGRESSOR_SETTINGS.epi_phase_encoding).upper(), raw_nii)
    ref_img = nib.load(str(cfg.rt_motion_ref_epi))
    require_same_grid(raw_img, ref_img)
    raw_data = np.asarray(raw_img.dataobj, dtype=np.float32)
    unwarp_dir = cfg.rt_unwarp_dir
    local_unwarped = unwarp_dir / f"vol_{volume_idx:05d}_uw_native.nii"
    corrected = None
    if handler.pyhysco_applier is not None:
        try:
            corrected = handler.pyhysco_applier.apply_volume(raw_data)
        except Exception:
            log.exception("[FMAP] Preloaded apply failed; using file-based PyHySCO.")
    if corrected is None:
        if not unwarp_volume(raw_nii, local_unwarped, cfg):
            return False
        corrected = np.asarray(nib.load(str(local_unwarped)).dataobj, dtype=np.float32).copy()
    img = nib.Nifti1Image(corrected, raw_img.affine)
    if bool(getattr(REGRESSOR_SETTINGS, "save_intermediate_unwarped", True)):
        nib.save(img, str(local_unwarped))
    log_step("FMAP", volume_idx, start_t=uw_t0, speed=speed)

    # ---------- 2) Motion correction (RtpVolreg) ----------
    t0 = time.perf_counter()
    mc_dir = cfg.rt_mc_dir
    mc_nii = mc_dir / f"vol_{volume_idx:05d}_mc.nii"

    # MC input is now corrected BOLD.
    data = np.asanyarray(img.dataobj).astype(np.float32)

    # Create a temporary NIfTI for RtpVolreg to work on
    tmp_img = nib.Nifti1Image(data, img.affine, img.header.copy())
    tmp_img.set_filename(str(mc_nii))

    # Run RTPSpy volreg (works in-place on tmp_img.dataobj)
    handler.volreg.do_proc(tmp_img, vol_idx=volume_idx - 1)

    # Extract corrected data and save with a FRESH header
    mc_data = np.asanyarray(tmp_img.dataobj).astype(np.float32)
    # Apply the estimated motion to geometric support, never re-register a mask
    # or threshold denoised intensities (valid residuals may be zero/negative).
    acquisition_support = getattr(handler, "acquisition_support", None)
    if acquisition_support is None or acquisition_support.shape != data.shape:
        acquisition_support = np.ones(data.shape, dtype=np.float32)
        handler.acquisition_support = acquisition_support
    mc_support_values = handler.volreg.resample_support(acquisition_support, volume_idx - 1)
    if mc_support_values.shape != mc_data.shape or not np.isfinite(mc_support_values).all():
        raise ValueError("Invalid motion-corrected field-of-view support.")
    mc_support = mc_support_values >= SUPPORT_THRESHOLD
    mc_data[~mc_support] = 0
    mc_img = nib.Nifti1Image(mc_data, ref_img.affine)  # fixed corrected reference grid
    nib.save(mc_img, str(mc_nii))

    # One on-disk product for fully corrected BOLD, shared by all consumers.
    mc_unwarped_nii = mc_nii
    mc_unwarped_img = mc_img

    # ----- 2b) MOTION + FD (ONLINE) -----
    # RtpVolreg exposes [roll pitch yaw dS dL dP]: rotations in degrees,
    # followed by translations in mm.
    try:
        motion_vec = np.asarray(handler.volreg._motion[volume_idx - 1]).astype(float)  # shape (6,)
    except Exception as e:
        log.error(f"[MC] Could not read motion for vol {volume_idx:05d}: {e}")
        motion_vec = np.zeros(6, dtype=float)

    # Save raw motion parameters (AFNI-style 1D)
    append_motion(handler.motion_file, motion_vec)

    # Compute delta motion relative to previous volume
    if handler.prev_motion is None:
        delta = np.zeros_like(motion_vec)
    else:
        delta = motion_vec - handler.prev_motion
    handler.prev_motion = motion_vec.copy()

    # Framewise displacement: translation plus radius-scaled rotation.
    fd_value = float(fd_from_rtpspy_delta(delta, handler.brain_radius_mm))

    # Optionally emulate pre_trial_scan_num behavior:
    if volume_idx <= handler.pre_trial_scans:
        fd_to_save = float("nan")
    else:
        fd_to_save = fd_value

    append_fd(handler.fd_file, volume_idx, fd_to_save)

    log_step(
        "MC",
        volume_idx,
        extra=f"FD={fd_to_save:.4f}" if np.isfinite(fd_to_save) else "FD=NaN (pre-trial)",
        start_t=t0,
        speed=speed,
    )

    # ---------- 2b.5) CENSOR FLAGS (FD + DVARS) ----------
    qc_t0 = time.perf_counter()
    fd_censor = 0
    dvars_censor = 0

    # --- FD censor: FD > thr plus +1 TR ---
    if REGRESSOR_SETTINGS.enable_fd_censor_reg:
        hit_fd = np.isfinite(fd_to_save) and (fd_value > REGRESSOR_SETTINGS.fd_thr)
        if handler.censor_next_fd:
            fd_censor = 1
            handler.censor_next_fd = False
        if hit_fd:
            fd_censor = 1
            if REGRESSOR_SETTINGS.censor_plus1:
                handler.censor_next_fd = True

    # --- DVARS censor: raw DVARS robust_z > thr plus +1 TR ---
    if REGRESSOR_SETTINGS.enable_dvars_censor_reg and handler.dvars_mask is not None:
        if handler.prev_mc_for_dvars is not None:
            dvars_val = compute_dvars(
                handler.prev_mc_for_dvars, mc_data, handler.dvars_mask,
                previous_support=getattr(handler, "prev_mc_support_for_dvars", None),
                current_support=mc_support)

            # robust z against prior DVARS history (exclude current)
            z = robust_z(dvars_val, handler.dvars_hist)

            # update history after computing z
            if np.isfinite(dvars_val):
                handler.dvars_hist.append(dvars_val)

            # save to a file
            append_dvars(
                handler.cfg.rt_work_dir / "dvars_rt.csv",
                volume_idx,
                fd_to_save,
                dvars_val,
                z
            )
            handler.last_dvars_val = dvars_val
            handler.last_dvars_z = z

            # optional: ignore first few DVARS for robust stats stability
            enough = (len(handler.dvars_hist) >= REGRESSOR_SETTINGS.dvars_warmup)
            hit_dvars = enough and np.isfinite(z) and (z > REGRESSOR_SETTINGS.dvars_thr_robust_z)

            if handler.censor_next_dvars:
                dvars_censor = 1
                handler.censor_next_dvars = False
            if hit_dvars:
                dvars_censor = 1
                if REGRESSOR_SETTINGS.censor_plus1:
                    handler.censor_next_dvars = True
        else:
            # first timepoint: no DVARS
            handler.dvars_hist.append(0.0)
            handler.last_dvars_val = float("nan")
            handler.last_dvars_z = float("nan")

    # update prev for next DVARS
    handler.prev_mc_for_dvars = mc_data.copy()
    handler.prev_mc_support_for_dvars = mc_support.copy()
    if speed is not None:
        speed.stage(volume_idx, "QC", time.perf_counter() - qc_t0)

    # ---------- 3) Space handling (EPI passthrough, EPI→T1, or EPI→T1→MNI) ----------
    t0 = time.perf_counter()
    analysis_space = str(REGRESSOR_SETTINGS.analysis_space).lower()
    mc_for_warp = mc_unwarped_nii
    t1_stage_nii = None
    pending_spatial_outputs = []
    score_input_nii = mc_for_warp
    score_input_orig_nii: Optional[Path] = None
    if cfg.enable_original_score and mc_unwarped_nii.exists():
        score_input_orig_nii = mc_unwarped_nii

    if analysis_space == "epi":
        bridge = session_epi_bridge(cfg)
        if bridge is None:
            log_step("TRANS", volume_idx, "skipped (preparation EPI grid)", start_t=t0, speed=speed)
        else:
            reference = resolve_analysis_reference(cfg, "epi")
            epi_dir = cfg.rt_work_dir / "epi_ref"
            epi_dir.mkdir(parents=True, exist_ok=True)

            def map_to_reference(source, suffix=""):
                destination = epi_dir / f"vol_{volume_idx:05d}_epi_ref{suffix}.nii"
                temporary = destination.with_name("." + destination.name)
                pending_spatial_outputs.append((temporary, destination))
                if handler.gpu_resampler is not None:
                    handler.gpu_resampler.resample_nifti_to_nifti(source, temporary)
                else:
                    run(["antsApplyTransforms", "-d", "3", "-i", str(source),
                         "-r", str(reference), "-o", str(temporary), "-t", str(bridge),
                         "-n", "Linear", "--float", "1"])
                require_same_grid(nib.load(str(temporary)), nib.load(str(reference)))
                return temporary

            score_input_nii = map_to_reference(mc_for_warp)
            if cfg.enable_original_score:
                score_input_orig_nii = map_to_reference(mc_unwarped_nii, "_orig")
            log_step("TRANS", volume_idx, "session EPI→reference EPI", start_t=t0, speed=speed)
    elif analysis_space == "t1":
        t1_dir = cfg.rt_work_dir / "t1"
        t1_dir.mkdir(parents=True, exist_ok=True)
        t1_final = t1_dir / f"vol_{volume_idx:05d}_t1.nii"
        t1_nii = t1_final.with_name("." + t1_final.name)
        pending_spatial_outputs.append((t1_nii, t1_final))

        epi2t1 = cfg.trans_dir / "epi2t1_Composite.h5"
        decoder_template = resolve_decoder_template(cfg)
        t1_ref = decoder_template

        if not decoder_template.exists():
            log.error(f"Decoder template not found at {decoder_template}")
            return False

        if not epi2t1.exists():
            log.error(f"Missing EPI→T1 transform in {cfg.trans_dir}")
            return False

        if handler.gpu_resampler is not None:
            handler.gpu_resampler.resample_nifti_to_nifti(mc_for_warp, t1_nii)
        else:
            cmd = [
                "antsApplyTransforms",
                "-d", "3",
                "-i", str(mc_for_warp),
                "-r", str(t1_ref),
                "-o", str(t1_nii),
                "-t", str(epi2t1),
                "-n", "Linear",
                "--float", "1",
            ]
            run(cmd)
        score_input_nii = t1_nii
        t1_stage_nii = t1_final

        # Optional: also save pre-denoise / pre-normalization score source.
        if cfg.enable_original_score:
            t1_orig_final = t1_dir / f"vol_{volume_idx:05d}_t1_orig.nii"
            t1_orig_nii = t1_orig_final.with_name("." + t1_orig_final.name)
            pending_spatial_outputs.append((t1_orig_nii, t1_orig_final))
            if handler.gpu_resampler is not None:
                handler.gpu_resampler.resample_nifti_to_nifti(mc_unwarped_nii, t1_orig_nii)
            else:
                cmd_orig = [
                    "antsApplyTransforms",
                    "-d", "3",
                    "-i", str(mc_unwarped_nii),
                    "-r", str(t1_ref),
                    "-o", str(t1_orig_nii),
                    "-t", str(epi2t1),
                    "-n", "Linear",
                    "--float", "1",
                ]
                run(cmd_orig)
            score_input_orig_nii = t1_orig_nii
        log_step("TRANS", volume_idx, "warp→T1", start_t=t0, speed=speed)
    elif analysis_space == "mni":
        mni_dir = cfg.rt_mni_dir
        mni_final = mni_dir / f"vol_{volume_idx:05d}_mni.nii"
        mni_nii = mni_final.with_name("." + mni_final.name)
        pending_spatial_outputs.append((mni_nii, mni_final))

        warp_t1_mni = t1_to_mni_transform(cfg)
        epi2t1 = cfg.trans_dir / "epi2t1_Composite.h5"
        decoder_template = resolve_decoder_template(cfg)

        if not decoder_template.exists():
            log.error(f"Decoder template not found at {decoder_template}")
            return False

        if not (warp_t1_mni.exists() and epi2t1.exists()):
            log.error(f"Missing transforms in {cfg.trans_dir}")
            return False

        if handler.gpu_resampler is not None:
            handler.gpu_resampler.resample_nifti_to_nifti(mc_for_warp, mni_nii)
        else:
            cmd = [
                "antsApplyTransforms",
                "-d", "3",
                "-i", str(mc_for_warp),
                "-r", str(decoder_template),
                "-o", str(mni_nii),
                "-t", str(warp_t1_mni),
                "-t", str(epi2t1),
                "-n", "Linear",
                "--float", "1",
            ]
            run(cmd)
        score_input_nii = mni_nii

        # Optional: also save pre-denoise / pre-normalization score source.
        if cfg.enable_original_score:
            mni_orig_final = mni_dir / f"vol_{volume_idx:05d}_mni_orig.nii"
            mni_orig_nii = mni_orig_final.with_name("." + mni_orig_final.name)
            pending_spatial_outputs.append((mni_orig_nii, mni_orig_final))
            if handler.gpu_resampler is not None:
                handler.gpu_resampler.resample_nifti_to_nifti(mc_unwarped_nii, mni_orig_nii)
            else:
                cmd_orig = [
                    "antsApplyTransforms",
                    "-d", "3",
                    "-i", str(mc_unwarped_nii),
                    "-r", str(decoder_template),
                    "-o", str(mni_orig_nii),
                    "-t", str(warp_t1_mni),
                    "-t", str(epi2t1),
                    "-n", "Linear",
                    "--float", "1",
                ]
                run(cmd_orig)
            score_input_orig_nii = mni_orig_nii
        log_step("TRANS", volume_idx, "warp→MNI", start_t=t0, speed=speed)
    else:
        log.error(
            "Unsupported analysis_space=%r. Expected 'epi', 't1', or 'mni'.",
            REGRESSOR_SETTINGS.analysis_space,
        )
        return False

    # Carry acquisition coverage to the response grid once, before smoothing.
    coverage_t0 = time.perf_counter()
    final_coverage = final_volume_support(cfg, handler, mc_support, ref_img.affine,
                                         volume_idx, analysis_space)
    spatial_output_support = final_coverage.copy()
    brain_mask = getattr(handler, "final_brain_mask", None)
    if brain_mask is None:
        brain_mask = np.ones(final_coverage.shape, dtype=bool)
    spatial_output_support &= brain_mask
    # The native MC product stays untouched even for an EPI-space passthrough.
    if score_input_nii != mc_unwarped_nii:
        zero_outside_support(score_input_nii, spatial_output_support)
    final_orig_support = spatial_output_support
    if score_input_orig_nii == mc_unwarped_nii:
        original_dir = cfg.rt_work_dir / "original"
        original_dir.mkdir(parents=True, exist_ok=True)
        score_input_orig_nii = original_dir / f"vol_{volume_idx:05d}_epi_orig.nii"
        nib.save(nib.Nifti1Image(np.where(final_orig_support, mc_data, 0), ref_img.affine),
                 str(score_input_orig_nii))
    elif score_input_orig_nii is not None:
        zero_outside_support(score_input_orig_nii, final_orig_support)
    for temporary, destination in pending_spatial_outputs:
        temporary.replace(destination)
        if score_input_nii == temporary:
            score_input_nii = destination
        if score_input_orig_nii == temporary:
            score_input_orig_nii = destination
    if speed is not None:
        speed.stage(volume_idx, "TRANS", time.perf_counter() - coverage_t0)

    # Both correlation QC and cGLM use the unsmoothed spatial output.
    spatial_image = nib.load(str(score_input_nii))
    correlation_qc = record_decoder_correlation(
        cfg, handler, volume_idx, spatial_image, spatial_output_support)

    # Keep experiment-mask selection before cGLM; smooth only its output below.
    smoother = getattr(handler, "smoother", None)
    if smoother is not None and smoother.output_kind == "masked":
        # Zero FWHM is still mask-only selection, with the existing raw preview.
        mask_t0 = time.perf_counter()
        folder = cfg.rt_work_dir / "masked"
        score_input_nii = smooth_file(smoother, score_input_nii,
            folder / f"vol_{volume_idx:05d}_masked.nii", support_mask=spatial_output_support)
        if score_input_orig_nii is not None:
            score_input_orig_nii = smooth_file(smoother, score_input_orig_nii,
                folder / f"vol_{volume_idx:05d}_masked_orig.nii", support_mask=final_orig_support)
        log_step("MASK", volume_idx, start_t=mask_t0, speed=speed)
    temporal_image = spatial_image
    temporal_data = np.asarray(temporal_image.dataobj, dtype=np.float32).copy()
    processing_mask = getattr(handler, "processing_mask", None)
    if processing_mask is None:
        processing_mask = brain_mask.copy()
    if smoother is not None:
        processing_mask = processing_mask & smoother.mask
    if processing_mask.shape != temporal_data.shape or final_coverage.shape != temporal_data.shape:
        raise ValueError("Temporal mask, acquisition support and final volume grids differ.")
    temporal_support = spatial_output_support & processing_mask
    temporal_data[~temporal_support] = 0
    # RTPSpy updates its image in-place; keep saved spatial/smoothed data intact.
    temporal_image = nib.Nifti1Image(temporal_data.copy(), temporal_image.affine)
    temporal_image.set_filename(str(score_input_nii))
    nuisance_means = (native_nuisance_means(handler, mc_data, mc_support)
                      if handler.motion_regressor is not None else None)

    # ---------- Temporal denoising on the final analysis grid ----------
    reg_t0 = time.perf_counter()
    if handler.motion_regressor is not None:
        handler.proc_src.proc_data = np.asanyarray(mc_unwarped_img.dataobj)
        cleaned, reg_ready = handler.motion_regressor.apply(
            temporal_image,
            volume_idx,
            fd_censor=fd_censor,
            dvars_censor=dvars_censor,
            support_mask=temporal_support,
            nuisance_means=nuisance_means,
        )
        processor = handler.motion_regressor
        cleaned = np.array(cleaned, copy=True)
        cleaned_support = getattr(processor, "output_support", None)
        cleaned_support = temporal_support if cleaned_support is None else cleaned_support
        cleaned[~cleaned_support] = 0
        reg_dir = cfg.rt_reg_dir
        reg_nii = reg_dir / f"vol_{volume_idx:05d}_reg.nii"
        nib.save(nib.Nifti1Image(cleaned, temporal_image.affine), str(reg_nii))
        log_step("REG", volume_idx, "denoising", start_t=reg_t0, speed=speed)
    else:
        processor = handler.voxel_normalizer
        # No GLM: normalization still uses the same dedicated following stage.
        cleaned = np.array(temporal_data, copy=True)
        cleaned_support = temporal_support.copy()
        norm_brain = getattr(processor, "brain_mask", None)
        if norm_brain is not None:
            cleaned_support &= norm_brain
        cleaned[~cleaned_support] = 0
        reg_ready = True
        reg_dir = cfg.rt_reg_dir
        reg_nii = reg_dir / f"vol_{volume_idx:05d}_reg.nii"
        nib.save(nib.Nifti1Image(cleaned, temporal_image.affine), str(reg_nii))
        log_step("REG", volume_idx, "skipped", start_t=reg_t0, speed=speed)

    if cleaned_support.shape != temporal_support.shape:
        raise ValueError("Denoising support and final analysis volume grids differ.")
    current_brain_support = brain_mask & final_coverage
    current_processing_support = temporal_support

    # ---------- Smooth residuals over cGLM's surviving coverage ----------
    # An earlier coverage loss permanently excludes a voxel from the fitted
    # history. Do not mix that voxel into its surviving neighbours afterward.
    if smoother is not None and smoother.output_kind == "smooth":
        smooth_t0 = time.perf_counter()
        kind = smoother.output_kind
        folder = cfg.rt_work_dir / kind
        score_input_nii = smooth_file(smoother, reg_nii,
            folder / f"vol_{volume_idx:05d}_{kind}.nii", support_mask=cleaned_support)
        cleaned = np.asarray(nib.load(str(score_input_nii)).dataobj, dtype=np.float32).copy()
        if score_input_orig_nii is not None:
            score_input_orig_nii = smooth_file(smoother, score_input_orig_nii,
                folder / f"vol_{volume_idx:05d}_{kind}_orig.nii", support_mask=final_orig_support)
        log_step("SMOOTH" if kind == "smooth" else "MASK", volume_idx, start_t=smooth_t0, speed=speed)

    # ---------- Voxel normalization after denoising and smoothing ----------
    norm_t0 = time.perf_counter()
    glm_ready = bool(reg_ready)
    normalizer = handler.voxel_normalizer
    reference_count = normalizer.reference_count
    if glm_ready:
        normalized = normalizer.transform(cleaned, cleaned_support)
        cleaned_support = normalizer.output_support.copy()
    else:
        normalized = np.zeros(cleaned.shape, dtype=np.float32)
        cleaned_support = np.zeros(cleaned.shape, dtype=bool)
    normalization_ready = bool(normalizer.ready)
    reg_ready = bool(glm_ready and normalization_ready and (cleaned_support & brain_mask).any())
    # Score against PRIOR samples. Reuse the first GLM fit's residual history
    # for startup instead of waiting for a second window of emitted residuals.
    was_calibrated = normalizer.ready
    if normalizer.mode == "mean":
        if current_brain_support.any():
            # Keep percent-of-raw-mean units: the denominator comes from raw
            # BOLD with the same spatial kernel, not zero-centred residuals.
            raw_reference = temporal_data
            if (smoother is not None and smoother.output_kind == "smooth"
                    and (normalizer.reference == "running"
                         or normalizer.reference_count < normalizer.ref_volumes)):
                raw_reference = smoother.apply(temporal_data, support_mask=current_processing_support)
            normalizer.observe(raw_reference, current_processing_support)
    elif glm_ready:
        # Preserve the GLM's valid support separately from normalization's
        # not-yet-calibrated output support.
        fit_support = getattr(processor, "output_support", None) if handler.motion_regressor is not None else current_processing_support
        fit_support = current_processing_support if fit_support is None else fit_support
        startup = (handler.motion_regressor.take_startup_reference()
                   if handler.motion_regressor is not None else None)
        if startup is not None:
            normalizer.initialize_from_moments(**startup)
            # The current residual is already represented in the first fit's
            # history (or outside the fixed first-N window). Do not add it twice.
        elif (handler.motion_regressor is None or normalizer.reference_count > 0) and (fit_support & brain_mask).any():
            # A fit can precede N valid design rows; wait for its startup
            # history snapshot instead of mixing partial emitted residuals in.
            normalizer.observe(cleaned, fit_support)
    norm_dir = cfg.rt_work_dir / "norm"
    norm_dir.mkdir(parents=True, exist_ok=True)
    norm_nii = norm_dir / f"vol_{volume_idx:05d}_norm.nii"
    norm_tmp = norm_nii.with_name("." + norm_nii.name)
    nib.save(nib.Nifti1Image(normalized, temporal_image.affine), str(norm_tmp))
    norm_tmp.replace(norm_nii)
    score_input_nii = norm_nii
    if not was_calibrated and normalizer.ready:
        save_normalization_reference(cfg, normalizer, temporal_image.affine, volume_idx)
    handler.normalization_last_volume = volume_idx
    log_step("NORM", volume_idx,
             f"{normalizer.mode}/{normalizer.reference}; prior_samples={reference_count}; ready={int(reg_ready)}",
             start_t=norm_t0, speed=speed)

    if speed is not None:
        speed.readiness(volume_idx, reg_ready)
    output_t0 = time.perf_counter()

    if handler.motion_regressor is not None:
        reg_names, reg_row = handler.motion_regressor.get_regressors(volume_idx)
        if reg_names and reg_row is not None:
            append_regressors(handler.cfg.rt_work_dir / "regressors_rt.csv", volume_idx, reg_names, reg_row)

    biopac_missing = False
    if handler.biopac_receiver is not None:
        biopac_missing = handler.biopac_receiver.was_missing(volume_idx)
    append_regression_status(
        handler.cfg.rt_work_dir / "regression_status_rt.csv",
        volume_idx,
        fd_to_save,
        handler.last_dvars_val,
        handler.last_dvars_z,
        fd_censor,
        dvars_censor,
        reg_ready,
        biopac_missing,
        glm_ready=glm_ready,
        normalization_ready=normalization_ready,
        normalization_reference_count=reference_count,
        reference_correlation=correlation_qc["reference_correlation"],
        reference_coverage_fraction=correlation_qc["reference_coverage_fraction"],
        correlation_status=correlation_qc["correlation_status"],
    )
    if REGRESSOR_SETTINGS.biopac_timelag and handler.biopac_receiver is not None:
        trigger_ts = handler.biopac_receiver.get_trigger_timestamp(volume_idx)
        if trigger_ts is not None:
            timelag = volume_timestamp - trigger_ts
            handler._biopac_timelag_sum += timelag
            handler._biopac_timelag_count += 1
            avg_timelag = handler._biopac_timelag_sum / handler._biopac_timelag_count
            append_biopac_timelag(
                handler._biopac_timelag_path,
                volume_idx,
                trigger_ts,
                volume_timestamp,
                timelag,
                avg_timelag,
            )

    if speed is not None:
        speed.stage(volume_idx, "OUTPUT", time.perf_counter() - output_t0)

    # GLM/normalization support already occupies the final grid: do not warp it again.
    final_support = cleaned_support & processing_mask
    support_dir = cfg.rt_work_dir / "support"
    support_dir.mkdir(parents=True, exist_ok=True)
    final_kind = "epi_ref" if analysis_space == "epi" and session_epi_bridge(cfg) is not None else analysis_space
    nib.save(nib.Nifti1Image(final_support.astype(np.uint8), temporal_image.affine),
             support_dir / f"vol_{volume_idx:05d}_{final_kind}.nii")
    if score_input_orig_nii is not None and not np.array_equal(final_orig_support, final_support):
        original_support_dir = cfg.rt_work_dir / "support_original"
        original_support_dir.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(final_orig_support.astype(np.uint8), temporal_image.affine),
                 original_support_dir / f"vol_{volume_idx:05d}_{final_kind}.nii")
    record_volume_support(cfg, handler, volume_idx, brain_mask, final_coverage,
                          cleaned_support, final_support)

    output_t0 = time.perf_counter()
    stream_kind = str(getattr(REGRESSOR_SETTINGS, "volume_stream_kind", "unwarped")).lower()
    stream_paths = {
        "raw": raw_nii,
        "mc": mc_nii,
        "unwarped": mc_nii,  # compatibility alias for fully corrected BOLD
        "reg": reg_nii,
        "norm": norm_nii,
        "score_input": score_input_nii,
    }
    handler.volume_streamer.publish(volume_idx, stream_paths.get(stream_kind), t1_stage_nii)

    if not cfg.enable_scoring:
        if handler.score_queue is not None:
            try:
                handler.score_queue.put_nowait(
                    {
                        "volume_idx": volume_idx,
                        "watchdog_timestamp": volume_timestamp,
                        "reg_ready": reg_ready,
                        "glm_ready": glm_ready,
                        "normalization_ready": normalization_ready,
                    }
                )
            except Exception as exc:
                log.error(f"[SCORE] Failed to enqueue volume {volume_idx:05d}: {exc}")
        if speed is not None:
            speed.stage(volume_idx, "OUTPUT", time.perf_counter() - output_t0)
        return True

    if speed is not None:
        speed.stage(volume_idx, "OUTPUT", time.perf_counter() - output_t0)

    # ---------- 4) Decoder scoring ----------
    t0 = time.perf_counter()
    try:
        # Load the warped volume (decoder/ROI space)
        score_img = nib.load(str(score_input_nii))
        score_data = np.asanyarray(score_img.dataobj)
        original_score = None
        if score_input_orig_nii is not None:
            score_orig_img = nib.load(str(score_input_orig_nii))
            score_orig_data = np.asanyarray(score_orig_img.dataobj)
            original_score = handler.scorer.score_from_array(score_orig_data)

        # Only accumulate baseline from *denoised* volumes
        if reg_ready and handler.scorer.baseline_count < handler.scorer.n_baseline:
            handler.scorer.accumulate_baseline(score_data)
            if handler.scorer.baseline_count == handler.scorer.n_baseline:
                handler.scorer.finalize_baseline()

        # Calibration zeros are unavailable features, not a neutral score.
        raw_score = handler.scorer.score_from_array(score_data) if reg_ready else float("nan")
        if reg_ready and np.isfinite(raw_score) and getattr(REGRESSOR_SETTINGS, "save_decoder_contributions", False):
            contributions_dir = cfg.rt_work_dir / "decoder_contributions"
            contributions_dir.mkdir(parents=True, exist_ok=True)
            contributions = handler.scorer.contributions_from_array(score_data)
            contributions_img = nib.Nifti1Image(contributions, score_img.affine, score_img.header.copy())
            contributions_img.set_data_dtype(np.float32)
            nib.save(contributions_img, contributions_dir / f"vol_{volume_idx:05d}_decoder_contributions.nii")
            contribution_metadata = contributions_dir / "decoder.json"
            if not contribution_metadata.exists():
                contribution_metadata.write_text(json.dumps({
                    "bias": float(handler.scorer.bias), "score": "score_raw",
                    "relation": "sum(voxel_contributions) + bias; subject to floating-point rounding",
                    "input_folder": "norm", "bias_included_in_image": False,
                }, indent=2) + "\n", encoding="utf-8")
        event_type = handler.score_event_tracker.for_volume(volume_idx)
        analysis_timestamp = time.time()
        timestamp = append_score(
            cfg.rt_work_dir / "scores.csv",
            volume_idx,
            raw_score,
            original_score=original_score,
            reg_ready=reg_ready,
            timestamp=analysis_timestamp,
            event_type=event_type,
        )
        z_score = None
        if handler.reference_score_stats is not None:
            stats = handler.reference_score_stats
            z_score = (raw_score - stats["mean"]) / stats["std"]
            append_score_z(
                cfg.rt_work_dir / "scores_z.csv",
                volume_idx,
                timestamp,
                raw_score,
                z_score,
                stats,
                reg_ready=reg_ready,
            )
        if handler.score_queue is not None:
            try:
                payload = {
                    "volume_idx": volume_idx,
                    "timestamp": timestamp,
                    "analysis_timestamp": analysis_timestamp,
                    "watchdog_timestamp": volume_timestamp,
                    "score_raw": raw_score,
                    "score_original": original_score,
                    "reg_ready": reg_ready,
                    "glm_ready": glm_ready,
                    "normalization_ready": normalization_ready,
                    "event_type": event_type,
                }
                if z_score is not None:
                    payload["score_z"] = z_score
                handler.score_queue.put_nowait(payload)
            except Exception as exc:
                log.error(f"[SCORE] Failed to enqueue score for vol {volume_idx:05d}: {exc}")

        if z_score is None:
            extra = f"raw={raw_score:.4f}"
        else:
            extra = f"raw={raw_score:.4f} z={z_score:.4f}"

        log_step("SCORE", volume_idx, extra, start_t=t0, speed=speed)

    except Exception as e:
        log.error(f"[SCORE] Failed scoring vol {volume_idx:05d}: {e}")
        return False

    return True






def unwarp_volume(raw_nii: Path, out_nii: Path, cfg: RTSessionConfig):
    if str(REGRESSOR_SETTINGS.fieldmap_method).lower() != "pyhysco":
        raise ValueError("unwarp -> MC supports PyHySCO only.")
    pe = str(REGRESSOR_SETTINGS.epi_phase_encoding).upper()
    spec = validate_bold(nib.load(str(raw_nii)), cfg.fmap_dir, pe, raw_nii)
    apply_pyhysco_fieldmap(
        epi_path=raw_nii, fieldmap_path=_preferred_pyhysco_fieldmap(cfg.fmap_dir),
        out_path=out_nii, phase_encoding_direction=spec["phase_encoding_axis"],
        polarity=1 if pe == "AP" else -1,
    )
    return True


def validate_run_provenance(cfg):
    """Do not mix old MC-first outputs or a different field/reference in a run."""
    folder = cfg.rt_work_dir
    marker = folder / "preprocessing_order.json"
    saved = json.loads(marker.read_text()) if marker.exists() else None
    if saved is not None and saved.get("coverage_policy") != COVERAGE_POLICY:
        raise ValueError("Run coverage handling changed. Use a fresh run output directory.")
    trans_dir = getattr(cfg, "trans_dir", None)
    if trans_dir is not None and (trans_dir / "registration_pending.json").exists():
        raise ValueError("Registration preparation is incomplete. Finish preparation before starting a run.")
    legacy_schema = saved is not None and not any(
        key in saved for key in ("analysis_space", "spatial_transform", "nuisance_masks"))
    if legacy_schema and trans_dir is not None and (trans_dir / "registration.json").exists():
        raise ValueError("Legacy run provenance cannot certify the new registration. Use a fresh run output directory.")
    payload = {"order": PIPELINE_ORDER, "coverage_policy": COVERAGE_POLICY,
               "temporal_processing_order": TEMPORAL_PROCESSING_ORDER,
               "normalization": normalization_spec(),
               "processing_mask": processing_mask_spec(cfg),
               "calibration": load_calibration(cfg.fmap_dir),
               "field": source_identity(_preferred_pyhysco_fieldmap(cfg.fmap_dir)),
               "reference": source_identity(cfg.rt_motion_ref_epi),
               "epi_phase_encoding": str(REGRESSOR_SETTINGS.epi_phase_encoding).upper()}
    space = str(REGRESSOR_SETTINGS.analysis_space).lower()
    interpolation = str(getattr(REGRESSOR_SETTINGS, "gpu_resampler_interpolation", "linear")).lower()
    payload["spatial_interpolation"] = interpolation
    payload["spatial_support"] = "nearest_brain_linear_availability_v2"
    if not legacy_schema:
        payload["analysis_space"] = space
    bridge = session_epi_bridge(cfg) if space == "epi" else None
    if not legacy_schema and (space in {"t1", "mni"} or bridge is not None):
        from gpu_ants_like_resampler import sampling_grid_identity
        transforms = [bridge] if bridge is not None else [cfg.trans_dir / "epi2t1_Composite.h5"]
        if space == "mni":
            transforms.insert(0, t1_to_mni_transform(cfg))
        payload["spatial_transform"] = sampling_grid_identity(
            cfg.rt_unwarped_analysis_ref_epi, resolve_analysis_reference(cfg, space), transforms)
        if bridge is not None:
            payload["epi_reference_mapped"] = True
    if not legacy_schema and bool(getattr(REGRESSOR_SETTINGS, "enable_motion_regression", False)):
        # These EPI-space masks can change when a registration is rebuilt even
        # when the final analysis stays in EPI space.
        payload["nuisance_masks"] = {}
        for setting, attribute in (("use_gs", "rt_unwarped_analysis_ref_mask"),
                                   ("use_wm", "rt_wm_mask"), ("use_vent", "rt_vent_mask")):
            if bool(getattr(REGRESSOR_SETTINGS, setting, False)):
                path = getattr(cfg, attribute)
                payload["nuisance_masks"][setting] = source_identity(path) if path.exists() else None
    fwhm = float(getattr(REGRESSOR_SETTINGS, "smoothing_fwhm_mm", 0))
    if not np.isfinite(fwhm) or fwhm < 0:
        raise ValueError("smoothing_fwhm_mm must be finite and >= 0.")
    mask_type = getattr(REGRESSOR_SETTINGS, "analysis_mask", "whole_brain")
    if final_output_kind(fwhm, mask_type) is not None:
        reference = resolve_analysis_reference(cfg, space)
        payload["final_masking"] = {"fwhm_mm": fwhm, "space": space,
            **mask_provenance(cfg, mask_type, space, reference,
                getattr(REGRESSOR_SETTINGS, "analysis_mask_file", None),
                getattr(REGRESSOR_SETTINGS, "analysis_mask_space", "final"))}
    if saved is not None:
        if saved != payload:
            raise ValueError("Run preprocessing, normalization, calibration/reference or spatial transform changed. Use a fresh run output directory.")
        if legacy_schema:
            log.info("[REGISTRATION] Reusing legacy run provenance with its original checks; "
                     "registration fingerprints remain unrecorded.")
        return  # Preserve legacy markers; never certify new fields retroactively.
    elif any((folder / name).exists() and any((folder / name).glob("*.nii*"))
             for name in ("mc", "unwarped", "reg", "norm", "epi_ref", "t1", "mni", "smooth", "masked")):
        raise ValueError("Legacy processed run detected. Use a fresh output directory for unwarp -> MC replay.")
    marker.write_text(json.dumps(payload, indent=2))


# ---------- Main ----------

def validate_nf_reference(cfg, settings=None):
    """Require the reference run's spatial files without preparing new assets."""
    from fmri_rt_preproc.session_alignment import validate_prepared_reference
    validate_prepared_reference(cfg, REGRESSOR_SETTINGS if settings is None else settings)


def prepare_nf_run(cfg, settings=None):
    """Check reference-run assets and prepare only the selected run's fieldmap.

    Spatial preparation belongs to RS/behavioral reference runs, never NF.
    Launchers use validate_nf_reference before opening and run this fieldmap
    readiness path inside their processing worker.
    """
    if settings is not None:
        REGRESSOR_SETTINGS.update(vars(settings))
    from fmri_rt_preproc.session_alignment import ensure_nf_session
    ensure_nf_session(cfg, REGRESSOR_SETTINGS, prepare_fieldmap=ensure_fieldmap_only)


def _process_recorded_snapshot(event_handler, existing):
    """Use the normal ordered workers, then complete their normal stop/export."""
    try:
        indices = [parse_dicom_name(path.name)[2] for path in existing]
        expected = set(range(1, len(indices) + 1))
        if not indices or len(set(indices)) != len(indices) or set(indices) != expected:
            raise ValueError("Recorded DICOM snapshot must be nonempty, unique and contiguous from volume 1.")
        for path in existing:
            event_handler.enqueue_path(path, backlog=True)
        while True:
            while event_handler.submit_pending():
                pass
            with event_handler._lock:
                completed = set(event_handler._processed_scans)
            if expected.issubset(completed):
                break
            time.sleep(0.01)
    finally:
        event_handler.stop()
        exporter = getattr(event_handler, "_qc_export_thread", None)
        if exporter is not None:
            exporter.join()


def run_rt_pipeline(cfg: RTSessionConfig, score_queue: Optional[object] = None):
    analysis_space = str(REGRESSOR_SETTINGS.analysis_space).lower()
    if analysis_space not in {"epi", "t1", "mni"}:
        raise ValueError(
            f"Unsupported analysis_space={REGRESSOR_SETTINGS.analysis_space!r}. "
            "Use 'epi', 't1', or 'mni'."
        )
    if getattr(cfg, "recorded_only", False) and REGRESSOR_SETTINGS.enable_biopac_physio:
        raise ValueError("recorded_only does not support live BIOPAC; disable enable_biopac_physio.")
    if cfg.enable_scoring and analysis_space in {"epi", "t1"} and cfg.decoder_template is None:
        raise ValueError(
            "EPI/T1-space scoring requires --decoder-template pointing to a decoder in the selected space."
        )
    decoder_template = resolve_decoder_template(cfg)

    validate_regression_status_schema(cfg.rt_work_dir / "regression_status_rt.csv")

    from fmri_rt_preproc.session_alignment import ensure_nf_session
    ensure_nf_session(cfg, REGRESSOR_SETTINGS, prepare_fieldmap=ensure_fieldmap_only)
    validate_run_provenance(cfg)

    cfg.reference_score_stats = load_reference_score_stats(cfg, cfg.reference_score_run)
    write_session_metadata(cfg, decoder_template)
    # Process existing DICOMs first (offline-style), but only for this run
    existing = []
    for path in sorted(cfg.incoming_dir.glob("*.dcm")):
        parsed = parse_dicom_name(path.name)
        if parsed is None:
            continue
        _, run_id, _ = parsed
        if run_id == int(cfg.run):
            existing.append(path)
    defer_biopac = bool(existing) or REGRESSOR_SETTINGS.biopac_start_online_only
    event_handler = DICOMHandler(
        cfg,
        score_queue=score_queue,
        start_biopac=not defer_biopac,
    )
    if getattr(cfg, "recorded_only", False):
        _process_recorded_snapshot(event_handler, existing)
        return
    observer = Observer()
    observer.schedule(event_handler, str(cfg.incoming_dir), recursive=False)

    # Prime backlog provenance before created/closed notifications can arrive.
    if event_handler.speed is not None:
        for path in existing:
            event_handler.speed.arrive(parse_dicom_name(path.name)[2], backlog=True)

    observer.start()
    # Handler initialization can take several seconds. Files written after the
    # initial snapshot but before the observer started have no queued event.
    # Subscribe first, then rescan before submitting any work. Overlapping
    # created/closed events use the handler's existing deduplication guards.
    existing = []
    for path in sorted(cfg.incoming_dir.glob("*.dcm")):
        parsed = parse_dicom_name(path.name)
        if parsed is not None and parsed[1] == int(cfg.run):
            existing.append(path)
    if existing:
        print(f"[RT] Found {len(existing)} existing DICOMs — processing offline first…")
        for f in existing:
            event_handler.enqueue_path(Path(f), backlog=True)
        while event_handler.submit_pending():
            continue

    if defer_biopac:
        event_handler.enable_online_mode()

    print("[RT] Switching to online mode.")
    try:
        event_handler.enable_online_mode()
        while True:
            while event_handler.submit_pending():
                continue
            time.sleep(0.2)
    except KeyboardInterrupt:
        observer.stop()
    finally:
        event_handler.stop()
    observer.join()

def compute_dvars(prev_vol: np.ndarray, cur_vol: np.ndarray, mask: np.ndarray,
                  previous_support=None, current_support=None) -> float:
    """RMS temporal difference over voxels acquired in both adjacent volumes."""
    valid = np.ones(cur_vol.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool).copy()
    for support in (previous_support, current_support):
        if support is not None:
            support = np.asarray(support)
            if support.shape != valid.shape or support.dtype.kind != "b":
                raise ValueError("DVARS support must be boolean and match the volume grid.")
            valid &= support
    if not valid.any():
        return float("nan")
    diff = cur_vol[valid] - prev_vol[valid]
    diff = diff.astype(np.float32)
    return float(np.sqrt(np.mean(diff * diff)))

def robust_z(x: float, history: list[float]) -> float:
    # robust z using median and MAD from history
    if len(history) < 5:
        return float("nan")
    med = float(np.median(history))
    mad = float(np.median(np.abs(np.asarray(history) - med)))
    if mad < 1e-8:
        return float("nan")
    return float((x - med) / (1.4826 * mad))

def append_dvars(qc_path: Path, volume_idx: int, fd: float, dvars: float, dvars_z: float):
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    exists = qc_path.exists()
    with open(qc_path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["volume_idx", "fd", "dvars", "robust_z"])
        w.writerow([volume_idx, fd, dvars, dvars_z])


def append_regressors(reg_path: Path, volume_idx: int, reg_names: list[str], reg_row: np.ndarray):
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    exists = reg_path.exists()
    with open(reg_path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["volume_idx", *reg_names])
        w.writerow([volume_idx, *reg_row.tolist()])


REGRESSION_STATUS_FIELDS = ["volume_idx", "fd", "dvars", "dvars_z", "fd_censor", "dvars_censor",
    "reg_ready", "biopac_missing", "glm_ready", "normalization_ready", "normalization_reference_count",
    "reference_correlation", "reference_coverage_fraction", "correlation_status"]


def validate_regression_status_schema(path):
    """Refuse mixed old/new CSV columns before preparation or new volume work."""
    path = Path(path)
    if not path.exists() or not path.stat().st_size:
        return
    with path.open(newline="", encoding="utf-8-sig") as stream:
        header = next(csv.reader(stream), [])
    if header != REGRESSION_STATUS_FIELDS:
        raise ValueError("Existing regression_status_rt.csv has a different column schema. Use a fresh run directory; existing QC was not changed.")


def append_regression_status(
    qc_path: Path,
    volume_idx: int,
    fd: float,
    dvars: float,
    dvars_z: float,
    fd_censor: int,
    dvars_censor: int,
    reg_ready: bool,
    biopac_missing: bool,
    *, glm_ready=None, normalization_ready=None, normalization_reference_count=None,
    reference_correlation=float("nan"), reference_coverage_fraction=float("nan"),
    correlation_status="unavailable",
):
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    validate_regression_status_schema(qc_path)
    exists = qc_path.exists() and qc_path.stat().st_size > 0
    with open(qc_path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(REGRESSION_STATUS_FIELDS)
        w.writerow([volume_idx, fd, dvars, dvars_z, fd_censor, dvars_censor, int(reg_ready), int(biopac_missing),
                    int(glm_ready) if glm_ready is not None else "",
                    int(normalization_ready) if normalization_ready is not None else "",
                    normalization_reference_count if normalization_reference_count is not None else "",
                    reference_correlation, reference_coverage_fraction, correlation_status])


def append_biopac_timelag(
    path: Path,
    volume_idx: int,
    trigger_timestamp: float,
    volume_timestamp: float,
    timelag_s: float,
    avg_timelag_s: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(
                [
                    "volume_idx",
                    "trigger_timestamp",
                    "volume_timestamp",
                    "timelag_s",
                    "avg_timelag_s",
                ]
            )
        w.writerow(
            [
                volume_idx,
                f"{trigger_timestamp:.6f}",
                f"{volume_timestamp:.6f}",
                f"{timelag_s:.6f}",
                f"{avg_timelag_s:.6f}",
            ]
        )


def main():
    parser = argparse.ArgumentParser(description="Real-time fMRI watcher pipeline")
    parser.add_argument("--sub", required=True, help="Subject ID, e.g. 00086")
    parser.add_argument("--day", required=True, help="Day/session, e.g. 3")
    parser.add_argument("--run", required=True, help="Run number, e.g. 4 (matches 000004 in DICOM name)")
    parser.add_argument(
        "--ap-block",
        type=int,
        default=None,
        help="AP block to use for b0",
    )
    parser.add_argument(
        "--pa-block",
        type=int,
        default=None,
        help="PA block to use for b0",
    )
    parser.add_argument(
        "--rs",
        dest="reference_score_run",
        help="Reference run ID for z-scoring (uses scores.csv from that run).",
    )
    parser.add_argument(
        "--incoming-root",
        required=False,
        default="/home/sin/DecNef_pain_Dec23/realtime/incoming/pain7T/20251105.20251105_00085.Kostya",
        help="Folder where scanner writes DICOMs in real-time.",
    )
    parser.add_argument(
        "--base-data",
        required=False,
        default="/SSD2/DecNef_py/data",
        help="Base preproc data folder (same as offline pipeline).",
    )
    parser.add_argument(
        "--decoder-template",
        required=False,
        help="Optional decoder template path to override the default.",
    )
    parser.add_argument(
        "--decoder-roi-txt",
        required=False,
        help="Optional ROI_DECODER-style text file used for scoring mask/weights. If omitted, scoring uses decoder NIfTI nonzero voxels.",
    )
    parser.add_argument(
        "--no-score",
        action="store_true",
        help="Disable decoder scoring (still runs motion correction + warps).",
    )
    parser.add_argument(
        "--enable-original-score",
        action="store_true",
        help="Also compute score_original from a separately warped pre-denoise/pre-normalization volume.",
    )
    parser.add_argument(
        "--biopac-enable",
        action="store_true",
        default=None,
        help="Enable BIOPAC RetroTS regressors via TCP.",
    )
    parser.add_argument(
        "--biopac-host",
        default=None,
        help="Host to bind BIOPAC receiver.",
    )
    parser.add_argument(
        "--biopac-port",
        type=int,
        default=None,
        help="Port to bind BIOPAC receiver.",
    )
    parser.add_argument(
        "--biopac-timeout",
        type=float,
        default=None,
        help="Seconds to wait for physio regressors before zero-fill.",
    )
    parser.add_argument(
        "--biopac-phys-reg",
        default=None,
        choices=["RICOR8", "RVT5", "RVT+RICOR13"],
        help="Physio regressor family to expect from BIOPAC stream.",
    )
    parser.add_argument(
        "--biopac-mode",
        default=None,
        choices=["tcp", "file"],
        help="BIOPAC input mode: tcp (listen on socket) or file (tail CSV).",
    )
    parser.add_argument(
        "--biopac-file",
        default=None,
        help="Path to BIOPAC regressors CSV when using --biopac-mode=file.",
    )
    parser.add_argument(
        "--biopac-poll",
        type=float,
        default=None,
        help="Polling interval (seconds) for file-backed BIOPAC buffer.",
    )
    parser.add_argument(
        "--biopac-handshake",
        action="store_true",
        default=None,
        help="Send a handshake with TR to the BIOPAC streamer.",
    )
    parser.add_argument(
        "--biopac-start-online",
        action="store_true",
        default=None,
        help="Defer BIOPAC receiver start until after offline DICOM processing.",
    )
    parser.add_argument(
        "--biopac-timelag",
        action="store_true",
        default=None,
        help="Log per-volume trigger-to-volume timelag and running average.",
    )
    parser.add_argument(
        "--analysis-space",
        choices=["epi", "t1", "mni"],
        default=None,
        help="Space for scoring/output volumes: epi (native EPI), t1 (apply EPI->T1), or mni (apply EPI->T1->MNI).",
    )
    parser.add_argument(
        "--voxel-norm-ref-volumes",
        type=int,
        default=None,
        help="Normalization reference length and minimum GLM startup count; overrides the settings file only when supplied.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum parallel processing workers for DICOM handling.",
    )
    parser.add_argument(
        "--rt-max-scan-length",
        type=int,
        default=None,
        help="Maximum TR count preallocated by RTPSpy regression.",
    )
    parser.add_argument(
        "--pipeline-engine",
        choices=["parallel_ordered", "legacy"],
        default=None,
        help="Pipeline execution engine.",
    )
    parser.add_argument(
        "--commit-wait-timeout-s",
        type=float,
        default=None,
        help="Seconds to wait before force-advancing ordered commit for stalled scans.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="Maximum retries per DICOM if processing fails.",
    )
    parser.add_argument(
        "--volume-streamer",
        action="store_true",
        default=None,
        help="Open a small best-effort local QC window showing selected live NIfTI volumes.",
    )
    parser.add_argument(
        "--volume-stream-kind",
        choices=["raw", "mc", "unwarped", "reg", "score_input"],
        default=None,
        help="Which per-volume NIfTI to show in the QC window. Defaults to settings-file/global config.",
    )
    parser.add_argument(
        "--volume-stream-every-n",
        type=int,
        default=None,
        help="Render only every Nth volume in the QC window to reduce system load. Defaults to settings-file/global config.",
    )

    parser.add_argument(
        "--settings-file",
        default=None,
        help="Optional JSON file with global runtime settings (TR, censor thresholds, BIOPAC defaults, etc.).",
    )
    from rt_global_settings import add_registration_arguments, apply_registration_arguments
    add_registration_arguments(parser)
    parser.add_argument("--preparation-only", action="store_true",
                        help="Process a preparation/RS run without the NF session gate.")
    parser.add_argument("--recorded-only", action="store_true",
                        help="Process existing contiguous DICOMs and return; do not start a watchdog.")
    args = parser.parse_args()

    if args.settings_file:
        loaded = load_regressor_settings(args.settings_file)
        REGRESSOR_SETTINGS.update(vars(loaded))
    apply_registration_arguments(REGRESSOR_SETTINGS, args)

    # Omitted flags must not replace values just loaded from the JSON profile.
    # Preserve the standalone CLI's historical BIOPAC-off default without a profile.
    if args.biopac_enable is None and not args.settings_file:
        args.biopac_enable = False
    overrides = {
        "enable_biopac_physio": args.biopac_enable,
        "biopac_host": args.biopac_host,
        "biopac_port": args.biopac_port,
        "biopac_timeout": args.biopac_timeout,
        "biopac_phys_reg": args.biopac_phys_reg,
        "biopac_handshake": args.biopac_handshake,
        "biopac_start_online_only": args.biopac_start_online,
        "biopac_mode": args.biopac_mode,
        "biopac_file": args.biopac_file,
        "biopac_poll_interval": args.biopac_poll,
        "biopac_timelag": args.biopac_timelag,
        "analysis_space": args.analysis_space,
        "voxel_norm_ref_volumes": args.voxel_norm_ref_volumes,
        "max_workers": args.max_workers,
        "rt_max_scan_length": args.rt_max_scan_length,
        "pipeline_engine": args.pipeline_engine,
        "commit_wait_timeout_s": (max(0.1, args.commit_wait_timeout_s)
                                  if args.commit_wait_timeout_s is not None else None),
        "max_retries": args.max_retries,
        "enable_volume_streamer": args.volume_streamer,
        "volume_stream_kind": args.volume_stream_kind,
        "volume_stream_every_n": args.volume_stream_every_n,
    }
    REGRESSOR_SETTINGS.update({key: value for key, value in overrides.items() if value is not None})

    cfg = RTSessionConfig(
        preparation_only=args.preparation_only,
        recorded_only=args.recorded_only,
        subject=args.sub,
        day=args.day,
        run=args.run,
        incoming_root=Path(args.incoming_root),
        base_data=Path(args.base_data),
        decoder_template=Path(args.decoder_template) if args.decoder_template else None,
        decoder_roi_txt=Path(args.decoder_roi_txt) if args.decoder_roi_txt else None,
        reference_score_run=args.reference_score_run,
        enable_scoring=not args.no_score,
        enable_original_score=args.enable_original_score,
        ap_block=args.ap_block,
        pa_block=args.pa_block,
    )

    if not cfg.incoming_dir.exists():
        raise FileNotFoundError(f"Incoming directory does not exist: {cfg.incoming_dir}")

    run_rt_pipeline(cfg)


if __name__ == "__main__":
    main()
