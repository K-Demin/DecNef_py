#!/usr/bin/env python
import time
import csv
import logging
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

import nibabel as nib
import numpy as np

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from fmri_rt_preproc.RTPSpy_tools.rtp_volreg import RtpVolreg
from fmri_rt_preproc.RTPSpy_tools.rtp_regress import RtpRegress
from fmri_rt_preproc.utils import run  # your existing run() wrapper

from decoder_score import DecoderScorer
from biopac_rt.biopac_receiver import (
    BiopacReceiverConfig,
    BiopacRetroTSReceiver,
    BiopacRetroTSFileBuffer,
)

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

# ---------- Regressor config (edit to control usage) ----------

@dataclass
class RegressorSettings:
    enable_motion_regression: bool = True
    mot_reg: str = "mot6"
    max_poly_order: float = np.inf
    TR: float = 1
    use_gs: bool = False
    use_wm: bool = True
    use_vent: bool = True

    # --- censor regressors ---
    enable_fd_censor_reg: bool = True
    fd_thr: float = 0.3            #  (units = mm)

    enable_dvars_censor_reg: bool = True
    dvars_thr_robust_z: float = 3.0  # robust z threshold

    censor_plus1: bool = True        # add +1 TR neighbor
    dvars_warmup: int = 20           # don’t compute robust stats until you have some history
    dvars_mask_source: str = "ref_mask"  # "ref_mask" uses cfg.rt_ref_mask

    # --- BIOPAC physio regressors (RETROTS) ---
    enable_biopac_physio: bool = True
    biopac_phys_reg: str = "RICOR8"
    biopac_host: str = "0.0.0.0"
    biopac_port: int = 15000
    biopac_timeout: float = 0.3
    biopac_handshake: bool = True
    biopac_start_online_only: bool = False
    biopac_mode: str = "tcp"
    biopac_file: Optional[Path] = None
    biopac_poll_interval: float = 0.05
    max_workers: int = 6
    max_retries: int = 3


REGRESSOR_SETTINGS = RegressorSettings(
    # How to use:
    # - enable_motion_regression: True/False to toggle RtpRegress entirely.
    # - mot_reg: one of {"None", "mot6", "mot12", "dmot6"} (RTPSpy-supported).
    # - max_poly_order: int >= 0 or np.inf (higher allows more polynomial terms).
    # - TR: float > 0 (seconds, used for polynomial regressor timing).
    # - use_gs/use_wm/use_vent: True/False to include each mask regressor when file exists.
    enable_motion_regression=True,
    mot_reg="mot6",
    max_poly_order=np.inf,
    TR=1,
    use_gs=False, # Probably it's better to avoid cause it correlates with global brain activity
    use_wm=True,
    use_vent=True,
)

def log_step(step: str, vol: int, extra: str = "", start_t=None):
    """Compact colored/clean log."""
    v = f"{vol:05d}"
    if start_t is not None:
        dt = time.time() - start_t
        log.info(f"[{step:<5}] vol {v}  {extra}  ({dt*1000:.1f} ms)")
    else:
        log.info(f"[{step:<5}] vol {v}  {extra}")


def append_score(csv_path: Path, volume_idx: int, raw_score: float) -> float:
    timestamp = time.time()
    exists = csv_path.exists()

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["volume_idx", "timestamp", "score_raw"])
        writer.writerow([volume_idx, timestamp, raw_score])
    return timestamp

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

class MotionRegressor:
    def __init__(
        self,
        volreg: RtpVolreg,
        gs_mask: Optional[Path] = None,
        wm_mask: Optional[Path] = None,
        vent_mask: Optional[Path] = None,
        mot_reg: str = "mot6",
        max_poly_order: float = np.inf,
        TR: float = 1,
        max_scan_length: int = 1000,
        enable_fd_censor_reg: bool = False,
        enable_dvars_censor_reg: bool = False,
        phys_reg: str = "None",
        rtp_physio: Optional[object] = None,
    ):
        kwargs = dict(
            mot_reg=mot_reg,
            volreg=volreg,
            TR=TR,
            wait_num=0,
            max_poly_order=max_poly_order,
            save_proc=False,
            online_saving=False,
            reg_retro_proc=False,
            max_scan_length=max_scan_length,
            spike_reg_num=200,
            phys_reg=phys_reg,
            rtp_physio=rtp_physio,
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
    ) -> tuple[np.ndarray, bool]:
        """
        Returns (cleaned_vol, regressed_bool). Always returns a tuple.
        """

        # Ensure regressor init/ready state
        if not self._ready:
            try:
                self._ready = bool(self._regress.ready_proc())
            except Exception as exc:
                log.error(f"[REG] Failed to prepare regressor: {exc}")
                return np.asanyarray(mc_img.dataobj), False

        if not self._ready:
            return np.asanyarray(mc_img.dataobj), False

        # Do regression
        try:
            prev_vol = getattr(self._regress, "_vol_num", 0)

            # RTPSpy modifies mc_img in-place
            self._regress.do_proc(
                mc_img,
                vol_idx=volume_idx - 1,
                fd_censor=fd_censor,
                dvars_censor=dvars_censor,
            )

            cur_vol = getattr(self._regress, "_vol_num", prev_vol + 1)
            regressed = cur_vol > getattr(self._regress, "wait_num", 0)

            cleaned = np.asarray(mc_img.dataobj, dtype=np.float32)
            return cleaned, regressed

        except Exception as exc:
            log.error(f"[REG] Motion regression failed at vol {volume_idx:05d}: {exc}")
            return np.asanyarray(mc_img.dataobj), False

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

    @property
    def subject_root(self) -> Path:
        return self.base_data / f"sub-{self.subject}"

    @property
    def day_root(self) -> Path:
        return self.subject_root / self.day

    @property
    def trans_dir(self) -> Path:
        # precomputed transforms live here (from offline pipeline)
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
    def rt_ref_epi(self) -> Path:
        """
        Global real-time reference EPI (set by offline preprocessor).
        """
        return self.day_root / "func" / "trans" / "epi_unwarped_mean.nii"

    @property
    def rt_ref_mask(self) -> Path:
        """
        Optional mask for the RT reference (not strictly needed here,
        but kept for completeness / future use).
        """
        return self.day_root / "func" / "trans" / "epi_mask_mean.nii"


def resolve_decoder_template(cfg: RTSessionConfig) -> Path:
    return cfg.decoder_template or (
        Path(cfg.base_data).parent
        / "decoders"
        / "rweights_NSF_grouppred_cvpcrTMP_nonzeros.nii"
    )


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
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


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
        self._executor = ThreadPoolExecutor(max_workers=REGRESSOR_SETTINGS.max_workers)

        # --- RTPSpy Volreg ---
        self.volreg = RtpVolreg(regmode='heptic')
        self.volreg.ignore_init = 0
        self.volreg.save_proc = False

        # --- NEW: reference is the global offline EPI mean ---
        self.ref_set = False
        ref_epi = self.cfg.rt_ref_epi
        if not ref_epi.exists():
            raise FileNotFoundError(
                f"RT reference EPI not found at {ref_epi}. "
                f"Run the offline preprocessing pipeline first so "
                f"rt_ref_epi.nii is created in {self.cfg.day_root / 'func'}."
            )
        self.volreg.set_ref_vol(str(ref_epi))
        self.ref_set = True

        # --- Motion / FD state ---
        self.motion_file = self.cfg.rt_work_dir / "motion_rt.1D"
        self.fd_file = self.cfg.rt_work_dir / "fd_rt.csv"
        self.prev_motion = None              # previous 6-vector
        self.brain_radius_mm = 50.0          # standard radius for FD
        self.pre_trial_scans = 0             # if you ever want NaNs for early scans

        # --- NEW: DVARS state ---
        self.prev_mc_for_dvars = None
        self.dvars_hist = []  # store prior DVARS to compute robust stats
        self.censor_next_fd = False
        self.censor_next_dvars = False
        self.last_dvars_val = float("nan")
        self.last_dvars_z = float("nan")

        # --- NEW: load mask for DVARS (use cfg.rt_ref_mask) ---
        self.dvars_mask = None
        if REGRESSOR_SETTINGS.enable_dvars_censor_reg:
            mpath = cfg.rt_ref_mask
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

        gs = resolve_mask("GS", cfg.rt_ref_mask, REGRESSOR_SETTINGS.use_gs)
        wm = resolve_mask("WM", cfg.rt_wm_mask, REGRESSOR_SETTINGS.use_wm)
        vent = resolve_mask("Vent", cfg.rt_vent_mask, REGRESSOR_SETTINGS.use_vent)

        used = {k: v for k, v in {"GS": gs, "WM": wm, "Vent": vent}.items() if v is not None}
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
            self.motion_regressor = MotionRegressor(
                self.volreg,
                gs_mask=gs,
                wm_mask=wm,
                vent_mask=vent,
                mot_reg=REGRESSOR_SETTINGS.mot_reg,
                max_poly_order=REGRESSOR_SETTINGS.max_poly_order,
                TR=REGRESSOR_SETTINGS.TR,
                max_scan_length=1000,  # or your typical max TR count for a run
                enable_fd_censor_reg=REGRESSOR_SETTINGS.enable_fd_censor_reg,
                enable_dvars_censor_reg=REGRESSOR_SETTINGS.enable_dvars_censor_reg,
                phys_reg=phys_reg,
                rtp_physio=rtp_physio,
            )
        else:
            log.info("[REG] Motion regression disabled by config.")
            self.motion_regressor = None

        # --- Source container for GS/WM/Vent regressors (RTPSpy expects mask_src_proc.proc_data) ---
        self.proc_src = ProcSrc()
        if self.motion_regressor is not None:
            self.motion_regressor._regress.mask_src_proc = self.proc_src

        # --- Decoder / scorer ---
        decoder_path = resolve_decoder_template(cfg)
        roi_txt = cfg.trans_dir / "ROI_DECODER.txt"

        self.scorer = DecoderScorer(
            decoder_path,
            roi_txt=roi_txt,
            n_baseline=20,   # keep your current baseline length
        )

    def stop(self):
        if self.biopac_receiver is not None:
            self.biopac_receiver.stop()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def start_biopac(self):
        if self.biopac_receiver is None:
            return
        if self._biopac_timeout is not None:
            self.biopac_receiver.config.timeout = self._biopac_timeout
        self.biopac_receiver.start()

    def on_created(self, event):
        if event.is_directory:
            return

        path = Path(event.src_path)
        log.info(f"[WATCHDOG] File detected: {path}")
        self.enqueue_path(path)

    def enqueue_path(self, path: Path) -> None:
        parsed = parse_dicom_name(path.name)
        if parsed is None:
            log.debug(f"[WATCHDOG] Ignoring non-matching file: {path.name}")
            return

        _, run_id, scan = parsed
        if run_id != self.current_run:
            log.debug(f"[WATCHDOG] Ignoring run {run_id}, expecting {self.current_run}")
            return
        if scan in self._processed_scans or scan in self._pending_scans:
            return
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
            volume_idx = self.next_volume_idx
            self.next_volume_idx += 1
            self._inflight_scans.add(scan)
        self._executor.submit(self._process_scan, path, scan, volume_idx)
        return True

    def _process_scan(self, path: Path, scan: int, volume_idx: int) -> None:
        log.info(f"[WATCHDOG] Processing volume idx {volume_idx} (run={self.current_run}, scan={scan})")
        ok = False
        for attempt in range(1, REGRESSOR_SETTINGS.max_retries + 1):
            ok = process_volume(self.cfg, self, path, volume_idx)
            if ok:
                break
            log.warning(
                "[WATCHDOG] Retry scan %s (attempt %s/%s)",
                scan,
                attempt,
                REGRESSOR_SETTINGS.max_retries,
            )
            time.sleep(0.2)
        if not ok:
            log.error("[WATCHDOG] Giving up on scan %s after %s attempts.", scan, REGRESSOR_SETTINGS.max_retries)
        with self._lock:
            self._inflight_scans.discard(scan)
            self.mark_processed(scan)

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


# ---------- Core processing hook (DICOM -> NIfTI -> MC) ----------

def process_volume(cfg: RTSessionConfig, handler: "DICOMHandler",
                   dicom_path: Path, volume_idx: int):
    """
    For each incoming DICOM:
      1) DICOM -> raw NIfTI (single volume)
      2) Motion correction with RTPSpy -> mc NIfTI
      3) Apply ANTs transforms (EPI->T1->MNI) -> mni NIfTI
    """

    # ---------- 1) DICOM -> raw NIfTI ----------
    t0 = time.time()

    raw_dir = cfg.rt_raw_dir
    raw_nii = raw_dir / f"vol_{volume_idx:05d}.nii"

    if not raw_nii.exists():
        if not wait_for_file_complete(dicom_path, timeout=5.0, interval=0.1):
            log.error(f"[DICOM] vol {volume_idx:05d} FAILED (file not stable)")
            return False
        for attempt in range(3):
            run([
                "dcm2niix",
                "-z", "n",  # no gzip
                "-s", "y",
                "-b", "n",
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
            return False

    log_step("DICOM", volume_idx, start_t=t0)

    # ---------- 1.5) APPLY FIELD MAP UNWARP BEFORE MC ----------
    t0 = time.time()
    unwarp_dir = cfg.rt_unwarp_dir
    unwarped_nii = unwarp_dir / f"vol_{volume_idx:05d}_uw.nii"

    if not unwarped_nii.exists():
        ok = unwarp_volume(raw_nii, unwarped_nii, cfg)
        if not ok:
            log.error(f"[FMAP] Failed unwarp for {raw_nii}")
            return False
        log_step("FMAP", volume_idx, start_t=t0)
    else:
        log.info(f"[FMAP] Unwarp exists for vol {volume_idx}")


    # ---------- 2) Motion correction (RtpVolreg) ----------
    t0 = time.time()
    mc_dir = cfg.rt_mc_dir
    mc_nii = mc_dir / f"vol_{volume_idx:05d}_mc.nii"

    # use UNWARPED as input to MC
    img = nib.load(str(unwarped_nii))
    data = np.asanyarray(img.dataobj).astype(np.float32)

    # Create a temporary NIfTI for RtpVolreg to work on
    tmp_img = nib.Nifti1Image(data, img.affine, img.header.copy())
    tmp_img.set_filename(str(mc_nii))

    # Run RTPSpy volreg (works in-place on tmp_img.dataobj)
    handler.volreg.do_proc(tmp_img, vol_idx=volume_idx - 1)

    # Extract corrected data and save with a FRESH header
    mc_data = np.asanyarray(tmp_img.dataobj).astype(np.float32)
    mc_img = nib.Nifti1Image(mc_data, img.affine)  # new clean header
    nib.save(mc_img, str(mc_nii))

    # ----- 2b) MOTION + FD (ONLINE) -----
    # RtpVolreg stores motion as [x y z rx ry rz] per volume (AFNI-style).
    # Typically translations in mm, rotations in degrees.
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

    # Convert rotations (rx, ry, rz) from degrees → radians, then to mm
    trans = delta[:3]                              # mm
    rot_deg = delta[3:]                            # degrees
    rot_rad = rot_deg * np.pi / 180.0              # radians
    disp_rot = handler.brain_radius_mm * rot_rad   # mm

    # Framewise displacement: sum of absolute displacement
    fd_value = float(np.sum(np.abs(np.concatenate([trans, disp_rot]))))

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
    )

    # ---------- 2b.5) CENSOR FLAGS (FD + DVARS) ----------
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
            dvars_val = compute_dvars(handler.prev_mc_for_dvars, mc_data, handler.dvars_mask)

            # robust z against prior DVARS history (exclude current)
            z = robust_z(dvars_val, handler.dvars_hist)

            # update history after computing z
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


    # ---------- 2c) Motion regression (RTPS_py) ----------
    reg_t0 = time.time()
    mc_for_warp = mc_nii
    if handler.motion_regressor is not None:
        handler.proc_src.proc_data = np.asanyarray(mc_img.dataobj)
        cleaned, reg_ready = handler.motion_regressor.apply(
            mc_img,
            volume_idx,
            fd_censor=fd_censor,
            dvars_censor=dvars_censor,
        )
        reg_dir = cfg.rt_reg_dir
        reg_nii = reg_dir / f"vol_{volume_idx:05d}_reg.nii"
        nib.save(nib.Nifti1Image(cleaned, img.affine), str(reg_nii))
        mc_for_warp = reg_nii
        log_step("REG", volume_idx, "motion", start_t=reg_t0)
    else:
        cleaned = np.asanyarray(mc_img.dataobj)
        reg_ready = True
        log_step("REG", volume_idx, "skipped", start_t=reg_t0)

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
    )

    # ---------- 3) Apply ANTs transforms to MNI ----------
    t0 = time.time()
    mni_dir = cfg.rt_mni_dir
    mni_nii = mni_dir / f"vol_{volume_idx:05d}_mni.nii"

    warp_t1_mni = cfg.subject_root / "anat" / "warp_T1_to_MNI_synth.nii"
    epi2t1 = cfg.trans_dir / "epi2t1_Composite.h5"
    decoder_template = resolve_decoder_template(cfg)

    if not decoder_template.exists():
        log.error(f"Decoder template not found at {decoder_template}")
        return False

    if not (warp_t1_mni.exists() and epi2t1.exists()):
        log.error(f"Missing transforms in {cfg.trans_dir}")
        return False

    cmd = [
        "bash", "-lc",
        f"""
          export ANTS_USE_GPU=1
          export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$(nproc)
          export OMP_NUM_THREADS=$(nproc)
          antsApplyTransforms \
            -d 3 \
            -i {mc_for_warp} \
            -r {decoder_template} \
            -o {mni_nii} \
            -t {warp_t1_mni} \
            -t {epi2t1} \
            -n Linear --float 1
        """
    ]
    run(cmd)
    log_step("ANTS", volume_idx, "warp→MNI", start_t=t0)

    # ---------- 4) Decoder scoring ----------
    t0 = time.time()
    try:
        # Load the warped volume (decoder/ROI space)
        mni_img = nib.load(str(mni_nii))
        mni_data = np.asanyarray(mni_img.dataobj)

        # Only accumulate baseline from *denoised* volumes
        if reg_ready and handler.scorer.baseline_count < handler.scorer.n_baseline:
            handler.scorer.accumulate_baseline(mni_data)
            if handler.scorer.baseline_count == handler.scorer.n_baseline:
                handler.scorer.finalize_baseline()

        # Always compute raw; z will be NaN until baseline_ready
        raw_score = handler.scorer.score_from_array(mni_data)
        timestamp = append_score(cfg.rt_work_dir / "scores.csv", volume_idx, raw_score)
        if handler.score_queue is not None:
            try:
                handler.score_queue.put_nowait(
                    {
                        "volume_idx": volume_idx,
                        "timestamp": timestamp,
                        "score_raw": raw_score,
                    }
                )
            except Exception as exc:
                log.error(f"[SCORE] Failed to enqueue score for vol {volume_idx:05d}: {exc}")

        extra = f"raw={raw_score:.4f}"

        log_step("SCORE", volume_idx, extra, start_t=t0)

    except Exception as e:
        log.error(f"[SCORE] Failed scoring vol {volume_idx:05d}: {e}")
        return False

    return True






def unwarp_volume(raw_nii: Path, out_nii: Path, cfg: RTSessionConfig):
    warp = cfg.day_root / "fmap" / "AP2PA_1InverseWarp.nii"
    affine = cfg.day_root / "fmap" / "AP2PA_0GenericAffine.mat"
    pa_mean = cfg.day_root / "fmap" / "PA_mean.nii.gz"

    if not warp.exists() or not affine.exists():
        log.error("[FMAP] Missing AP→PA warp or affine")
        return False

    cmd = [
        "bash", "-lc",
        f"""
        export ANTS_USE_GPU=1
        export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$(nproc)
        export OMP_NUM_THREADS=$(nproc)
        antsApplyTransforms \
            -d 3 \
            -e 3 \
            -i {raw_nii} \
            -r {pa_mean} \
            -o {out_nii} \
            -t {warp} \
            -t {affine} --float 1
        """
    ]
    run(cmd)
    return True


def wait_for_file_complete(path: Path, timeout: float = 5.0, interval: float = 0.1) -> bool:
    """
    Wait until file size is stable for two consecutive checks.
    """
    deadline = time.monotonic() + max(0.0, timeout)
    last_size = -1
    stable_hits = 0
    while time.monotonic() < deadline:
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            size = -1
        if size > 0 and size == last_size:
            stable_hits += 1
            if stable_hits >= 2:
                return True
        else:
            stable_hits = 0
        last_size = size
        time.sleep(interval)
    return False




# ---------- Main ----------

def run_rt_pipeline(cfg: RTSessionConfig, score_queue: Optional[object] = None):
    decoder_template = resolve_decoder_template(cfg)
    write_session_metadata(cfg, decoder_template)
    # Process existing DICOMs first (offline-style), but only for this run
    existing = sorted(cfg.incoming_dir.glob("*.dcm"))
    defer_biopac = REGRESSOR_SETTINGS.biopac_start_online_only and bool(existing)
    event_handler = DICOMHandler(
        cfg,
        score_queue=score_queue,
        start_biopac=not defer_biopac,
    )
    observer = Observer()
    observer.schedule(event_handler, str(cfg.incoming_dir), recursive=False)

    observer.start()
    if existing:
        print(f"[RT] Found {len(existing)} existing DICOMs — processing offline first…")
        for f in existing:
            event_handler.enqueue_path(Path(f))
        while event_handler.submit_pending():
            continue

    if defer_biopac:
        event_handler.start_biopac()

    print("[RT] Switching to online mode.")
    try:
        while True:
            while event_handler.submit_pending():
                continue
            time.sleep(0.2)
    except KeyboardInterrupt:
        observer.stop()
    finally:
        event_handler.stop()
    observer.join()

def compute_dvars(prev_vol: np.ndarray, cur_vol: np.ndarray, mask: np.ndarray) -> float:
    diff = (cur_vol - prev_vol)
    if mask is not None:
        diff = diff[mask]
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
):
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    exists = qc_path.exists()
    with open(qc_path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(
                [
                    "volume_idx",
                    "fd",
                    "dvars",
                    "dvars_z",
                    "fd_censor",
                    "dvars_censor",
                    "reg_ready",
                    "biopac_missing",
                ]
            )
        w.writerow([volume_idx, fd, dvars, dvars_z, fd_censor, dvars_censor, int(reg_ready), int(biopac_missing)])


def main():
    parser = argparse.ArgumentParser(description="Real-time fMRI watcher pipeline")
    parser.add_argument("--sub", required=True, help="Subject ID, e.g. 00086")
    parser.add_argument("--day", required=True, help="Day/session, e.g. 3")
    parser.add_argument("--run", required=True, help="Run number, e.g. 4 (matches 000004 in DICOM name)")
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
        "--biopac-enable",
        action="store_true",
        help="Enable BIOPAC RetroTS regressors via TCP.",
    )
    parser.add_argument(
        "--biopac-host",
        default=REGRESSOR_SETTINGS.biopac_host,
        help="Host to bind BIOPAC receiver.",
    )
    parser.add_argument(
        "--biopac-port",
        type=int,
        default=REGRESSOR_SETTINGS.biopac_port,
        help="Port to bind BIOPAC receiver.",
    )
    parser.add_argument(
        "--biopac-timeout",
        type=float,
        default=REGRESSOR_SETTINGS.biopac_timeout,
        help="Seconds to wait for physio regressors before zero-fill.",
    )
    parser.add_argument(
        "--biopac-phys-reg",
        default=REGRESSOR_SETTINGS.biopac_phys_reg,
        choices=["RICOR8", "RVT5", "RVT+RICOR13"],
        help="Physio regressor family to expect from BIOPAC stream.",
    )
    parser.add_argument(
        "--biopac-mode",
        default=REGRESSOR_SETTINGS.biopac_mode,
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
        default=REGRESSOR_SETTINGS.biopac_poll_interval,
        help="Polling interval (seconds) for file-backed BIOPAC buffer.",
    )
    parser.add_argument(
        "--biopac-handshake",
        action="store_true",
        default=REGRESSOR_SETTINGS.biopac_handshake,
        help="Send a handshake with TR to the BIOPAC streamer.",
    )
    parser.add_argument(
        "--biopac-start-online",
        action="store_true",
        default=REGRESSOR_SETTINGS.biopac_start_online_only,
        help="Defer BIOPAC receiver start until after offline DICOM processing.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=REGRESSOR_SETTINGS.max_workers,
        help="Maximum parallel processing workers for DICOM handling.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=REGRESSOR_SETTINGS.max_retries,
        help="Maximum retries per DICOM if processing fails.",
    )
    args = parser.parse_args()

    REGRESSOR_SETTINGS.enable_biopac_physio = args.biopac_enable
    REGRESSOR_SETTINGS.biopac_host = args.biopac_host
    REGRESSOR_SETTINGS.biopac_port = args.biopac_port
    REGRESSOR_SETTINGS.biopac_timeout = args.biopac_timeout
    REGRESSOR_SETTINGS.biopac_phys_reg = args.biopac_phys_reg
    REGRESSOR_SETTINGS.biopac_handshake = args.biopac_handshake
    REGRESSOR_SETTINGS.biopac_start_online_only = args.biopac_start_online
    REGRESSOR_SETTINGS.biopac_mode = args.biopac_mode
    REGRESSOR_SETTINGS.biopac_file = Path(args.biopac_file) if args.biopac_file else None
    REGRESSOR_SETTINGS.biopac_poll_interval = args.biopac_poll
    REGRESSOR_SETTINGS.max_workers = args.max_workers
    REGRESSOR_SETTINGS.max_retries = args.max_retries

    cfg = RTSessionConfig(
        subject=args.sub,
        day=args.day,
        run=args.run,
        incoming_root=Path(args.incoming_root),
        base_data=Path(args.base_data),
        decoder_template=Path(args.decoder_template) if args.decoder_template else None,
    )

    if not cfg.incoming_dir.exists():
        raise FileNotFoundError(f"Incoming directory does not exist: {cfg.incoming_dir}")

    run_rt_pipeline(cfg)


if __name__ == "__main__":
    main()
