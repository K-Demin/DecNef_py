#!/usr/bin/env python
import time
import csv
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass

import nibabel as nib
import numpy as np

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from fmri_rt_preproc.RTPSpy_tools.rtp_volreg import RtpVolreg
from fmri_rt_preproc.RTPSpy_tools.rtp_regress import RtpRegress
from fmri_rt_preproc.utils import run  # your existing run() wrapper

from decoder_score import DecoderScorer

log = logging.getLogger("rt_pipeline")
logging.basicConfig(level=logging.INFO)

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

def log_step(step: str, vol: int, extra: str = "", start_t=None):
    """Compact colored/clean log."""
    v = f"{vol:05d}"
    if start_t is not None:
        dt = time.time() - start_t
        log.info(f"[{step:<5}] vol {v}  {extra}  ({dt*1000:.1f} ms)")
    else:
        log.info(f"[{step:<5}] vol {v}  {extra}")


def append_score(csv_path: Path, volume_idx: int, raw_score: float, z_score: float):
    timestamp = time.time()
    exists = csv_path.exists()

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["volume_idx", "timestamp", "score_raw", "score_z"])
        writer.writerow([volume_idx, timestamp, raw_score, z_score])

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


class MotionRegressor:
    """Wrapper around RTPSpy's regression module for online motion denoising."""

    def __init__(self, volreg: RtpVolreg):
        self._regress = RtpRegress(
            mot_reg="mot6",
            volreg=volreg,
            wait_num=0,  # regress from the first usable volume
            save_proc=False,
            online_saving=False,
        )
        self._ready = False

    def apply(self, mc_img: nib.Nifti1Image, volume_idx: int) -> np.ndarray:
        """Run motion regression using the cumulative RTPSpy pipeline."""

        if not self._ready:
            try:
                self._ready = bool(self._regress.ready_proc())
            except Exception as exc:  # pragma: no cover - safety guard
                log.error(f"[REG] Failed to prepare regressor: {exc}")
                return np.asanyarray(mc_img.dataobj)

        if not self._ready:
            return np.asanyarray(mc_img.dataobj)

        try:
            self._regress.do_proc(mc_img, vol_idx=volume_idx - 1)
            if self._regress.proc_data is None:
                return np.asanyarray(mc_img.dataobj)
            return np.asarray(self._regress.proc_data, dtype=np.float32)
        except Exception as exc:  # pragma: no cover - runtime safeguard
            log.error(f"[REG] Motion regression failed at vol {volume_idx:05d}: {exc}")
            return np.asanyarray(mc_img.dataobj)


# ---------- Simple config for this RT session ----------

@dataclass
class RTSessionConfig:
    subject: str
    day: str
    run: str
    incoming_root: Path
    base_data: Path

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
        return self.day_root / "func" / "rt_ref_epi.nii"

    @property
    def rt_ref_mask(self) -> Path:
        """
        Optional mask for the RT reference (not strictly needed here,
        but kept for completeness / future use).
        """
        return self.day_root / "func" / "rt_ref_epi_mask.nii"


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
    def __init__(self, cfg: RTSessionConfig):
        super().__init__()
        self.cfg = cfg
        self.current_run = int(cfg.run)
        self.next_volume_idx = 1

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
        self.motion_regressor = MotionRegressor(self.volreg)

        # --- Decoder / scorer ---
        decoder_path = Path(cfg.base_data).parent / "decoders" / "rweights_NSF_grouppred_cvpcrTMP_nonzeros.nii"
        roi_txt = cfg.trans_dir / "ROI_DECODER.txt"

        self.scorer = DecoderScorer(
            decoder_path,
            roi_txt=roi_txt,
            n_baseline=20,   # keep your current baseline length
        )

    def on_created(self, event):
        if event.is_directory:
            return

        path = Path(event.src_path)
        log.info(f"[WATCHDOG] File detected: {path}")
        self.process_file(path)

    def process_file(self, path: Path):
        parsed = parse_dicom_name(path.name)
        if parsed is None:
            log.debug(f"[WATCHDOG] Ignoring non-matching file: {path.name}")
            return

        series_id, run_id, scan = parsed
        if run_id != self.current_run:
            log.debug(f"[WATCHDOG] Ignoring run {run_id}, expecting {self.current_run}")
            return

        volume_idx = self.next_volume_idx
        self.next_volume_idx += 1

        log.info(f"[WATCHDOG] Processing volume idx {volume_idx} (run={run_id}, scan={scan})")
        process_volume(self.cfg, self, path, volume_idx)


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
        if not produced:
            log.error(f"[DICOM] vol {volume_idx:05d} FAILED (no output)")
            return

        raw_nii = produced[0]

    log_step("DICOM", volume_idx, start_t=t0)

    # ---------- 1.5) APPLY FIELD MAP UNWARP BEFORE MC ----------
    t0 = time.time()
    unwarp_dir = cfg.rt_unwarp_dir
    unwarped_nii = unwarp_dir / f"vol_{volume_idx:05d}_uw.nii"

    if not unwarped_nii.exists():
        ok = unwarp_volume(raw_nii, unwarped_nii, cfg)
        if not ok:
            log.error(f"[FMAP] Failed unwarp for {raw_nii}")
            return
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

    # ---------- 2c) Motion regression (RTPS_py) ----------
    reg_t0 = time.time()
    mc_for_warp = mc_nii
    cleaned = handler.motion_regressor.apply(mc_img, volume_idx)
    if not np.may_share_memory(cleaned, mc_data):
        reg_nii = mc_dir / f"vol_{volume_idx:05d}_mc_reg.nii"
        nib.save(nib.Nifti1Image(cleaned, img.affine), str(reg_nii))
        mc_for_warp = reg_nii
        log_step("REG", volume_idx, "motion", start_t=reg_t0)

    # ---------- 3) Apply ANTs transforms to MNI ----------
    t0 = time.time()
    mni_dir = cfg.rt_mni_dir
    mni_nii = mni_dir / f"vol_{volume_idx:05d}_mni.nii"

    warp_t1_mni = cfg.subject_root / "anat" / "warp_T1_to_MNI_synth.nii"
    epi2t1 = cfg.trans_dir / "epi2t1_Composite.h5"
    decoder_template = Path(cfg.base_data).parent / "decoders" / "rweights_NSF_grouppred_cvpcrTMP_nonzeros.nii"

    if not decoder_template.exists():
        log.error(f"Decoder template not found at {decoder_template}")
        return

    if not (warp_t1_mni.exists() and epi2t1.exists()):
        log.error(f"Missing transforms in {cfg.trans_dir}")
        return

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
        # Load the warped volume (decoder space) once
        mni_img = nib.load(str(mni_nii))
        mni_data = np.asanyarray(mni_img.dataobj)

        # First N volumes → baseline accumulation
        if handler.scorer.baseline_count < handler.scorer.n_baseline:
            handler.scorer.accumulate_baseline(mni_data)
            if handler.scorer.baseline_count == handler.scorer.n_baseline:
                handler.scorer.finalize_baseline()

        raw_score, z_score = handler.scorer.score_from_array(mni_data, use_z=True)
        append_score(cfg.rt_work_dir / "scores.csv", volume_idx, raw_score, z_score)
        log_step("SCORE", volume_idx, f"raw={raw_score:.4f} z={z_score:.4f}", start_t=t0)

    except Exception as e:
        log.error(f"[SCORE] Failed scoring vol {volume_idx:05d}: {e}")





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




# ---------- Main ----------

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
    args = parser.parse_args()

    cfg = RTSessionConfig(
        subject=args.sub,
        day=args.day,
        run=args.run,
        incoming_root=Path(args.incoming_root),
        base_data=Path(args.base_data),
    )

    if not cfg.incoming_dir.exists():
        raise FileNotFoundError(f"Incoming directory does not exist: {cfg.incoming_dir}")

    event_handler = DICOMHandler(cfg)
    observer = Observer()
    observer.schedule(event_handler, str(cfg.incoming_dir), recursive=False)

    # Process existing DICOMs first (offline-style), but only for this run
    existing = sorted(cfg.incoming_dir.glob("*.dcm"))
    if existing:
        print(f"[RT] Found {len(existing)} existing DICOMs — processing offline first…")
        for f in existing:
            event_handler.process_file(Path(f))

    print("[RT] Switching to online mode.")
    observer.start()

    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
