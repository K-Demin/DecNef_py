# fmri_rt_preproc/prep_surface_rois.py

from __future__ import annotations
import argparse
import logging
from pathlib import Path

from fmri_rt_preproc.utils import run, ensure_dir, run_in_conda

log = logging.getLogger("prep_surface_rois")
logging.basicConfig(level=logging.INFO)


def find_t1(anat_dir: Path) -> Path:
    """Prefer T1_N4.nii, fallback to T1.nii / T1.nii.gz."""
    candidates = [
        anat_dir / "T1_N4.nii",
        anat_dir / "T1.nii",
        anat_dir / "T1.nii.gz",
    ]
    for c in candidates:
        if c.exists():
            log.info("Using T1 image: %s", c)
            return c
    raise FileNotFoundError(
        f"No T1 found in {anat_dir}. Tried: "
        + ", ".join(str(c) for c in candidates)
    )


def run_fastsurfer_full(
    t1_path: Path,
    sid: str,
    fs_subjects_dir: Path,
    fs_license: Path,
    threads: str = "max",
    device: str = "cuda",
    conda_env: str = "fastsurfer",
) -> None:
    """Run full FastSurfer (not seg_only) in a dedicated conda env."""
    ensure_dir(fs_subjects_dir)

    subj_dir = fs_subjects_dir / sid
    aseg = subj_dir / "mri" / "aseg.mgz"

    if aseg.exists():
        log.info("FastSurfer already completed for %s – skipping.", sid)
        return

    cmd_str = (
        f"run_fastsurfer.sh "
        f"--t1 {t1_path} "
        f"--sid {sid} "
        f"--sd {fs_subjects_dir} "
        f"--fs_license {fs_license} "
        f"--threads {threads} "
        f"--3T "
        f"--device {device}"
    )
    log.info("Running FastSurfer in conda env '%s': %s", conda_env, cmd_str)
    run_in_conda(conda_env, cmd_str)


def make_dkt_rois(sid: str, fs_subjects_dir: Path) -> None:
    """
    Create atlas-based surface-derived volumetric ROIs from
    aparc.DKTatlas+aseg.deep.mgz:

      - ROI_LPFC
      - ROI_sensorimotor
      - ROI_EVC
    """
    subj_dir = fs_subjects_dir / sid
    mri_dir = subj_dir / "mri"
    dkt_path = mri_dir / "aparc.DKTatlas+aseg.deep.mgz"

    if not dkt_path.exists():
        raise FileNotFoundError(
            f"Expected DKT atlas file not found: {dkt_path}\n"
            "Make sure FastSurfer produced aparc.DKTatlas+aseg.deep.mgz."
        )

    log.info("Creating cortical ROIs from %s", dkt_path)

    # --- LPFC ---
    roi_lpfc_mgz = mri_dir / "ROI_LPFC.mgz"
    if not roi_lpfc_mgz.exists():
        run([
            "mri_binarize",
            "--i", str(dkt_path),
            "--match",
            "1027", "1003", "2027", "2003", "1020", "2020", "1018", "2018",
            "--o", str(roi_lpfc_mgz),
        ])
        log.info("Created %s", roi_lpfc_mgz)

    # --- Sensorimotor (M1 + S1) ---
    roi_sens_mgz = mri_dir / "ROI_sensorimotor.mgz"
    if not roi_sens_mgz.exists():
        run([
            "mri_binarize",
            "--i", str(dkt_path),
            "--match",
            "1024", "2024", "1022", "2022",
            "--o", str(roi_sens_mgz),
        ])
        log.info("Created %s", roi_sens_mgz)

    # --- Early visual cortex (EVC) ---
    roi_evc_mgz = mri_dir / "ROI_EVC.mgz"
    if not roi_evc_mgz.exists():
        run([
            "mri_binarize",
            "--i", str(dkt_path),
            "--match",
            "1021", "2021", "1005", "2005", "1013", "2013",
            "--o", str(roi_evc_mgz),
        ])
        log.info("Created %s", roi_evc_mgz)

    # Also create NIfTI versions (more convenient for ANTs/Python)
    for mgz_path in [roi_lpfc_mgz, roi_sens_mgz, roi_evc_mgz]:
        nii_path = mgz_path.with_suffix(".nii.gz")
        if not nii_path.exists():
            run(["mri_convert", str(mgz_path), str(nii_path)])
            log.info("Converted %s -> %s", mgz_path.name, nii_path.name)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run full FastSurfer and create atlas-based "
            "surface-derived cortical volumetric ROIs (LPFC, sensorimotor, EVC)."
        )
    )
    parser.add_argument(
        "--root", type=Path, required=True,
        help="Root data dir, e.g. /SSD2/DecNef_py/data",
    )
    parser.add_argument(
        "--subj", type=str, required=True,
        help="Subject ID without 'sub-' prefix (e.g. 00085).",
    )
    parser.add_argument(
        "--day", type=str, required=True,
        help="Day/session name (e.g. 2_copy).",
    )
    parser.add_argument(
        "--fs-license", type=Path,
        default=Path("/usr/local/freesurfer/7.4.1/license.txt"),
        help="Path to FreeSurfer license file (default: /usr/local/freesurfer/7.4.1/license.txt).",
    )
    parser.add_argument(
        "--threads", type=str, default="max",
        help='Threads for FastSurfer (default: "max").',
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help='FastSurfer device (default: "cuda").',
    )
    parser.add_argument(
        "--conda-env", type=str, default="fastsurfer",
        help="Conda env with FastSurfer installed (default: fastsurfer).",
    )

    args = parser.parse_args()

    # infer layout: /root/sub-<subj>/anat
    subj_root = args.root / f"sub-{args.subj}"
    anat_dir = subj_root / "anat"
    fs_subjects_dir = anat_dir / "fastsurfer"
    sid = f"{args.subj}_day-{args.day}"

    log.info("Subject root: %s", subj_root)
    log.info("Anat dir    : %s", anat_dir)
    log.info("FastSurfer sd: %s", fs_subjects_dir)
    log.info("FS subject ID: %s", sid)

    t1_path = find_t1(anat_dir)

    # 1) full FastSurfer
    run_fastsurfer_full(
        t1_path=t1_path,
        sid=sid,
        fs_subjects_dir=fs_subjects_dir,
        fs_license=args.fs_license,
        threads=args.threads,
        device=args.device,
        conda_env=args.conda_env,
    )

    # 2) ROIs in T1 space
    make_dkt_rois(
        sid=sid,
        fs_subjects_dir=fs_subjects_dir,
    )

    log.info("Done: FastSurfer + DKT atlas-based cortical ROIs ready.")


if __name__ == "__main__":
    main()
