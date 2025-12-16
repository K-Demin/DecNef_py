# run_preproc.py
import logging
import argparse
from pathlib import Path
import shutil
import time

from fmri_rt_preproc.config import SubjectDayConfig
from fmri_rt_preproc.pipeline import FMRIRealtimePreprocessor
from fmri_rt_preproc.utils import ensure_dir, run  # <- to call fslmerge

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("fmri_rt_preproc")

BASE_DATA = Path(__file__).resolve().parent / "data"


def _convert_dicoms_if_needed(dicom_dir: Path, out_prefix: str) -> list[Path]:
    """Convert DICOMs in ``dicom_dir`` to NIfTI using dcm2niix if present."""

    # Accept both lower- and upper-case extensions so the caller doesn't have to
    # normalize filenames.
    dicoms = sorted(
        f
        for f in dicom_dir.iterdir()
        if f.is_file() and f.suffix.lower() == ".dcm"
    )
    if not dicoms:
        return []

    log.info("No NIfTI files found – converting %d DICOMs in %s", len(dicoms), dicom_dir)

    ensure_dir(dicom_dir)
    run([
        "dcm2niix",
        "-z",
        "y",
        "-b",
        "n",
        "-f",
        out_prefix,
        "-o",
        str(dicom_dir),
        str(dicom_dir),
    ])

    converted = sorted(dicom_dir.glob(f"{out_prefix}*.nii*"))
    log.info("Converted %d NIfTI file(s) from DICOMs", len(converted))
    return converted


def _parse_dicom_name(name: str):
    """
    Parse a Siemens-like DICOM filename: 001_000004_000003.dcm
    Returns (series_id, run_id, scan).
    """

    stem = Path(name).stem
    parts = stem.split("_")
    if len(parts) != 3:
        return None
    try:
        series_id = int(parts[0])
        run_id = int(parts[1])
        scan = int(parts[2])
    except ValueError:
        return None
    return series_id, run_id, scan


def _dicoms_for_block(incoming_dir: Path, block_id: int) -> list[Path]:
    """Collect DICOMs in ``incoming_dir`` matching the given block/run id."""

    if block_id is None:
        return []

    dicoms: list[Path] = []
    for f in sorted(incoming_dir.iterdir()):
        if not f.is_file() or f.suffix.lower() != ".dcm":
            continue
        parsed = _parse_dicom_name(f.name)
        if parsed is None:
            continue
        _, run_id, _ = parsed
        if run_id == block_id:
            dicoms.append(f)
    return dicoms


def _stage_convert_to_target(
    incoming_dir: Path, block_id: int, dest_path: Path
) -> Path:
    """Stage a specific block's DICOMs and convert them to a fixed NIfTI name."""

    ensure_dir(dest_path.parent)

    if dest_path.exists():
        log.info("Found existing file %s; skipping conversion", dest_path)
        return dest_path

    dicoms = _dicoms_for_block(incoming_dir, block_id)
    if not dicoms:
        raise FileNotFoundError(
            f"No DICOMs found in {incoming_dir} for block/run {block_id}."
        )

    log.info(
        "Staging %d DICOM(s) for block %s into %s", len(dicoms), block_id, dest_path.parent
    )
    for src in dicoms:
        dst = dest_path.parent / src.name
        if not dst.exists():
            shutil.copy2(src, dst)

    dest_prefix = dest_path.name.split(".")[0]
    converted = _convert_dicoms_if_needed(dest_path.parent, dest_prefix)
    if not converted:
        raise FileNotFoundError(
            f"Failed to convert staged DICOMs for block/run {block_id} in {dest_path.parent}"
        )

    # Normalize the converted file to the expected name (e.g., AP.nii.gz)
    chosen = converted[0]
    if chosen != dest_path:
        if dest_path.exists():
            dest_path.unlink()
        chosen.rename(dest_path)

    return dest_path


def _find_structural(anat_dir: Path) -> Path:
    """Locate the first structural (T1) scan, converting DICOMs if necessary."""

    ensure_dir(anat_dir)

    candidates = sorted(anat_dir.glob("T1*.nii*"))
    if not candidates:
        converted = _convert_dicoms_if_needed(anat_dir, "T1")
        candidates = sorted(converted)

    if not candidates:
        raise FileNotFoundError(
            f"No structural scan found in {anat_dir}. "
            "Expected T1*.nii* or DICOM files."
        )

    log.info("Using structural scan: %s", candidates[0])
    return candidates[0]


def _find_fieldmap(fmap_dir: Path, prefix: str) -> Path:
    """Locate AP/PA fieldmap, converting DICOMs if necessary."""

    ensure_dir(fmap_dir)
    candidates = sorted(fmap_dir.glob(f"{prefix}*.nii*"))
    if not candidates:
        converted = _convert_dicoms_if_needed(fmap_dir, prefix)
        candidates = sorted(converted)

    if not candidates:
        raise FileNotFoundError(
            f"No {prefix} fieldmap found in {fmap_dir}. "
            f"Expected {prefix}*.nii* or DICOM files."
        )

    log.info("Using %s fieldmap: %s", prefix, candidates[0])
    return candidates[0]


def main():
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", required=True, help="Subject ID, e.g. 0001")
    parser.add_argument("--day", required=True, help="Session/day label, e.g. 3")
    parser.add_argument(
        "--epi-pattern",
        default="*.nii*",
        help="Glob pattern for raw EPI files inside the run dir (default: '*.nii*').",
    )
    parser.add_argument(
        "--incoming-root",
        type=Path,
        help="Optional folder containing raw DICOMs to stage/convert (Siemens 001_XXX_YYY.dcm naming).",
    )
    parser.add_argument(
        "--struct-block",
        type=int,
        help="Block/run id for the structural (UNI/T1) DICOMs inside incoming-root (required when using --incoming-root).",
    )
    parser.add_argument(
        "--ap-block",
        type=int,
        help="Block/run id for the AP fieldmap DICOMs inside incoming-root (required when using --incoming-root).",
    )
    parser.add_argument(
        "--pa-block",
        type=int,
        help="Block/run id for the PA fieldmap DICOMs inside incoming-root (required when using --incoming-root).",
    )
    parser.add_argument(
        "--no-save-config",
        action="store_true",
        help="Do not write config.json (use only in-memory config).",
    )
    args = parser.parse_args()

    # data/sub-0001/3
    day_root = BASE_DATA / f"sub-{args.sub}" / args.day
    func_root = day_root / "func"
    run_dir = func_root / "trans"   # we treat 'trans' as the single run folder

    if args.incoming_root:
        incoming_dir = args.incoming_root
        if not incoming_dir.exists():
            raise FileNotFoundError(f"Incoming directory does not exist: {incoming_dir}")

        if None in (args.struct_block, args.ap_block, args.pa_block):
            raise ValueError(
                "When providing --incoming-root you must also set --struct-block, "
                "--ap-block, and --pa-block so the correct DICOM runs can be staged."
            )

        anat_dir = day_root.parent / "anat"
        fmap_dir = day_root / "fmap"

        _stage_convert_to_target(incoming_dir, args.struct_block, anat_dir / "T1.nii.gz")
        _stage_convert_to_target(incoming_dir, args.ap_block, fmap_dir / "AP.nii.gz")
        _stage_convert_to_target(incoming_dir, args.pa_block, fmap_dir / "PA.nii.gz")

    # ------------------------------------------------------------------
    # 1) Collect ALL EPI files in this run and merge into a single 4D file
    # ------------------------------------------------------------------
    epi_files = sorted(run_dir.glob(args.epi_pattern))
    if not epi_files:
        epi_files = _convert_dicoms_if_needed(run_dir, "epi")

    if not epi_files:
        raise FileNotFoundError(
            f"No EPI files found in {run_dir} matching pattern '{args.epi_pattern}' "
            "or convertible DICOMs."
        )

    log.info(f"Found {len(epi_files)} EPI files in {run_dir}")
    for f in epi_files:
        log.info(f"  EPI: {f.name}")

    merged_epi = run_dir / "epi_4d.nii.gz"
    if not merged_epi.exists():
        log.info(f"Merging EPI files into 4D: {merged_epi}")
        run(["fslmerge", "-t", str(merged_epi)] + [str(f) for f in epi_files])
    else:
        log.info(f"Using existing merged EPI: {merged_epi}")

    # ------------------------------------------------------------------
    # 2) Build a config for this single run
    # ------------------------------------------------------------------
    # NOTE: we don't have args.run anymore; just give a fixed run_id
    t1_path = _find_structural(day_root.parent / "anat")
    fmap_dir = day_root / "fmap"
    ap_path = _find_fieldmap(fmap_dir, "AP")
    pa_path = _find_fieldmap(fmap_dir, "PA")

    cfg = SubjectDayConfig.for_single_run(
        subject_id=args.sub,
        day_id=args.day,
        root=day_root,
        run_id="trans",      # arbitrary ID for this synthetic run
        epi_file=merged_epi,
        t1_file=t1_path,
        ap_file=ap_path,
        pa_file=pa_path,
    )

    # Optionally save config.json so you can re-use it later
    config_path = day_root / "config.json"
    if not args.no_save_config:
        cfg.to_json(config_path)
        log.info(f"Saved config to {config_path}")

    # ------------------------------------------------------------------
    # 3) Run the pipeline (anat + fmap + this one run)
    # ------------------------------------------------------------------
    pipe = FMRIRealtimePreprocessor(cfg)
    pipe.run_all()

    log.info("Preprocessing finished.")
    end = time.time()
    print("Elapsed:", end - start, "seconds")

    # QC plot using decoder (unchanged)
    if cfg.runs:
        run_cfg = cfg.runs[0]
        run_dir = run_cfg.epi_file.parent
        decoder_path = Path("/SSD2/DecNef_py/decoders/rweights_NSF_grouppred_cvpcrTMP_nonzeros.nii")
        if decoder_path.exists():
            pipe.qc_plot_epi_in_mni(run_dir, decoder_path=decoder_path)


if __name__ == "__main__":
    main()
