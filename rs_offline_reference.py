"""Process recorded DICOMs through normal RS preprocessing, then pool scores.

Example: --sub 55555 --day offline --incoming-root data/sub-55555/dcm
  --settings-file sara_settings_v2.json --decoder-template DECODER.nii.gz
  --run 8 6 7 --run 9 6 7 --run 10 6 7 --run 13 11 12

Each --run is BOLD_SERIES AP_SERIES PA_SERIES. Series choices are explicit;
the script does not infer fieldmaps from acquisition order or folder names.
Outputs use data/sub-SUB/DAY/func/SERIES and func/pooled_reference.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
import re

from fmri_rt_preproc.offline_assets import file_identity
from rt_global_settings import load_regressor_settings

log = logging.getLogger(__name__)
NAME = re.compile(r"(\d+)_(\d+)_(\d+)\.dcm$")
PROCESSING_ERROR = re.compile(r"\bERROR\b|Traceback \(most recent call last\)")


def snapshot(incoming, series):
    """A fixed available recording; contiguity does not prove intended duration."""
    records = {}
    for path in sorted(incoming.glob("*.dcm")):
        match = NAME.fullmatch(path.name)
        if match is None or int(match[2]) != series:
            continue
        index = int(match[3])
        stat = path.stat()
        if index in records or stat.st_size == 0:
            raise ValueError(f"Duplicate/empty DICOM in series {series}: {path}")
        records[index] = dict(path=str(path.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    if not records or set(records) != set(range(1, len(records) + 1)):
        raise ValueError(f"Series {series} must contain a nonempty contiguous DICOM snapshot starting at1")
    return [records[index] for index in sorted(records)]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _worker_settings(settings):
    payload = vars(settings).copy()
    payload["enable_volume_streamer"] = False  # Batch has no live image window.
    return payload


def _run_worker(config, payload, log_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("MPLBACKEND", "Agg")
    with Path(log_path).open("w", encoding="utf-8") as stream, redirect_stdout(stream), redirect_stderr(stream):
        logging.basicConfig(level=logging.INFO, stream=stream, force=True)
        import rt_pipeline as rt
        rt.REGRESSOR_SETTINGS.update(payload)
        rt.run_rt_pipeline(rt.RTSessionConfig(**config))


def run_recordings(args):
    settings = load_regressor_settings(args.settings_file)
    if settings.enable_biopac_physio:
        raise ValueError("Recorded reference processing does not provide live BIOPAC; disable it in JSON")
    if Path(args.day).name != args.day or args.day in {".", ".."}:
        raise ValueError("--day must be one session-folder name")
    if Path(args.pool_name).name != args.pool_name or args.pool_name in {".", ".."}:
        raise ValueError("--pool-name must be one folder name")
    runs = [tuple(value) for value in args.run]
    if any(min(value) < 1 for value in runs) or len({value[0] for value in runs}) != len(runs):
        raise ValueError("BOLD/AP/PA series must be positive; each BOLD series may be selected once")
    incoming = Path(args.incoming_root).expanduser().resolve(strict=True)
    decoder = Path(args.decoder_template).expanduser().resolve(strict=True)
    roi = Path(args.decoder_roi_txt).expanduser().resolve(strict=True) if args.decoder_roi_txt else None
    base_data = Path(args.base_data).expanduser().resolve()
    folders = [base_data / f"sub-{args.sub}" / args.day / "func" / str(run) for run, _, _ in runs]
    pool_folder = folders[0].parent / args.pool_name
    if pool_folder in folders:
        raise ValueError("Pooled reference name collides with a BOLD run")
    for folder in [*folders, pool_folder]:
        if folder.exists():
            raise FileExistsError(f"Use fresh run/reference folders; existing output was preserved: {folder}")
    snapshots = {number: snapshot(incoming, number) for values in runs for number in values}
    if len(snapshots[runs[0][0]]) < 30:
        raise ValueError("The first reference run must include scans11–30 for normal preparation")
    model = dict(decoder=file_identity(decoder), roi_txt=file_identity(roi) if roi else None)
    worker_settings = _worker_settings(settings)
    effective_settings = settings.to_json_dict()
    effective_settings["enable_volume_streamer"] = False

    # Exactly the shared RS preparation call, once for this session's fixed MC
    # reference. Subsequent selected pairs are prepared by ensure_fieldmap_only.
    from rs_realtime_parallel import _run_preproc, _plot_qc
    first, ap, pa = runs[0]
    _run_preproc(args.sub, args.day, base_data, incoming, args.struct_block, ap, pa, first,
        registration_initialization=settings.registration_initialization,
        registration_syn=settings.registration_syn, registration_direction=settings.registration_direction,
        registration_threads=settings.registration_threads,
        registration_bridge_initialization=settings.registration_bridge_initialization,
        offline_assets=settings.offline_assets, registration_qc_policy=settings.registration_qc_policy,
        epi_reference_n4=settings.epi_reference_n4, settings_file=args.settings_file)
    context = mp.get_context("spawn")
    for (run, ap, pa), folder in zip(runs, folders):
        folder.mkdir(parents=True)
        sources = {str(number): snapshots[number] for number in (run, ap, pa)}
        receipt = dict(version=1, status="running", subject=args.sub, run=run,
            expected_volumes=len(snapshots[run]), settings=effective_settings,
            operational_overrides={"enable_volume_streamer": False},
            **model, inputs_snapshot=sources,
            snapshot_scope="all available contiguous recorded files; not proof of intended acquisition duration")
        receipt_path = folder / "recorded_reference.json"
        write_json(receipt_path, receipt)
        config = dict(subject=args.sub, day=args.day, run=str(run), incoming_root=incoming,
            base_data=base_data, decoder_template=decoder, decoder_roi_txt=roi,
            enable_scoring=True, preparation_only=True, recorded_only=True,
            test_speed=args.test_speed, ap_block=ap, pa_block=pa)
        log_path = folder / "processing.log"
        log.info("Processing recorded run %s (%s volumes), AP=%s PA=%s -> %s",
                 run, len(snapshots[run]), ap, pa, folder)
        process = context.Process(target=_run_worker, args=(config, worker_settings, log_path))
        process.start()
        try:
            while process.is_alive():
                process.join(timeout=0.25)
                # A handled regression error can invalidate history without
                # crashing the worker. Stop a failed batch; never pool it.
                if log_path.exists() and PROCESSING_ERROR.search(log_path.read_text(encoding="utf-8", errors="replace")):
                    if process.is_alive():
                        process.terminate()
                    process.join()
                    break
        finally:
            if process.is_alive():
                process.terminate()
                process.join()
        errors = [line for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                  if PROCESSING_ERROR.search(line)]
        unchanged = all(snapshot(incoming, number) == snapshots[number] for number in (run, ap, pa))
        model_unchanged = model == dict(decoder=file_identity(decoder), roi_txt=file_identity(roi) if roi else None)
        receipt.update(status="complete" if process.exitcode == 0 and not errors and unchanged and model_unchanged else "failed",
            worker_exit_code=process.exitcode, processing_error_count=len(errors),
            error_examples=errors[:10], inputs_unchanged=unchanged, decoder_unchanged=model_unchanged)
        write_json(receipt_path, receipt)
        if receipt["status"] != "complete":
            raise RuntimeError(f"Recorded run {run} failed; inspect {log_path}. No pooled reference was created.")
        # Normal RS completion outputs, not a separate diagnostic/QC workflow.
        _plot_qc(folder)
        if args.test_speed:
            from rt_speed import plot_speed
            plot_speed(folder / "speed_test.csv", tr=settings.TR)
        log.info("Recorded run %s finished: %s", run, folder)

    import rt_pipeline as rt
    from fmri_rt_preproc.reference_pool import pool_native_references
    rt.REGRESSOR_SETTINGS.update(vars(settings))
    result = pool_native_references(folders, pool_folder,
        expected_volumes=[len(snapshots[run]) for run, _, _ in runs],
        settings=settings, subject=args.sub, decoder_template=decoder, decoder_roi_txt=roi, runtime=rt)
    print("Use the pooled reference with --rs", pool_folder)
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sub", required=True)
    parser.add_argument("--day", required=True)
    parser.add_argument("--incoming-root", type=Path, required=True)
    parser.add_argument("--base-data", type=Path, default=Path("data"))
    parser.add_argument("--settings-file", type=Path, required=True)
    parser.add_argument("--decoder-template", type=Path, required=True)
    parser.add_argument("--decoder-roi-txt", type=Path)
    parser.add_argument("--run", nargs=3, type=int, action="append", required=True,
                        metavar=("BOLD_SERIES", "AP_SERIES", "PA_SERIES"))
    parser.add_argument("--struct-block", type=int)
    parser.add_argument("--pool-name", default="pooled_reference")
    parser.add_argument("--test-speed", action="store_true")
    return parser


def main(argv=None):
    logging.basicConfig(level=logging.INFO)
    run_recordings(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
