"""Prepare and check the current session EPI bridge before starting NF."""
import argparse
from pathlib import Path

from rt_global_settings import load_regressor_settings, add_registration_arguments, apply_registration_arguments


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sub", required=True)
    parser.add_argument("--day", required=True)
    parser.add_argument("--reference-run", required=True,
                        help="Current session DICOM series containing the reference BOLD (scans 11-30).")
    parser.add_argument("--incoming-root", required=True, type=Path)
    parser.add_argument("--base-data", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--ap-block", type=int)
    parser.add_argument("--pa-block", type=int)
    parser.add_argument("--settings-file")
    add_registration_arguments(parser)
    args = parser.parse_args(argv)
    settings = load_regressor_settings(args.settings_file)
    apply_registration_arguments(settings, args)
    from rt_pipeline import RTSessionConfig, prepare_nf_run
    cfg = RTSessionConfig(subject=args.sub, day=args.day, run=args.reference_run,
                          incoming_root=args.incoming_root, base_data=args.base_data,
                          enable_scoring=False, ap_block=args.ap_block, pa_block=args.pa_block)
    try:
        prepare_nf_run(cfg, settings)
    except (ValueError, RuntimeError) as error:
        marker = cfg.day_root / "func" / "session_trans" / "session_alignment.json"
        if marker.exists():
            print(f"Session assets are saved in {marker.parent}. Inspect registration_qc_*.png/json.")
            print(f'After visual review: python check_registration.py --trans "{marker.parent}" '
                  '--approve --reviewer YOUR_NAME')
        raise SystemExit(str(error)) from error
    print(f"Session registration QC passed: {cfg.trans_dir}")


if __name__ == "__main__":
    main()
