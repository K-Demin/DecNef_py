"""Check an existing reference preparation before starting NF.

Run standard RS or the behavioral reference first. This compatibility command
does not collect EPIs, fit transforms or create a replacement session reference.
"""
import argparse
from pathlib import Path

from rt_global_settings import load_regressor_settings, add_registration_arguments, apply_registration_arguments


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sub", required=True)
    parser.add_argument("--day", required=True)
    parser.add_argument("--reference-run", required=True,
                        help="Reference run ID (retained for command compatibility; spatial files come from DAY/func/trans).")
    parser.add_argument("--incoming-root", required=True, type=Path)
    parser.add_argument("--base-data", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--ap-block", type=int)
    parser.add_argument("--pa-block", type=int)
    parser.add_argument("--settings-file")
    add_registration_arguments(parser)
    args = parser.parse_args(argv)
    settings = load_regressor_settings(args.settings_file)
    apply_registration_arguments(settings, args)
    from rt_pipeline import RTSessionConfig, validate_nf_reference
    cfg = RTSessionConfig(subject=args.sub, day=args.day, run=args.reference_run,
                          incoming_root=args.incoming_root, base_data=args.base_data,
                          enable_scoring=False, ap_block=args.ap_block, pa_block=args.pa_block)
    try:
        validate_nf_reference(cfg, settings)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from error
    print(f"Prepared reference files are ready: {cfg.trans_dir}")


if __name__ == "__main__":
    main()
