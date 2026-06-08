from __future__ import annotations

import argparse
import json
from pathlib import Path

import rs_pca_runtime as pca_rt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create or verify the permanent blinded PCA condition key for one subject."
    )
    parser.add_argument("--sub", required=True, help="Subject ID, e.g. 00085")
    parser.add_argument(
        "--base-data",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Base data folder containing sub-* folders.",
    )
    parser.add_argument(
        "--roi-labels",
        default="LPFC,Sensorimotor,EVC",
        help="Comma-separated ROI labels for condition mapping.",
    )
    parser.add_argument(
        "--direction-labels",
        default="up,down",
        help="Comma-separated modulation directions.",
    )
    parser.add_argument(
        "--condition-symbols",
        default="A,B,C,D,E,F",
        help="Comma-separated blinded labels shown to the subject.",
    )
    parser.add_argument(
        "--pca-target-pc",
        default="PC01",
        help="PC to modulate for each ROI condition.",
    )
    parser.add_argument(
        "--condition-seed",
        type=int,
        default=None,
        help="Optional seed for the order in which conditions are assigned to runs.",
    )
    parser.add_argument(
        "--symbol-seed",
        type=int,
        default=None,
        help="Optional seed for the hidden A-F to ROI/direction assignment.",
    )
    parser.add_argument(
        "--condition-private-key",
        type=Path,
        default=None,
        help="Private JSON path. Defaults to sub-*/pca_condition_key_private.json.",
    )
    parser.add_argument(
        "--condition-public-schedule",
        type=Path,
        default=None,
        help="Public JSON path. Defaults to sub-*/pca_condition_schedule_public.json.",
    )
    args = parser.parse_args()

    subject_root = args.base_data / f"sub-{args.sub}"
    default_private, default_public = pca_rt.default_condition_paths(subject_root)
    private_path = args.condition_private_key or default_private
    public_path = args.condition_public_schedule or default_public

    schedule = pca_rt.load_or_create_condition_schedule(
        private_path=private_path,
        public_path=public_path,
        roi_labels=pca_rt.parse_csv_list(args.roi_labels),
        direction_labels=pca_rt.parse_csv_list(args.direction_labels),
        symbols=pca_rt.parse_csv_list(args.condition_symbols),
        target_pc=args.pca_target_pc,
        condition_seed=args.condition_seed,
        symbol_seed=args.symbol_seed,
    )

    public_payload = {
        "private_key": str(private_path),
        "public_schedule": str(public_path),
        "created_at": schedule.get("created_at"),
        "schema_version": schedule.get("schema_version"),
        "target_pc": schedule.get("target_pc"),
        "order": schedule.get("order", []),
        "symbols": [c["symbol"] for c in schedule.get("conditions", [])],
    }
    print(json.dumps(public_payload, indent=2))


if __name__ == "__main__":
    main()
